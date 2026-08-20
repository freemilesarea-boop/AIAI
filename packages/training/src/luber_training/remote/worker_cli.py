"""The worker's command line — what the control plane actually invokes.

    python -m luber_training.remote --root /opt/luber/worker <command>

Every command prints one JSON envelope on stdout and exits. That is the
whole protocol: no daemon, no port, no persistent connection. The
control plane runs this over SSH, reads a line of JSON, and moves on;
the trainer it launched keeps running regardless.

Diagnostics go to stderr, so stdout is always parseable. A worker that
printed a warning into its own reply would break every caller, and the
callers are on a different machine.

Nothing here decides anything about legitimacy. The verbs are received,
verify, run, report — and where the worker cannot establish a fact, it
reports that it could not rather than choosing a plausible answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from luber_training.remote.execution import ExecutionState
from luber_training.remote.manifest import RemoteArtifactManifest
from luber_training.remote.paths import RemoteRoots
from luber_training.remote.protocol import REMOTE_PROTOCOL_VERSION, Envelope, RemoteCommand
from luber_training.remote.secrets import redact
from luber_training.remote.streams import MetricStream
from luber_training.remote.worker import RemoteWorker, WorkerError


def _emit(envelope: Envelope) -> int:
    """One JSON object on stdout, and nothing else, ever."""
    print(json.dumps(envelope.to_dict(), ensure_ascii=False, sort_keys=True, default=str))
    return 0 if envelope.ok else 1


def _worker(args: argparse.Namespace) -> RemoteWorker:
    return RemoteWorker(Path(args.root).expanduser())


def _worker_id(worker: RemoteWorker) -> str | None:
    try:
        identity, _ = worker.identity()
        return identity.worker_id
    except Exception:
        return None


# ── commands ─────────────────────────────────────────────────────────


def cmd_init(args: argparse.Namespace) -> int:
    worker = _worker(args)
    roots = RemoteRoots.under(args.base) if args.base else None
    config = worker.initialise(
        worker_name=args.name,
        roots=roots,
        trainer_root=args.trainer_root,
        repository_root=args.repository_root,
    )
    identity, _ = worker.identity()
    return _emit(
        Envelope(
            ok=True,
            command="init",
            worker_id=identity.worker_id,
            payload={"config": config.to_dict(), "identity": identity.to_dict()},
        )
    )


def cmd_identity(args: argparse.Namespace) -> int:
    worker = _worker(args)
    identity, changed = worker.identity()
    return _emit(
        Envelope(
            ok=True,
            command=RemoteCommand.IDENTITY.value,
            worker_id=identity.worker_id,
            payload={"identity": identity.to_dict(), "host_fingerprint_changed": changed},
        )
    )


def cmd_probe(args: argparse.Namespace) -> int:
    worker = _worker(args)
    payload = worker.describe()
    if args.output:
        destination = Path(args.output).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        payload = {**payload, "written_to": str(destination)}
    return _emit(
        Envelope(
            ok=True,
            command=RemoteCommand.PROBE.value,
            worker_id=payload["identity"]["worker_id"],
            payload=payload,
        )
    )


def cmd_heartbeat(args: argparse.Namespace) -> int:
    worker = _worker(args)
    beat = worker.heartbeat()
    return _emit(
        Envelope(
            ok=True,
            command=RemoteCommand.HEARTBEAT.value,
            worker_id=beat.worker_id,
            payload=beat.to_dict(),
        )
    )


def cmd_prepare(args: argparse.Namespace) -> int:
    worker = _worker(args)
    state = worker.prepare(
        run_id=args.run_id,
        plan_sha256=args.plan_sha256,
        manifest_sha256=args.manifest_sha256,
    )
    lease = worker.lease(args.run_id)
    return _emit(
        Envelope(
            ok=True,
            command=RemoteCommand.PREPARE.value,
            worker_id=_worker_id(worker),
            payload={
                "state": state.to_dict(),
                "lease": lease.to_dict() if lease else None,
                "run_root": str(worker.layout(args.run_id).root),
            },
        )
    )


def cmd_receive(args: argparse.Namespace) -> int:
    """Record the manifest that describes what was just transferred.

    Separate from the transfer itself: bytes arrive by whatever
    transport the operator configured, and this is the worker being told
    what those bytes were supposed to be. Verification happens at
    preflight, against this record.
    """
    worker = _worker(args)
    layout = worker.layout(args.run_id)
    layout.ensure()
    manifest = RemoteArtifactManifest.read(Path(args.manifest).expanduser())
    if manifest.run_id != args.run_id:
        return _emit(
            Envelope(
                ok=False,
                command=RemoteCommand.RECEIVE.value,
                worker_id=_worker_id(worker),
                error=(
                    f"the manifest describes run {manifest.run_id}, not {args.run_id}; "
                    "refusing to associate one run's artifacts with another"
                ),
            )
        )
    manifest.write(layout.manifest_json)

    state = ExecutionState.read(layout)
    if state is not None:
        state.manifest_sha256 = manifest.digest()
        state.detail = f"{len(manifest.entries)} artifact(s) recorded"
        state.write(layout)

    return _emit(
        Envelope(
            ok=True,
            command=RemoteCommand.RECEIVE.value,
            worker_id=_worker_id(worker),
            payload={
                "manifest_sha256": manifest.digest(),
                "entries": len(manifest.entries),
                "total_bytes": manifest.total_bytes,
            },
        )
    )


def cmd_preflight(args: argparse.Namespace) -> int:
    worker = _worker(args)
    report = worker.preflight(
        args.run_id,
        minimum_free_disk_mb=args.minimum_free_disk_mb,
        require_code_match=not args.allow_code_mismatch,
    )
    return _emit(
        Envelope(
            ok=report.passed,
            command=RemoteCommand.PREFLIGHT.value,
            worker_id=report.worker_id,
            payload=report.to_dict(),
            error=None if report.passed else "; ".join(report.blocking_reasons),
        )
    )


def cmd_start(args: argparse.Namespace) -> int:
    worker = _worker(args)
    state = worker.start(args.run_id)
    return _emit(
        Envelope(
            ok=True,
            command=RemoteCommand.START.value,
            worker_id=_worker_id(worker),
            payload=state.to_dict(),
        )
    )


def cmd_status(args: argparse.Namespace) -> int:
    worker = _worker(args)
    state = worker.status(args.run_id)
    return _emit(
        Envelope(
            ok=True,
            command=RemoteCommand.STATUS.value,
            worker_id=_worker_id(worker),
            payload=state.to_dict(),
        )
    )


def cmd_logs(args: argparse.Namespace) -> int:
    worker = _worker(args)
    chunk = worker.logs(args.run_id, stream=args.stream, offset=args.offset, limit=args.limit)
    payload = chunk.to_dict()
    payload["text"] = redact(payload["text"])
    return _emit(
        Envelope(
            ok=True,
            command=RemoteCommand.LOGS.value,
            worker_id=_worker_id(worker),
            payload=payload,
        )
    )


def cmd_metrics(args: argparse.Namespace) -> int:
    worker = _worker(args)
    stream = MetricStream(line_cursor=args.cursor)
    events = stream.read(worker.metrics_path(args.run_id), limit=args.limit)
    return _emit(
        Envelope(
            ok=True,
            command=RemoteCommand.METRICS.value,
            worker_id=_worker_id(worker),
            payload={
                "events": [event.to_dict() for event in events],
                "next_cursor": stream.line_cursor,
                "count": len(events),
            },
        )
    )


def cmd_cancel(args: argparse.Namespace) -> int:
    worker = _worker(args)
    state = worker.cancel(args.run_id, grace_seconds=args.grace_seconds)
    return _emit(
        Envelope(
            ok=True,
            command=RemoteCommand.CANCEL.value,
            worker_id=_worker_id(worker),
            payload=state.to_dict(),
        )
    )


def cmd_checkpoints(args: argparse.Namespace) -> int:
    worker = _worker(args)
    return _emit(
        Envelope(
            ok=True,
            command=RemoteCommand.CHECKPOINTS.value,
            worker_id=_worker_id(worker),
            payload={"checkpoints": worker.checkpoints(args.run_id)},
        )
    )


def cmd_result(args: argparse.Namespace) -> int:
    worker = _worker(args)
    result = worker.result(args.run_id)
    return _emit(
        Envelope(
            ok=True,
            command=RemoteCommand.RESULT.value,
            worker_id=result.worker_id,
            payload=result.to_dict(),
        )
    )


def cmd_cleanup(args: argparse.Namespace) -> int:
    worker = _worker(args)
    return _emit(
        Envelope(
            ok=True,
            command=RemoteCommand.CLEANUP.value,
            worker_id=_worker_id(worker),
            payload=worker.cleanup(args.run_id, remove_dataset=args.remove_dataset),
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m luber_training.remote",
        description=(
            "The LUBER remote training worker. Receives approved artifacts, verifies them, "
            "runs the trainer it was given, and reports. Decides nothing."
        ),
    )
    parser.add_argument("--root", default="./luber-worker", help="this worker's root directory")
    parser.add_argument(
        "--protocol-version",
        default=REMOTE_PROTOCOL_VERSION,
        help="the control plane's protocol version",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="initialise this machine as a worker")
    init.add_argument("--name", required=True)
    init.add_argument("--base", help="base directory for the conventional root layout")
    init.add_argument("--trainer-root", help="where ACE-Step is installed")
    init.add_argument("--repository-root", help="where LUBER is checked out")
    init.set_defaults(func=cmd_init)

    sub.add_parser(RemoteCommand.IDENTITY.value, help="this worker's stable identity").set_defaults(
        func=cmd_identity
    )

    probe = sub.add_parser(RemoteCommand.PROBE.value, help="measure this machine")
    probe.add_argument("--output", help="also write the report to a file")
    probe.set_defaults(func=cmd_probe)

    sub.add_parser(RemoteCommand.HEARTBEAT.value, help="report life").set_defaults(
        func=cmd_heartbeat
    )

    prepare = sub.add_parser(RemoteCommand.PREPARE.value, help="claim a run and make room")
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--plan-sha256", required=True)
    prepare.add_argument("--manifest-sha256", required=True)
    prepare.set_defaults(func=cmd_prepare)

    receive = sub.add_parser(RemoteCommand.RECEIVE.value, help="record a transferred manifest")
    receive.add_argument("--run-id", required=True)
    receive.add_argument("--manifest", required=True)
    receive.set_defaults(func=cmd_receive)

    preflight = sub.add_parser(RemoteCommand.PREFLIGHT.value, help="verify before training")
    preflight.add_argument("--run-id", required=True)
    preflight.add_argument("--minimum-free-disk-mb", type=int)
    preflight.add_argument(
        "--allow-code-mismatch",
        action="store_true",
        help="do not require the worker's LUBER commit to match the dispatch",
    )
    preflight.set_defaults(func=cmd_preflight)

    start = sub.add_parser(RemoteCommand.START.value, help="launch the trainer")
    start.add_argument("--run-id", required=True)
    start.set_defaults(func=cmd_start)

    status = sub.add_parser(RemoteCommand.STATUS.value, help="execution state")
    status.add_argument("--run-id", required=True)
    status.set_defaults(func=cmd_status)

    logs = sub.add_parser(RemoteCommand.LOGS.value, help="incremental log read")
    logs.add_argument("--run-id", required=True)
    logs.add_argument("--stream", default="stdout", choices=["stdout", "stderr"])
    logs.add_argument("--offset", type=int, default=0)
    logs.add_argument("--limit", type=int, default=262_144)
    logs.set_defaults(func=cmd_logs)

    metrics = sub.add_parser(RemoteCommand.METRICS.value, help="incremental metric read")
    metrics.add_argument("--run-id", required=True)
    metrics.add_argument("--cursor", type=int, default=0)
    metrics.add_argument("--limit", type=int, default=2000)
    metrics.set_defaults(func=cmd_metrics)

    cancel = sub.add_parser(RemoteCommand.CANCEL.value, help="stop the trainer gracefully")
    cancel.add_argument("--run-id", required=True)
    cancel.add_argument("--grace-seconds", type=float, default=60.0)
    cancel.set_defaults(func=cmd_cancel)

    checkpoints = sub.add_parser(RemoteCommand.CHECKPOINTS.value, help="what the trainer wrote")
    checkpoints.add_argument("--run-id", required=True)
    checkpoints.set_defaults(func=cmd_checkpoints)

    result = sub.add_parser(RemoteCommand.RESULT.value, help="the run's result manifest")
    result.add_argument("--run-id", required=True)
    result.set_defaults(func=cmd_result)

    cleanup = sub.add_parser(RemoteCommand.CLEANUP.value, help="remove scratch, never evidence")
    cleanup.add_argument("--run-id", required=True)
    cleanup.add_argument(
        "--remove-dataset",
        action="store_true",
        help="also remove the transferred dataset (only after the run has finished)",
    )
    cleanup.set_defaults(func=cmd_cleanup)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from luber_training.remote.protocol import check_protocol

        check_protocol(args.protocol_version, peer="control plane")
        return int(args.func(args))
    except WorkerError as exc:
        return _emit(Envelope(ok=False, command=args.command, error=redact(str(exc))))
    except Exception as exc:
        # The caller is parsing stdout on another machine. A traceback
        # there would be an unparseable reply, so the traceback goes to
        # stderr and the reply stays well-formed.
        import traceback

        traceback.print_exc(file=sys.stderr)
        return _emit(
            Envelope(
                ok=False,
                command=args.command,
                error=redact(f"{type(exc).__name__}: {exc}"),
            )
        )


if __name__ == "__main__":
    raise SystemExit(main())
