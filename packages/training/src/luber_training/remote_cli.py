"""The control-plane side of remote execution, as operator commands.

    python -m luber_training remote <command>

Split into its own module rather than growing `cli.py`, because these
verbs move data to a machine somebody else owns and that boundary is
worth being able to read in one place.

Operator-only, like everything else in this package. There is no HTTP
surface here, no role, and no path by which a LUBER account reaches any
of it: SSH configuration, worker host details, checkpoint paths and
dataset transfer controls exist only in this program.

The order of the verbs is the order an operator uses them, and each one
refuses rather than improvises when its preconditions are not met.
`stage` will not assemble a transfer whose rights gate fails.
`dispatch` will not launch when it cannot establish that nothing is
already running. `collect` will not register a checkpoint whose bytes do
not hash to what the worker reported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from luber_training.entities import CheckpointKind, FailureCode, RunStatus, TrainingWorker
from luber_training.orchestrator import Orchestrator
from luber_training.registry import Registry
from luber_training.remote.backend import RemoteGpuBackend, failure_code_for
from luber_training.remote.capabilities import to_worker_class
from luber_training.remote.client import (
    LocalWorkerClient,
    RemoteWorkerClient,
    SshWorkerClient,
    WorkerEndpoint,
    WorkerUnreachable,
)
from luber_training.remote.collect import (
    CollectionError,
    collect_run,
    plan_remote_retention,
    register_collected,
)
from luber_training.remote.identity import LivenessPolicy
from luber_training.remote.manifest import RemoteArtifactManifest, disk_requirement
from luber_training.remote.protocol import REMOTE_PROTOCOL_VERSION, ReconcileOutcome
from luber_training.remote.result import RemoteResult
from luber_training.remote.secrets import (
    EnvironmentSecretResolver,
    FileSecretResolver,
    NullSecretResolver,
    SecretResolver,
    redact,
)
from luber_training.remote.ssh import SshArtifactTransport
from luber_training.remote.staging import (
    StagingError,
    StagingInputs,
    build_staging,
    verify_staging,
)
from luber_training.remote.streams import merge_into
from luber_training.remote.transport import ArtifactTransport, LocalArtifactTransport

#: Where staged runs and collected checkpoints live by default. Both are
#: gitignored; both hold copies of licensed audio and trained weights.
DEFAULT_STAGING_ROOT = "./remote_staging"
DEFAULT_COLLECT_ROOT = "./collected_checkpoints"


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str))


def _fail(message: str) -> int:
    print(json.dumps({"ok": False, "error": redact(message)}, indent=2, ensure_ascii=False))
    return 1


def _orchestrator(args: argparse.Namespace) -> Orchestrator:
    return Orchestrator(
        Registry(Path(args.registry).expanduser()),
        repository_root=Path(args.repository).expanduser() if args.repository else Path.cwd(),
    )


def _secrets(args: argparse.Namespace) -> SecretResolver:
    """The resolver an operator configured, or none at all.

    `NullSecretResolver` is the default rather than a guess: a component
    that unexpectedly asks for a credential should fail loudly instead
    of finding one somewhere nobody intended.
    """
    if getattr(args, "secret_dir", None):
        return FileSecretResolver(
            Path(args.secret_dir).expanduser(),
            repository_root=Path(args.repository).expanduser() if args.repository else Path.cwd(),
        )
    if getattr(args, "secrets_from_env", False):
        return EnvironmentSecretResolver()
    return NullSecretResolver()


def _endpoint(args: argparse.Namespace) -> WorkerEndpoint:
    return WorkerEndpoint(
        worker_root=args.worker_root,
        python_executable=args.remote_python,
        host=getattr(args, "host", None),
        user=getattr(args, "user", None),
        port=getattr(args, "port", None),
        ssh_key_ref=getattr(args, "ssh_key_ref", None),
        known_hosts_ref=getattr(args, "known_hosts_ref", None),
    )


def _client(args: argparse.Namespace) -> RemoteWorkerClient:
    if args.transport == "local":
        return LocalWorkerClient(Path(args.worker_root).expanduser())
    return SshWorkerClient(_endpoint(args), secrets=_secrets(args))


def _transport(args: argparse.Namespace) -> ArtifactTransport:
    """A transport rooted at the worker's *run root*.

    Rooting it there rather than at the filesystem root is what makes
    every transfer path `<run_id>/...`, so one run cannot address
    another's directory even if a manifest asked it to.
    """
    if args.transport == "local":
        from luber_training.remote.worker import RemoteWorker

        worker = RemoteWorker(Path(args.worker_root).expanduser())
        return LocalArtifactTransport(Path(worker.config().roots.run_root))
    return SshArtifactTransport(_endpoint(args), args.remote_run_root, secrets=_secrets(args))


def _backend(args: argparse.Namespace) -> RemoteGpuBackend:
    return RemoteGpuBackend(
        _client(args),
        _transport(args),
        liveness=LivenessPolicy(),
        minimum_free_disk_mb=getattr(args, "minimum_free_disk_mb", None),
        allow_code_mismatch=getattr(args, "allow_code_mismatch", False),
    )


# ── worker ───────────────────────────────────────────────────────────


def cmd_worker_register_remote(args: argparse.Namespace) -> int:
    """Probe a remote machine and record what it actually is.

    The classification comes from the probe, never from the operator. A
    host becomes CUDA_TRAINING by demonstrating CUDA through torch on
    that machine, and there is no flag here that asserts it instead.
    """
    client = _client(args)
    try:
        described = client.probe_worker()
    except WorkerUnreachable as exc:
        return _fail(str(exc))

    identity = described["identity"]
    capabilities = described["capabilities"]
    orchestrator = _orchestrator(args)

    from luber_training.entities import WorkerCapabilities

    known = set(WorkerCapabilities.__dataclass_fields__)
    reported = {
        key: value for key, value in _worker_capabilities(capabilities).items() if key in known
    }

    worker = TrainingWorker(
        # The worker's own id, minted on the machine and stable across
        # its reboots. Not a fresh one issued by the control plane,
        # which would drift every time this command ran.
        worker_id=identity["worker_id"],
        name=identity["worker_name"],
        backend_type=identity["backend_type"],
        host_identity=identity["host_fingerprint"],
        worker_class=to_worker_class(described["classification"]),
        capabilities=WorkerCapabilities(**reported),
        max_concurrent_runs=args.max_concurrent_runs,
        ssh_key_ref=getattr(args, "ssh_key_ref", None),
        credential_ref=getattr(args, "known_hosts_ref", None),
        software_environment={
            "remote_protocol_version": described["protocol_version"],
            "capability_signature": described["capability_signature"],
            "remote_classification": described["classification"],
        },
    )
    orchestrator.register_worker(worker)
    _print(
        {
            "worker": worker.to_dict(),
            "host_fingerprint_changed": described.get("host_fingerprint_changed"),
            "note": (
                "the classification came from the worker's own probe; nothing here can "
                "promote a machine to CUDA_TRAINING without torch demonstrating CUDA on it"
            ),
        }
    )
    return 0


def _worker_capabilities(payload: dict[str, Any]) -> dict[str, Any]:
    """The Phase 25 capability fields, out of a remote report."""
    return {
        "gpu_vendor": payload.get("gpu_vendor"),
        "gpu_model": payload.get("gpu_model"),
        "gpu_count": payload.get("gpu_count"),
        "vram_total_mb": payload.get("vram_total_mb"),
        "system_ram_mb": payload.get("system_ram_mb"),
        "cpu_count": payload.get("cpu_count"),
        "cuda_available": payload.get("cuda_available"),
        "cuda_version": payload.get("cuda_version"),
        "driver_version": payload.get("driver_version"),
        "torch_version": payload.get("torch_version"),
        "python_version": payload.get("python_version"),
        "bf16_supported": payload.get("bf16_supported"),
        "free_disk_mb": payload.get("free_disk_mb"),
        "reported_by": (
            f"luber-remote probe ({payload.get('os_name')} {payload.get('architecture')})"
        ),
        "reported_at": payload.get("probed_at"),
    }


def cmd_worker_verify(args: argparse.Namespace) -> int:
    """Confirm a registered worker is still the machine we verified."""
    orchestrator = _orchestrator(args)
    worker = orchestrator.get_worker(args.worker_id)
    client = _client(args)

    try:
        described = client.probe_worker()
    except WorkerUnreachable as exc:
        _print({"ok": False, "worker_id": args.worker_id, "reachable": False, "error": str(exc)})
        return 1

    identity = described["identity"]
    problems: list[str] = []
    if identity["worker_id"] != worker.worker_id:
        problems.append(
            f"the machine reports worker id {identity['worker_id']}, the registry has "
            f"{worker.worker_id}"
        )
    if identity["host_fingerprint"] != worker.host_identity:
        problems.append(
            "the host fingerprint has changed; this machine may have been rebuilt since "
            "it was registered"
        )
    if described["protocol_version"] != REMOTE_PROTOCOL_VERSION:
        problems.append(
            f"the worker speaks {described['protocol_version']}, this build speaks "
            f"{REMOTE_PROTOCOL_VERSION}"
        )
    recorded = worker.software_environment.get("capability_signature")
    if recorded and recorded != described["capability_signature"]:
        problems.append(
            "the capability signature has changed: the hardware or the installed "
            "software is not what was registered"
        )

    _print(
        {
            "ok": not problems,
            "worker_id": worker.worker_id,
            "reachable": True,
            "classification": described["classification"],
            "active_run_id": described.get("active_run_id"),
            "problems": problems,
        }
    )
    return 0 if not problems else 1


def cmd_worker_heartbeat(args: argparse.Namespace) -> int:
    client = _client(args)
    try:
        beat = client.heartbeat()
    except WorkerUnreachable as exc:
        return _fail(str(exc))

    policy = LivenessPolicy()
    _print(
        {
            "heartbeat": beat,
            "liveness": policy.evaluate(beat.get("timestamp")),
            "policy": policy.to_dict(),
        }
    )
    return 0


# ── staging ──────────────────────────────────────────────────────────


def _staging_inputs(args: argparse.Namespace) -> StagingInputs:
    evaluation_only: frozenset[str] = frozenset()
    if getattr(args, "evaluation_only", None):
        lines = Path(args.evaluation_only).expanduser().read_text(encoding="utf-8").splitlines()
        evaluation_only = frozenset(
            line.strip() for line in lines if line.strip() and not line.startswith("#")
        )
    return StagingInputs(
        dataset_build_dir=Path(args.dataset_build).expanduser(),
        curation_build_dir=Path(args.curation_build).expanduser(),
        audio_root=Path(args.audio_root).expanduser(),
        evaluation_only_ids=evaluation_only,
    )


def cmd_run_stage(args: argparse.Namespace) -> int:
    """Assemble a run's transfer set, re-checking that it may be sent."""
    orchestrator = _orchestrator(args)
    plan = orchestrator.compile_plan(args.run_id)

    from luber_training.plan import capture_environment

    environment = capture_environment(
        orchestrator.repository_root, ace_step_commit=plan.config.ace_step_commit
    )

    try:
        result = build_staging(
            plan=plan,
            inputs=_staging_inputs(args),
            staging_root=Path(args.staging_root).expanduser(),
            environment_lock=environment.to_dict(),
        )
    except StagingError as exc:
        # Rights and leakage violations arrive here. Nothing was
        # written, and nothing left the machine.
        return _fail(str(exc))

    plan_summary = result.manifest.transfer_bytes()
    _print(
        {
            **result.to_dict(),
            "disk_requirement": disk_requirement(
                type(
                    "Plan",
                    (),
                    {
                        "upload_bytes": plan_summary,
                        "total_bytes": result.manifest.total_bytes,
                        "deduplicated_bytes": plan_summary,
                    },
                )()
            ).to_dict(),
            "next": "run verify-staging, then run dispatch",
        }
    )
    return 0


def cmd_run_verify_staging(args: argparse.Namespace) -> int:
    """Recheck a staged tree before anything is uploaded."""
    orchestrator = _orchestrator(args)
    plan = orchestrator.compile_plan(args.run_id)
    staging_dir = Path(args.staging_root).expanduser() / args.run_id

    problems = verify_staging(staging_dir, plan=plan)

    # The gates again, on the manifest that is about to be sent. Twice
    # is deliberate: `stage` checked what it built, and this checks what
    # is on disk now, which is what will actually move.
    gate_summary: dict[str, Any] = {}
    curated = staging_dir / "metadata" / "curated_manifest.jsonl"
    if curated.is_file():
        from luber_training.remote.staging import revalidate_before_transfer

        records = [
            json.loads(line)
            for line in curated.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        report = revalidate_before_transfer(
            records, evaluation_only_ids=_staging_inputs(args).evaluation_only_ids
        )
        gate_summary = report.to_dict()
        if not report.passed:
            problems.append(f"gates: {report.first_failure.detail if report.first_failure else ''}")

    _print(
        {
            "ok": not problems,
            "run_id": args.run_id,
            "staging_dir": str(staging_dir),
            "problems": problems,
            "gates": gate_summary,
        }
    )
    return 0 if not problems else 1


# ── dispatch and supervision ─────────────────────────────────────────


def cmd_run_dispatch(args: argparse.Namespace) -> int:
    """Transfer, verify, and start — reconciling first, always."""
    orchestrator = _orchestrator(args)
    plan = orchestrator.compile_plan(args.run_id)
    worker = orchestrator.get_worker(args.worker_id)
    staging_dir = Path(args.staging_root).expanduser() / args.run_id

    manifest_path = staging_dir / "artifact_manifest.json"
    if not manifest_path.is_file():
        return _fail(f"{args.run_id} has not been staged; run `remote run stage` first")
    manifest = RemoteArtifactManifest.read(manifest_path)

    backend = _backend(args)
    try:
        result = backend.dispatch(plan, worker, manifest=manifest, staging_dir=staging_dir)
    except Exception as exc:
        return _fail(f"{type(exc).__name__}: {exc}")

    _print(result.to_dict())
    return 0 if result.launched or result.reconcile.worker_state else 1


def cmd_run_remote_status(args: argparse.Namespace) -> int:
    backend = _backend(args)
    orchestrator = _orchestrator(args)
    plan = orchestrator.compile_plan(args.run_id)
    status = backend.status(plan)
    reconcile = backend.reconcile(args.run_id)
    _print({"status": status.to_dict(), "reconcile": reconcile.to_dict()})
    return 0


def cmd_run_remote_logs(args: argparse.Namespace) -> int:
    """Read from a cursor. Never the whole file."""
    client = _client(args)
    try:
        payload = client.logs(args.run_id, stream=args.stream, offset=args.offset, limit=args.limit)
    except WorkerUnreachable as exc:
        return _fail(str(exc))
    _print(
        {
            "stream": payload["stream"],
            "offset": payload["offset"],
            "next_offset": payload["next_offset"],
            "eof": payload["eof"],
            "size_bytes": payload["size_bytes"],
            "text": payload["text"],
        }
    )
    return 0


def cmd_run_remote_metrics(args: argparse.Namespace) -> int:
    """Pull new metric events into the run's local metrics file."""
    orchestrator = _orchestrator(args)
    plan = orchestrator.compile_plan(args.run_id)
    backend = _backend(args)
    events = backend.collect_metrics(plan)

    written = merge_into(orchestrator.metric_writer(args.run_id).path, events)
    _print(
        {
            "collected": len(events),
            "newly_recorded": written,
            "note": "events already recorded are not written twice",
        }
    )
    return 0


def cmd_run_remote_cancel(args: argparse.Namespace) -> int:
    orchestrator = _orchestrator(args)
    plan = orchestrator.compile_plan(args.run_id)
    backend = _backend(args)
    status = backend.cancel(plan)

    run = orchestrator.get_run(args.run_id)
    if not run.is_terminal and status.status == RunStatus.CANCELLED.value:
        orchestrator.transition_run(
            args.run_id,
            RunStatus.CANCELLED.value,
            error_code=FailureCode.CANCELLED_BY_OPERATOR.value,
            error_message="cancelled by operator on the worker",
        )
    _print({**status.to_dict(), "run_status": orchestrator.get_run(args.run_id).status})
    return 0


def cmd_run_reconcile(args: argparse.Namespace) -> int:
    """Ask the worker what actually happened, and record it. Idempotent.

    The only command that resolves a lost run, and it never launches
    anything. Where the worker cannot say how a trainer ended, the run
    stays as it is and the report says UNKNOWN — a state nobody
    established is not one to write into a registry.
    """
    orchestrator = _orchestrator(args)
    backend = _backend(args)
    report = backend.reconcile(args.run_id)
    run = orchestrator.get_run(args.run_id)

    applied: str | None = None
    if not run.is_terminal and report.run_status and args.apply:
        target = report.run_status
        if target != run.status and run.can_transition_to(target):
            orchestrator.transition_run(
                args.run_id,
                target,
                error_code=(
                    failure_code_for(report)
                    if target in (RunStatus.FAILED.value, RunStatus.LOST.value)
                    else None
                ),
                error_message=report.detail or None,
            )
            applied = target

    _print(
        {
            "reconcile": report.to_dict(),
            "run_status": orchestrator.get_run(args.run_id).status,
            "applied": applied,
            "note": (
                "UNREACHABLE means the trainer may still be running. Nothing is launched "
                "for this run until the worker answers"
                if report.outcome == ReconcileOutcome.UNREACHABLE.value
                else ""
            ),
        }
    )
    return 0


# ── collection ───────────────────────────────────────────────────────


def cmd_run_collect(args: argparse.Namespace) -> int:
    """Bring checkpoints back, verify them, then register them.

    Registration happens only for checkpoints whose local bytes hash to
    what the worker reported. A failed collection registers nothing and
    leaves the remote copy alone, so the retry has something to fetch.
    """
    orchestrator = _orchestrator(args)
    client = _client(args)
    transport = _transport(args)

    try:
        result = RemoteResult.from_dict(client.collect_result(args.run_id))
    except WorkerUnreachable as exc:
        return _fail(str(exc))

    destination = Path(args.collect_root).expanduser() / args.run_id
    report = collect_run(transport, result, destination_root=destination)

    registered: list[dict[str, Any]] = []
    problems: list[str] = []
    for collected in report.collected:
        if not collected.ok:
            problems.append(f"{collected.checkpoint_id}: {collected.problem}")
            continue
        remote = next(
            item for item in result.checkpoints if item.checkpoint_id == collected.checkpoint_id
        )
        try:
            checkpoint = register_collected(
                orchestrator,
                run_id=args.run_id,
                collected=collected,
                remote=remote,
                kind=args.kind,
            )
        except CollectionError as exc:
            problems.append(str(exc))
            continue
        registered.append(checkpoint.to_dict())

    _print(
        {
            "ok": report.ok and not problems,
            "collection": report.to_dict(),
            "registered": registered,
            "problems": problems,
            "retention": [decision.to_dict() for decision in plan_remote_retention(report)],
            "note": (
                "a checkpoint is READY only after its local bytes hashed to the digest the "
                "worker reported. Nothing here creates an evaluation candidate"
            ),
        }
    )
    return 0 if report.ok and not problems else 1


def cmd_run_verify_remote(args: argparse.Namespace) -> int:
    """Check the worker still holds what this run believes it does."""
    orchestrator = _orchestrator(args)
    plan = orchestrator.compile_plan(args.run_id)
    client = _client(args)
    transport = _transport(args)

    problems: list[str] = []
    try:
        described = client.probe_worker()
    except WorkerUnreachable as exc:
        return _fail(str(exc))

    if described["protocol_version"] != REMOTE_PROTOCOL_VERSION:
        problems.append(f"protocol mismatch: worker speaks {described['protocol_version']}")

    staged = Path(args.staging_root).expanduser() / args.run_id / "artifact_manifest.json"
    if not staged.is_file():
        problems.append("this run has not been staged locally, so there is nothing to compare")
    else:
        manifest = RemoteArtifactManifest.read(staged)
        if manifest.training_plan_sha256 != plan.digest():
            problems.append(
                f"the staged manifest cites plan {manifest.training_plan_sha256[:12]}, the "
                f"run compiles to {plan.digest()[:12]}"
            )
        scoped = RemoteArtifactManifest(
            run_id=manifest.run_id, training_plan_sha256=manifest.training_plan_sha256
        )
        for entry in manifest.entries:
            scoped.entries.append(
                type(entry)(
                    artifact_id=entry.artifact_id,
                    role=entry.role,
                    target_path=f"{args.run_id}/{entry.target_path}",
                    sha256=entry.sha256,
                    size_bytes=entry.size_bytes,
                    required=entry.required,
                    track_id=entry.track_id,
                )
            )
        for path, problem in transport.verify_manifest(scoped).items():
            problems.append(f"{path}: {problem}")

    try:
        status = client.status(args.run_id)
    except Exception:
        status = {}

    _print(
        {
            "ok": not problems,
            "run_id": args.run_id,
            "worker_id": described["identity"]["worker_id"],
            "worker_state": status.get("state"),
            "problems": problems,
        }
    )
    return 0 if not problems else 1


def cmd_run_cleanup(args: argparse.Namespace) -> int:
    client = _client(args)
    try:
        payload = client.cleanup(args.run_id, remove_dataset=args.remove_dataset)
    except WorkerUnreachable as exc:
        return _fail(str(exc))
    _print(payload)
    return 0


def add_remote_parser(sub: Any) -> None:
    """Attach the `remote` command group to the training CLI."""
    remote = sub.add_parser("remote", help="remote GPU execution (operator only)")
    remote_sub = remote.add_subparsers(dest="remote_group", required=True)

    connection = argparse.ArgumentParser(add_help=False)
    connection.add_argument(
        "--transport",
        default="local",
        choices=["local", "ssh"],
        help="local drives a worker in another directory; ssh drives a remote host",
    )
    connection.add_argument("--worker-root", default="./luber-worker", help="the worker's root")
    connection.add_argument("--remote-python", default="python", help="interpreter on the worker")
    connection.add_argument("--remote-run-root", help="ssh: the worker's run root")
    connection.add_argument("--host", help="ssh: hostname")
    connection.add_argument("--user", help="ssh: username")
    connection.add_argument("--port", type=int, help="ssh: port")
    connection.add_argument("--ssh-key-ref", help="a NAME, never a key")
    connection.add_argument("--known-hosts-ref", help="a NAME, never a file's contents")
    connection.add_argument("--secret-dir", help="directory the secret resolver reads")
    connection.add_argument(
        "--secrets-from-env", action="store_true", help="resolve secrets from LUBER_SECRET_*"
    )

    # ── worker ──
    worker = remote_sub.add_parser("worker", help="remote workers").add_subparsers(
        dest="action", required=True
    )
    register = worker.add_parser("register-remote", parents=[connection])
    register.add_argument("--max-concurrent-runs", type=int, default=1)
    register.set_defaults(func=cmd_worker_register_remote)

    verify_worker = worker.add_parser("verify", parents=[connection])
    verify_worker.add_argument("--worker-id", required=True)
    verify_worker.set_defaults(func=cmd_worker_verify)

    heartbeat = worker.add_parser("heartbeat", parents=[connection])
    heartbeat.set_defaults(func=cmd_worker_heartbeat)

    # ── run ──
    run = remote_sub.add_parser("run", help="remote runs").add_subparsers(
        dest="action", required=True
    )

    staging = argparse.ArgumentParser(add_help=False)
    staging.add_argument("--staging-root", default=DEFAULT_STAGING_ROOT)

    stage = run.add_parser("stage", parents=[staging])
    stage.add_argument("--run-id", required=True)
    stage.add_argument("--dataset-build", required=True, help="Phase 23 build directory")
    stage.add_argument("--curation-build", required=True, help="Phase 24 curation directory")
    stage.add_argument("--audio-root", required=True, help="where the approved audio lives")
    stage.add_argument("--evaluation-only", help="file of track ids barred from training")
    stage.set_defaults(func=cmd_run_stage)

    verify_staging_parser = run.add_parser("verify-staging", parents=[staging])
    verify_staging_parser.add_argument("--run-id", required=True)
    verify_staging_parser.add_argument("--dataset-build", required=True)
    verify_staging_parser.add_argument("--curation-build", required=True)
    verify_staging_parser.add_argument("--audio-root", required=True)
    verify_staging_parser.add_argument("--evaluation-only")
    verify_staging_parser.set_defaults(func=cmd_run_verify_staging)

    dispatch = run.add_parser("dispatch", parents=[connection, staging])
    dispatch.add_argument("--run-id", required=True)
    dispatch.add_argument("--worker-id", required=True)
    dispatch.add_argument("--minimum-free-disk-mb", type=int)
    dispatch.add_argument(
        "--allow-code-mismatch",
        action="store_true",
        help="do not require the worker's LUBER commit to match this one",
    )
    dispatch.set_defaults(func=cmd_run_dispatch)

    remote_status = run.add_parser("remote-status", parents=[connection])
    remote_status.add_argument("--run-id", required=True)
    remote_status.set_defaults(func=cmd_run_remote_status)

    remote_logs = run.add_parser("remote-logs", parents=[connection])
    remote_logs.add_argument("--run-id", required=True)
    remote_logs.add_argument("--stream", default="stdout", choices=["stdout", "stderr"])
    remote_logs.add_argument("--offset", type=int, default=0)
    remote_logs.add_argument("--limit", type=int, default=262_144)
    remote_logs.set_defaults(func=cmd_run_remote_logs)

    remote_metrics = run.add_parser("remote-metrics", parents=[connection])
    remote_metrics.add_argument("--run-id", required=True)
    remote_metrics.set_defaults(func=cmd_run_remote_metrics)

    remote_cancel = run.add_parser("remote-cancel", parents=[connection])
    remote_cancel.add_argument("--run-id", required=True)
    remote_cancel.set_defaults(func=cmd_run_remote_cancel)

    reconcile = run.add_parser("reconcile", parents=[connection])
    reconcile.add_argument("--run-id", required=True)
    reconcile.add_argument(
        "--apply", action="store_true", help="record what the worker reported on the run"
    )
    reconcile.set_defaults(func=cmd_run_reconcile)

    collect = run.add_parser("collect", parents=[connection])
    collect.add_argument("--run-id", required=True)
    collect.add_argument("--collect-root", default=DEFAULT_COLLECT_ROOT)
    collect.add_argument(
        "--kind",
        default=CheckpointKind.ADAPTER.value,
        choices=[kind.value for kind in CheckpointKind],
        help="MOCK for anything a synthetic trainer produced",
    )
    collect.set_defaults(func=cmd_run_collect)

    verify_remote = run.add_parser("verify-remote", parents=[connection, staging])
    verify_remote.add_argument("--run-id", required=True)
    verify_remote.set_defaults(func=cmd_run_verify_remote)

    cleanup = run.add_parser("cleanup", parents=[connection])
    cleanup.add_argument("--run-id", required=True)
    cleanup.add_argument("--remove-dataset", action="store_true")
    cleanup.set_defaults(func=cmd_run_cleanup)


__all__ = ["DEFAULT_COLLECT_ROOT", "DEFAULT_STAGING_ROOT", "add_remote_parser"]
