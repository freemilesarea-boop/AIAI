"""Operator command line for training orchestration.

    python -m luber_training --registry ./training-registry <command>

Deliberately operator-only and local. There is no HTTP surface, no user
account and no role: an ordinary LUBER account cannot launch training,
cancel a run, reach a training dataset or download a checkpoint, because
none of those paths exist outside this CLI. A weak role check bolted
onto the consumer API would be worse than the absence of one — it would
imply a boundary that a bug could cross.

`run start --backend dry-run` exercises every gate, the plan compiler
and the full lifecycle without training anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from luber_training import registry as registry_module
from luber_training.backends import DRY_RUN, REMOTE_GPU, LocalDryRunBackend
from luber_training.config import PRESET_INTENT, PRESETS, Precision, TrainingConfig, preset
from luber_training.entities import (
    CheckpointKind,
    ModelBaseline,
    TrainingDatasetRef,
    TrainingStrategySupport,
    TrainingWorker,
    WorkerCapabilities,
    WorkerClass,
)
from luber_training.gates import GateInputs, run_all
from luber_training.ids import EntityKind, new_id
from luber_training.orchestrator import Orchestrator
from luber_training.probe import probe_worker
from luber_training.registry import Registry
from luber_training.trainer_adapter import compile_command


def _orchestrator(args: argparse.Namespace) -> Orchestrator:
    registry = Registry(Path(args.registry).expanduser())
    return Orchestrator(
        registry,
        artifacts_root=Path(args.artifacts).expanduser() if args.artifacts else None,
        repository_root=Path(args.repository).expanduser() if args.repository else Path.cwd(),
    )


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str))


def _evaluation_only(path: str | None) -> frozenset[str]:
    if not path:
        return frozenset()
    lines = Path(path).expanduser().read_text(encoding="utf-8").splitlines()
    return frozenset(line.strip() for line in lines if line.strip() and not line.startswith("#"))


def _gate_inputs(args: argparse.Namespace) -> GateInputs:
    dataset_dir = Path(args.dataset_build).expanduser()
    curation_dir = Path(args.curation_build).expanduser()
    return GateInputs(
        dataset_lock_path=dataset_dir / "dataset_lock.json",
        dataset_manifest_path=dataset_dir / "dataset_manifest.jsonl",
        curation_lock_path=curation_dir / "curation_lock.json",
        curated_manifest_path=curation_dir / "curated_manifest.jsonl",
        evaluation_only_ids=_evaluation_only(getattr(args, "evaluation_only", None)),
        allow_self_generated=getattr(args, "allow_self_generated", False),
    )


# ── baseline ─────────────────────────────────────────────────────────


def cmd_baseline_register(args: argparse.Namespace) -> int:
    orchestrator = _orchestrator(args)
    baseline = ModelBaseline(
        model_id=new_id(EntityKind.MODEL),
        provider=args.provider,
        model_family=args.family,
        model_name=args.name,
        model_version=args.version,
        upstream_commit=args.upstream_commit,
        architecture=args.architecture,
        training_strategy_support=[
            TrainingStrategySupport.LORA.value,
            TrainingStrategySupport.LOKR.value,
        ],
        checkpoint_reference=args.checkpoint_reference,
    )
    _print(orchestrator.register_baseline(baseline).to_dict())
    return 0


def cmd_baseline_list(args: argparse.Namespace) -> int:
    _print(_orchestrator(args).registry.list_all("models"))
    return 0


# ── experiment ───────────────────────────────────────────────────────


def cmd_experiment_create(args: argparse.Namespace) -> int:
    orchestrator = _orchestrator(args)
    experiment = orchestrator.create_experiment(
        name=args.name,
        hypothesis=args.hypothesis,
        base_model_id=args.base_model_id,
        description=args.description or "",
        operator=args.operator or "",
        tags=args.tag or [],
    )
    _print(experiment.to_dict())
    return 0


def cmd_experiment_list(args: argparse.Namespace) -> int:
    _print(_orchestrator(args).registry.list_all("experiments"))
    return 0


# ── worker ───────────────────────────────────────────────────────────


def cmd_worker_probe(args: argparse.Namespace) -> int:
    """Capture this machine's real capabilities.

    Writes `worker_capabilities.json`. On a machine without NVIDIA it
    does not invoke `nvidia-smi` at all and reports the GPU fields as
    unknown rather than as absent hardware.
    """
    capabilities, classification = probe_worker()
    payload = {
        "worker_class": classification,
        "capabilities": capabilities.to_dict(),
    }
    destination = (
        Path(args.output).expanduser() if args.output else Path("worker_capabilities.json")
    )
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    _print(payload)
    print(f"\n  written to {destination}", file=sys.stderr)
    return 0


def cmd_worker_register(args: argparse.Namespace) -> int:
    orchestrator = _orchestrator(args)
    capabilities = WorkerCapabilities()
    worker_class = args.worker_class or WorkerClass.UNVERIFIED.value

    if args.capabilities:
        payload = json.loads(Path(args.capabilities).expanduser().read_text(encoding="utf-8"))
        reported = payload.get("capabilities", payload)
        known = set(WorkerCapabilities.__dataclass_fields__)
        capabilities = WorkerCapabilities(
            **{key: value for key, value in reported.items() if key in known}
        )
        # The probe's own classification wins unless the operator
        # overrode it. A worker does not become GPU-ready by being
        # registered with an optimistic flag.
        if not args.worker_class:
            worker_class = str(payload.get("worker_class", WorkerClass.UNVERIFIED.value))

    worker = TrainingWorker(
        worker_id=new_id(EntityKind.WORKER),
        name=args.name,
        backend_type=args.backend,
        host_identity=args.host_identity,
        worker_class=worker_class,
        capabilities=capabilities,
        max_concurrent_runs=args.max_concurrent_runs,
        ssh_key_ref=args.ssh_key_ref,
        credential_ref=args.credential_ref,
    )
    _print(orchestrator.register_worker(worker).to_dict())
    return 0


def cmd_worker_list(args: argparse.Namespace) -> int:
    _print(_orchestrator(args).registry.list_all("workers"))
    return 0


# ── run ──────────────────────────────────────────────────────────────


def _preset_config(args: argparse.Namespace) -> TrainingConfig:
    """The preset, with an explicitly requested precision applied.

    `--precision` exists because "auto" is not portable. On Apple
    silicon the installed trainer resolves it to fp16 and no step
    completes — Phase 33 measured that — so a run there has to name
    bf16 or fp32. The operator names it: nothing in this code path
    substitutes one dtype for another on a machine's behalf, because a
    precision quietly changed underneath a run is a different run
    reported as the same one.
    """
    config = preset(args.preset)
    requested = getattr(args, "precision", None)
    if requested:
        config = config.with_overrides(precision=requested)
    return config


def cmd_run_create(args: argparse.Namespace) -> int:
    orchestrator = _orchestrator(args)
    curation_dir = Path(args.curation_build).expanduser()
    dataset_dir = Path(args.dataset_build).expanduser()

    dataset_lock = json.loads((dataset_dir / "dataset_lock.json").read_text(encoding="utf-8"))
    curation_lock = json.loads((curation_dir / "curation_lock.json").read_text(encoding="utf-8"))

    dataset_ref = TrainingDatasetRef(
        dataset_id=str(dataset_lock.get("dataset_id", "")),
        dataset_lock_sha256=str(curation_lock.get("source_dataset_lock_sha256") or ""),
        curation_id=str(curation_lock.get("curation_id", "")),
        curation_lock_sha256=str(curation_lock.get("curated_manifest_sha256", "")),
        curated_manifest_sha256=str(curation_lock.get("curated_manifest_sha256", "")),
        manifest_artifact_ref=f"curation://{curation_lock.get('curation_id')}/curated_manifest",
        sampling_weights_sha256=curation_lock.get("sampling_weights_sha256"),
        selected_track_count=int(curation_lock.get("selected_track_count") or 0),
        selected_hours=float(curation_lock.get("selected_hours") or 0.0),
    )

    run = orchestrator.create_run(
        experiment_id=args.experiment_id,
        dataset_ref=dataset_ref,
        config=_preset_config(args),
        execution_backend=args.backend,
        worker_id=args.worker_id,
        parent_run_id=args.parent_run_id,
        resume_from_checkpoint_id=args.resume_from,
    )
    _print(run.to_dict())
    return 0


def cmd_run_validate(args: argparse.Namespace) -> int:
    orchestrator = _orchestrator(args)
    run, report = orchestrator.validate_run(
        args.run_id, _gate_inputs(args), worker_id=args.worker_id
    )
    _print({"run": run.to_dict(), "gates": report.to_dict()})
    return 0 if report.passed else 1


def cmd_run_start(args: argparse.Namespace) -> int:
    orchestrator = _orchestrator(args)
    run = orchestrator.get_run(args.run_id)
    if run.execution_backend != DRY_RUN:
        print(
            f"backend {run.execution_backend!r} has no implementation in this phase; "
            "only the dry-run backend can execute",
            file=sys.stderr,
        )
        return 2

    plan = orchestrator.compile_plan(args.run_id)
    worker = orchestrator.get_worker(run.worker_id or args.worker_id)
    backend = LocalDryRunBackend()

    gate_report = run_all(_gate_inputs(args))
    preflight = orchestrator.preflight(
        args.run_id,
        plan,
        worker,
        backend,
        gate_report,
        require_clean_repository=not args.allow_dirty,
    )
    if args.preflight_output:
        Path(args.preflight_output).expanduser().write_text(
            json.dumps(preflight.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if not preflight.ok:
        _print(preflight.to_dict())
        return 1

    orchestrator.start_run(args.run_id, plan, worker, backend)
    events = backend.collect_metrics(plan)
    orchestrator.record_metrics(args.run_id, events)
    run = orchestrator.complete_run(args.run_id)
    _print(
        {
            "run": run.to_dict(),
            "plan_sha256": plan.digest(),
            "metrics_recorded": len(events),
            "preflight": preflight.to_dict(),
            "note": "dry run: nothing was trained and no checkpoint was produced",
        }
    )
    return 0


def cmd_run_status(args: argparse.Namespace) -> int:
    orchestrator = _orchestrator(args)
    run = orchestrator.get_run(args.run_id)
    _print(
        {
            "run": run.to_dict(),
            "checkpoints": [c.to_dict() for c in orchestrator.run_checkpoints(args.run_id)],
            "audit": orchestrator.registry.audit_events(args.run_id),
        }
    )
    return 0


def cmd_run_cancel(args: argparse.Namespace) -> int:
    orchestrator = _orchestrator(args)
    plan = orchestrator.compile_plan(args.run_id)
    run = orchestrator.cancel_run(args.run_id, plan, LocalDryRunBackend())
    _print(run.to_dict())
    return 0


def cmd_run_list(args: argparse.Namespace) -> int:
    _print(_orchestrator(args).registry.list_all("runs"))
    return 0


def cmd_run_bundle(args: argparse.Namespace) -> int:
    _print(_orchestrator(args).run_bundle(args.run_id))
    return 0


def cmd_run_command(args: argparse.Namespace) -> int:
    """Show the trainer invocation a plan compiles to. Never runs it."""
    orchestrator = _orchestrator(args)
    plan = orchestrator.compile_plan(args.run_id)
    command = compile_command(plan, trainer_root=args.trainer_root)
    _print(
        {
            **command.to_dict(),
            "display": command.display(),
            "note": "compiled for inspection; Phase 25 does not execute the trainer",
        }
    )
    return 0


# ── checkpoints and candidates ───────────────────────────────────────


def cmd_checkpoint_list(args: argparse.Namespace) -> int:
    orchestrator = _orchestrator(args)
    records = (
        [c.to_dict() for c in orchestrator.run_checkpoints(args.run_id)]
        if args.run_id
        else orchestrator.registry.list_all("checkpoints")
    )
    _print(records)
    return 0


def cmd_candidate_create(args: argparse.Namespace) -> int:
    orchestrator = _orchestrator(args)
    candidate = orchestrator.create_candidate(args.checkpoint_id, notes=args.notes or "")
    _print(
        {
            **candidate.to_dict(),
            "note": (
                "a candidate is a request for evaluation evidence. Nothing here promotes "
                "a model, and promotion requires the evaluation phase."
            ),
        }
    )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Registry integrity: every reference resolves, no orphans."""
    orchestrator = _orchestrator(args)
    registry = orchestrator.registry
    problems: list[str] = []

    models = set(registry.list_ids("models"))
    experiments = {e["experiment_id"]: e for e in registry.list_all("experiments")}
    runs = {r["run_id"]: r for r in registry.list_all("runs")}

    for experiment in experiments.values():
        if experiment["base_model_id"] not in models:
            problems.append(
                f"experiment {experiment['experiment_id']} cites unknown model "
                f"{experiment['base_model_id']}"
            )
    for run in runs.values():
        if run["experiment_id"] not in experiments:
            problems.append(f"run {run['run_id']} cites unknown experiment")
        if run["base_model_id"] not in models:
            problems.append(f"run {run['run_id']} cites unknown model")
        parent = run.get("parent_run_id")
        if parent and parent not in runs:
            problems.append(f"run {run['run_id']} cites unknown parent {parent}")
    for checkpoint in registry.list_all("checkpoints"):
        if checkpoint["run_id"] not in runs:
            problems.append(f"checkpoint {checkpoint['checkpoint_id']} cites unknown run")
    for candidate in registry.list_all("candidates"):
        if candidate["checkpoint_id"] not in set(registry.list_ids("checkpoints")):
            problems.append(f"candidate {candidate['candidate_id']} cites unknown checkpoint")
        # The invariant that must never break: a MOCK artifact cannot
        # become an evaluation candidate.
        try:
            referenced = orchestrator.get_checkpoint(candidate["checkpoint_id"])
        except Exception:
            continue
        if referenced.kind == CheckpointKind.MOCK.value:
            problems.append(f"candidate {candidate['candidate_id']} references a MOCK artifact")

    _print(
        {
            "ok": not problems,
            "problems": problems,
            "counts": {name: len(registry.list_ids(name)) for name in registry_module.COLLECTIONS},
        }
    )
    return 0 if not problems else 1


def cmd_presets(args: argparse.Namespace) -> int:
    _print(
        {
            name: {
                "intent": PRESET_INTENT[name],
                "config_sha256": PRESETS[name]().digest(),
                "vram_requirement": "UNKNOWN_REQUIREMENT",
            }
            for name in sorted(PRESETS)
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m luber_training",
        description=(
            "Operator training orchestration. Decides what may be trained and records "
            "what happened; performs no training itself."
        ),
    )
    parser.add_argument("--registry", default="./training-registry", help="registry directory")
    parser.add_argument("--artifacts", help="run artifact root (default: <registry>/training_runs)")
    parser.add_argument("--repository", help="LUBER repository root, for the code version lock")
    sub = parser.add_subparsers(dest="group", required=True)

    # baseline
    baseline = sub.add_parser("baseline", help="model baseline registry").add_subparsers(
        dest="action", required=True
    )
    register = baseline.add_parser("register")
    register.add_argument("--provider", required=True)
    register.add_argument("--family", required=True)
    register.add_argument("--name", required=True)
    register.add_argument("--version", required=True)
    register.add_argument("--upstream-commit", required=True)
    register.add_argument("--architecture", required=True)
    register.add_argument("--checkpoint-reference")
    register.set_defaults(func=cmd_baseline_register)
    baseline.add_parser("list").set_defaults(func=cmd_baseline_list)

    # experiment
    experiment = sub.add_parser("experiment", help="experiment registry").add_subparsers(
        dest="action", required=True
    )
    create = experiment.add_parser("create")
    create.add_argument("--name", required=True)
    create.add_argument("--hypothesis", required=True)
    create.add_argument("--base-model-id", required=True)
    create.add_argument("--description")
    create.add_argument("--operator")
    create.add_argument("--tag", action="append")
    create.set_defaults(func=cmd_experiment_create)
    experiment.add_parser("list").set_defaults(func=cmd_experiment_list)

    # worker
    worker = sub.add_parser("worker", help="training workers").add_subparsers(
        dest="action", required=True
    )
    probe = worker.add_parser("probe", help="capture this machine's capabilities")
    probe.add_argument("--output")
    probe.set_defaults(func=cmd_worker_probe)
    worker_register = worker.add_parser("register")
    worker_register.add_argument("--name", required=True)
    worker_register.add_argument("--backend", required=True, choices=[DRY_RUN, REMOTE_GPU])
    worker_register.add_argument("--host-identity", required=True)
    worker_register.add_argument("--capabilities", help="worker_capabilities.json from the probe")
    worker_register.add_argument("--worker-class", choices=[c.value for c in WorkerClass])
    worker_register.add_argument("--max-concurrent-runs", type=int, default=1)
    worker_register.add_argument("--ssh-key-ref", help="a NAME, never a key")
    worker_register.add_argument("--credential-ref", help="a NAME, never a secret")
    worker_register.set_defaults(func=cmd_worker_register)
    worker.add_parser("list").set_defaults(func=cmd_worker_list)

    # run
    run = sub.add_parser("run", help="training runs").add_subparsers(dest="action", required=True)
    gates = argparse.ArgumentParser(add_help=False)
    gates.add_argument("--dataset-build", required=True, help="Phase 23 build directory")
    gates.add_argument("--curation-build", required=True, help="Phase 24 curation directory")
    gates.add_argument("--evaluation-only", help="file of track ids barred from training")
    gates.add_argument(
        "--allow-self-generated",
        action="store_true",
        help="admit this project's own model output (default: false)",
    )

    run_create = run.add_parser("create", parents=[gates])
    run_create.add_argument("--experiment-id", required=True)
    run_create.add_argument("--preset", default="LORA_STANDARD", choices=sorted(PRESETS))
    run_create.add_argument(
        "--precision",
        choices=[item.value for item in Precision],
        help=(
            "override the preset's precision. Required on Apple silicon, where 'auto' "
            "resolves to fp16 and the trainer cannot complete a step"
        ),
    )
    run_create.add_argument("--backend", default=DRY_RUN, choices=[DRY_RUN, REMOTE_GPU])
    run_create.add_argument("--worker-id")
    run_create.add_argument("--parent-run-id")
    run_create.add_argument("--resume-from", help="checkpoint id to resume from")
    run_create.set_defaults(func=cmd_run_create)

    run_validate = run.add_parser("validate", parents=[gates])
    run_validate.add_argument("--run-id", required=True)
    run_validate.add_argument("--worker-id")
    run_validate.set_defaults(func=cmd_run_validate)

    run_start = run.add_parser("start", parents=[gates])
    run_start.add_argument("--run-id", required=True)
    run_start.add_argument("--worker-id")
    run_start.add_argument("--backend", default=DRY_RUN, choices=[DRY_RUN])
    run_start.add_argument("--preflight-output")
    run_start.add_argument(
        "--allow-dirty", action="store_true", help="permit an unidentified working tree"
    )
    run_start.set_defaults(func=cmd_run_start)

    run_status = run.add_parser("status")
    run_status.add_argument("--run-id", required=True)
    run_status.set_defaults(func=cmd_run_status)

    run_cancel = run.add_parser("cancel")
    run_cancel.add_argument("--run-id", required=True)
    run_cancel.set_defaults(func=cmd_run_cancel)

    run.add_parser("list").set_defaults(func=cmd_run_list)

    run_bundle = run.add_parser("bundle", help="the reproducibility bundle")
    run_bundle.add_argument("--run-id", required=True)
    run_bundle.set_defaults(func=cmd_run_bundle)

    run_command = run.add_parser("command", help="show the compiled trainer invocation")
    run_command.add_argument("--run-id", required=True)
    run_command.add_argument("--trainer-root", default="/opt/ace-step")
    run_command.set_defaults(func=cmd_run_command)

    # checkpoint / candidate
    checkpoint = sub.add_parser("checkpoint", help="checkpoint registry").add_subparsers(
        dest="action", required=True
    )
    checkpoint_list = checkpoint.add_parser("list")
    checkpoint_list.add_argument("--run-id")
    checkpoint_list.set_defaults(func=cmd_checkpoint_list)

    candidate = sub.add_parser("candidate", help="evaluation candidates").add_subparsers(
        dest="action", required=True
    )
    candidate_create = candidate.add_parser("create")
    candidate_create.add_argument("--checkpoint-id", required=True)
    candidate_create.add_argument("--notes")
    candidate_create.set_defaults(func=cmd_candidate_create)

    # Remote GPU execution. In its own module because these verbs move
    # data to a machine somebody else owns.
    from luber_training.remote_cli import add_remote_parser

    add_remote_parser(sub)

    # The execution-readiness gate and the bounded canary. In their own
    # module because `canary ace-step` is the one verb here that starts
    # a real trainer — bounded, and only ever bounded.
    from luber_training.preflight_cli import add_preflight_parsers

    add_preflight_parsers(sub)

    sub.add_parser("presets", help="available training presets").set_defaults(func=cmd_presets)
    sub.add_parser("verify", help="registry integrity").set_defaults(func=cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
