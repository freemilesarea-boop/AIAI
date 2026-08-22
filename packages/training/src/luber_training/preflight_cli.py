"""Operator verbs for the execution-readiness gate and the canary.

In its own module for the same reason `remote_cli` is: these verbs
touch a real trainer. `preflight` starts nothing and is safe to run at
any time; `canary ace-step` starts the installed trainer, inside bounds
this module cannot widen.

Nothing here can launch an unrestricted run. The canary's length comes
from :class:`~luber_training.canary.CanaryEnvelope`, which validates
against module ceilings on construction, and there is no flag that
raises them — `--samples` is clamped by the envelope, not by the parser.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from luber_hardware import (
    ComputeDevice,
    ExecutionLocation,
    MachineCapability,
    ProbeError,
    probe_machine,
    resolve_precision,
)
from luber_training.canary import (
    CanaryEnvelope,
    CanaryStatus,
    ace_step_canary,
    cleanup_workspace,
    default_workspace,
    orchestration_canary,
)
from luber_training.capacity import capacity_report
from luber_training.capacity_policy import (
    DEFAULT_POLICY,
    CapacityQualification,
    device_total_for,
    qualify,
)
from luber_training.entities import TrainingWorker
from luber_training.gates import GateReport, run_all
from luber_training.memory import MIB
from luber_training.memory_profiler import (
    DEFAULT_PROBE_ENCODER_LENGTH,
    DEFAULT_PROFILE_TIMEOUT_SECONDS,
    PRODUCTION_LATENT_LENGTH,
    ProbeShape,
    ProfileRequest,
    bounded_envelope_for,
    identity_for,
    load_profiles,
    profile_memory,
    render_markdown,
    write_profile,
)
from luber_training.orchestrator import Orchestrator
from luber_training.plan import TrainingPlan
from luber_training.preflight import PreflightIntent, PreflightStatus
from luber_training.preflight_collect import (
    collect_dataset_evidence,
    collect_remote_evidence,
    collect_storage_evidence,
    collect_trainer_evidence,
    synthetic_dataset_evidence,
)
from luber_training.trainer_adapter import compile_command

#: What the canary writes into, when nobody names somewhere.
DEFAULT_CANARY_DIRNAME = "canary"

#: Where memory profiles are kept, beneath the registry root.
#:
#: A directory rather than a file: profiles are evidence records, one
#: per configuration identity, and a bf16 measurement must not overwrite
#: an fp32 one. They are operational artifacts and are never committed.
MEMORY_PROFILE_DIRNAME = "memory_profiles"


def _profile_directory(args: argparse.Namespace) -> Path:
    return Path(args.registry).expanduser() / MEMORY_PROFILE_DIRNAME


def _capacity_decision(
    args: argparse.Namespace,
    plan: TrainingPlan,
    capability: MachineCapability,
    device: str,
    *,
    latent_length: int,
    encoder_length: int,
) -> Any:
    """Ask the qualifier what the stored profiles permit.

    The request is built from the plan and the *asked-about* shape, not
    from whatever a profile happened to measure. A profile qualifies a
    request; a request never adopts a profile's shape to make itself
    qualify.
    """
    shape = ProbeShape(latent_length=latent_length, encoder_length=encoder_length)
    identity = identity_for(plan, shape, model_variant=args.model_variant)
    requested = identity.to_dict()
    requested["torch_version"] = capability.torch_version
    return qualify(
        device=device,
        requested=requested,
        profiles=load_profiles(_profile_directory(args)),
        host_total_bytes=(
            None if capability.memory_total_mb is None else capability.memory_total_mb * MIB
        ),
        device_total_bytes=device_total_for(capability, device),
        runs_control_plane=args.runs_control_plane,
        policy=DEFAULT_POLICY,
    )


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str))


def _capability(python_executable: str | None, label: str | None = None) -> MachineCapability:
    try:
        return probe_machine(python_executable or None, label=label)
    except ProbeError as exc:
        # A named interpreter that cannot be reached is a fact about the
        # deployment, not a crash. It arrives as a capability with
        # nothing measured, which is what makes the preflight UNVERIFIED
        # rather than silently local.
        return MachineCapability(
            label=label or "unreachable-interpreter",
            notes=(f"the configured interpreter could not be probed: {exc}",),
        )


def _worker(orchestrator: Orchestrator, worker_id: str | None) -> TrainingWorker | None:
    if not worker_id:
        return None
    try:
        return orchestrator.get_worker(worker_id)
    except (KeyError, FileNotFoundError):
        return None


def _gate_report(args: argparse.Namespace) -> GateReport | None:
    """Run the real Phase 25 gates when the builds were named.

    Never fabricated. Without both build directories there is no gate
    report, and the preflight reports the rights position as
    unestablished rather than assuming it.
    """
    if not (getattr(args, "dataset_build", None) and getattr(args, "curation_build", None)):
        return None
    from luber_training.cli import _gate_inputs

    return run_all(_gate_inputs(args))


def _plan(orchestrator: Orchestrator, args: argparse.Namespace) -> TrainingPlan:
    return orchestrator.compile_plan(args.run_id, execution_device=args.device)


# ── preflight ────────────────────────────────────────────────────────


def cmd_preflight(args: argparse.Namespace) -> int:
    """Whether this machine can execute this plan. Starts nothing."""
    from luber_training.cli import _orchestrator

    orchestrator = _orchestrator(args)
    plan = _plan(orchestrator, args)
    run = orchestrator.get_run(args.run_id)
    worker = _worker(orchestrator, run.worker_id)
    capability = _capability(args.python)
    trainer_root = Path(args.trainer_root).expanduser() if args.trainer_root else None

    argv: list[str] = []
    if trainer_root is not None:
        command = compile_command(
            plan,
            trainer_root=str(trainer_root),
            python_executable=args.python or "python",
            model_dir=args.model_dir,
        )
        # The program name is not part of what the parser reads.
        argv = command.argv[2:]

    trainer = collect_trainer_evidence(
        plan,
        trainer_root=trainer_root,
        python_executable=args.python,
        argv=argv,
    )
    gate_report = _gate_report(args)
    dataset = collect_dataset_evidence(
        plan,
        curated_manifest_path=(
            Path(args.curation_build).expanduser() / "curated_manifest.jsonl"
            if args.curation_build
            else None
        ),
        locks_verified=None if gate_report is None else gate_report.passed,
        check_source_files=args.check_files,
    )
    output_dir = Path(run.output_directory or orchestrator.artifacts_root / args.run_id)
    storage = collect_storage_evidence(
        output_dir=output_dir,
        checkpoint_dir=output_dir / "checkpoints",
    )
    remote = collect_remote_evidence(worker, location=args.location)
    device = args.device or plan.requirements.execution_device or ComputeDevice.CPU.value
    capacity = capacity_report(
        capability,
        device=device,
        free_disk_mb=storage.free_disk_mb,
        disk_measured_by="the machine running this preflight",
    )

    decision = _capacity_decision(
        args,
        plan,
        capability,
        device,
        latent_length=args.latent_length,
        encoder_length=args.encoder_length,
    )
    result = orchestrator.training_preflight(
        args.run_id,
        plan,
        capability=capability,
        execution_location=args.location,
        intent=args.intent,
        worker=worker,
        gate_report=gate_report,
        dataset=dataset,
        trainer=trainer,
        storage=storage,
        remote=remote,
        capacity=capacity,
        capacity_decision=decision,
    )
    if args.json:
        _print(result.to_dict())
    else:
        print(result.render())
    return 0 if result.status == PreflightStatus.READY.value else 1


# ── canary ───────────────────────────────────────────────────────────


def cmd_canary_orchestration(args: argparse.Namespace) -> int:
    """Prove LUBER's half of a canary. Starts no trainer."""
    from luber_training.cli import _orchestrator

    orchestrator = _orchestrator(args)
    plan = _plan(orchestrator, args)
    envelope = CanaryEnvelope(max_samples=args.samples, resume=args.resume)
    result = orchestration_canary(
        plan,
        envelope,
        trainer_root=args.trainer_root or "${LUBER_TRAINER_ROOT}",
        python_executable=args.python or "python",
        execution_location=args.location,
        model_dir=args.model_dir,
    )
    orchestrator.record_canary(args.run_id, result.to_dict())
    _print(result.to_dict())
    return 0


def cmd_canary_ace_step(args: argparse.Namespace) -> int:
    """Run the installed trainer, bounded, on synthetic tensors.

    This is the only verb in LUBER that starts a real trainer, and it
    can only start a bounded one: the envelope is constructed here and
    validated against the module ceilings, so a request outside them
    raises before any process exists.
    """
    from luber_training.cli import _orchestrator

    orchestrator = _orchestrator(args)
    plan = _plan(orchestrator, args)
    run = orchestrator.get_run(args.run_id)
    trainer_root = Path(args.trainer_root).expanduser()
    python_executable = Path(args.python).expanduser()
    workspace = (
        Path(args.workspace).expanduser()
        if args.workspace
        else Path(run.output_directory or orchestrator.artifacts_root / args.run_id)
        / DEFAULT_CANARY_DIRNAME
    )

    capability = _capability(str(python_executable))
    device = args.device or plan.requirements.execution_device or ComputeDevice.CPU.value
    precision = resolve_precision(
        capability, device=device, requested=plan.config.precision, allow_unverified=True
    )
    envelope = CanaryEnvelope(
        max_samples=args.samples,
        resume=args.resume,
        wall_clock_seconds=float(args.timeout),
    )
    result = ace_step_canary(
        plan,
        envelope,
        trainer_root=trainer_root,
        python_executable=python_executable,
        model_dir=Path(args.model_dir).expanduser(),
        workspace=workspace,
        execution_location=args.location,
        resolved_precision=precision.precision,
        gate_report=_gate_report(args),
        dataset_dir=Path(args.dataset_dir).expanduser() if args.dataset_dir else None,
    )
    orchestrator.record_canary(args.run_id, result.to_dict())

    if args.record_preflight:
        dataset = synthetic_dataset_evidence(plan, workspace / "dataset")
        storage = collect_storage_evidence(
            dataset_dir=workspace / "dataset",
            output_dir=workspace / "output",
            checkpoint_dir=workspace / "output" / "checkpoints",
        )
        trainer = collect_trainer_evidence(
            plan,
            trainer_root=trainer_root,
            python_executable=python_executable,
            argv=result.command[2:],
        )
        orchestrator.training_preflight(
            args.run_id,
            plan,
            capability=capability,
            execution_location=args.location,
            intent=PreflightIntent.CANARY.value,
            dataset=dataset,
            trainer=trainer,
            storage=storage,
            canary=result.as_evidence(),
            capacity=capacity_report(
                capability,
                device=device,
                free_disk_mb=storage.free_disk_mb,
                checkpoint_bytes=(result.checkpoint or {}).get("size_bytes"),
            ),
        )

    _print(result.to_dict())
    if args.cleanup:
        # A canary's checkpoint is a model that learned noise. Removing
        # it by default would hide the evidence; keeping it forever is
        # how one ends up committed. So it is a flag, and the record of
        # what happened survives either way in the registry.
        cleanup_workspace(workspace)
    return 0 if result.status == CanaryStatus.PASSED.value else 1


# ── memory profile and capacity ──────────────────────────────────────


def cmd_profile_memory(args: argparse.Namespace) -> int:
    """Measure what one configuration costs, inside the real trainer.

    Bounded exactly as a canary is, and instrumented from inside the
    trainer process because that is the only place `torch.mps` will
    answer. Nothing is reduced to make it fit: the shape asked for is
    the shape measured, and a smaller one is a different profile.
    """
    from luber_training.cli import _orchestrator

    orchestrator = _orchestrator(args)
    plan = _plan(orchestrator, args)
    trainer_root = Path(args.trainer_root).expanduser()
    python_executable = Path(args.python).expanduser()
    shape = ProbeShape(
        latent_length=args.latent_length,
        encoder_length=args.encoder_length,
        samples=args.samples,
    )
    workspace = (
        Path(args.workspace).expanduser()
        if args.workspace
        else default_workspace(trainer_root, f"profile-{args.run_id}")
    )

    request = ProfileRequest(
        plan=plan,
        shape=shape,
        trainer_root=trainer_root,
        python_executable=python_executable,
        model_dir=Path(args.model_dir).expanduser(),
        workspace=workspace,
        envelope=bounded_envelope_for(shape),
        model_variant=args.model_variant,
        timeout_seconds=float(args.timeout),
        sample_interval_seconds=float(args.sample_interval),
        gate_report=_gate_report(args),
        dataset_dir=Path(args.dataset_dir).expanduser() if args.dataset_dir else None,
        runs_control_plane=args.runs_control_plane,
        luber_commit=_luber_commit(args),
        measure_resume=args.measure_resume,
    )
    profile = profile_memory(request)

    directory = _profile_directory(args)
    path = write_profile(profile, directory)
    (directory / f"{profile.profile_id}.md").write_text(render_markdown(profile), encoding="utf-8")
    orchestrator.record_memory_profile(args.run_id, profile.to_dict())

    if args.cleanup:
        cleanup_workspace(workspace)
    _print({"profile": str(path), **profile.to_dict()})
    return 0 if profile.completed else 1


def cmd_capacity(args: argparse.Namespace) -> int:
    """What the stored profiles say about running this configuration."""
    from luber_training.cli import _orchestrator

    orchestrator = _orchestrator(args)
    plan = _plan(orchestrator, args)
    capability = _capability(args.python)
    device = args.device or plan.requirements.execution_device or ComputeDevice.CPU.value
    decision = _capacity_decision(
        args,
        plan,
        capability,
        device,
        latent_length=args.latent_length,
        encoder_length=args.encoder_length,
    )
    _print(
        {
            **decision.to_dict(),
            "policy": DEFAULT_POLICY.to_dict(),
            "profiles_considered": len(load_profiles(_profile_directory(args))),
        }
    )
    return 0 if decision.qualification == CapacityQualification.QUALIFIED.value else 1


def _luber_commit(args: argparse.Namespace) -> str | None:
    import subprocess

    root = Path(args.repository).expanduser() if args.repository else Path.cwd()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (result.stdout.strip() or None) if result.returncode == 0 else None


# ── parser ───────────────────────────────────────────────────────────


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--device",
        choices=[item.value for item in ComputeDevice],
        help="compile the plan for this compute device (Phase 32 placement)",
    )
    parser.add_argument(
        "--location",
        choices=[item.value for item in ExecutionLocation],
        default=ExecutionLocation.LOCAL.value,
    )
    parser.add_argument("--trainer-root", help="the ACE-Step installation")
    parser.add_argument("--python", help="the interpreter that runs training")
    parser.add_argument("--model-dir", help="root holding the base model weights")
    parser.add_argument("--dataset-build", help="dataset build directory, for the gates")
    parser.add_argument("--curation-build", help="curation build directory, for the gates")
    parser.add_argument("--evaluation-only", help="file of track ids that may never train")
    parser.add_argument("--allow-self-generated", action="store_true")
    parser.add_argument("--model-variant", default="turbo")
    parser.add_argument(
        "--latent-length",
        type=int,
        default=PRODUCTION_LATENT_LENGTH,
        help=(
            "latent frames per sample. Defaults to the production maximum "
            f"({PRODUCTION_LATENT_LENGTH} frames = 240s at 25 frames/s); a profile does "
            "not qualify a longer one than it measured"
        ),
    )
    parser.add_argument("--encoder-length", type=int, default=DEFAULT_PROBE_ENCODER_LENGTH)
    parser.add_argument(
        "--runs-control-plane",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether this machine also serves the API, the database and the queue",
    )


def add_preflight_parsers(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Attach `preflight` and `canary` to the operator CLI."""
    preflight = sub.add_parser(
        "preflight", help="whether a machine can execute a plan (starts nothing)"
    )
    _add_common(preflight)
    preflight.add_argument(
        "--intent",
        choices=[item.value for item in PreflightIntent],
        default=PreflightIntent.CANARY.value,
        help=(
            "CANARY clears a bounded run; FULL_TRAINING additionally requires a measured "
            "memory requirement, which nothing has, so it reports UNVERIFIED"
        ),
    )
    preflight.add_argument(
        "--check-files",
        action="store_true",
        help="verify every referenced source file exists on this machine",
    )
    preflight.add_argument("--json", action="store_true")
    preflight.set_defaults(func=cmd_preflight)

    canary = sub.add_parser("canary", help="a bounded training run, never a full one")
    canary_sub = canary.add_subparsers(dest="action", required=True)

    orchestration = canary_sub.add_parser(
        "orchestration", help="compile and bound the plan; start no trainer"
    )
    _add_common(orchestration)
    orchestration.add_argument("--samples", type=int, default=2)
    orchestration.add_argument("--resume", action="store_true")
    orchestration.set_defaults(func=cmd_canary_orchestration)

    ace = canary_sub.add_parser(
        "ace-step",
        help="run the installed trainer for one bounded epoch on synthetic tensors",
    )
    _add_common(ace)
    ace.add_argument("--samples", type=int, default=2, help="bounded by the canary envelope")
    ace.add_argument(
        "--resume",
        action="store_true",
        help="save, stop, reload and continue, then check the step counter advanced",
    )
    ace.add_argument("--timeout", type=float, default=1800.0, help="wall clock, seconds")
    ace.add_argument("--workspace", help="where the canary writes; defaults inside the run")
    ace.add_argument(
        "--dataset-dir",
        help=(
            "gate-cleared tensors to use instead of the synthetic fixture; refused unless "
            "every gate passed"
        ),
    )
    ace.add_argument(
        "--record-preflight",
        action="store_true",
        help="write a training preflight citing this canary's result",
    )
    ace.add_argument(
        "--cleanup",
        action="store_true",
        help="delete the canary workspace, including the checkpoint it produced",
    )
    ace.set_defaults(func=cmd_canary_ace_step)

    profile = sub.add_parser(
        "profile-memory",
        help="measure what one configuration costs, inside the real trainer",
    )
    _add_common(profile)
    profile.add_argument("--samples", type=int, default=2)
    profile.add_argument(
        "--timeout", type=float, default=DEFAULT_PROFILE_TIMEOUT_SECONDS, help="wall clock"
    )
    profile.add_argument("--sample-interval", type=float, default=0.25)
    profile.add_argument("--workspace", help="where the profile writes; beneath the trainer root")
    profile.add_argument("--dataset-dir", help="gate-cleared tensors instead of a fixture")
    profile.add_argument(
        "--measure-resume",
        action="store_true",
        help="run a second bounded leg from the checkpoint, and record its peak separately",
    )
    profile.add_argument("--cleanup", action="store_true")
    profile.set_defaults(func=cmd_profile_memory)

    capacity = sub.add_parser(
        "capacity", help="what the stored profiles permit for this configuration"
    )
    _add_common(capacity)
    capacity.set_defaults(func=cmd_capacity)


__all__ = [
    "DEFAULT_CANARY_DIRNAME",
    "add_preflight_parsers",
    "cmd_canary_ace_step",
    "cmd_canary_orchestration",
    "cmd_capacity",
    "cmd_preflight",
    "cmd_profile_memory",
]
