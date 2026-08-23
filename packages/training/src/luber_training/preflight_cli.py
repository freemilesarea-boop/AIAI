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
from luber_training.pilot import (
    PILOT_MAX_OPTIMIZER_STEPS,
    PILOT_MAX_SEGMENT_STEPS,
    PILOT_MAX_WALL_CLOCK_SECONDS,
    PilotOutcome,
    PilotStepBudget,
)
from luber_training.pilot_runner import (
    PILOT_SUBDIR,
    PilotRequest,
    render_dataset_report,
    run_pilot,
    verify_pilot_dataset,
    write_pilot_artifacts,
)
from luber_training.pilot_runner import (
    identity_for as pilot_identity_for,
)
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
    shape = ProbeShape(
        latent_length=latent_length,
        encoder_length=encoder_length,
        latent_shape_count=max(1, int(getattr(args, "latent_shape_count", 1) or 1)),
    )
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
    # Naming the tensors is what lets this verb answer "can the trainer
    # read its input" and "does the path survive the trainer's own root
    # check" at all. Without it both stay UNKNOWN, and a preflight that
    # cannot see the data it is clearing has no business reporting
    # READY — so the two questions arrive together or not at all.
    dataset_dir = (
        Path(args.dataset_dir).expanduser() if getattr(args, "dataset_dir", None) else None
    )
    storage = collect_storage_evidence(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        checkpoint_dir=output_dir / "checkpoints",
        trainer_root=trainer_root,
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


# ── pilot ────────────────────────────────────────────────────────────


def _pilot_request(args: argparse.Namespace, orchestrator: Orchestrator) -> PilotRequest:
    """Assemble a pilot from the run, the machine and the stored evidence.

    Everything a pilot is allowed to vary comes from the plan. The
    ceilings do not appear here at all — they are module constants in
    `luber_training.pilot`, and there is deliberately no argument that
    reaches them.
    """
    plan = _plan(orchestrator, args)
    run = orchestrator.get_run(args.run_id)
    trainer_root = Path(args.trainer_root).expanduser()
    capability = _capability(args.python)
    device = args.device or plan.requirements.execution_device or ComputeDevice.CPU.value
    workspace = (
        Path(args.workspace).expanduser()
        if args.workspace
        else default_workspace(trainer_root, f"{PILOT_SUBDIR}-{args.run_id}")
    )
    return PilotRequest(
        plan=plan,
        dataset_dir=Path(args.dataset_dir).expanduser(),
        trainer_root=trainer_root,
        python_executable=Path(args.python).expanduser(),
        model_dir=Path(args.model_dir).expanduser(),
        workspace=workspace,
        dataset_id=args.dataset_id or (run.dataset_ref.dataset_id or "pilot"),
        latent_length=args.latent_length,
        encoder_length=args.encoder_length,
        seed=args.seed,
        model_variant=args.model_variant,
        gate_report=_gate_report(args),
        capacity=_capacity_decision(
            args,
            plan,
            capability,
            device,
            latent_length=args.latent_length,
            encoder_length=args.encoder_length,
        ),
        preflight_status=args.preflight_status,
        allow_synthetic=args.allow_synthetic_fixture,
        segment_timeout_seconds=float(args.timeout),
        measure_resume=args.resume,
    )


def cmd_pilot_prepare(args: argparse.Namespace) -> int:
    """Say what a pilot would do, and start nothing.

    The step budget, the dataset verdict and the capacity position, all
    computed before any process exists. An operator reads this and knows
    whether the pilot will be refused before spending anything on it.
    """
    from luber_training.cli import _orchestrator

    orchestrator = _orchestrator(args)
    request = _pilot_request(args, orchestrator)
    verdict = verify_pilot_dataset(
        request.dataset_dir,
        gate_report=request.gate_report,
        allow_synthetic=request.allow_synthetic,
    )
    try:
        budget = PilotStepBudget.for_ceiling(
            samples=max(1, verdict.sample_count),
            micro_batch_size=request.plan.config.batch_size,
            gradient_accumulation=request.plan.config.gradient_accumulation,
            ceiling=PILOT_MAX_SEGMENT_STEPS,
        )
        budget_payload: dict[str, Any] = budget.to_dict()
        identity = pilot_identity_for(request, budget, verdict.manifest_digest or "")
        identity_payload: dict[str, Any] = {
            "pilot_id": identity.pilot_id(),
            "digest": identity.digest(),
            **identity.to_dict(),
        }
        report = render_dataset_report(verdict, identity)
    except Exception as exc:
        budget_payload = {"error": str(exc)}
        identity_payload = {}
        report = ""

    _print(
        {
            "dataset": verdict.to_dict(),
            "step_budget": budget_payload,
            "identity": identity_payload,
            "capacity": None if request.capacity is None else request.capacity.to_dict(),
            "ceilings": {
                "optimizer_steps": PILOT_MAX_OPTIMIZER_STEPS,
                "segment_steps": PILOT_MAX_SEGMENT_STEPS,
                "wall_clock_seconds": PILOT_MAX_WALL_CLOCK_SECONDS,
            },
            "dataset_report": report,
            "note": "prepare starts nothing",
        }
    )
    return 0 if verdict.permitted else 1


def cmd_pilot_run(args: argparse.Namespace) -> int:
    """Run one bounded pilot: two segments, a checkpoint, a resume.

    The only verb in LUBER that trains on real music, and it can only
    train a bounded amount of it. Every ceiling is a module constant and
    no argument here reaches one.
    """
    from luber_training.cli import _orchestrator

    orchestrator = _orchestrator(args)
    request = _pilot_request(args, orchestrator)
    result = run_pilot(request)

    directory = Path(request.workspace)
    paths = write_pilot_artifacts(result, directory)
    orchestrator.record_pilot(args.run_id, result.to_dict())

    if args.cleanup:
        from luber_training.canary import cleanup_workspace

        cleanup_workspace(request.workspace)

    _print({**result.to_dict(), "artifacts": {key: str(value) for key, value in paths.items()}})
    return 0 if result.outcome == PilotOutcome.COMPLETED_VALID_SIGNAL.value else 1


def cmd_pilot_status(args: argparse.Namespace) -> int:
    """The recorded pilot for a run, if one has been run."""
    from luber_training.cli import _orchestrator

    orchestrator = _orchestrator(args)
    run = orchestrator.get_run(args.run_id)
    directory = Path(run.output_directory or orchestrator.artifacts_root / args.run_id)
    path = directory / "pilot.json"
    if not path.is_file():
        _print({"available": False, "reason": "no pilot has been run for this run"})
        return 1
    _print(json.loads(path.read_text(encoding="utf-8")))
    return 0


def cmd_pilot_verify(args: argparse.Namespace) -> int:
    """Re-check a recorded pilot's evidence without re-running it."""
    from luber_training.cli import _orchestrator
    from luber_training.pilot import PilotTrainingResult, classify_signal

    orchestrator = _orchestrator(args)
    run = orchestrator.get_run(args.run_id)
    directory = Path(run.output_directory or orchestrator.artifacts_root / args.run_id)
    path = directory / "pilot.json"
    if not path.is_file():
        _print({"available": False, "reason": "no pilot has been run for this run"})
        return 1
    result = PilotTrainingResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
    signal, detail = classify_signal(
        loss=result.loss,
        parameters=result.parameters,
        gradients=result.gradients,
        expected_steps=result.expected_steps,
        completed_steps=result.completed_steps,
    )
    _print(
        {
            "pilot_id": result.pilot_id,
            "recorded_signal": result.signal,
            "recomputed_signal": signal,
            "agrees": signal == result.signal,
            "detail": detail,
            "within_budget": result.within_budget,
            "dataset_kind": result.dataset_kind,
            "artifact_class": list(result.artifact_class),
        }
    )
    return 0 if signal == result.signal else 1


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
        "--latent-shape-count",
        type=int,
        default=1,
        help=(
            "how many distinct latent lengths the dataset holds. Metal keeps an allocator "
            "working set per shape, so a profile measured over one shape does not qualify "
            "a run over many — Phase 36 lost four runs to exactly that"
        ),
    )
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
    preflight.add_argument(
        "--dataset-dir",
        help=(
            "preprocessed tensors the run will read. Without it the storage checks stay "
            "UNKNOWN, because nothing was looked at"
        ),
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

    pilot = sub.add_parser("pilot", help="a bounded real LoRA pilot: tens of steps, never more")
    pilot_sub = pilot.add_subparsers(dest="action", required=True)

    def _add_pilot(parser: argparse.ArgumentParser) -> None:
        _add_common(parser)
        parser.add_argument("--dataset-dir", required=True, help="preprocessed pilot tensors")
        parser.add_argument("--dataset-id", help="stable id for the pilot dataset")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--workspace", help="where the pilot writes; beneath the trainer root")
        parser.add_argument(
            "--preflight-status",
            help="the Phase 33 preflight status; anything but READY blocks the pilot",
        )
        parser.add_argument(
            "--allow-synthetic-fixture",
            action="store_true",
            help=(
                "run against a synthetic fixture to check the mechanism. The result is "
                "stamped SYNTHETIC_FIXTURE and is never real-data evidence"
            ),
        )
        parser.add_argument(
            "--timeout",
            type=float,
            default=PILOT_MAX_WALL_CLOCK_SECONDS,
            help="wall clock for one segment, clamped by the module ceiling",
        )
        parser.add_argument(
            "--resume",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="run a second segment from the checkpoint the first one wrote",
        )

    prepare = pilot_sub.add_parser("prepare", help="what a pilot would do; starts nothing")
    _add_pilot(prepare)
    prepare.set_defaults(func=cmd_pilot_prepare)

    pilot_run = pilot_sub.add_parser("run", help="run one bounded pilot")
    _add_pilot(pilot_run)
    pilot_run.add_argument("--cleanup", action="store_true")
    pilot_run.set_defaults(func=cmd_pilot_run)

    status = pilot_sub.add_parser("status", help="the recorded pilot for a run")
    _add_common(status)
    status.set_defaults(func=cmd_pilot_status)

    verify = pilot_sub.add_parser(
        "verify", help="re-check a recorded pilot's evidence without re-running it"
    )
    _add_common(verify)
    verify.set_defaults(func=cmd_pilot_verify)


__all__ = [
    "DEFAULT_CANARY_DIRNAME",
    "add_preflight_parsers",
    "cmd_canary_ace_step",
    "cmd_canary_orchestration",
    "cmd_capacity",
    "cmd_pilot_prepare",
    "cmd_pilot_run",
    "cmd_pilot_status",
    "cmd_pilot_verify",
    "cmd_preflight",
    "cmd_profile_memory",
]
