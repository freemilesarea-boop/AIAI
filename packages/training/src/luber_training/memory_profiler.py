"""Running a bounded memory profile, and deciding what it is evidence of.

The control-plane half of Phase 34. It builds a workload, hands it to
:mod:`luber_training._memory_probe` under the trainer's own interpreter,
bounds it with a wall clock, and turns what comes back into a
:class:`~luber_training.memory.TrainingMemoryProfile`.

The design decision that shapes everything here is **sequence length**.

The VAE downsamples 48 kHz audio by 2·4·4·6·10 = 1920, which is 25
latent frames a second, and preprocessing caps a track at 240 seconds —
so a production sample is around 6000 frames. Phase 33's canary ran at
64 frames, about two and a half seconds. Activation memory scales with
that dimension, so the canary's peak is a real measurement of a workload
production does not resemble, and using it to qualify a real run would
be worse than having no measurement: it would look like evidence.

A profile therefore carries the length it measured, says how
representative that is, and never qualifies a longer one.

Nothing here changes a configuration to make it fit. If a smaller
workload is wanted, that is a different profile with a different
identity, and the caller asks for it by name.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from luber_hardware import ComputeDevice
from luber_training import _memory_probe
from luber_training.canary import (
    CANARY_ABSOLUTE_WALL_CLOCK_SECONDS,
    CanaryEnvelope,
    DatasetVerdict,
    bound_plan,
    generate_synthetic_fixture,
    verify_canary_dataset,
    within,
)
from luber_training.gates import GateReport
from luber_training.memory import (
    MEMORY_PROBE_PROTOCOL_VERSION,
    MemoryDomain,
    MemoryProfileIdentity,
    MemorySnapshot,
    ProfileOutcome,
    ProfileStage,
    Representativeness,
    RuntimeIdentity,
    TrainingMemoryProfile,
    classify_memory_failure,
    summarise_peaks,
)
from luber_training.plan import TrainingPlan
from luber_training.trainer_adapter import compile_command

#: Latent frames per second of audio.
#:
#: 48 000 Hz divided by the VAE's downsampling product
#: (2·4·4·6·10 = 1920), read from `checkpoints/vae/config.json`. Derived
#: arithmetic over two published numbers, not a measurement — and the
#: derivation is here so it can be checked rather than trusted.
LATENT_FRAMES_PER_SECOND = 25.0

#: What preprocessing will actually produce for a full track:
#: `preprocess_audio_files(max_duration=240.0)` at the rate above.
PRODUCTION_MAX_DURATION_SECONDS = 240.0
PRODUCTION_LATENT_LENGTH = int(PRODUCTION_MAX_DURATION_SECONDS * LATENT_FRAMES_PER_SECOND)

#: A conditioning length for the probe.
#:
#: Production encoder length depends on how much text and lyric a track
#: carries and nothing in this repository has measured its distribution,
#: so this is a **stated probe parameter** rather than a claim about
#: production. It is part of the profile identity, and a profile does
#: not qualify a longer one.
DEFAULT_PROBE_ENCODER_LENGTH = 256

#: Share of the production length below which a profile is not evidence
#: about production at all. Two thirds: a measurement on most of a track
#: says something about a whole one; a measurement on a hundredth does
#: not.
REPRESENTATIVE_RATIO = 0.66
PARTIAL_RATIO = 0.25

#: How long a bounded profile may run. The trainer must load a 4.5 GB
#: model before it can take a step, and a long sequence takes longer per
#: step, so this is more generous than the canary's — and still bounded
#: by the canary module's absolute ceiling.
DEFAULT_PROFILE_TIMEOUT_SECONDS = 2400.0

#: Where a profile keeps its workspace, beneath the trainer root.
#: ACE-Step validates `--dataset-dir` against its working directory, so
#: a dataset anywhere else is refused after the model has loaded.
PROFILE_SUBDIR = "profile"

#: Defaults for the in-process safety boundary. Both are observations of
#: a reading that already exists, not predictions.
DEFAULT_HOST_AVAILABLE_FLOOR_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_MPS_RECOMMENDED_MAX_FRACTION = 0.92


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ProfilerError(RuntimeError):
    """Raised when a profile cannot be attempted as asked."""


@dataclass(frozen=True)
class ProbeShape:
    """The tensor dimensions a profile measures.

    Named separately from the training config because the trainer does
    not have a flag for them — they are properties of the *data*, fixed
    when the preprocessed tensors were written. A profile that did not
    record them would be a memory figure with no idea what it was a
    memory figure for.
    """

    latent_length: int
    encoder_length: int = DEFAULT_PROBE_ENCODER_LENGTH
    samples: int = 2

    @property
    def latent_seconds(self) -> float:
        return self.latent_length / LATENT_FRAMES_PER_SECOND

    def representativeness(self) -> tuple[str, str]:
        """How much like a production sample this shape is.

        Judged only on the sequence dimension, because that is the one
        this repository can compare against a known production value.
        The conditioning length is recorded and not judged — nothing has
        measured what production lyrics come to.
        """
        ratio = self.latent_length / PRODUCTION_LATENT_LENGTH
        detail = (
            f"{self.latent_length} latent frames ≈ {self.latent_seconds:.0f}s of audio, "
            f"against a production maximum of {PRODUCTION_LATENT_LENGTH} frames "
            f"({PRODUCTION_MAX_DURATION_SECONDS:.0f}s at {LATENT_FRAMES_PER_SECOND:g} "
            f"frames/s). Conditioning length {self.encoder_length} is a stated probe "
            "parameter: production encoder length depends on the text and lyrics a track "
            "carries and nothing has measured its distribution"
        )
        if ratio >= REPRESENTATIVE_RATIO:
            return Representativeness.REPRESENTATIVE.value, detail
        if ratio >= PARTIAL_RATIO:
            return Representativeness.PARTIALLY_REPRESENTATIVE.value, detail
        return Representativeness.NOT_REPRESENTATIVE.value, detail

    def to_dict(self) -> dict[str, Any]:
        return {
            "latent_length": self.latent_length,
            "latent_seconds": round(self.latent_seconds, 2),
            "encoder_length": self.encoder_length,
            "samples": self.samples,
        }


def identity_for(
    plan: TrainingPlan,
    shape: ProbeShape,
    *,
    model_variant: str = "turbo",
) -> MemoryProfileIdentity:
    """The configuration a profile of this plan would belong to.

    Everything comes from the plan and the shape. The micro batch is the
    config's `batch_size` because that is what the trainer hands the
    DataLoader unchanged — gradient accumulation is recorded beside it
    and multiplies the *effective* batch without holding more
    activations at once.
    """
    config = plan.config
    device = plan.requirements.execution_device or ComputeDevice.CPU.value
    return MemoryProfileIdentity(
        device=device,
        precision=config.precision,
        optimizer=config.optimizer_type,
        strategy=config.strategy,
        micro_batch_size=config.batch_size,
        gradient_accumulation=config.gradient_accumulation,
        gradient_checkpointing=config.gradient_checkpointing,
        lora_rank=config.rank,
        lora_alpha=config.alpha,
        target_modules=tuple(config.target_modules),
        attention_type=config.attention_type,
        latent_length=shape.latent_length,
        encoder_length=shape.encoder_length,
        model_variant=model_variant,
        base_model_upstream_commit=plan.base_model_upstream_commit,
        ace_step_commit=config.ace_step_commit,
        num_devices=config.num_devices,
        offload_encoder=config.offload_encoder,
    )


def profile_id_for(identity: MemoryProfileIdentity, plan_digest: str) -> str:
    """A readable, deterministic name for one profile.

    Device, precision and length up front because those are what an
    operator scans a directory for; the identity digest so two profiles
    that differ anywhere cannot collide.
    """
    return (
        f"{identity.device.lower()}-{identity.precision}-"
        f"b{identity.micro_batch_size}-r{identity.lora_rank}-"
        f"t{identity.latent_length}-{identity.digest()[:12]}"
    )


@dataclass
class ProfileRequest:
    """Everything one bounded profiling run needs."""

    plan: TrainingPlan
    shape: ProbeShape
    trainer_root: Path
    python_executable: Path
    model_dir: Path
    workspace: Path
    envelope: CanaryEnvelope
    model_variant: str = "turbo"
    timeout_seconds: float = DEFAULT_PROFILE_TIMEOUT_SECONDS
    sample_interval_seconds: float = _memory_probe.DEFAULT_SAMPLE_INTERVAL
    dataset_dir: Path | None = None
    gate_report: GateReport | None = None
    runs_control_plane: bool = True
    luber_commit: str | None = None
    #: Measure the resume path as well, in its own leg.
    #:
    #: A second bounded invocation against the checkpoint the first one
    #: wrote. Resume loads optimizer state that a fresh start allocates
    #: from nothing, so the two are not the same measurement and the
    #: profile keeps them apart.
    measure_resume: bool = False
    host_available_floor_bytes: int = DEFAULT_HOST_AVAILABLE_FLOOR_BYTES
    mps_recommended_max_fraction: float = DEFAULT_MPS_RECOMMENDED_MAX_FRACTION


def _blocked(
    request: ProfileRequest, identity: MemoryProfileIdentity, reason: str
) -> TrainingMemoryProfile:
    representativeness, detail = request.shape.representativeness()
    return TrainingMemoryProfile(
        profile_id=profile_id_for(identity, request.plan.digest()),
        plan_digest=request.plan.digest(),
        identity=identity,
        runtime=RuntimeIdentity(
            ace_step_commit=identity.ace_step_commit,
            luber_commit=request.luber_commit,
            device_class=identity.device,
        ),
        outcome=ProfileOutcome.BLOCKED.value,
        representativeness=representativeness,
        representativeness_detail=detail,
        failure_reason=reason,
        started_at=_now(),
        finished_at=_now(),
    )


def profile_memory(request: ProfileRequest) -> TrainingMemoryProfile:
    """Measure one configuration, inside the real trainer, under a clock.

    The workload is the real trainer on synthetic tensors of the
    requested shape. Synthetic because the shapes are what memory
    depends on and no recording is needed to produce them — and because
    a profile must not require material nobody is allowed to train on.
    Rights are checked all the same: a directory that is neither a
    synthetic fixture nor gate-cleared is refused, exactly as the canary
    refuses it.
    """
    identity = identity_for(request.plan, request.shape, model_variant=request.model_variant)

    if not Path(request.trainer_root).is_dir():
        return _blocked(request, identity, f"no trainer is installed at {request.trainer_root}")
    if not Path(request.python_executable).is_file():
        return _blocked(request, identity, f"no interpreter at {request.python_executable}")
    if not Path(request.model_dir).is_dir():
        return _blocked(
            request,
            identity,
            f"the base model root {request.model_dir} does not exist. A profile never "
            "downloads weights: it runs against what is installed or it does not run",
        )
    if request.timeout_seconds > CANARY_ABSOLUTE_WALL_CLOCK_SECONDS:
        raise ProfilerError(
            f"a bounded profile may run for up to {CANARY_ABSOLUTE_WALL_CLOCK_SECONDS:.0f}s; "
            f"{request.timeout_seconds:.0f}s was asked for"
        )

    workspace = Path(request.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    fixture_dir = request.dataset_dir or (workspace / "dataset")
    output_dir = workspace / "output"

    if not within(fixture_dir, Path(request.trainer_root)):
        return _blocked(
            request,
            identity,
            f"the profile dataset at {fixture_dir} is outside the trainer's working "
            f"directory ({request.trainer_root}); ACE-Step refuses it after the model has "
            "loaded",
        )

    verdict: DatasetVerdict
    if request.dataset_dir is None:
        verdict = generate_synthetic_fixture(
            python_executable=request.python_executable,
            trainer_root=Path(request.trainer_root),
            destination=fixture_dir,
            envelope=request.envelope,
            latent_length=request.shape.latent_length,
            encoder_length=request.shape.encoder_length,
        )
    else:
        verdict = verify_canary_dataset(
            fixture_dir, envelope=request.envelope, gate_report=request.gate_report
        )
    if not verdict.permitted:
        return _blocked(request, identity, verdict.detail)

    bounded = bound_plan(
        request.plan,
        request.envelope,
        dataset_dir=fixture_dir,
        output_dir=output_dir,
        model_dir=Path(request.model_dir),
    )
    command = compile_command(
        bounded,
        trainer_root=str(request.trainer_root),
        python_executable=str(request.python_executable),
        model_variant=request.model_variant,
    )
    # argv[0] is the interpreter; the probe *is* the interpreter, so what
    # it needs is the script and its arguments.
    trainer_argv = command.argv[1:]

    result_path = workspace / "memory_probe.json"
    payload = {
        "argv": trainer_argv,
        "result_path": str(result_path),
        "sample_interval": request.sample_interval_seconds,
        "device": identity.device,
        "ace_step_commit": identity.ace_step_commit,
        "luber_commit": request.luber_commit,
        "limits": {
            "host_available_floor_bytes": request.host_available_floor_bytes,
            "mps_recommended_max_fraction": request.mps_recommended_max_fraction,
        },
    }

    started_at = _now()
    started = time.perf_counter()
    timed_out = False
    log_path = workspace / "profile.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    script = Path(_memory_probe.__file__).resolve()
    with log_path.open("wb") as handle:
        process = subprocess.Popen(
            [str(request.python_executable), str(script)],
            cwd=str(request.trainer_root),
            stdin=subprocess.PIPE,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        try:
            assert process.stdin is not None
            process.stdin.write(json.dumps(payload).encode("utf-8"))
            process.stdin.close()
            process.wait(timeout=request.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate(process)
        except BrokenPipeError:
            _terminate(process)
    wall_seconds = round(time.perf_counter() - started, 3)

    tail = _tail(log_path)
    document = _read_result(result_path)
    profile = _build_profile(
        request,
        identity,
        document=document,
        timed_out=timed_out,
        wall_seconds=wall_seconds,
        started_at=started_at,
        log_tail=tail,
    )

    if request.measure_resume and profile.completed:
        profile.resume_peak_bytes = _measure_resume(
            request, bounded, output_dir, workspace, identity
        )
        if profile.resume_peak_bytes is not None:
            profile.not_observed.pop(ProfileStage.RESUME_LOADED.value, None)
            profile.not_observed.pop(ProfileStage.RESUME_STEP_COMPLETE.value, None)
    return profile


def _measure_resume(
    request: ProfileRequest,
    bounded: TrainingPlan,
    output_dir: Path,
    workspace: Path,
    identity: MemoryProfileIdentity,
) -> int | None:
    """Run a second bounded leg from the checkpoint the first one wrote.

    Reported as its own number rather than folded into the peak: a fresh
    start and a resume allocate differently, and an operator planning a
    long run that will be interrupted needs to know the larger of the
    two rather than an average of them.
    """
    from luber_training.canary import latest_checkpoint

    checkpoint = latest_checkpoint(output_dir)
    if checkpoint is None:
        return None

    resume_envelope = CanaryEnvelope(max_samples=request.envelope.max_samples, resume=True)
    resumed = replace(
        bounded, config=bounded.config.with_overrides(epochs=resume_envelope.max_epochs)
    )
    command = compile_command(
        resumed,
        trainer_root=str(request.trainer_root),
        python_executable=str(request.python_executable),
        model_variant=request.model_variant,
        resume_from=str(checkpoint),
    )
    result_path = workspace / "memory_probe_resume.json"
    payload = {
        "argv": command.argv[1:],
        "result_path": str(result_path),
        "sample_interval": request.sample_interval_seconds,
        "device": identity.device,
        "ace_step_commit": identity.ace_step_commit,
        "luber_commit": request.luber_commit,
        "limits": {
            "host_available_floor_bytes": request.host_available_floor_bytes,
            "mps_recommended_max_fraction": request.mps_recommended_max_fraction,
        },
    }
    log_path = workspace / "profile-resume.log"
    script = Path(_memory_probe.__file__).resolve()
    with log_path.open("wb") as handle:
        process = subprocess.Popen(
            [str(request.python_executable), str(script)],
            cwd=str(request.trainer_root),
            stdin=subprocess.PIPE,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        try:
            assert process.stdin is not None
            process.stdin.write(json.dumps(payload).encode("utf-8"))
            process.stdin.close()
            process.wait(timeout=request.timeout_seconds)
        except subprocess.TimeoutExpired:
            _terminate(process)
            return None
        except BrokenPipeError:
            _terminate(process)
            return None

    document = _read_result(result_path)
    if document is None or document.get("outcome") != ProfileOutcome.COMPLETED.value:
        return None
    snapshots = [
        MemorySnapshot.from_dict(item)
        for item in document.get("snapshots") or []
        if isinstance(item, dict)
    ]
    peaks = summarise_peaks(snapshots, device=identity.device)
    accelerator = next((peak for peak in peaks if peak.domain != MemoryDomain.HOST.value), None)
    return accelerator.peak_bytes if accelerator is not None else None


def _terminate(process: subprocess.Popen[bytes]) -> None:
    """Stop a profile that outran its clock, group and all."""
    try:
        os.killpg(os.getpgid(process.pid), 15)
    except OSError:
        process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), 9)
        except OSError:
            process.kill()
        process.wait(timeout=30)


def _tail(path: Path, lines: int = 40) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(line for line in content[-lines:] if line.strip())


def _read_result(path: Path) -> dict[str, Any] | None:
    """Read the probe's document, refusing anything malformed.

    A profile is evidence, and evidence that cannot be parsed is not
    evidence with a default — it is absent. Nothing here evaluates what
    it reads, and an unknown protocol version is refused rather than
    interpreted.
    """
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(document, dict):
        return None
    version = str(document.get("protocol_version", ""))
    if version != MEMORY_PROBE_PROTOCOL_VERSION:
        return {
            "outcome": ProfileOutcome.FAILED.value,
            "failure_reason": (
                f"the probe reported protocol {version!r}, and this build reads "
                f"{MEMORY_PROBE_PROTOCOL_VERSION!r}"
            ),
            "snapshots": [],
        }
    return document


def _build_profile(
    request: ProfileRequest,
    identity: MemoryProfileIdentity,
    *,
    document: dict[str, Any] | None,
    timed_out: bool,
    wall_seconds: float,
    started_at: str,
    log_tail: str,
) -> TrainingMemoryProfile:
    representativeness, representativeness_detail = request.shape.representativeness()
    profile = TrainingMemoryProfile(
        profile_id=profile_id_for(identity, request.plan.digest()),
        plan_digest=request.plan.digest(),
        identity=identity,
        runtime=RuntimeIdentity(
            ace_step_commit=identity.ace_step_commit,
            luber_commit=request.luber_commit,
            device_class=identity.device,
        ),
        representativeness=representativeness,
        representativeness_detail=representativeness_detail,
        started_at=started_at,
        finished_at=_now(),
        wall_seconds=wall_seconds,
        sampler_interval_seconds=request.sample_interval_seconds,
    )

    if timed_out:
        profile.outcome = ProfileOutcome.PROFILE_TIMEOUT.value
        profile.failure_reason = (
            f"the profile exceeded its {request.timeout_seconds:.0f}s wall clock and was "
            "stopped. A killed run's peak is a peak of however far it got"
        )
        profile.failure_kind = classify_memory_failure(log_tail)
        return profile

    if document is None:
        profile.outcome = ProfileOutcome.FAILED.value
        profile.failure_reason = (
            "the probe wrote no readable profile document; the trainer log tail is the only "
            "evidence of what happened"
        )
        profile.failure_kind = classify_memory_failure(log_tail)
        return profile

    snapshots = [
        MemorySnapshot.from_dict(item)
        for item in document.get("snapshots") or []
        if isinstance(item, dict)
    ]
    profile.snapshots = snapshots
    profile.peaks = summarise_peaks(
        snapshots,
        device=identity.device,
        runtime_peaks={
            str(key): int(value)
            for key, value in (document.get("runtime_peaks") or {}).items()
            if isinstance(value, (int, float))
        },
    )
    profile.not_observed = {
        str(key): str(value) for key, value in (document.get("not_observed") or {}).items()
    }
    profile.optimizer_steps = (
        int(document["optimizer_steps"])
        if isinstance(document.get("optimizer_steps"), (int, float))
        else None
    )
    profile.checkpoint_peak_bytes = _stage_peak(
        snapshots, identity.device, ProfileStage.CHECKPOINT_COMPLETE
    )
    profile.resume_peak_bytes = _stage_peak(
        snapshots, identity.device, ProfileStage.RESUME_STEP_COMPLETE
    )

    runtime = document.get("runtime_identity") or {}
    if isinstance(runtime, dict):
        profile.runtime = RuntimeIdentity.from_dict(
            {**runtime, "luber_commit": request.luber_commit or runtime.get("luber_commit")}
        )

    outcome = str(document.get("outcome", ProfileOutcome.FAILED.value))
    profile.outcome = (
        outcome
        if outcome in {item.value for item in ProfileOutcome}
        else ProfileOutcome.FAILED.value
    )
    profile.failure_reason = str(document.get("failure_reason", ""))
    profile.failure_kind = classify_memory_failure(
        profile.failure_reason or document.get("traceback_tail") or log_tail
    )

    if document.get("truncated"):
        profile.not_observed["SAMPLING"] = (
            "the snapshot cap was reached and sampling stopped; the peak is drawn from what "
            "was recorded before that"
        )
    if document.get("sampler_running_after_stop"):
        profile.not_observed["SAMPLER"] = "the sampler thread did not stop cleanly"
    for stage in ProfileStage:
        observed = set(document.get("observed_stages") or [])
        if stage.value not in observed and stage.value not in profile.not_observed:
            profile.not_observed[stage.value] = (
                "not reached by this workload" if profile.completed else "the run did not get here"
            )
    return profile


def _stage_peak(snapshots: list[MemorySnapshot], device: str, stage: ProfileStage) -> int | None:
    """The highest figure recorded at one stage, in the device's domain."""
    from luber_training.memory import MemoryDomain

    domain = (
        MemoryDomain.APPLE_UNIFIED.value
        if device == ComputeDevice.MPS.value
        else MemoryDomain.CUDA_DEVICE.value
        if device == ComputeDevice.CUDA.value
        else MemoryDomain.HOST.value
    )
    values = [
        snapshot.value_for(domain)
        for snapshot in snapshots
        if snapshot.stage == stage.value and snapshot.value_for(domain) is not None
    ]
    return max(values) if values else None  # type: ignore[type-var]


# ── artifacts ────────────────────────────────────────────────────────

PROFILE_JSON_NAME = "training_memory_profile.json"
PROFILE_MARKDOWN_NAME = "training_memory_profile.md"


def write_profile(profile: TrainingMemoryProfile, directory: Path) -> Path:
    """Persist a profile as an operational artifact.

    One file per profile identity, so evidence for bf16 does not
    overwrite evidence for fp32 and a second machine's measurement does
    not overwrite the first's. These are runtime records and belong in a
    registry directory, never in git.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{profile.profile_id}.json"
    path.write_text(
        json.dumps(profile.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_profiles(directory: Path) -> list[TrainingMemoryProfile]:
    """Every readable profile in a directory, newest last.

    A file this build cannot parse is skipped rather than raising: one
    stale document should not make every other measurement unreadable.
    """
    if not directory.is_dir():
        return []
    profiles: list[TrainingMemoryProfile] = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        try:
            profiles.append(TrainingMemoryProfile.from_dict(payload))
        except Exception:
            continue
    return profiles


def render_markdown(profile: TrainingMemoryProfile) -> str:
    """An operator-readable form of one profile.

    Every figure carries its domain and its kind. Apple's numbers are
    called unified memory, because that is what they are, and a reader
    who skims must not come away thinking the machine has 24 GB of VRAM.
    """
    from luber_training.memory import MIB, MemoryDomain

    identity = profile.identity
    lines = [
        f"# Training memory profile — {profile.profile_id}",
        "",
        f"- outcome: **{profile.outcome}**",
        f"- representativeness: **{profile.representativeness}**",
        f"  - {profile.representativeness_detail}",
        f"- plan digest: `{profile.plan_digest[:16]}`",
        f"- identity digest: `{profile.identity_digest[:16]}`",
        f"- measured: {profile.started_at} → {profile.finished_at} "
        f"({profile.wall_seconds}s wall clock)",
        "",
        "## Configuration",
        "",
        f"- device **{identity.device}**, precision **{identity.precision}**, "
        f"optimizer **{identity.optimizer}**",
        f"- micro batch **{identity.micro_batch_size}**, gradient accumulation "
        f"**{identity.gradient_accumulation}** (effective batch "
        f"{identity.effective_batch_size})",
        f"- LoRA rank {identity.lora_rank}, alpha {identity.lora_alpha}, "
        f"modules {', '.join(identity.target_modules)}",
        f"- gradient checkpointing: {identity.gradient_checkpointing}",
        f"- latent length **{identity.latent_length}** frames "
        f"(≈{identity.latent_seconds(LATENT_FRAMES_PER_SECOND):.0f}s of audio), "
        f"encoder length {identity.encoder_length}",
        f"- model variant {identity.model_variant}, ACE-Step `{identity.ace_step_commit[:12]}`",
        "",
        "## Peaks",
        "",
        "| domain | kind | peak | baseline | growth |",
        "|---|---|---|---|---|",
    ]
    for peak in profile.peaks:
        label = (
            "unified memory (shared with the OS — not VRAM)"
            if peak.domain == MemoryDomain.APPLE_UNIFIED.value
            else peak.domain
        )
        lines.append(
            f"| {label} | {peak.kind} | "
            f"{'—' if peak.peak_bytes is None else f'{peak.peak_bytes // MIB} MiB'} | "
            f"{'—' if peak.baseline_bytes is None else f'{peak.baseline_bytes // MIB} MiB'} | "
            f"{'—' if peak.growth_bytes is None else f'{peak.growth_bytes // MIB} MiB (DERIVED)'} |"
        )

    lines += [
        "",
        "## Stages observed",
        "",
    ]
    seen: set[str] = set()
    for snapshot in profile.snapshots:
        if snapshot.stage in seen or snapshot.stage == "SAMPLE":
            continue
        seen.add(snapshot.stage)
        lines.append(f"- {snapshot.stage} at {snapshot.elapsed_seconds:.1f}s")
    if profile.not_observed:
        lines += ["", "## Not observed", ""]
        for stage, reason in sorted(profile.not_observed.items()):
            lines.append(f"- {stage}: {reason}")

    lines += [
        "",
        "## What this does not say",
        "",
        "A memory profile says what a configuration cost on one machine. It says nothing "
        "about music quality, convergence, generalisation, or whether the resulting model "
        "is any good. It qualifies no configuration other than the one it measured.",
        "",
    ]
    return "\n".join(lines)


def bounded_envelope_for(shape: ProbeShape) -> CanaryEnvelope:
    """A canary envelope sized for a profiling run of this shape.

    The same ceilings as everywhere else — a profile is a canary with
    instrumentation, not a licence to train for longer.
    """
    return CanaryEnvelope(max_samples=shape.samples, resume=False)


__all__ = [
    "DEFAULT_PROBE_ENCODER_LENGTH",
    "DEFAULT_PROFILE_TIMEOUT_SECONDS",
    "LATENT_FRAMES_PER_SECOND",
    "PARTIAL_RATIO",
    "PRODUCTION_LATENT_LENGTH",
    "PRODUCTION_MAX_DURATION_SECONDS",
    "PROFILE_JSON_NAME",
    "PROFILE_MARKDOWN_NAME",
    "PROFILE_SUBDIR",
    "REPRESENTATIVE_RATIO",
    "ProbeShape",
    "ProfileRequest",
    "ProfilerError",
    "bounded_envelope_for",
    "identity_for",
    "load_profiles",
    "profile_id_for",
    "profile_memory",
    "render_markdown",
    "write_profile",
]
