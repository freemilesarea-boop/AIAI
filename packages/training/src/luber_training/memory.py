"""What a training run actually costs in memory, and how we know.

Phase 33 could say only that the memory requirement was UNKNOWN. This
module is the vocabulary for saying something else — and most of its
design is about the ways that saying something else goes wrong.

**Three domains, never merged.** Host RAM, Apple unified memory and
CUDA device memory answer different questions and are reported in
different fields. A resident-set figure on Apple silicon is not "the
memory the model used", because the GPU allocates from the same pool the
process already counts; a CUDA `memory_allocated` is not host RSS. There
is no field that could hold either.

**Peaks come in two kinds.** CUDA keeps its own high-water mark
(`max_memory_allocated`) — that is a `RUNTIME_PEAK`, exact by
construction. The pinned torch has **no** MPS equivalent, so the highest
value a sampler happened to observe is a `SAMPLED_PEAK`: a lower bound
on the truth, and labelled as one everywhere it appears. Calling a
sampled maximum an exact peak would be the fabrication this phase exists
to avoid.

**A measurement belongs to one configuration.** Batch size, precision,
LoRA rank, gradient checkpointing and — the one that matters most here —
the latent sequence length all change what a run costs. So a profile
carries an identity digest over exactly those fields, and a profile
whose identity does not match a request does not qualify it. A figure
measured on 2.5 seconds of latents says nothing about four minutes.

**Nothing here allocates anything.** This module has no torch import and
never touches a device; it is the shape the trainer-side probe reports
into and the shape the qualifier reads.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from luber_training.capacity import EvidenceSource

MEMORY_PROFILE_SCHEMA_VERSION = "luber-training-memory-profile/1"

#: The protocol version the trainer-side probe stamps on what it emits.
#: A document from a newer probe is refused rather than read with its new
#: fields ignored — the fields it added would be exactly the ones a stale
#: reader needed.
MEMORY_PROBE_PROTOCOL_VERSION = "luber-memory-probe/1"

#: Bytes per mebibyte, in one place, so a reader never has to check
#: whether a figure was divided by 1000 or 1024 somewhere.
MIB = 1024 * 1024


class MemoryDomain(StrEnum):
    """Which pool a figure came out of.

    Kept apart because they are not comparable. A Mac's unified memory
    is shared with the operating system and with the process's own host
    allocations, so the same bytes can appear in two places; a CUDA
    card's memory is separate from the host's and does not.
    """

    HOST = "HOST"
    #: Apple's unified memory, as `torch.mps` reports it. Never VRAM.
    APPLE_UNIFIED = "APPLE_UNIFIED"
    CUDA_DEVICE = "CUDA_DEVICE"


class PeakKind(StrEnum):
    """How a peak was arrived at.

    ``RUNTIME_PEAK`` is a high-water mark the runtime itself kept.
    ``SAMPLED_PEAK`` is the largest value a sampler observed, which is a
    **lower bound**: anything that rose and fell between two samples was
    never seen. The pinned torch offers no MPS peak counter, so every
    Apple figure in this project is sampled.
    """

    RUNTIME_PEAK = "RUNTIME_PEAK"
    SAMPLED_PEAK = "SAMPLED_PEAK"
    NOT_AVAILABLE = "NOT_AVAILABLE"


class ProfileStage(StrEnum):
    """Points in the real trainer's lifecycle worth a measurement.

    Every one of these is observed by wrapping a callable the installed
    trainer already calls, so a stage that appears in a profile really
    happened. A stage nothing could observe is recorded in
    ``not_observed`` with the reason rather than omitted — an absent
    stage and an unobservable one look identical otherwise.
    """

    BASELINE = "BASELINE"
    RUNTIME_INITIALIZED = "RUNTIME_INITIALIZED"
    MODEL_LOADED = "MODEL_LOADED"
    LORA_ATTACHED = "LORA_ATTACHED"
    OPTIMIZER_CREATED = "OPTIMIZER_CREATED"
    BATCH_READY = "BATCH_READY"
    FORWARD_COMPLETE = "FORWARD_COMPLETE"
    BACKWARD_COMPLETE = "BACKWARD_COMPLETE"
    OPTIMIZER_STEP_COMPLETE = "OPTIMIZER_STEP_COMPLETE"
    CHECKPOINT_BEGIN = "CHECKPOINT_BEGIN"
    CHECKPOINT_COMPLETE = "CHECKPOINT_COMPLETE"
    RESUME_LOADED = "RESUME_LOADED"
    RESUME_STEP_COMPLETE = "RESUME_STEP_COMPLETE"
    FINAL = "FINAL"


#: The order stages are expected in. Used to check a profile is coherent
#: rather than to reorder it: a profile whose stages arrived out of order
#: is a profile something went wrong in, and silently sorting it would
#: hide that.
STAGE_ORDER: tuple[str, ...] = tuple(item.value for item in ProfileStage)


class MemoryFailureKind(StrEnum):
    """Why a memory-related failure happened, where it can be told.

    ``UNKNOWN_MEMORY_FAILURE`` is not a catch-all for every exception —
    it is for a failure that *looks* memory-shaped without matching a
    known signature. An ordinary `RuntimeError` stays an ordinary
    failure and is classified ``NOT_A_MEMORY_FAILURE``, because calling
    every crash an OOM would make the OOM code meaningless.
    """

    HOST_OOM = "HOST_OOM"
    MPS_OOM = "MPS_OOM"
    CUDA_OOM = "CUDA_OOM"
    UNKNOWN_MEMORY_FAILURE = "UNKNOWN_MEMORY_FAILURE"
    NOT_A_MEMORY_FAILURE = "NOT_A_MEMORY_FAILURE"


#: Signatures each runtime actually emits. Every entry is a string one of
#: these runtimes prints, not a phrase that sounded plausible.
_CUDA_SIGNATURES: tuple[str, ...] = (
    "cuda out of memory",
    "torch.cuda.outofmemoryerror",
    "cublas_status_alloc_failed",
    "cudaerrormemoryallocation",
)
_MPS_SIGNATURES: tuple[str, ...] = (
    "mps backend out of memory",
    "mpsndarray.mm",
    "failed to allocate",
    "total bytes of mps memory",
)
_HOST_SIGNATURES: tuple[str, ...] = (
    "cannot allocate memory",
    "std::bad_alloc",
    "memoryerror",
    "killed process",
    "out of memory: killed",
)
#: Memory-shaped but unattributable. `DefaultCPUAllocator` can fail for
#: reasons other than exhaustion, so it lands here rather than in HOST.
_GENERIC_SIGNATURES: tuple[str, ...] = (
    "out of memory",
    "defaultcpuallocator: can't allocate",
)


def classify_memory_failure(text: str | None) -> str:
    """What kind of memory failure this text describes, if any.

    Ordered most specific first, and deliberately conservative: a
    SIGKILL is not on any list, because the kernel OOM killer and
    `kill -9` are indistinguishable from outside the process.
    """
    if not text:
        return MemoryFailureKind.NOT_A_MEMORY_FAILURE.value
    haystack = text.lower()
    for signatures, kind in (
        (_CUDA_SIGNATURES, MemoryFailureKind.CUDA_OOM),
        (_MPS_SIGNATURES, MemoryFailureKind.MPS_OOM),
        (_HOST_SIGNATURES, MemoryFailureKind.HOST_OOM),
        (_GENERIC_SIGNATURES, MemoryFailureKind.UNKNOWN_MEMORY_FAILURE),
    ):
        if any(signature in haystack for signature in signatures):
            return kind.value
    return MemoryFailureKind.NOT_A_MEMORY_FAILURE.value


# ── one measurement ──────────────────────────────────────────────────


@dataclass(frozen=True)
class MemorySnapshot:
    """Every memory figure available at one moment, by domain.

    ``None`` means the runtime does not expose it or nobody could read
    it. It never means zero, and no field here is filled from another
    domain's number.
    """

    stage: str
    #: Seconds since the probe started. Not a wall-clock time: a profile
    #: is about a run's shape, and an absolute timestamp would make two
    #: otherwise identical profiles differ.
    elapsed_seconds: float = 0.0
    #: Free text for a stage that happened more than once — which step,
    #: which checkpoint.
    note: str = ""

    # ── host ──
    host_rss_bytes: int | None = None
    host_available_bytes: int | None = None
    system_total_bytes: int | None = None

    # ── Apple unified memory, as torch.mps reports it ──
    mps_current_allocated_bytes: int | None = None
    mps_driver_allocated_bytes: int | None = None
    mps_recommended_max_bytes: int | None = None

    # ── CUDA device ──
    cuda_allocated_bytes: int | None = None
    cuda_reserved_bytes: int | None = None
    cuda_peak_allocated_bytes: int | None = None
    cuda_peak_reserved_bytes: int | None = None
    cuda_total_bytes: int | None = None
    cuda_free_bytes: int | None = None

    def value_for(self, domain: str) -> int | None:
        """The figure this snapshot offers for *domain*, or None.

        Apple's number is the driver allocation rather than the current
        allocation: the driver figure is what the process is holding
        from the system, and the current figure excludes everything the
        caching allocator has kept back but not handed out.
        """
        if domain == MemoryDomain.HOST.value:
            return self.host_rss_bytes
        if domain == MemoryDomain.APPLE_UNIFIED.value:
            return self.mps_driver_allocated_bytes
        if domain == MemoryDomain.CUDA_DEVICE.value:
            return self.cuda_reserved_bytes
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "note": self.note,
            "host_rss_bytes": self.host_rss_bytes,
            "host_available_bytes": self.host_available_bytes,
            "system_total_bytes": self.system_total_bytes,
            "mps_current_allocated_bytes": self.mps_current_allocated_bytes,
            "mps_driver_allocated_bytes": self.mps_driver_allocated_bytes,
            "mps_recommended_max_bytes": self.mps_recommended_max_bytes,
            "cuda_allocated_bytes": self.cuda_allocated_bytes,
            "cuda_reserved_bytes": self.cuda_reserved_bytes,
            "cuda_peak_allocated_bytes": self.cuda_peak_allocated_bytes,
            "cuda_peak_reserved_bytes": self.cuda_peak_reserved_bytes,
            "cuda_total_bytes": self.cuda_total_bytes,
            "cuda_free_bytes": self.cuda_free_bytes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MemorySnapshot:
        byte_fields = {
            name: _optional_int(payload.get(name))
            for name in cls.__dataclass_fields__
            if name.endswith("_bytes")
        }
        return cls(
            stage=str(payload.get("stage") or ""),
            elapsed_seconds=float(payload.get("elapsed_seconds") or 0.0),
            note=str(payload.get("note") or ""),
            **byte_fields,
        )


# ── what the measurement is about ────────────────────────────────────


@dataclass(frozen=True)
class MemoryProfileIdentity:
    """The configuration a memory measurement belongs to.

    Every field changes what a run costs. Nothing volatile is here —
    no timestamp, no hostname, no free-memory reading, no pid — so two
    runs of the same configuration produce the same identity and a
    profile can be looked up by what it measured rather than by when.

    ``latent_length`` is the field this phase exists around. Activation
    memory scales with the sequence being trained on, and the Phase 33
    canary ran at 64 frames — about 2.5 seconds at the VAE's 25 Hz —
    where a production track is thousands. A profile that ignored it
    would qualify four minutes of audio with evidence from two seconds.
    """

    device: str
    precision: str
    optimizer: str
    strategy: str
    micro_batch_size: int
    gradient_accumulation: int
    gradient_checkpointing: bool
    lora_rank: int
    lora_alpha: int
    target_modules: tuple[str, ...]
    attention_type: str
    #: Latent frames per sample — the trained sequence length.
    latent_length: int
    #: Conditioning sequence length.
    encoder_length: int
    model_variant: str
    base_model_upstream_commit: str
    ace_step_commit: str
    num_devices: int = 1
    offload_encoder: bool = False
    #: How many *distinct* latent lengths the dataset holds.
    #:
    #: Phase 36 found this the hard way. Metal keeps an allocator
    #: working set per tensor shape, so a dataset of 24 different
    #: lengths reached 29 GiB where four tracks at the same maximum
    #: length peaked at 9.4 — and every attempt died at the same step.
    #: `latent_length` alone said the two workloads were identical.
    #: One means a fixed-shape dataset; a profile measured at one shape
    #: does not qualify a run over many.
    latent_shape_count: int = 1

    @property
    def effective_batch_size(self) -> int:
        """What the optimizer sees, as distinct from what memory sees.

        Gradient accumulation multiplies the effective batch without
        holding more activations at once, so the two numbers are
        recorded separately and the memory-relevant one is the micro
        batch.
        """
        return self.micro_batch_size * max(1, self.gradient_accumulation)

    def latent_seconds(self, frames_per_second: float = 25.0) -> float:
        """The audio duration this sequence length corresponds to.

        The rate is the VAE's: 48 kHz divided by its downsampling
        product (2·4·4·6·10 = 1920) is 25 frames a second. Passed in
        rather than hard-wired so a different VAE does not make this
        quietly wrong.
        """
        return self.latent_length / frames_per_second if frames_per_second else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "precision": self.precision,
            "optimizer": self.optimizer,
            "strategy": self.strategy,
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation": self.gradient_accumulation,
            "effective_batch_size": self.effective_batch_size,
            "gradient_checkpointing": self.gradient_checkpointing,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "target_modules": list(self.target_modules),
            "attention_type": self.attention_type,
            "latent_length": self.latent_length,
            "encoder_length": self.encoder_length,
            "latent_shape_count": self.latent_shape_count,
            "model_variant": self.model_variant,
            "base_model_upstream_commit": self.base_model_upstream_commit,
            "ace_step_commit": self.ace_step_commit,
            "num_devices": self.num_devices,
            "offload_encoder": self.offload_encoder,
        }

    def digest(self) -> str:
        """A fingerprint of the configuration, and of nothing else."""
        payload = {
            key: value
            for key, value in self.to_dict().items()
            # Derived from two fields already present; hashing it too
            # would make the digest depend on the same fact twice.
            if key != "effective_batch_size"
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
                "utf-8"
            )
        ).hexdigest()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MemoryProfileIdentity:
        return cls(
            device=str(payload.get("device", "")),
            precision=str(payload.get("precision", "")),
            optimizer=str(payload.get("optimizer", "")),
            strategy=str(payload.get("strategy", "")),
            micro_batch_size=int(payload.get("micro_batch_size", 0)),
            gradient_accumulation=int(payload.get("gradient_accumulation", 0)),
            gradient_checkpointing=bool(payload.get("gradient_checkpointing", False)),
            lora_rank=int(payload.get("lora_rank", 0)),
            lora_alpha=int(payload.get("lora_alpha", 0)),
            target_modules=tuple(payload.get("target_modules") or ()),
            attention_type=str(payload.get("attention_type", "")),
            latent_length=int(payload.get("latent_length", 0)),
            encoder_length=int(payload.get("encoder_length", 0)),
            # Profiles written before Phase 37 measured one shape,
            # because that is all a fixture generator produces. Reading
            # them as 1 is the truth about them, not a default.
            latent_shape_count=int(payload.get("latent_shape_count", 1)),
            model_variant=str(payload.get("model_variant", "")),
            base_model_upstream_commit=str(payload.get("base_model_upstream_commit", "")),
            ace_step_commit=str(payload.get("ace_step_commit", "")),
            num_devices=int(payload.get("num_devices", 1)),
            offload_encoder=bool(payload.get("offload_encoder", False)),
        )


@dataclass(frozen=True)
class RuntimeIdentity:
    """The software a measurement was taken against.

    Reproducibility only. There is no field here for a hostname, a
    username or a path, so a profile cannot carry one — and a profile
    travels: it is written on a worker, read on a control plane and
    quoted in an operator report.
    """

    python_version: str | None = None
    torch_version: str | None = None
    ace_step_commit: str | None = None
    luber_commit: str | None = None
    platform_class: str | None = None
    device_class: str | None = None
    probe_protocol_version: str = MEMORY_PROBE_PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "ace_step_commit": self.ace_step_commit,
            "luber_commit": self.luber_commit,
            "platform_class": self.platform_class,
            "device_class": self.device_class,
            "probe_protocol_version": self.probe_protocol_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RuntimeIdentity:
        return cls(
            python_version=_optional_str(payload.get("python_version")),
            torch_version=_optional_str(payload.get("torch_version")),
            ace_step_commit=_optional_str(payload.get("ace_step_commit")),
            luber_commit=_optional_str(payload.get("luber_commit")),
            platform_class=_optional_str(payload.get("platform_class")),
            device_class=_optional_str(payload.get("device_class")),
            probe_protocol_version=str(
                payload.get("probe_protocol_version") or MEMORY_PROBE_PROTOCOL_VERSION
            ),
        )


def _optional_str(value: Any) -> str | None:
    return str(value) if isinstance(value, str) and value.strip() else None


# ── the peak ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DomainPeak:
    """The highest figure seen in one domain, and how it was obtained."""

    domain: str
    kind: str
    peak_bytes: int | None = None
    #: Which stage the peak was observed at, where the sampler could
    #: attribute it. Sampling between stage markers cannot always.
    peak_stage: str | None = None
    baseline_bytes: int | None = None
    total_bytes: int | None = None
    #: How many samples the figure was drawn from. One sample is not a
    #: peak, and a reader should be able to see that.
    sample_count: int = 0
    detail: str = ""

    @property
    def source(self) -> str:
        """The evidence class this peak belongs to.

        A runtime high-water mark is MEASURED. A sampled maximum is
        MEASURED as well — the samples are real readings — but its
        `kind` says it is a lower bound, and nothing downstream may
        round that away.
        """
        if self.peak_bytes is None:
            return EvidenceSource.UNKNOWN.value
        return EvidenceSource.MEASURED.value

    @property
    def growth_bytes(self) -> int | None:
        """Peak minus baseline: what the run itself added.

        DERIVED, not measured — it is arithmetic over two readings, and
        it is reported as such wherever it is used.
        """
        if self.peak_bytes is None or self.baseline_bytes is None:
            return None
        return max(0, self.peak_bytes - self.baseline_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "kind": self.kind,
            "source": self.source,
            "peak_bytes": self.peak_bytes,
            "peak_mib": None if self.peak_bytes is None else self.peak_bytes // MIB,
            "peak_stage": self.peak_stage,
            "baseline_bytes": self.baseline_bytes,
            "total_bytes": self.total_bytes,
            "growth_bytes": self.growth_bytes,
            "growth_source": (
                EvidenceSource.UNKNOWN.value
                if self.growth_bytes is None
                else EvidenceSource.DERIVED.value
            ),
            "sample_count": self.sample_count,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DomainPeak:
        return cls(
            domain=str(payload.get("domain", "")),
            kind=str(payload.get("kind", PeakKind.NOT_AVAILABLE.value)),
            peak_bytes=_optional_int(payload.get("peak_bytes")),
            peak_stage=_optional_str(payload.get("peak_stage")),
            baseline_bytes=_optional_int(payload.get("baseline_bytes")),
            total_bytes=_optional_int(payload.get("total_bytes")),
            sample_count=int(payload.get("sample_count") or 0),
            detail=str(payload.get("detail", "")),
        )


def _optional_int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


class ProfileOutcome(StrEnum):
    """How a profiling run ended."""

    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    #: The bounded profile exceeded its wall clock. Not a success, and
    #: not evidence: a killed run's peak is a peak of however far it got.
    PROFILE_TIMEOUT = "PROFILE_TIMEOUT"
    #: Something outside the profiler stopped it — no trainer, no
    #: interpreter, no permitted data.
    BLOCKED = "BLOCKED"
    NOT_RUN = "NOT_RUN"


class Representativeness(StrEnum):
    """Whether a profile's workload resembles a production one.

    The distinction Phase 34 turns on. A profile measured on a sequence
    a hundredth of production length is a real measurement of an
    unrealistic workload, and using its peak to qualify a real run would
    be worse than having no measurement at all — it would look like
    evidence.
    """

    REPRESENTATIVE = "REPRESENTATIVE"
    PARTIALLY_REPRESENTATIVE = "PARTIALLY_REPRESENTATIVE"
    NOT_REPRESENTATIVE = "NOT_REPRESENTATIVE"
    UNKNOWN = "UNKNOWN"


@dataclass
class TrainingMemoryProfile:
    """One bounded measurement of what a configuration costs.

    A record of something that happened, not a decision. Whether the
    numbers are good enough for a given run is
    :mod:`luber_training.capacity_policy`'s question, and keeping the two
    apart is what lets a profile be re-read against a policy that has
    since changed.
    """

    profile_id: str
    plan_digest: str
    identity: MemoryProfileIdentity
    runtime: RuntimeIdentity
    outcome: str = ProfileOutcome.NOT_RUN.value
    representativeness: str = Representativeness.UNKNOWN.value
    representativeness_detail: str = ""
    snapshots: list[MemorySnapshot] = field(default_factory=list)
    peaks: list[DomainPeak] = field(default_factory=list)
    not_observed: dict[str, str] = field(default_factory=dict)
    #: Whether the checkpoint write and the resume were separately
    #: observed, and what they cost — both are their own peaks.
    checkpoint_peak_bytes: int | None = None
    resume_peak_bytes: int | None = None
    optimizer_steps: int | None = None
    started_at: str = ""
    finished_at: str = ""
    wall_seconds: float | None = None
    failure_reason: str = ""
    failure_kind: str = MemoryFailureKind.NOT_A_MEMORY_FAILURE.value
    sampler_interval_seconds: float | None = None
    schema_version: str = MEMORY_PROFILE_SCHEMA_VERSION

    @property
    def completed(self) -> bool:
        return self.outcome == ProfileOutcome.COMPLETED.value

    @property
    def identity_digest(self) -> str:
        return self.identity.digest()

    def peak_for(self, domain: str) -> DomainPeak | None:
        for peak in self.peaks:
            if peak.domain == domain:
                return peak
        return None

    def stages_in_order(self) -> bool:
        """Whether the observed stages appear in a coherent order.

        Repeats are fine — a step happens many times — but a stage that
        first appears before one it must follow means the profile is
        describing something other than what it claims.
        """
        seen: list[int] = []
        for snapshot in self.snapshots:
            if snapshot.stage not in STAGE_ORDER:
                continue
            position = STAGE_ORDER.index(snapshot.stage)
            if (
                seen
                and position < max(seen)
                and snapshot.stage
                not in {
                    ProfileStage.FORWARD_COMPLETE.value,
                    ProfileStage.BACKWARD_COMPLETE.value,
                    ProfileStage.OPTIMIZER_STEP_COMPLETE.value,
                    ProfileStage.BATCH_READY.value,
                    ProfileStage.CHECKPOINT_BEGIN.value,
                    ProfileStage.CHECKPOINT_COMPLETE.value,
                }
            ):
                return False
            seen.append(position)
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "plan_digest": self.plan_digest,
            "identity_digest": self.identity_digest,
            "identity": self.identity.to_dict(),
            "runtime": self.runtime.to_dict(),
            "outcome": self.outcome,
            "completed": self.completed,
            "representativeness": self.representativeness,
            "representativeness_detail": self.representativeness_detail,
            "snapshots": [item.to_dict() for item in self.snapshots],
            "peaks": [item.to_dict() for item in self.peaks],
            "not_observed": dict(sorted(self.not_observed.items())),
            "checkpoint_peak_bytes": self.checkpoint_peak_bytes,
            "resume_peak_bytes": self.resume_peak_bytes,
            "optimizer_steps": self.optimizer_steps,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "wall_seconds": self.wall_seconds,
            "failure_reason": self.failure_reason,
            "failure_kind": self.failure_kind,
            "sampler_interval_seconds": self.sampler_interval_seconds,
            "note": (
                "A memory profile says what a configuration cost. It says nothing about "
                "music quality, convergence or whether the resulting model is any good."
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainingMemoryProfile:
        version = str(payload.get("schema_version", ""))
        if version != MEMORY_PROFILE_SCHEMA_VERSION:
            raise ProfileFormatError(
                f"memory profile schema {version!r} is not {MEMORY_PROFILE_SCHEMA_VERSION!r}; "
                "a profile from a different build is refused rather than read with its "
                "unknown fields ignored"
            )
        identity = payload.get("identity")
        if not isinstance(identity, dict):
            raise ProfileFormatError("a memory profile must carry its configuration identity")
        return cls(
            profile_id=str(payload.get("profile_id", "")),
            plan_digest=str(payload.get("plan_digest", "")),
            identity=MemoryProfileIdentity.from_dict(identity),
            runtime=RuntimeIdentity.from_dict(payload.get("runtime") or {}),
            outcome=str(payload.get("outcome", ProfileOutcome.NOT_RUN.value)),
            representativeness=str(
                payload.get("representativeness", Representativeness.UNKNOWN.value)
            ),
            representativeness_detail=str(payload.get("representativeness_detail", "")),
            snapshots=[
                MemorySnapshot.from_dict(item)
                for item in payload.get("snapshots") or []
                if isinstance(item, dict)
            ],
            peaks=[
                DomainPeak.from_dict(item)
                for item in payload.get("peaks") or []
                if isinstance(item, dict)
            ],
            not_observed={
                str(key): str(value) for key, value in (payload.get("not_observed") or {}).items()
            },
            checkpoint_peak_bytes=_optional_int(payload.get("checkpoint_peak_bytes")),
            resume_peak_bytes=_optional_int(payload.get("resume_peak_bytes")),
            optimizer_steps=_optional_int(payload.get("optimizer_steps")),
            started_at=str(payload.get("started_at", "")),
            finished_at=str(payload.get("finished_at", "")),
            wall_seconds=(
                float(payload["wall_seconds"])
                if isinstance(payload.get("wall_seconds"), (int, float))
                else None
            ),
            failure_reason=str(payload.get("failure_reason", "")),
            failure_kind=str(
                payload.get("failure_kind", MemoryFailureKind.NOT_A_MEMORY_FAILURE.value)
            ),
            sampler_interval_seconds=(
                float(payload["sampler_interval_seconds"])
                if isinstance(payload.get("sampler_interval_seconds"), (int, float))
                else None
            ),
        )


class ProfileFormatError(ValueError):
    """Raised when a document is not a memory profile this build reads."""


def summarise_peaks(
    snapshots: list[MemorySnapshot],
    *,
    device: str,
    runtime_peaks: dict[str, int] | None = None,
) -> list[DomainPeak]:
    """Turn a series of snapshots into one peak per applicable domain.

    Host is always applicable. The accelerator domain follows the device
    — an Apple run has no CUDA peak and a CUDA run has no Apple one, and
    emitting an empty row for the other would invite somebody to read a
    zero as a measurement.
    """
    from luber_hardware import ComputeDevice

    domains = [MemoryDomain.HOST.value]
    if device == ComputeDevice.MPS.value:
        domains.append(MemoryDomain.APPLE_UNIFIED.value)
    elif device == ComputeDevice.CUDA.value:
        domains.append(MemoryDomain.CUDA_DEVICE.value)

    peaks: list[DomainPeak] = []
    for domain in domains:
        values = [
            (snapshot.value_for(domain), snapshot.stage)
            for snapshot in snapshots
            if snapshot.value_for(domain) is not None
        ]
        baseline = next(
            (
                snapshot.value_for(domain)
                for snapshot in snapshots
                if snapshot.stage == ProfileStage.BASELINE.value
                and snapshot.value_for(domain) is not None
            ),
            None,
        )
        total = _total_for(domain, snapshots)

        runtime_peak = (runtime_peaks or {}).get(domain)
        if runtime_peak is not None:
            peaks.append(
                DomainPeak(
                    domain=domain,
                    kind=PeakKind.RUNTIME_PEAK.value,
                    peak_bytes=runtime_peak,
                    baseline_bytes=baseline,
                    total_bytes=total,
                    sample_count=len(values),
                    detail="high-water mark kept by the runtime itself",
                )
            )
            continue

        if not values:
            peaks.append(
                DomainPeak(
                    domain=domain,
                    kind=PeakKind.NOT_AVAILABLE.value,
                    baseline_bytes=baseline,
                    total_bytes=total,
                    detail="the runtime exposed no figure for this domain",
                )
            )
            continue

        peak_bytes, peak_stage = max(values, key=lambda item: item[0] or 0)
        peaks.append(
            DomainPeak(
                domain=domain,
                kind=PeakKind.SAMPLED_PEAK.value,
                peak_bytes=peak_bytes,
                peak_stage=peak_stage,
                baseline_bytes=baseline,
                total_bytes=total,
                sample_count=len(values),
                detail=(
                    "the largest value a sampler observed. This is a lower bound: "
                    "anything that rose and fell between two samples was never seen"
                ),
            )
        )
    return peaks


def _total_for(domain: str, snapshots: list[MemorySnapshot]) -> int | None:
    for snapshot in snapshots:
        if domain in (MemoryDomain.HOST.value, MemoryDomain.APPLE_UNIFIED.value):
            if snapshot.system_total_bytes is not None:
                return snapshot.system_total_bytes
        elif snapshot.cuda_total_bytes is not None:
            return snapshot.cuda_total_bytes
    return None


__all__ = [
    "MEMORY_PROBE_PROTOCOL_VERSION",
    "MEMORY_PROFILE_SCHEMA_VERSION",
    "MIB",
    "STAGE_ORDER",
    "DomainPeak",
    "MemoryDomain",
    "MemoryFailureKind",
    "MemoryProfileIdentity",
    "MemorySnapshot",
    "PeakKind",
    "ProfileFormatError",
    "ProfileOutcome",
    "ProfileStage",
    "Representativeness",
    "RuntimeIdentity",
    "TrainingMemoryProfile",
    "classify_memory_failure",
    "summarise_peaks",
]
