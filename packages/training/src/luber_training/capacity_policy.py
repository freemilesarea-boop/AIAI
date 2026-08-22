"""Whether a measurement is enough to let a larger run start.

A profile says what a configuration cost. This says whether that is
good enough — and the two are deliberately separate objects, because a
policy changes and a measurement does not. A profile taken today can be
re-judged tomorrow against a stricter reserve without being re-run.

Four ideas do the work.

**Peak below total is not safe.** A machine that is exactly full is a
machine whose next allocation fails, and on the planned topology the
same Mac is also the API, Postgres, Redis and the orchestrator. So a
policy reserves a fraction and a floor, and the reserve is larger when
the target is carrying the control plane.

**A sampled peak is a lower bound.** The pinned torch gives no MPS peak
counter, so an Apple peak is the largest value a sampler happened to
catch — anything that rose and fell between two samples was never seen.
Sampled evidence therefore gets a larger safety margin than a runtime
high-water mark, and the two margins are separate numbers with separate
reasons.

**Evidence belongs to a configuration.** A profile qualifies a request
only when every memory-relevant field matches. Batch 1 does not qualify
batch 4; rank 8 does not qualify rank 64; bf16 does not qualify fp32;
MPS does not qualify CUDA; and two seconds of latents does not qualify
four minutes. Where they differ, the qualification is UNVERIFIED and the
differing fields are named.

**UNKNOWN never becomes QUALIFIED.** Absent evidence is absent. The only
route to QUALIFIED is an applicable, completed, representative profile
whose measured peak satisfies the policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from luber_hardware import ComputeDevice
from luber_training.capacity import CapacityEvidence, EvidenceSource
from luber_training.memory import (
    MIB,
    DomainPeak,
    MemoryDomain,
    PeakKind,
    ProfileOutcome,
    Representativeness,
    TrainingMemoryProfile,
)

#: Bump when the same profile against the same machine would be judged
#: differently. A stored decision cites it, so a reader can tell "the
#: hardware changed" from "we started reserving more".
CAPACITY_POLICY_VERSION = "capacity-policy-v1"

GIB = 1024 * MIB


class CapacityQualification(StrEnum):
    """Whether the evidence permits a run of this configuration."""

    QUALIFIED = "QUALIFIED"
    #: Satisfies the policy, but with little room. Worth an operator's
    #: attention before a long run; distinct from QUALIFIED because
    #: "just fits" and "fits" lead to different decisions.
    MARGIN_LOW = "MARGIN_LOW"
    #: Measured evidence says the policy cannot be met.
    INSUFFICIENT = "INSUFFICIENT"
    #: No applicable evidence. Not a failure of the hardware — a gap in
    #: what anybody has measured.
    UNVERIFIED = "UNVERIFIED"


class Applicability(StrEnum):
    """Whether a profile is evidence about a particular request."""

    APPLICABLE = "APPLICABLE"
    #: The configuration differs in a way that changes memory.
    CONFIGURATION_MISMATCH = "CONFIGURATION_MISMATCH"
    #: The runtime it was measured against has since moved.
    STALE_RUNTIME = "STALE_RUNTIME"
    #: The profile did not finish, so its peak is a peak of however far
    #: it got.
    INCOMPLETE = "INCOMPLETE"
    #: The workload it measured does not resemble a production one.
    NOT_REPRESENTATIVE = "NOT_REPRESENTATIVE"


@dataclass(frozen=True)
class CapacityPolicy:
    """How much room a run must leave, and how much a peak is trusted.

    Every default here is a **choice**, not a measurement, and none of
    them is claimed to be universally optimal. They are conservative
    because the failure they prevent — a control plane swapped out
    mid-run on the machine somebody is watching — is worse than the
    failure they cause, which is a run somebody has to re-qualify with a
    smaller configuration.
    """

    version: str = CAPACITY_POLICY_VERSION

    #: Host RAM held back from a training workload.
    host_headroom_fraction: float = 0.20
    host_headroom_floor_bytes: int = 4 * GIB

    #: Accelerator memory held back. Lower than the host fraction
    #: because on CUDA the device pool has no operating system in it —
    #: and on Apple the *host* reserve is already protecting the same
    #: physical memory.
    device_headroom_fraction: float = 0.15
    device_headroom_floor_bytes: int = 2 * GIB

    #: Extra reserve when the target is also serving the API, the
    #: database and the queue, as the planned Mac mini will be.
    control_plane_reserve_bytes: int = 4 * GIB

    #: What a sampled peak is multiplied by before it is judged.
    #:
    #: A sampled maximum is a lower bound: the true peak may have
    #: occurred between two samples. 1.25 is a margin, not a
    #: measurement of how much was missed — nothing can measure that
    #: from outside — and it is stated rather than folded into the peak.
    sampled_peak_margin: float = 1.25

    #: What a runtime high-water mark is multiplied by. Smaller, because
    #: the runtime did not miss anything; the margin covers allocator
    #: fragmentation and run-to-run variation rather than blindness.
    runtime_peak_margin: float = 1.10

    #: Above this share of the budget, a pass is reported MARGIN_LOW.
    margin_low_ratio: float = 0.85

    def host_reserve(self, total_bytes: int, *, runs_control_plane: bool) -> int:
        reserve = max(
            int(total_bytes * self.host_headroom_fraction), self.host_headroom_floor_bytes
        )
        if runs_control_plane:
            reserve += self.control_plane_reserve_bytes
        return reserve

    def device_reserve(self, total_bytes: int) -> int:
        return max(
            int(total_bytes * self.device_headroom_fraction), self.device_headroom_floor_bytes
        )

    def margin_for(self, kind: str) -> float:
        return (
            self.runtime_peak_margin
            if kind == PeakKind.RUNTIME_PEAK.value
            else self.sampled_peak_margin
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "host_headroom_fraction": self.host_headroom_fraction,
            "host_headroom_floor_bytes": self.host_headroom_floor_bytes,
            "device_headroom_fraction": self.device_headroom_fraction,
            "device_headroom_floor_bytes": self.device_headroom_floor_bytes,
            "control_plane_reserve_bytes": self.control_plane_reserve_bytes,
            "sampled_peak_margin": self.sampled_peak_margin,
            "runtime_peak_margin": self.runtime_peak_margin,
            "margin_low_ratio": self.margin_low_ratio,
            "note": (
                "Conservative defaults, chosen rather than measured. They are not claimed "
                "to be optimal for any particular machine."
            ),
        }


DEFAULT_POLICY = CapacityPolicy()

#: Runtime facts that make a profile stale when they move.
#:
#: A different ACE-Step revision may allocate differently; a different
#: torch may change the allocator. A different *date* changes nothing,
#: which is why time is not on this list.
STALENESS_FIELDS: tuple[str, ...] = ("ace_step_commit", "torch_version")


@dataclass
class DomainVerdict:
    """One domain judged against the policy."""

    domain: str
    qualification: str
    peak_bytes: int | None = None
    peak_kind: str = PeakKind.NOT_AVAILABLE.value
    required_bytes: int | None = None
    total_bytes: int | None = None
    reserved_bytes: int | None = None
    budget_bytes: int | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "qualification": self.qualification,
            "peak_bytes": self.peak_bytes,
            "peak_kind": self.peak_kind,
            "required_bytes": self.required_bytes,
            "total_bytes": self.total_bytes,
            "reserved_bytes": self.reserved_bytes,
            "budget_bytes": self.budget_bytes,
            "detail": self.detail,
        }


@dataclass
class CapacityDecision:
    """Whether this machine may run this configuration, and on what evidence."""

    qualification: str
    device: str
    policy_version: str = CAPACITY_POLICY_VERSION
    profile_id: str | None = None
    identity_digest: str | None = None
    applicability: str | None = None
    applicability_detail: str = ""
    reasons: list[str] = field(default_factory=list)
    domains: list[DomainVerdict] = field(default_factory=list)
    evidence: list[CapacityEvidence] = field(default_factory=list)
    measured_at: str | None = None

    @property
    def permits_full_training(self) -> bool:
        """Only a clean pass or a narrow one. Never UNVERIFIED."""
        return self.qualification in (
            CapacityQualification.QUALIFIED.value,
            CapacityQualification.MARGIN_LOW.value,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "qualification": self.qualification,
            "device": self.device,
            "policy_version": self.policy_version,
            "profile_id": self.profile_id,
            "identity_digest": self.identity_digest,
            "applicability": self.applicability,
            "applicability_detail": self.applicability_detail,
            "reasons": list(self.reasons),
            "domains": [item.to_dict() for item in self.domains],
            "evidence": [item.to_dict() for item in self.evidence],
            "measured_at": self.measured_at,
            "note": (
                "QUALIFIED requires an applicable, completed, representative profile whose "
                "measured peak satisfies the policy. Absent evidence is UNVERIFIED and "
                "never becomes QUALIFIED."
            ),
        }


# ── applicability ────────────────────────────────────────────────────

#: Identity fields whose difference makes a profile inapplicable, with
#: the reason each one matters. Named individually rather than compared
#: as a digest so a refusal can say *which* field differed.
MEMORY_RELEVANT_FIELDS: dict[str, str] = {
    "device": "a different device allocates from a different pool",
    "precision": "a different dtype changes the size of every tensor",
    "optimizer": "optimizer state is per-parameter and differs by optimizer",
    "micro_batch_size": "activations are held for every sample in a micro batch",
    "gradient_checkpointing": "checkpointing trades activation memory for recomputation",
    "lora_rank": "rank sets the size of the trained parameters and their optimizer state",
    "lora_alpha": "part of the adapter's shape",
    "target_modules": "which modules carry an adapter decides how many there are",
    "attention_type": "which attention blocks are adapted",
    "latent_length": "activation memory scales with the sequence being trained on",
    "encoder_length": "cross-attention memory scales with the conditioning length",
    "model_variant": "a different model has a different parameter count",
    "base_model_upstream_commit": "a different model revision may have a different shape",
    "num_devices": "memory per device depends on how many there are",
    "offload_encoder": "offloading moves components off the training device",
}

#: Fields where a profile measured at a *larger* value also covers a
#: smaller request. Everything else must match exactly.
#:
#: Only the two sequence dimensions qualify, and only downward: a run
#: measured on 6000 latent frames tells you about 3000, because the
#: shorter run allocates less of the same thing. The reverse is exactly
#: the extrapolation this module exists to refuse.
MONOTONIC_FIELDS: frozenset[str] = frozenset({"latent_length", "encoder_length"})


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def applicability(
    profile: TrainingMemoryProfile,
    requested: dict[str, Any],
    *,
    require_representative: bool = True,
) -> tuple[str, str]:
    """Whether *profile* is evidence about *requested*, and why not.

    ``requested`` is an identity mapping — the same shape
    :meth:`MemoryProfileIdentity.to_dict` produces — so a caller can ask
    about a configuration that has never been profiled.
    """
    if profile.outcome != ProfileOutcome.COMPLETED.value:
        return (
            Applicability.INCOMPLETE.value,
            f"the profile ended {profile.outcome}, so its peak is a peak of however far "
            "it got rather than of the workload",
        )

    measured = profile.identity.to_dict()
    differences: list[str] = []
    for field_name, why in MEMORY_RELEVANT_FIELDS.items():
        want = requested.get(field_name)
        have = measured.get(field_name)
        if want is None or have == want:
            continue
        if field_name in MONOTONIC_FIELDS:
            measured_value = _as_int(have)
            wanted_value = _as_int(want)
            if measured_value is not None and wanted_value is not None:
                if measured_value >= wanted_value:
                    continue
            differences.append(
                f"{field_name}: measured at {have}, asked about {want} — {why}, and a "
                "shorter measurement does not cover a longer run"
            )
            continue
        differences.append(f"{field_name}: measured {have}, requested {want} — {why}")

    if differences:
        return Applicability.CONFIGURATION_MISMATCH.value, "; ".join(differences)

    runtime = profile.runtime.to_dict()
    stale = [
        name
        for name in STALENESS_FIELDS
        if requested.get(name) is not None and runtime.get(name) != requested.get(name)
    ]
    if stale:
        return (
            Applicability.STALE_RUNTIME.value,
            "the runtime has moved since this was measured: "
            + ", ".join(
                f"{name} was {runtime.get(name)}, now {requested.get(name)}" for name in stale
            ),
        )

    if require_representative and profile.representativeness in (
        Representativeness.NOT_REPRESENTATIVE.value,
        Representativeness.UNKNOWN.value,
    ):
        return (
            Applicability.NOT_REPRESENTATIVE.value,
            f"the profiled workload is {profile.representativeness}: "
            f"{profile.representativeness_detail}",
        )

    return Applicability.APPLICABLE.value, "every memory-relevant field matches"


# ── qualification ────────────────────────────────────────────────────


def _judge_domain(
    peak: DomainPeak,
    *,
    policy: CapacityPolicy,
    total_bytes: int | None,
    runs_control_plane: bool,
) -> DomainVerdict:
    if peak.peak_bytes is None:
        return DomainVerdict(
            domain=peak.domain,
            qualification=CapacityQualification.UNVERIFIED.value,
            peak_kind=peak.kind,
            total_bytes=total_bytes,
            detail="no peak was measured for this domain",
        )
    if not total_bytes:
        return DomainVerdict(
            domain=peak.domain,
            qualification=CapacityQualification.UNVERIFIED.value,
            peak_bytes=peak.peak_bytes,
            peak_kind=peak.kind,
            detail="the machine has not reported how much memory this domain has",
        )

    margin = policy.margin_for(peak.kind)
    required = int(peak.peak_bytes * margin)
    reserve = (
        policy.host_reserve(total_bytes, runs_control_plane=runs_control_plane)
        if peak.domain == MemoryDomain.HOST.value
        else policy.device_reserve(total_bytes)
    )
    budget = max(0, total_bytes - reserve)

    if required > budget:
        qualification = CapacityQualification.INSUFFICIENT.value
        detail = (
            f"{required // MIB} MiB required ({peak.peak_bytes // MIB} MiB measured x "
            f"{margin:g} {peak.kind} margin) against a {budget // MIB} MiB budget "
            f"({total_bytes // MIB} MiB total less {reserve // MIB} MiB reserved)"
        )
    elif required > int(budget * policy.margin_low_ratio):
        qualification = CapacityQualification.MARGIN_LOW.value
        detail = (
            f"{required // MIB} MiB required against a {budget // MIB} MiB budget — inside "
            f"the policy but above {policy.margin_low_ratio:.0%} of it"
        )
    else:
        qualification = CapacityQualification.QUALIFIED.value
        detail = (
            f"{required // MIB} MiB required against a {budget // MIB} MiB budget "
            f"({total_bytes // MIB} MiB total less {reserve // MIB} MiB reserved)"
        )

    return DomainVerdict(
        domain=peak.domain,
        qualification=qualification,
        peak_bytes=peak.peak_bytes,
        peak_kind=peak.kind,
        required_bytes=required,
        total_bytes=total_bytes,
        reserved_bytes=reserve,
        budget_bytes=budget,
        detail=detail,
    )


def _worst(qualifications: list[str]) -> str:
    for candidate in (
        CapacityQualification.INSUFFICIENT.value,
        CapacityQualification.UNVERIFIED.value,
        CapacityQualification.MARGIN_LOW.value,
    ):
        if candidate in qualifications:
            return candidate
    return CapacityQualification.QUALIFIED.value


def qualify(
    *,
    device: str,
    requested: dict[str, Any],
    profiles: list[TrainingMemoryProfile],
    host_total_bytes: int | None,
    device_total_bytes: int | None = None,
    runs_control_plane: bool = False,
    policy: CapacityPolicy | None = None,
    require_representative: bool = True,
) -> CapacityDecision:
    """The one place that decides whether a machine may take a run.

    Central by design. Memory-safety logic scattered across a scheduler,
    a preflight and a console would eventually disagree, and the copy
    that mattered would be whichever one the button used.

    On Apple silicon the host and unified-memory domains describe the
    *same* physical pool, so the two verdicts are evaluated
    independently and never added together. Summing them would
    double-count; taking the larger and ignoring the other would drop
    the operating system's share. Both must pass on their own budget.
    """
    active = policy or DEFAULT_POLICY
    candidates = [
        (profile, applicability(profile, requested, require_representative=require_representative))
        for profile in profiles
    ]
    applicable = [
        (profile, detail)
        for profile, (verdict, detail) in candidates
        if verdict == Applicability.APPLICABLE.value
    ]

    if not applicable:
        reasons = [
            f"{profile.profile_id or 'profile'}: {verdict} — {detail}"
            for profile, (verdict, detail) in candidates
        ] or ["no memory profile has been recorded for this device"]
        return CapacityDecision(
            qualification=CapacityQualification.UNVERIFIED.value,
            device=device,
            policy_version=active.version,
            applicability=(candidates[0][1][0] if candidates else None),
            applicability_detail=reasons[0],
            reasons=reasons,
            evidence=[
                CapacityEvidence(
                    name="training_memory_requirement_mb",
                    source=EvidenceSource.UNKNOWN.value,
                    detail=(
                        "no applicable memory profile exists for this configuration, so "
                        "what it needs is unknown"
                    ),
                )
            ],
        )

    # The most conservative applicable profile: the one whose measured
    # peak is highest. Picking the friendliest would be choosing the
    # evidence to fit the answer.
    profile, detail = max(
        applicable,
        key=lambda item: (
            max((peak.peak_bytes or 0) for peak in item[0].peaks) if item[0].peaks else 0
        ),
    )

    verdicts: list[DomainVerdict] = []
    for peak in profile.peaks:
        total = (
            host_total_bytes
            if peak.domain == MemoryDomain.HOST.value
            else (
                device_total_bytes
                if device_total_bytes is not None
                else (peak.total_bytes or host_total_bytes)
            )
        )
        if peak.domain == MemoryDomain.APPLE_UNIFIED.value:
            # Apple's own recommended working-set maximum where the
            # runtime reported one: it is the runtime's statement about
            # what it will hand out, which is a tighter and more honest
            # ceiling than the machine's total.
            total = peak.total_bytes or total
        verdicts.append(
            _judge_domain(
                peak,
                policy=active,
                total_bytes=total,
                runs_control_plane=runs_control_plane,
            )
        )

    qualification = _worst([item.qualification for item in verdicts])
    evidence: list[CapacityEvidence] = []
    for verdict in verdicts:
        if verdict.peak_bytes is None:
            continue
        evidence.append(
            CapacityEvidence(
                name=f"measured_peak_{verdict.domain.lower()}_mb",
                source=EvidenceSource.MEASURED.value,
                value_mb=verdict.peak_bytes // MIB,
                detail=verdict.detail,
                measured_by=f"bounded memory profile {profile.profile_id}",
                unified_memory=verdict.domain == MemoryDomain.APPLE_UNIFIED.value,
            )
        )
        if verdict.required_bytes is not None:
            evidence.append(
                CapacityEvidence(
                    name=f"required_with_margin_{verdict.domain.lower()}_mb",
                    source=EvidenceSource.DERIVED.value,
                    value_mb=verdict.required_bytes // MIB,
                    detail=verdict.detail,
                    derivation=(
                        f"{verdict.peak_bytes // MIB} MiB {verdict.peak_kind} x "
                        f"{active.margin_for(verdict.peak_kind):g} safety margin"
                    ),
                    unified_memory=verdict.domain == MemoryDomain.APPLE_UNIFIED.value,
                )
            )

    return CapacityDecision(
        qualification=qualification,
        device=device,
        policy_version=active.version,
        profile_id=profile.profile_id,
        identity_digest=profile.identity_digest,
        applicability=Applicability.APPLICABLE.value,
        applicability_detail=detail,
        reasons=[item.detail for item in verdicts],
        domains=verdicts,
        evidence=evidence,
        measured_at=profile.finished_at or None,
    )


def device_total_for(capability: Any, device: str) -> int | None:
    """The accelerator's total memory in bytes, where the concept applies.

    Reads Phase 32's capability rather than measuring anything. Apple's
    figure is the machine's unified memory and is treated as such by
    every caller: it is not a dedicated pool.
    """
    if device == ComputeDevice.CUDA.value:
        total = getattr(capability, "cuda_device_memory_mb", None)
    else:
        total = getattr(capability, "memory_total_mb", None)
    return None if total is None else int(total) * MIB


__all__ = [
    "CAPACITY_POLICY_VERSION",
    "DEFAULT_POLICY",
    "GIB",
    "MEMORY_RELEVANT_FIELDS",
    "MONOTONIC_FIELDS",
    "STALENESS_FIELDS",
    "Applicability",
    "CapacityDecision",
    "CapacityPolicy",
    "CapacityQualification",
    "DomainVerdict",
    "applicability",
    "device_total_for",
    "qualify",
]
