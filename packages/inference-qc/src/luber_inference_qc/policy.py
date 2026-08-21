"""How many attempts, on what evidence, and the ceiling on all of it.

The default is one candidate. Generating three songs and picking the
technically cleanest would cost three times the inference for a
selection that cannot judge whether any of them is *good* — it can only
say which is least broken. When the first candidate is not broken, the
comparison buys nothing and costs everything.

So the shape is: generate one, measure it, and spend a second inference
only when the measurement found something a second attempt could
plausibly fix. That is what makes this a reliability feature rather than
a quality-search feature, and the difference is the compute bill.

Three profiles, and the difference between them is entirely in this
file:

``STRICT_REPRODUCIBLE`` — exactly one provider call. A quality failure
is reported as a failure. Someone who supplied a seed and expects that
seed's output gets that seed's output or an honest error, never a
different song from a different seed.

``CONSERVATIVE`` — one candidate, one retry, and only for the failures
where a retry is obviously the right response: the file did not decode,
it is silent, it collapsed.

``STANDARD`` — the default. One candidate, up to two retries, and
duration and BPM failures justify one too.

``EXPERIMENTAL_MULTI_CANDIDATE`` — three candidates up front, for
comparing model behaviour. Not a consumer default and documented as not
being one.

Budgets are hard. Every profile carries a maximum provider call count
that nothing can exceed, because the failure mode this protects against
— a deterministic defect that every attempt reproduces — is exactly the
one where a per-failure retry rule would spend the whole budget.

There is deliberately no "best effort" switch. When the budget runs out
the generation fails, because every candidate that *could* have been
delivered was already eligible and would already have been selected —
the only thing such a switch could add is the delivery of a candidate
this engine measured and rejected, which is the outcome the whole phase
exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from luber_inference_qc.findings import Finding


class PolicyProfile(StrEnum):
    STRICT_REPRODUCIBLE = "STRICT_REPRODUCIBLE"
    CONSERVATIVE = "CONSERVATIVE"
    STANDARD = "STANDARD"
    EXPERIMENTAL_MULTI_CANDIDATE = "EXPERIMENTAL_MULTI_CANDIDATE"


@dataclass(frozen=True)
class CandidatePolicy:
    """One profile's worth of decisions, versioned with the retry policy."""

    name: str = PolicyProfile.STANDARD.value

    #: How many candidates to generate before looking at any of them.
    #: One, in every profile a consumer request uses.
    initial_candidate_count: int = 1
    #: Ceiling on candidates including retries.
    maximum_candidate_count: int = 3
    #: Ceiling on rounds of retry decisions.
    maximum_retry_rounds: int = 2
    #: The hard one. Nothing generates past this, whatever the findings.
    maximum_total_provider_calls: int = 3
    #: Wall-clock ceiling on the whole candidate phase, in seconds.
    #: ``None`` means the provider's own timeout is the only bound.
    maximum_elapsed_seconds: float | None = None

    #: Which findings justify spending another inference. A finding not
    #: named here is recorded and lived with — the candidate is either
    #: eligible despite it or the generation fails, but no more compute
    #: is spent.
    retry_findings: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                Finding.INVALID_AUDIO.value,
                Finding.NON_FINITE_SAMPLES.value,
                Finding.SILENT_OUTPUT.value,
                Finding.NEAR_SILENT.value,
                Finding.EARLY_COLLAPSE.value,
                Finding.SEVERE_CLIPPING.value,
                Finding.SPECTRAL_COLLAPSE.value,
                Finding.PHASE_UNSAFE.value,
                Finding.DURATION_SHORT.value,
                Finding.DURATION_LONG.value,
                Finding.CONTROL_BPM_MISMATCH.value,
                Finding.PROVIDER_TIMEOUT.value,
                Finding.PROVIDER_ERROR.value,
            }
        )
    )

    #: Stop early when this many consecutive attempts produce the same
    #: critical finding. A defect the model reproduces deterministically
    #: is not going to be fixed by the third attempt, and spending the
    #: budget to confirm that is the waste this exists to prevent.
    repeated_failure_limit: int = 2

    #: Whether ranking runs at all. With one candidate there is nothing
    #: to rank, but the flag exists so a profile can generate several
    #: and still deliver the first eligible one.
    candidate_selection_enabled: bool = True

    #: Whether a retry may use a different seed. False is what makes
    #: STRICT_REPRODUCIBLE strict.
    allow_seed_variation: bool = True

    def with_overrides(self, **kwargs: Any) -> CandidatePolicy:
        unknown = sorted(set(kwargs) - set(self.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown policy field(s): {', '.join(unknown)}")
        return replace(self, **kwargs)

    def retries_on(self, code: str) -> bool:
        return code in self.retry_findings

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "initial_candidate_count": self.initial_candidate_count,
            "maximum_candidate_count": self.maximum_candidate_count,
            "maximum_retry_rounds": self.maximum_retry_rounds,
            "maximum_total_provider_calls": self.maximum_total_provider_calls,
            "maximum_elapsed_seconds": self.maximum_elapsed_seconds,
            "retry_findings": sorted(self.retry_findings),
            "repeated_failure_limit": self.repeated_failure_limit,
            "candidate_selection_enabled": self.candidate_selection_enabled,
            "allow_seed_variation": self.allow_seed_variation,
        }


def strict_reproducible() -> CandidatePolicy:
    """One call, one answer, no substitution.

    A quality failure returns a failure. Delivering a different seed's
    output to someone who asked for a specific one would be answering a
    different question and calling it success.
    """
    return CandidatePolicy(
        name=PolicyProfile.STRICT_REPRODUCIBLE.value,
        initial_candidate_count=1,
        maximum_candidate_count=1,
        maximum_retry_rounds=0,
        maximum_total_provider_calls=1,
        retry_findings=frozenset(),
        candidate_selection_enabled=False,
        allow_seed_variation=False,
    )


def conservative() -> CandidatePolicy:
    """Retry only where a retry is obviously right."""
    return CandidatePolicy(
        name=PolicyProfile.CONSERVATIVE.value,
        initial_candidate_count=1,
        maximum_candidate_count=2,
        maximum_retry_rounds=1,
        maximum_total_provider_calls=2,
        retry_findings=frozenset(
            {
                Finding.INVALID_AUDIO.value,
                Finding.NON_FINITE_SAMPLES.value,
                Finding.SILENT_OUTPUT.value,
                Finding.EARLY_COLLAPSE.value,
                Finding.PROVIDER_TIMEOUT.value,
                Finding.PROVIDER_ERROR.value,
            }
        ),
    )


def standard() -> CandidatePolicy:
    """The default. One candidate; retry a measurable defect twice."""
    return CandidatePolicy()


def experimental_multi_candidate() -> CandidatePolicy:
    """Three candidates up front. For experiments, not for customers.

    Costs three inferences per request whatever the first one produced,
    which is why no consumer path selects it.
    """
    return CandidatePolicy(
        name=PolicyProfile.EXPERIMENTAL_MULTI_CANDIDATE.value,
        initial_candidate_count=3,
        maximum_candidate_count=4,
        maximum_retry_rounds=1,
        maximum_total_provider_calls=4,
    )


PROFILES: dict[str, Any] = {
    PolicyProfile.STRICT_REPRODUCIBLE.value: strict_reproducible,
    PolicyProfile.CONSERVATIVE.value: conservative,
    PolicyProfile.STANDARD.value: standard,
    PolicyProfile.EXPERIMENTAL_MULTI_CANDIDATE.value: experimental_multi_candidate,
}


def profile(name: str) -> CandidatePolicy:
    key = name.strip().upper()
    if key not in PROFILES:
        raise ValueError(f"unknown candidate policy {name!r}. Known: {', '.join(sorted(PROFILES))}")
    built: CandidatePolicy = PROFILES[key]()
    validate(built)
    return built


def validate(policy: CandidatePolicy) -> None:
    """Refuse a policy that cannot be honoured as written."""
    if policy.initial_candidate_count < 1:
        raise ValueError("initial_candidate_count must be at least 1")
    if policy.maximum_candidate_count < policy.initial_candidate_count:
        raise ValueError("maximum_candidate_count cannot be below initial_candidate_count")
    if policy.maximum_total_provider_calls < policy.initial_candidate_count:
        raise ValueError(
            "maximum_total_provider_calls cannot be below initial_candidate_count: the "
            "first round alone would exceed the budget"
        )
    if policy.maximum_retry_rounds < 0:
        raise ValueError("maximum_retry_rounds must not be negative")
    if policy.repeated_failure_limit < 1:
        raise ValueError("repeated_failure_limit must be at least 1")
    if policy.maximum_retry_rounds > 0 and not policy.retry_findings:
        raise ValueError(
            "a policy that allows retry rounds but names no retryable finding would never "
            "retry; say maximum_retry_rounds=0 instead"
        )


@dataclass
class Budget:
    """What has been spent, and the ceilings it is spent against.

    Mutable and passed by reference on purpose: the controller, the
    planner and the trace all have to agree about how much is left, and
    three copies would eventually disagree at the moment it mattered.
    """

    policy: CandidatePolicy
    provider_calls_used: int = 0
    candidates_generated: int = 0
    retry_rounds: int = 0
    elapsed_seconds: float = 0.0

    @property
    def calls_remaining(self) -> int:
        return max(0, self.policy.maximum_total_provider_calls - self.provider_calls_used)

    def can_call(self) -> str | None:
        """Why another provider call cannot be made, or ``None``.

        Deliberately narrower than `exhausted`. This asks whether the
        *call* can be paid for; whether another retry may be *planned* is
        a separate question with a separate counter, and conflating them
        costs a retry.

        The difference is where the counters are incremented. Calls and
        candidates are counted after they happen, so a ``>=`` comparison
        here means "there is nothing left". Retry rounds are counted when
        the round is decided on, before its call is made — so including
        that ceiling here would refuse the very round that was just
        approved, and a policy advertising two retries would deliver one.
        """
        if self.provider_calls_used >= self.policy.maximum_total_provider_calls:
            return (
                f"provider call budget spent: {self.provider_calls_used} of "
                f"{self.policy.maximum_total_provider_calls}"
            )
        if self.candidates_generated >= self.policy.maximum_candidate_count:
            return (
                f"candidate limit reached: {self.candidates_generated} of "
                f"{self.policy.maximum_candidate_count}"
            )
        limit = self.policy.maximum_elapsed_seconds
        if limit is not None and self.elapsed_seconds >= limit:
            return f"time budget spent: {self.elapsed_seconds:.1f}s of {limit:.1f}s"
        return None

    def exhausted(self) -> str | None:
        """Which ceiling stops another retry from being planned, or ``None``.

        Returns the reason rather than a boolean so the trace can say
        *which* limit stopped it. "Budget exhausted" without that is a
        message nobody can act on.
        """
        blocked = self.can_call()
        if blocked is not None:
            return blocked
        if self.retry_rounds >= self.policy.maximum_retry_rounds:
            return f"retry rounds spent: {self.retry_rounds} of {self.policy.maximum_retry_rounds}"
        return None

    def record_call(self, *, produced_candidate: bool) -> None:
        """One provider call happened. Counted whatever it returned.

        A call that timed out still cost the inference, and a budget that
        only counted successes would let a failing provider be retried
        without limit.
        """
        self.provider_calls_used += 1
        if produced_candidate:
            self.candidates_generated += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_calls_used": self.provider_calls_used,
            "candidates_generated": self.candidates_generated,
            "retry_rounds": self.retry_rounds,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "maximum_total_provider_calls": self.policy.maximum_total_provider_calls,
            "maximum_candidate_count": self.policy.maximum_candidate_count,
            "maximum_retry_rounds": self.policy.maximum_retry_rounds,
        }


__all__ = [
    "PROFILES",
    "Budget",
    "CandidatePolicy",
    "PolicyProfile",
    "conservative",
    "experimental_multi_candidate",
    "profile",
    "standard",
    "strict_reproducible",
    "validate",
]
