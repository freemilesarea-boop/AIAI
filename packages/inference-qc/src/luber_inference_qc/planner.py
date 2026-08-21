"""Whether to spend another inference, and what to change if so.

The planner answers one question — is another attempt worth it — and it
is deliberately reluctant. Every retry costs a full generation, and the
default policy allows at most two, so each decision has to be justified
by something measured rather than by hope that the next one will be
better.

Four rules, in the order they are applied.

**Budget first.** Nothing is planned that the budget cannot pay for.
Checking the findings first and the budget second would produce a plan
that then has to be abandoned, and a trace full of retries that never
happened.

**Non-retryable failures stop immediately.** A misconfigured provider
answers the same way every time. Retrying it spends inference to
reproduce an error.

**The same critical failure twice is a pattern.** A defect the model
produces deterministically is not going to be fixed by the third
attempt, and burning the rest of the budget to confirm that is the waste
this rule exists to avoid.

**The change is a seed, and only a seed.** The planner does not rewrite
the prompt, adjust the duration, change the key, or touch a provider
parameter it has not verified exists. A retry that quietly altered the
request would answer a different question and call it a recovery — and
the user would have no way to know their song came from a prompt they
did not write.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from luber_inference_qc.candidate import CandidateGeneration
from luber_inference_qc.findings import NON_RETRYABLE
from luber_inference_qc.identity import derive_seed
from luber_inference_qc.policy import Budget, CandidatePolicy


class RetryDecision(StrEnum):
    """What to do after a candidate was judged."""

    #: This candidate is deliverable. Stop and select it.
    NO_RETRY = "NO_RETRY"
    #: Try again with the same request and a different seed.
    RETRY_SAME_REQUEST_NEW_SEED = "RETRY_SAME_REQUEST_NEW_SEED"
    #: Try again with the identical request, seed included. Only for a
    #: transport failure, where the request never reached the model.
    RETRY_IDENTICAL_REQUEST = "RETRY_IDENTICAL_REQUEST"
    #: Nothing further will help. Fail the generation.
    FAIL_GENERATION = "FAIL_GENERATION"


@dataclass(frozen=True)
class RetryPlan:
    """The decision, the reason, and the seed for the next attempt."""

    decision: str
    reason: str
    next_seed: int | None = None
    next_attempt_index: int | None = None

    @property
    def should_retry(self) -> bool:
        return self.decision in {
            RetryDecision.RETRY_SAME_REQUEST_NEW_SEED.value,
            RetryDecision.RETRY_IDENTICAL_REQUEST.value,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "next_seed": self.next_seed,
            "next_attempt_index": self.next_attempt_index,
        }


class AdaptiveRetryPlanner:
    """Decides whether another attempt is justified."""

    def __init__(self, policy: CandidatePolicy) -> None:
        self.policy = policy

    def plan(
        self,
        *,
        candidate: CandidateGeneration,
        history: list[CandidateGeneration],
        budget: Budget,
        base_seed: int | None,
        request_sha256: str,
    ) -> RetryPlan:
        """What to do after *candidate* was measured.

        ``history`` is every attempt so far including this one, because
        the repeated-failure rule needs to see the pattern rather than
        the last data point.
        """
        if candidate.eligible:
            return RetryPlan(
                decision=RetryDecision.NO_RETRY.value,
                reason="the candidate is eligible for delivery",
            )

        codes = candidate.finding_codes()

        # A failure no further attempt can fix. Checked before the budget
        # so the trace records *why* nothing was tried, not merely that
        # there was money left.
        blocking_non_retryable = codes & NON_RETRYABLE
        if blocking_non_retryable:
            return RetryPlan(
                decision=RetryDecision.FAIL_GENERATION.value,
                reason=(
                    f"{', '.join(sorted(blocking_non_retryable))} is not fixed by another "
                    "attempt; retrying would spend an inference to reproduce the same error"
                ),
            )

        exhausted = budget.exhausted()
        if exhausted is not None:
            return RetryPlan(
                decision=RetryDecision.FAIL_GENERATION.value,
                reason=f"no budget remains: {exhausted}",
            )

        justifying = sorted(code for code in codes if self.policy.retries_on(code))
        if not justifying:
            return RetryPlan(
                decision=RetryDecision.FAIL_GENERATION.value,
                reason=(
                    "nothing this candidate failed on is something the "
                    f"{self.policy.name} policy retries: "
                    f"{', '.join(sorted(codes)) or 'no findings'}"
                ),
            )

        repeated = self._repeated_failure(history, justifying)
        if repeated is not None:
            return RetryPlan(
                decision=RetryDecision.FAIL_GENERATION.value,
                reason=repeated,
            )

        # A transport failure means the request never produced audio, so
        # the same seed is still the right one to ask for — changing it
        # would silently turn a delivery failure into a different song.
        transport_only = justifying and all(code.startswith("PROVIDER_") for code in justifying)
        if transport_only or not self.policy.allow_seed_variation:
            return RetryPlan(
                decision=RetryDecision.RETRY_IDENTICAL_REQUEST.value,
                reason=(
                    f"{', '.join(justifying)}: the request did not produce audio, so the "
                    "identical request is retried"
                ),
                next_seed=base_seed,
                next_attempt_index=candidate.attempt_index + 1,
            )

        next_index = candidate.attempt_index + 1
        return RetryPlan(
            decision=RetryDecision.RETRY_SAME_REQUEST_NEW_SEED.value,
            reason=(
                f"{', '.join(justifying)}: the same request is retried with a different "
                "seed, and nothing else about it changes"
            ),
            next_seed=derive_seed(base_seed, next_index, request_sha256),
            next_attempt_index=next_index,
        )

    def _repeated_failure(
        self, history: list[CandidateGeneration], justifying: list[str]
    ) -> str | None:
        """Whether the same critical failure has now happened enough times.

        Counts consecutive trailing attempts sharing a code, so an
        intermittent failure that alternates with a different one still
        gets its retries — it is the *deterministic* defect this catches.
        """
        limit = self.policy.repeated_failure_limit
        if len(history) < limit:
            return None

        for code in justifying:
            recent = history[-limit:]
            if all(code in item.finding_codes() for item in recent):
                return (
                    f"{code} has now occurred in {limit} consecutive attempts; the model is "
                    "reproducing it deterministically and further attempts would spend the "
                    "budget to confirm that"
                )
        return None


__all__ = ["AdaptiveRetryPlanner", "RetryDecision", "RetryPlan"]
