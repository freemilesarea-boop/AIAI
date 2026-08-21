"""When another inference is worth buying, and when it is not.

The planner's job is to be reluctant. Every one of these tests is about
a case where retrying looks superficially reasonable and is actually a
waste — a misconfigured provider that will answer the same way, a defect
the model reproduces every time, a failure the policy was never told to
retry. The cost of getting them wrong is real compute, spent to
reproduce an error somebody already has.
"""

from __future__ import annotations

import pytest

from luber_inference_qc import (
    AdaptiveRetryPlanner,
    Budget,
    CandidateGeneration,
    CandidateStatus,
    Finding,
    QCFinding,
    RetryDecision,
    Severity,
)
from luber_inference_qc.policy import conservative, standard, strict_reproducible

DIGEST = "d" * 64


def _candidate(index: int, *codes: Finding, status=CandidateStatus.REJECTED) -> CandidateGeneration:
    return CandidateGeneration(
        candidate_id=f"cand_{index:02d}",
        generation_id="gen",
        attempt_index=index,
        request_sha256=DIGEST,
        status=status.value,
        findings=[
            QCFinding(code=code.value, severity=Severity.CRITICAL.value, detail=code.value)
            for code in codes
        ],
    )


def _plan(policy, candidate, history=None, budget=None, base_seed=1234):
    return AdaptiveRetryPlanner(policy).plan(
        candidate=candidate,
        history=history if history is not None else [candidate],
        budget=budget or Budget(policy=policy),
        base_seed=base_seed,
        request_sha256=DIGEST,
    )


# ── the happy path costs nothing ─────────────────────────────────────


def test_an_eligible_candidate_is_not_retried():
    plan = _plan(standard(), _candidate(0, status=CandidateStatus.ELIGIBLE))
    assert plan.decision == RetryDecision.NO_RETRY.value
    assert plan.should_retry is False


# ── what justifies another attempt ───────────────────────────────────


def test_a_silent_candidate_is_retried_with_a_different_seed():
    plan = _plan(standard(), _candidate(0, Finding.SILENT_OUTPUT))
    assert plan.decision == RetryDecision.RETRY_SAME_REQUEST_NEW_SEED.value
    assert plan.next_seed not in (None, 1234)
    assert plan.next_attempt_index == 1


def test_the_retry_changes_the_seed_and_nothing_else():
    """A retry that altered the request would answer a different question.

    There is no prompt rewrite, no duration adjustment, no provider
    parameter the planner has not verified exists — the plan carries a
    seed and an attempt index, and there is nowhere else for a change to
    hide.
    """
    plan = _plan(standard(), _candidate(0, Finding.EARLY_COLLAPSE))
    assert set(plan.to_dict()) == {"decision", "reason", "next_seed", "next_attempt_index"}


def test_a_transport_failure_is_retried_with_the_identical_request():
    """The request never produced audio, so the same seed is still right.

    Changing it would quietly turn a delivery failure into a different
    song, and the user would have no way to know.
    """
    plan = _plan(standard(), _candidate(0, Finding.PROVIDER_TIMEOUT))
    assert plan.decision == RetryDecision.RETRY_IDENTICAL_REQUEST.value
    assert plan.next_seed == 1234


# ── what does not ────────────────────────────────────────────────────


def test_a_misconfigured_provider_is_not_retried():
    """It answers the same way every time. Retrying reproduces the error."""
    plan = _plan(standard(), _candidate(0, Finding.PROVIDER_MISCONFIGURED))
    assert plan.decision == RetryDecision.FAIL_GENERATION.value
    assert "not fixed by another attempt" in plan.reason


def test_a_non_retryable_failure_is_named_before_the_budget_is_consulted():
    """So the trace says *why* nothing was tried, not merely that there
    was money left."""
    spent = Budget(policy=standard(), provider_calls_used=99)
    plan = _plan(standard(), _candidate(0, Finding.PROVIDER_MISCONFIGURED), budget=spent)
    assert Finding.PROVIDER_MISCONFIGURED.value in plan.reason
    assert "budget" not in plan.reason


def test_a_failure_the_policy_does_not_retry_stops_the_run():
    plan = _plan(conservative(), _candidate(0, Finding.CONTROL_BPM_MISMATCH))
    assert plan.decision == RetryDecision.FAIL_GENERATION.value
    assert "CONSERVATIVE policy retries" in plan.reason


def test_the_same_failure_twice_running_is_treated_as_deterministic():
    """The third attempt is not going to be the one that works."""
    history = [_candidate(0, Finding.SILENT_OUTPUT), _candidate(1, Finding.SILENT_OUTPUT)]
    plan = _plan(standard(), history[-1], history=history)
    assert plan.decision == RetryDecision.FAIL_GENERATION.value
    assert "consecutive attempts" in plan.reason


def test_an_intermittent_failure_still_gets_its_retries():
    """It is the deterministic defect the rule catches, not any repeat."""
    history = [_candidate(0, Finding.SILENT_OUTPUT), _candidate(1, Finding.EARLY_COLLAPSE)]
    plan = _plan(standard(), history[-1], history=history)
    assert plan.should_retry is True


def test_nothing_is_planned_that_the_budget_cannot_pay_for():
    """A plan that has to be abandoned leaves a trace full of retries
    that never happened."""
    policy = standard()
    spent = Budget(policy=policy, provider_calls_used=policy.maximum_total_provider_calls)
    plan = _plan(policy, _candidate(0, Finding.SILENT_OUTPUT), budget=spent)
    assert plan.decision == RetryDecision.FAIL_GENERATION.value
    assert "no budget remains" in plan.reason


@pytest.mark.parametrize("code", [Finding.SILENT_OUTPUT, Finding.EARLY_COLLAPSE])
def test_the_reproducible_profile_never_retries_whatever_broke(code):
    plan = _plan(strict_reproducible(), _candidate(0, code))
    assert plan.decision == RetryDecision.FAIL_GENERATION.value
