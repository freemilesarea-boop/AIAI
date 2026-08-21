"""The ceilings, and the fact that nothing can spend past them.

Every number here is money. A policy that permitted one more provider
call than it says would be a bill nobody authorised, and the failure
mode it guards against — a deterministic defect that every attempt
reproduces — is precisely the one where a per-failure retry rule spends
everything it is given.
"""

from __future__ import annotations

import pytest

from luber_inference_qc import Budget, CandidatePolicy, Finding, PolicyProfile, profile
from luber_inference_qc.policy import (
    conservative,
    experimental_multi_candidate,
    standard,
    strict_reproducible,
    validate,
)

ALL_PROFILES = [name.value for name in PolicyProfile]


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_every_profile_is_internally_consistent(name):
    validate(profile(name))


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_no_profile_can_generate_past_its_call_budget(name):
    policy = profile(name)
    assert policy.initial_candidate_count <= policy.maximum_total_provider_calls
    assert policy.maximum_candidate_count <= policy.maximum_total_provider_calls


def test_the_default_costs_one_inference_when_nothing_is_wrong():
    """The whole economics of the phase.

    Generating several and picking the technically cleanest would cost
    several times the inference for a comparison that cannot say whether
    any of them is good.
    """
    assert standard().initial_candidate_count == 1


def test_the_consumer_default_is_standard():
    assert profile("standard").name == PolicyProfile.STANDARD.value
    assert profile("StAnDaRd").name == PolicyProfile.STANDARD.value


def test_the_reproducible_profile_makes_exactly_one_call():
    policy = strict_reproducible()
    assert policy.maximum_total_provider_calls == 1
    assert policy.maximum_retry_rounds == 0
    assert policy.retry_findings == frozenset()
    # A different seed's output is a different song. Someone who named a
    # seed gets that seed's output or an honest error.
    assert policy.allow_seed_variation is False


def test_the_conservative_profile_retries_only_the_obvious_failures():
    policy = conservative()
    assert Finding.SILENT_OUTPUT.value in policy.retry_findings
    assert Finding.DURATION_SHORT.value not in policy.retry_findings
    assert Finding.CONTROL_BPM_MISMATCH.value not in policy.retry_findings


def test_the_multi_candidate_profile_is_documented_as_not_a_default():
    policy = experimental_multi_candidate()
    assert policy.initial_candidate_count == 3
    assert "EXPERIMENTAL" in policy.name
    assert experimental_multi_candidate.__doc__ is not None
    assert "not for customers" in experimental_multi_candidate.__doc__


def test_an_unknown_profile_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown candidate policy"):
        profile("aggressive")


# ── refusing a policy that cannot be honoured ────────────────────────


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"initial_candidate_count": 0}, "at least 1"),
        ({"maximum_candidate_count": 0}, "below initial_candidate_count"),
        ({"maximum_total_provider_calls": 0}, "first round alone would exceed"),
        ({"maximum_retry_rounds": -1}, "must not be negative"),
        ({"repeated_failure_limit": 0}, "at least 1"),
        ({"retry_findings": frozenset()}, "would never retry"),
    ],
)
def test_a_policy_that_contradicts_itself_is_refused(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate(CandidatePolicy().with_overrides(**overrides))


def test_an_unknown_policy_field_is_refused_rather_than_ignored():
    """A typo that silently did nothing would be a policy nobody applied."""
    with pytest.raises(ValueError, match="unknown policy field"):
        CandidatePolicy().with_overrides(maximum_retries=5)


# ── the budget ───────────────────────────────────────────────────────


def test_a_call_that_returned_nothing_still_costs_a_call():
    """A timeout bought the inference. A budget that only counted
    successes would let a failing provider be retried without limit."""
    budget = Budget(policy=standard())
    budget.record_call(produced_candidate=False)
    assert budget.provider_calls_used == 1
    assert budget.candidates_generated == 0


def test_exhaustion_names_the_ceiling_it_hit():
    """ "Budget exhausted" on its own is a message nobody can act on."""
    budget = Budget(policy=standard())
    assert budget.exhausted() is None
    for _ in range(standard().maximum_total_provider_calls):
        budget.record_call(produced_candidate=True)
    assert "provider call budget spent" in (budget.exhausted() or "")


def test_the_time_ceiling_is_optional_and_honoured_when_set():
    policy = standard().with_overrides(maximum_elapsed_seconds=10.0)
    budget = Budget(policy=policy, elapsed_seconds=11.0)
    assert "time budget spent" in (budget.exhausted() or "")


def test_calls_remaining_never_goes_negative():
    budget = Budget(policy=strict_reproducible())
    budget.record_call(produced_candidate=True)
    budget.record_call(produced_candidate=True)
    assert budget.calls_remaining == 0
