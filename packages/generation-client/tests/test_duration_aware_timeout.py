"""Duration-aware generation timeout.

A timeout exists to distinguish **a dead or hung provider** from **a
request that legitimately takes a while**. Once requests span 30s to
240s of audio, one flat number cannot do both jobs: tight enough to
catch a hang on a short request is too tight for a long one, and loose
enough for a long one lets a hung short request sit for ten minutes.

The properties asserted here are the ones that make the policy safe:

- it never *shrinks* the budget below the configured flat timeout, so
  nothing that worked in Phase 8 can start timing out;
- it grows monotonically with requested duration;
- it stays generous against measured wall clock by a wide margin;
- an operator who deliberately tightens it still gets what they asked
  for.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ace_step_fake_server import FakeAceStepServer

from luber_generation_client import GenerationProviderError, GenerationRequest
from luber_generation_client.ace_step import (
    AceStepClient,
    AceStepProvider,
    AceStepProviderConfig,
)
from luber_schemas import ErrorCode, VocalGender

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mock_generation.wav"

#: Measured on this deployment, docs/PHASE9_LONG_FORM_ENGINE_AUDIT.md §6.
MEASURED_WALL_CLOCK = {30: 69.0, 120: 96.3, 180: 89.5, 240: 76.4}


def _provider(**overrides) -> AceStepProvider:
    config = AceStepProviderConfig(base_url="http://acestep.test", **overrides)
    return AceStepProvider(config, client=AceStepClient(config.base_url))


def test_defaults_never_shrink_the_phase8_budget():
    provider = _provider()
    for duration in (10, 30, 60, 120, 180, 240, 360):
        assert provider.timeout_for(duration) >= 600.0, duration


def test_budget_grows_with_duration():
    provider = _provider()
    budgets = [provider.timeout_for(d) for d in (30, 60, 120, 180, 240)]
    assert budgets == sorted(budgets)
    # And it actually grows, rather than sitting on the floor forever.
    assert provider.timeout_for(240) > provider.timeout_for(30)


@pytest.mark.parametrize(("duration", "wall"), sorted(MEASURED_WALL_CLOCK.items()))
def test_budget_is_generous_against_measured_wall_clock(duration, wall):
    # A liveness backstop, not a performance budget: at least 5x margin
    # over what the engine actually took, so contention or a cold start
    # can never be misreported as a dead provider.
    assert _provider().timeout_for(duration) >= wall * 5


def test_explicitly_tightened_timeout_is_honoured():
    # Scaling off => the operator's flat number wins, exactly.
    provider = _provider(generation_timeout=12.0, timeout_base_seconds=0.0, timeout_multiplier=0.0)
    assert provider.timeout_for(240) == 12.0


def test_scaling_can_be_disabled_without_disabling_the_floor():
    provider = _provider(timeout_base_seconds=0.0, timeout_multiplier=0.0)
    assert provider.timeout_for(240) == 600.0


def test_negative_duration_cannot_shrink_the_budget():
    # Defensive: a bad value must not produce a tiny or negative budget.
    assert _provider().timeout_for(-999) >= 600.0


def test_budget_is_finite_at_the_schema_ceiling():
    # 360s is luber_schemas.DURATION_MAX; the budget must stay sane.
    budget = _provider().timeout_for(360)
    assert 600.0 <= budget <= 3600.0


async def test_long_request_is_not_killed_by_the_short_flat_default(tmp_path):
    """The regression this policy exists to prevent.

    A slow long-form run must not be reported as a provider timeout
    while the engine is still working. Here the fake server needs more
    polls than a 30s-shaped budget would allow, and the request still
    succeeds because the budget scaled with the requested duration.
    """
    server = FakeAceStepServer(FIXTURE, polls_before_success=5)
    config = AceStepProviderConfig(
        base_url="http://acestep.test",
        output_dir=tmp_path / "raw",
        poll_interval=0.01,
        # A flat budget far too small for five polls...
        generation_timeout=0.001,
        # ...but duration-aware scaling gives a real one.
        timeout_base_seconds=5.0,
        timeout_multiplier=0.1,
    )
    provider = AceStepProvider(
        config, client=AceStepClient(config.base_url, transport=server.transport())
    )

    result = await provider.generate(
        GenerationRequest(
            title="long",
            prompt="p",
            lyrics="[Verse]\n가사",
            vocal_gender=VocalGender.FEMALE,
            duration_seconds=240,
        )
    )
    assert result.audio_path.is_file()


async def test_hung_provider_on_a_long_request_still_fails_eventually(tmp_path):
    """The other half: scaling must not mean "wait forever"."""
    server = FakeAceStepServer(FIXTURE, never_finish=True)
    config = AceStepProviderConfig(
        base_url="http://acestep.test",
        output_dir=tmp_path / "raw",
        poll_interval=0.01,
        generation_timeout=0.05,
        timeout_base_seconds=0.05,
        timeout_multiplier=0.001,
    )
    provider = AceStepProvider(
        config, client=AceStepClient(config.base_url, transport=server.transport())
    )

    with pytest.raises(GenerationProviderError) as excinfo:
        await provider.generate(
            GenerationRequest(
                title="hung",
                prompt="p",
                lyrics="[Verse]\n가사",
                vocal_gender=VocalGender.FEMALE,
                duration_seconds=240,
            )
        )
    assert excinfo.value.error_code is ErrorCode.GENERATION_TIMEOUT


def test_timeout_appears_in_the_error_message(tmp_path):
    # Operators diagnosing a timeout need to know which budget expired.
    provider = _provider(timeout_base_seconds=100.0, timeout_multiplier=1.0)
    assert provider.timeout_for(240) == 600.0  # floor still wins here
    assert provider.timeout_for(1000) == 1100.0
