"""The candidate loop, driven by a provider whose answers are scripted.

No model runs here. The controller is handed a callable that returns
audio, so a sequence of outcomes — broken, broken, fine — can be written
down and the loop's response asserted exactly. That is the only way to
test a retry policy: with a real provider the second attempt might
succeed by luck and the test would pass without proving anything.

What is asserted throughout is spending. The healthy path costs one
call; a retry costs one more and no more; an exhausted budget fails
rather than delivering something this engine already rejected.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest
import qc_fixtures as fx

from luber_inference_qc import Budget, CandidateStatus, Finding, QCTrace, RequestExpectation
from luber_inference_qc.controller import (
    CandidateGenerationController,
    Event,
    ProviderCallFailed,
)
from luber_inference_qc.policy import conservative, standard, strict_reproducible
from luber_inference_qc.trace import Outcome
from luber_inference_qc.workspace import CandidateWorkspace

pytestmark = pytest.mark.anyio

EXPECTATION = RequestExpectation(duration_seconds=12.0)
DIGEST = "d" * 64


@dataclass
class Rendered:
    """What a provider call returns, as far as the controller cares."""

    audio_path: Path
    seed_used: int | None = None


class ScriptedProvider:
    """Answers a written-down sequence of outcomes, one per call.

    Each entry is either a fixture factory or a `ProviderCallFailed` to
    raise. The last entry repeats, so a script can say "broken, then fine
    forever" without knowing how many attempts the policy allows.
    """

    def __init__(self, directory: Path, *script) -> None:
        self.directory = directory
        self.script = list(script)
        self.seeds: list[int | None] = []

    async def __call__(self, seed: int | None):
        index = min(len(self.seeds), len(self.script) - 1)
        self.seeds.append(seed)
        entry = self.script[index]
        if isinstance(entry, ProviderCallFailed):
            raise entry
        path = entry(self.directory / f"call-{len(self.seeds):02d}.wav")
        return Rendered(audio_path=path, seed_used=seed)

    @property
    def calls(self) -> int:
        return len(self.seeds)


def _controller(tmp_path: Path, policy=None, **kwargs) -> CandidateGenerationController:
    return CandidateGenerationController(
        policy=policy or standard(),
        workspace=CandidateWorkspace(tmp_path / "workspace", "gen-0001"),
        **kwargs,
    )


async def _run(controller, provider, **kwargs):
    return await controller.run(
        generation_id="gen-0001",
        request_sha256=DIGEST,
        expectation=EXPECTATION,
        generate=provider,
        **kwargs,
    )


# ── the healthy path ─────────────────────────────────────────────────


async def test_a_good_first_candidate_costs_exactly_one_call(tmp_path, audio_dir):
    """No second candidate is generated to compare against it.

    The comparison could only say which is less broken, and neither is.
    """
    provider = ScriptedProvider(audio_dir, fx.healthy)
    result = await _run(_controller(tmp_path), provider)

    assert result.selected is True
    assert provider.calls == 1
    assert result.budget.provider_calls_used == 1
    assert result.budget.retry_rounds == 0
    assert result.trace.outcome == Outcome.SELECTED
    assert result.winner_path is not None and result.winner_path.is_file()


async def test_the_provider_result_reaches_the_caller_for_post_processing(tmp_path, audio_dir):
    provider = ScriptedProvider(audio_dir, fx.healthy)
    result = await _run(_controller(tmp_path), provider, base_seed=4242)
    assert isinstance(result.winner_result, Rendered)
    assert result.winner.seed == 4242


# ── retry that works ─────────────────────────────────────────────────


async def test_a_broken_first_candidate_is_retried_and_the_good_one_wins(tmp_path, audio_dir):
    provider = ScriptedProvider(audio_dir, fx.silent, fx.healthy)
    result = await _run(_controller(tmp_path), provider, base_seed=1234)

    assert result.selected is True
    assert provider.calls == 2
    assert result.winner.attempt_index == 1
    assert result.budget.retry_rounds == 1
    # The first attempt used the seed that was asked for; the second was
    # derived, and derived deterministically.
    assert provider.seeds[0] == 1234
    assert provider.seeds[1] not in (None, 1234)


async def test_the_rejected_attempt_stays_in_the_trace_with_its_reason(tmp_path, audio_dir):
    """Failures are recorded, not hidden. A retry spike has to be
    explainable after the fact."""
    provider = ScriptedProvider(audio_dir, fx.silent, fx.healthy)
    result = await _run(_controller(tmp_path), provider)

    loser, winner = result.trace.candidates
    assert loser.status == CandidateStatus.REJECTED.value
    assert Finding.SILENT_OUTPUT.value in loser.finding_codes()
    assert winner.retry_reason and Finding.SILENT_OUTPUT.value in winner.retry_reason
    assert winner.parent_candidate_id == loser.candidate_id
    assert winner.attribution == "QUALITY_RETRY"


async def test_a_transport_failure_is_retried_with_the_same_seed(tmp_path, audio_dir):
    provider = ScriptedProvider(
        audio_dir,
        ProviderCallFailed(
            "the request timed out", retryable=True, error_code="GENERATION_TIMEOUT"
        ),
        fx.healthy,
    )
    result = await _run(_controller(tmp_path), provider, base_seed=777)

    assert result.selected is True
    assert provider.seeds == [777, 777]
    assert Finding.PROVIDER_TIMEOUT.value in result.trace.candidates[0].finding_codes()


# ── retry that does not ──────────────────────────────────────────────


async def test_a_deterministic_defect_stops_before_the_budget_is_spent(tmp_path, audio_dir):
    """Silent twice running is a pattern, and the third call is not
    bought to confirm it."""
    provider = ScriptedProvider(audio_dir, fx.silent)
    result = await _run(_controller(tmp_path), provider)

    assert result.selected is False
    assert provider.calls == 2
    assert provider.calls < standard().maximum_total_provider_calls
    assert result.failure_finding == Finding.SILENT_OUTPUT.value
    assert "consecutive attempts" in result.trace.outcome_detail


async def test_nothing_rejected_is_ever_delivered(tmp_path, audio_dir):
    """The outcome the whole phase exists to prevent.

    Every candidate was measured and rejected. There is no "best effort"
    that hands one of them over anyway.
    """
    provider = ScriptedProvider(audio_dir, fx.silent, fx.near_silent, fx.spectral_collapse)
    result = await _run(_controller(tmp_path), provider)

    assert result.selected is False
    assert result.winner is None and result.winner_path is None
    assert result.trace.selection is None
    assert result.trace.outcome in {Outcome.RETRY_EXHAUSTED, Outcome.ALL_CANDIDATES_REJECTED}


async def test_the_call_budget_is_never_exceeded(tmp_path, audio_dir):
    """Every distinct failure justifies a retry, so only the ceiling stops it."""
    provider = ScriptedProvider(audio_dir, fx.silent, fx.near_silent, fx.spectral_collapse)
    policy = standard().with_overrides(repeated_failure_limit=99)
    result = await _run(_controller(tmp_path, policy), provider)

    assert provider.calls == policy.maximum_total_provider_calls
    assert result.selected is False
    assert result.trace.exhausted is True


async def test_a_policy_that_allows_two_retries_performs_two(tmp_path, audio_dir):
    """The ceilings are advertised, so they have to be the ones applied.

    The two counters are incremented at opposite ends of a round —
    calls after they happen, retry rounds when they are decided on — so
    a single comparison against both would refuse the round it had just
    approved and quietly halve the policy.
    """
    provider = ScriptedProvider(audio_dir, fx.silent, fx.near_silent, fx.spectral_collapse)
    policy = standard().with_overrides(repeated_failure_limit=99)
    result = await _run(_controller(tmp_path, policy), provider)

    assert result.budget.retry_rounds == policy.maximum_retry_rounds == 2
    assert provider.calls == 3


async def test_a_misconfigured_provider_is_not_retried(tmp_path, audio_dir):
    provider = ScriptedProvider(
        audio_dir,
        ProviderCallFailed(
            "no model on this host", retryable=False, error_code="MODEL_LOAD_FAILED"
        ),
    )
    result = await _run(_controller(tmp_path), provider)

    assert provider.calls == 1
    assert result.selected is False
    assert result.failure_finding == Finding.PROVIDER_MISCONFIGURED.value
    assert result.trace.candidates[0].provider_error_code == "MODEL_LOAD_FAILED"


async def test_the_reproducible_profile_fails_rather_than_substituting(tmp_path, audio_dir):
    """A different seed's output is a different song."""
    provider = ScriptedProvider(audio_dir, fx.silent, fx.healthy)
    result = await _run(_controller(tmp_path, strict_reproducible()), provider, base_seed=5)

    assert provider.calls == 1
    assert result.selected is False


async def test_a_policy_that_does_not_retry_a_finding_does_not_spend_on_it(tmp_path, audio_dir):
    provider = ScriptedProvider(audio_dir, fx.truncated, fx.healthy)
    result = await _run(
        _controller(tmp_path, conservative()),
        provider,
    )
    assert provider.calls == 1
    assert result.selected is False
    assert "CONSERVATIVE policy retries" in result.trace.outcome_detail


# ── several candidates at once ───────────────────────────────────────


async def test_the_multi_candidate_profile_ranks_what_it_generated(tmp_path, audio_dir):
    policy = standard().with_overrides(
        name="TEST_MULTI",
        initial_candidate_count=3,
        maximum_candidate_count=3,
        maximum_total_provider_calls=3,
    )
    provider = ScriptedProvider(audio_dir, fx.healthy)
    result = await _run(_controller(tmp_path, policy), provider)

    assert provider.calls == 3
    assert result.selected is True
    assert len(result.trace.selection.ranking) == 3
    # Everything tied, so the earliest attempt was kept.
    assert result.winner.attempt_index == 0


# ── crash and resume ─────────────────────────────────────────────────


async def test_a_candidate_that_survived_a_crash_is_reused_rather_than_rebought(
    tmp_path, audio_dir
):
    """The case the workspace exists for: the worker died after the
    expensive call and before the cheap check that follows it."""
    first = ScriptedProvider(audio_dir, fx.healthy)
    original = await _run(_controller(tmp_path, standard()), first)
    recorded = [item.to_dict() for item in original.trace.candidates]

    second = ScriptedProvider(audio_dir, fx.healthy)
    resumed = await _run(_controller(tmp_path), second, resume_from=recorded)

    assert second.calls == 0
    assert resumed.selected is True
    assert resumed.budget.provider_calls_used == 0
    assert resumed.winner.raw_sha256 == original.winner.raw_sha256


async def test_a_half_written_survivor_is_regenerated_rather_than_trusted(tmp_path, audio_dir):
    first = ScriptedProvider(audio_dir, fx.healthy)
    original = await _run(_controller(tmp_path, standard()), first)
    recorded = [item.to_dict() for item in original.trace.candidates]

    # The file survived the crash, but not intact.
    original.winner_path.write_bytes(b"truncated during the crash")

    second = ScriptedProvider(audio_dir, fx.healthy)
    resumed = await _run(_controller(tmp_path), second, resume_from=recorded)

    assert second.calls == 1
    assert resumed.selected is True


async def test_resume_without_a_recorded_digest_regenerates(tmp_path, audio_dir):
    """A file that looks like a candidate is not one without the hash
    that proves it."""
    provider = ScriptedProvider(audio_dir, fx.healthy)
    result = await _run(
        _controller(tmp_path), provider, resume_from=[{"attempt_index": 0, "raw_sha256": None}]
    )
    assert provider.calls == 1
    assert result.selected is True


# ── the record, written as it happens ────────────────────────────────


async def test_the_trace_is_persisted_after_every_attempt(tmp_path, audio_dir):
    """A crash between the provider returning and QC finishing still
    leaves the record that the call was made."""
    written: list[tuple[int, str]] = []

    async def sink(trace: QCTrace, budget: Budget) -> None:
        written.append((budget.provider_calls_used, trace.outcome))

    provider = ScriptedProvider(audio_dir, fx.silent, fx.healthy)
    await _run(_controller(tmp_path, on_trace=sink), provider)

    # Generated, judged, generated, judged, selected — at minimum.
    assert len(written) >= 5
    assert written[0][0] == 1


async def test_every_decision_is_emitted_as_a_countable_event(tmp_path, audio_dir):
    """Free-form log text cannot answer "how often did we retry"."""
    events: list[str] = []
    provider = ScriptedProvider(audio_dir, fx.silent, fx.healthy)
    await _run(_controller(tmp_path, on_event=lambda name, payload: events.append(name)), provider)

    assert Event.CANDIDATE_REJECTED in events
    assert Event.RETRY_PLANNED in events
    assert Event.RETRY_STARTED in events
    assert Event.CANDIDATE_SELECTED in events


# ── concurrency ──────────────────────────────────────────────────────


async def test_concurrent_generations_do_not_read_each_others_candidates(tmp_path, audio_dir):
    """Workspaces are scoped per generation, so a stale directory can
    never be mistaken for another run's attempt."""

    async def one(index: int):
        controller = CandidateGenerationController(
            policy=standard(),
            workspace=CandidateWorkspace(tmp_path / "workspace", f"gen-{index:04d}"),
        )
        provider = ScriptedProvider(audio_dir / f"run-{index}", fx.healthy)
        return await controller.run(
            generation_id=f"gen-{index:04d}",
            request_sha256=DIGEST,
            expectation=EXPECTATION,
            generate=provider,
            base_seed=index,
        )

    results = await asyncio.gather(*(one(index) for index in range(4)))

    assert all(result.selected for result in results)
    paths = {result.winner_path for result in results}
    assert len(paths) == 4
    for index, result in enumerate(results):
        assert result.trace.generation_id == f"gen-{index:04d}"
        assert len(result.trace.candidates) == 1
