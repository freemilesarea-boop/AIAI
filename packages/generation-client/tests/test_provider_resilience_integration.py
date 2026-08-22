"""Resilience through the whole service: what a user actually gets.

The unit tests in `packages/provider-resilience` prove the breaker
decides correctly. These prove the decisions reach the parts of the
system somebody can see — how many times the provider was called, what
error code the row ends up with, and whether the routing story is
recoverable afterwards.

Providers here are deterministic doubles. A resilience policy cannot be
tested against a real provider that might succeed by luck: the test
would pass without proving the circuit did anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from luber_audio_utils import LocalAudioStorage
from luber_database import ResilienceRepository
from luber_generation_client import (
    GenerationRequest,
    GenerationResult,
    GenerationService,
    MockGenerationProvider,
    MusicGenerationProvider,
)
from luber_generation_client.errors import GenerationProviderError
from luber_generation_client.resilience import ResilienceGate
from luber_generation_client.resilience_factory import build_resilience_gate
from luber_provider_resilience import (
    CircuitIdentity,
    CircuitPolicy,
    CircuitState,
    DurableCircuitStore,
    FailoverMode,
)
from luber_schemas import ErrorCode, GenerationStatus

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mock_generation.wav"
TEXT_TO_MUSIC = CircuitIdentity("primary", "TEXT_TO_MUSIC")


class DeadProvider(MusicGenerationProvider):
    """Never answers. The outage this whole phase is about."""

    name = "primary"

    def __init__(self) -> None:
        self.calls = 0

    @property
    def supports_reference_audio(self) -> bool:
        return True

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls += 1
        raise GenerationProviderError(
            "the engine did not answer", error_code=ErrorCode.GENERATION_TIMEOUT
        )


class RejectingProvider(MusicGenerationProvider):
    """Refuses the request itself. Never provider evidence."""

    name = "primary"

    def __init__(self) -> None:
        self.calls = 0

    @property
    def supports_reference_audio(self) -> bool:
        return True

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls += 1
        raise GenerationProviderError(
            "that reference cannot be used",
            error_code=ErrorCode.REFERENCE_AUDIO_UNAVAILABLE,
        )


async def _create(repository, **overrides):
    payload = {
        "title": "TEST SONG",
        "prompt": "Dreamy Korean indie pop",
        "lyrics": "[Verse]\n테스트 가사",
        "vocal_gender": "female",
        "duration_requested": 20,
        "status": GenerationStatus.QUEUED.value,
        "language": "ko",
        **overrides,
    }
    generation = await repository.create_generation(**payload)
    await repository.create_job(
        generation.id, queue_name="luber:generation", status=GenerationStatus.QUEUED.value
    )
    return generation


def _service(repository, provider, tmp_path, gate=None):
    return GenerationService(
        repository,
        provider,
        LocalAudioStorage(tmp_path / "storage"),
        candidate_workspace_dir=str(tmp_path / "candidates"),
        resilience=gate,
    )


def _gate(engine, providers, *, threshold=3, failover=FailoverMode.DISABLED.value):
    return ResilienceGate.build(
        providers,
        store=DurableCircuitStore(ResilienceRepository(engine)),
        failover=failover,
        circuit_policy=CircuitPolicy(consecutive_failure_threshold=threshold),
    )


# ── the point of the phase ───────────────────────────────────────────


async def test_an_open_circuit_stops_the_provider_being_called_at_all(repository, engine, tmp_path):
    """The whole value with one provider: a dead engine is asked three
    times, not three times per request forever."""
    provider = DeadProvider()
    service = _service(repository, provider, tmp_path, _gate(engine, {"primary": provider}))

    for _ in range(6):
        generation = await _create(repository)
        await service.execute(generation.id, worker_id="w")

    assert provider.calls == 3, "the circuit opened and the rest were refused"

    stored = await ResilienceRepository(engine).load(TEXT_TO_MUSIC.key())
    assert stored["state"] == CircuitState.OPEN.value


async def test_a_refused_request_fails_with_a_typed_code_not_a_timeout(
    repository, engine, tmp_path
):
    provider = DeadProvider()
    service = _service(repository, provider, tmp_path, _gate(engine, {"primary": provider}))

    codes = []
    for _ in range(5):
        generation = await _create(repository)
        status = await service.execute(generation.id, worker_id="w")
        row = await repository.get_generation(generation.id)
        codes.append((status, row.error_code))

    assert all(status is GenerationStatus.FAILED for status, _ in codes)
    # The first exhausts the budget against a real timeout; the rest are
    # refused before the provider is touched.
    assert codes[0][1] == ErrorCode.GENERATION_TIMEOUT.value
    assert {code for _, code in codes[1:]} == {ErrorCode.PROVIDER_BUSY.value}


async def test_a_request_level_failure_never_opens_the_circuit(repository, engine, tmp_path):
    """Ten users with bad references must not take the provider offline
    for everybody else."""
    provider = RejectingProvider()
    service = _service(repository, provider, tmp_path, _gate(engine, {"primary": provider}))

    for _ in range(10):
        generation = await _create(repository)
        await service.execute(generation.id, worker_id="w")

    stored = await ResilienceRepository(engine).load(TEXT_TO_MUSIC.key())
    assert stored is None or stored["state"] == CircuitState.CLOSED.value


async def test_the_routing_story_is_recorded_on_the_generation(repository, engine, tmp_path):
    provider = DeadProvider()
    service = _service(repository, provider, tmp_path, _gate(engine, {"primary": provider}))

    generation = await _create(repository)
    await service.execute(generation.id, worker_id="w")

    trace = json.loads((await repository.get_generation(generation.id)).inference_qc_trace)
    resilience = trace["resilience"]

    assert resilience["providers_attempted"] == ["primary"]
    assert resilience["decisions"], "every routing decision is recorded"
    assert resilience["narrative"], "and rendered as sentences"
    assert any("primary" in line for line in resilience["narrative"])


async def test_a_healthy_generation_is_unchanged_and_records_a_success(
    repository, engine, tmp_path
):
    """The fast path: one call, one selection, no extra provider work."""
    provider = MockGenerationProvider(FIXTURE)
    gate = _gate(engine, {"primary": provider})
    service = _service(repository, provider, tmp_path, gate)

    generation = await _create(repository)
    status = await service.execute(generation.id, worker_id="w")

    assert status is GenerationStatus.COMPLETED
    trace = json.loads((await repository.get_generation(generation.id)).inference_qc_trace)
    assert len(trace["resilience"]["attempts"]) == 1
    assert trace["resilience"]["attempts"][0]["outcome"] == "SUCCEEDED"
    assert trace["resilience"]["provider_failovers"] == 0

    stored = await ResilienceRepository(engine).load(TEXT_TO_MUSIC.key())
    assert stored["state"] == CircuitState.CLOSED.value


# ── Phase 29 is untouched ────────────────────────────────────────────


async def test_without_a_gate_the_service_behaves_exactly_as_before(repository, tmp_path):
    provider = MockGenerationProvider(FIXTURE)
    service = _service(repository, provider, tmp_path, None)

    generation = await _create(repository)
    status = await service.execute(generation.id, worker_id="w")

    assert status is GenerationStatus.COMPLETED
    trace = json.loads((await repository.get_generation(generation.id)).inference_qc_trace)
    assert trace["resilience"] is None, "no routing record when nothing is routing"


async def test_the_quality_retry_budget_is_still_the_only_attempt_budget(
    repository, engine, tmp_path
):
    """Phase 31 redirects attempts; it does not add any.

    With the circuit effectively disabled (threshold 99) a dead provider
    costs exactly what it cost before this phase existed: Phase 29 tries
    the identical request once more, sees the same PROVIDER_TIMEOUT, and
    its repeated-failure rule stops there. Two calls — not two multiplied
    by anything the router does.
    """
    provider = DeadProvider()
    service = _service(
        repository, provider, tmp_path, _gate(engine, {"primary": provider}, threshold=99)
    )

    generation = await _create(repository)
    await service.execute(generation.id, worker_id="w")

    assert provider.calls == 2, "the Phase 29 count, unchanged by routing"


# ── configuration ────────────────────────────────────────────────────


class _Settings:
    generation_provider = "mock"
    mock_fixture_path = str(FIXTURE)
    provider_resilience_enabled = True
    provider_failover_mode = "DISABLED"
    provider_preference = ""
    provider_maximum_per_generation = 2
    ace_step_model = "test"


def test_the_factory_returns_nothing_when_resilience_is_off(engine):
    settings = _Settings()
    settings.provider_resilience_enabled = False
    assert build_resilience_gate(settings, repository=ResilienceRepository(engine)) is None


def test_the_factory_refuses_to_route_to_a_test_double(engine):
    """`mock` returns a committed fixture. Routing to it during an
    outage would deliver the same two seconds of audio to everybody,
    successfully, with nothing saying so."""
    gate = build_resilience_gate(_Settings(), repository=ResilienceRepository(engine))
    assert gate is None


def test_an_unknown_failover_mode_is_refused_rather_than_ignored(engine):
    settings = _Settings()
    settings.provider_failover_mode = "SOMETIMES"
    with pytest.raises(ValueError, match="unknown failover mode"):
        build_resilience_gate(settings, repository=ResilienceRepository(engine))


# ── the real provider, without calling it ────────────────────────────


class _AceSettings(_Settings):
    """Settings a deployment would actually use, minus the engine."""

    generation_provider = "ace_step"
    ace_step_base_url = "http://127.0.0.1:8019"
    ace_step_api_key = "sk-not-a-real-key"
    ace_step_request_timeout = 60.0
    ace_step_generation_timeout = 600.0
    ace_step_poll_interval = 2.0
    ace_step_output_dir = "data/raw-model-output"
    ace_step_inference_steps = 8
    ace_step_thinking = False


async def test_the_real_provider_profiles_with_every_capability_it_has(engine):
    """A safe smoke test: builds the real provider and asks what it can
    do. No request is made, no model is loaded, no GPU is touched.

    This is the test that would have caught the capability bug. The ABC
    declares `supports_reference_audio` as a **property** and
    `supports_audio_to_audio()` as a method; a profiler that accepted
    only callables read the property as "no" and every real deployment
    came back unable to take a reference track — a capability lost in
    silence, which is the failure this phase exists to prevent.
    """
    from luber_generation_client.resilience_factory import profiles_from_settings

    profiles = await profiles_from_settings(_AceSettings())

    assert len(profiles) == 1
    profile = profiles[0]
    assert profile.name == "ace_step"
    assert profile.revision == "test"
    assert "TEXT_TO_MUSIC" in profile.capabilities
    assert "REFERENCE_CONDITIONED" in profile.capabilities, (
        "the reference input is a property, not a method, and both shapes must be read"
    )


async def test_a_reference_request_routes_to_the_real_provider():
    """The capability and the request have to agree end to end.

    Profiled from the provider, required by the request, matched by the
    router. A mismatch anywhere refuses a reference-conditioned
    generation that the engine can serve perfectly well.
    """
    from luber_generation_client.resilience import needs_for
    from luber_generation_client.resilience_factory import profiles_from_settings
    from luber_provider_resilience import check_equivalence

    class _WithReference:
        edit_kind = None
        reference_audio_id = "a-reference"
        duration_requested = 20
        lyrics = "[Verse]\n테스트 가사"
        instrumental = False
        bpm = None
        key_scale = None

    needs = needs_for(_WithReference())
    assert needs.task_type == "REFERENCE_CONDITIONED"
    assert needs.has_reference

    profile = (await profiles_from_settings(_AceSettings()))[0]
    verdict = check_equivalence(profile, needs)

    assert verdict.equivalent, verdict.explain()
