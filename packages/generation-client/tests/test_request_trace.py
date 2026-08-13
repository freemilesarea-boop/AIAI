"""Request-trace persistence at the service boundary.

The trace answers one question after the fact: *what did LUBER actually
send to the provider?* Three properties make it trustworthy, and each is
asserted here rather than assumed:

- it is written **before** the provider runs, so a failed generation is
  as inspectable as a successful one;
- writing it can **never** fail the generation the user asked for;
- it carries **no** credentials, hostnames, or local paths.
"""

from __future__ import annotations

import json
from pathlib import Path

from luber_audio_utils import LocalAudioStorage
from luber_generation_client import (
    GenerationRequest,
    GenerationResult,
    GenerationService,
    MockGenerationProvider,
    MusicGenerationProvider,
)
from luber_generation_client.ace_step import (
    AceStepClient,
    AceStepProvider,
    AceStepProviderConfig,
)
from luber_schemas import GenerationStatus

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mock_generation.wav"

SECRET_API_KEY = "sk-live-DO-NOT-LEAK-2f4a9c"
SECRET_BASE_URL = "http://internal-gpu-07.acme.invalid:8001"


class _TracingMockProvider(MockGenerationProvider):
    """Mock audio, real trace — lets the service path be exercised fast."""

    def describe_request(self, request: GenerationRequest) -> dict[str, object]:
        return {
            "provider": "mock",
            "payload": {
                "audio_duration": float(request.duration_seconds),
                "bpm": request.bpm,
                "key_scale": request.key_scale,
                "time_signature": request.time_signature,
            },
        }


class _FailingProvider(MusicGenerationProvider):
    name = "failing"

    def describe_request(self, request: GenerationRequest) -> dict[str, object]:
        return {"provider": "failing", "payload": {"bpm": request.bpm}}

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        raise RuntimeError("synthetic provider crash")


class _UntraceableProvider(MockGenerationProvider):
    """A provider whose trace builder is broken."""

    def describe_request(self, request: GenerationRequest) -> dict[str, object]:
        raise RuntimeError("synthetic trace failure")


async def _create(repository, **overrides):
    defaults = dict(
        title="TRACE TEST",
        prompt="Dreamy Korean indie pop",
        lyrics="[Verse]\n테스트 가사",
        vocal_gender="female",
        duration_requested=30,
        status=GenerationStatus.QUEUED.value,
        language="ko",
    )
    defaults.update(overrides)
    gen = await repository.create_generation(**defaults)
    await repository.create_job(
        gen.id, queue_name="luber:generation", status=GenerationStatus.QUEUED.value
    )
    return gen


def _service(repository, provider, tmp_path):
    return GenerationService(repository, provider, LocalAudioStorage(tmp_path))


async def test_trace_is_persisted_for_a_successful_generation(repository, tmp_path):
    gen = await _create(repository, bpm=128, key_scale="C major", time_signature="4")
    final = await _service(repository, _TracingMockProvider(FIXTURE), tmp_path).execute(gen.id)

    assert final is GenerationStatus.COMPLETED
    fetched = await repository.get_generation(gen.id)
    trace = json.loads(fetched.request_trace)
    assert trace["payload"]["bpm"] == 128
    assert trace["payload"]["key_scale"] == "C major"
    assert trace["payload"]["time_signature"] == "4"


async def test_trace_is_persisted_even_when_the_generation_fails(repository, tmp_path):
    # The whole point: a failed run must stay inspectable.
    gen = await _create(repository, bpm=90)
    final = await _service(repository, _FailingProvider(), tmp_path).execute(gen.id)

    assert final is GenerationStatus.FAILED
    fetched = await repository.get_generation(gen.id)
    assert fetched.request_trace is not None
    assert json.loads(fetched.request_trace)["payload"]["bpm"] == 90


async def test_a_broken_trace_never_fails_the_generation(repository, tmp_path):
    gen = await _create(repository)
    final = await _service(repository, _UntraceableProvider(FIXTURE), tmp_path).execute(gen.id)

    assert final is GenerationStatus.COMPLETED
    fetched = await repository.get_generation(gen.id)
    assert fetched.request_trace is None


async def test_provider_without_a_trace_records_none(repository, tmp_path):
    # Phase 7-era providers: no trace, no error, no behaviour change.
    gen = await _create(repository)
    final = await _service(repository, MockGenerationProvider(FIXTURE), tmp_path).execute(gen.id)

    assert final is GenerationStatus.COMPLETED
    fetched = await repository.get_generation(gen.id)
    assert fetched.request_trace is None


async def test_advanced_controls_travel_from_the_row_to_the_provider(repository, tmp_path):
    captured: list[GenerationRequest] = []

    class _Capturing(MockGenerationProvider):
        async def generate(self, request):
            captured.append(request)
            return await super().generate(request)

    gen = await _create(repository, bpm=142, key_scale="F# minor", time_signature="6")
    await _service(repository, _Capturing(FIXTURE), tmp_path).execute(gen.id)

    assert captured[0].bpm == 142
    assert captured[0].key_scale == "F# minor"
    assert captured[0].time_signature == "6"


async def test_unset_controls_stay_unset_through_the_service(repository, tmp_path):
    captured: list[GenerationRequest] = []

    class _Capturing(MockGenerationProvider):
        async def generate(self, request):
            captured.append(request)
            return await super().generate(request)

    gen = await _create(repository)  # a Phase 7-shaped row
    await _service(repository, _Capturing(FIXTURE), tmp_path).execute(gen.id)

    assert captured[0].bpm is None
    assert captured[0].key_scale is None
    assert captured[0].time_signature is None


async def test_persisted_ace_step_trace_carries_no_secrets(repository, tmp_path):
    """End-to-end sanitization: real provider, real service, stored row."""
    from ace_step_fake_server import FakeAceStepServer

    server = FakeAceStepServer(FIXTURE)
    config = AceStepProviderConfig(
        base_url=SECRET_BASE_URL,
        api_key=SECRET_API_KEY,
        output_dir=tmp_path / "raw",
        poll_interval=0.01,
        generation_timeout=5.0,
    )
    provider = AceStepProvider(
        config,
        client=AceStepClient(config.base_url, api_key=config.api_key, transport=server.transport()),
    )

    gen = await _create(repository, bpm=128, key_scale="C major", time_signature="4")
    await _service(repository, provider, tmp_path).execute(gen.id)

    fetched = await repository.get_generation(gen.id)
    stored = fetched.request_trace
    assert stored is not None

    # The generation-relevant facts are there…
    trace = json.loads(stored)
    assert trace["payload"]["bpm"] == 128
    assert trace["payload"]["key_scale"] == "C major"

    # …and nothing that could authenticate, locate, or expose the host.
    for secret in (SECRET_API_KEY, SECRET_BASE_URL, "internal-gpu-07", str(tmp_path)):
        assert secret not in stored
    for key in ("api_key", "base_url", "Authorization", "Bearer", "password", "token"):
        assert key not in stored
