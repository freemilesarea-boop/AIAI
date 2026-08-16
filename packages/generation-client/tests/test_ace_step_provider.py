from pathlib import Path

import pytest
from ace_step_fake_server import FakeAceStepServer

from luber_generation_client import GenerationProviderError, GenerationRequest
from luber_generation_client.ace_step import (
    ACE_STEP_VERSION,
    AceStepClient,
    AceStepProvider,
    AceStepProviderConfig,
)
from luber_generation_client.ace_step.client import AceStepApiError
from luber_schemas import ErrorCode, VocalGender

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mock_generation.wav"


def _provider(server: FakeAceStepServer, tmp_path: Path, **config_overrides) -> AceStepProvider:
    config_kwargs: dict = {
        "base_url": "http://acestep.test",
        "output_dir": tmp_path / "raw",
        "poll_interval": 0.01,
        "generation_timeout": 5.0,
    }
    config_kwargs.update(config_overrides)
    config = AceStepProviderConfig(**config_kwargs)
    client = AceStepClient(config.base_url, transport=server.transport())
    return AceStepProvider(config, client=client)


def _request(**overrides):
    defaults = dict(
        title="PHASE 2 REAL TEST",
        prompt="Dreamy Korean indie pop with warm electric piano",
        lyrics="[Verse]\n오늘 밤 너를 생각해",
        vocal_gender=VocalGender.FEMALE,
        duration_seconds=30,
        language="ko",
        seed=777,
    )
    defaults.update(overrides)
    return GenerationRequest(**defaults)


async def test_provider_full_flow_returns_real_audio(tmp_path):
    server = FakeAceStepServer(FIXTURE, polls_before_success=2)
    provider = _provider(server, tmp_path)

    result = await provider.generate(_request())

    # Real bytes downloaded from the (fake) server, parsed as WAV.
    assert result.audio_path.is_file()
    assert result.audio_path.read_bytes() == FIXTURE.read_bytes()
    assert result.duration_seconds == pytest.approx(2.0)
    assert result.sample_rate == 48000
    assert result.provider == "ace_step"
    assert result.model_name == "acestep-v15-turbo"  # from task result dit_model
    assert result.model_version == ACE_STEP_VERSION
    assert result.seed_used == 12345  # server-reported seed wins

    # Request mapping used only verified upstream fields.
    payload = server.release_payloads[0]
    assert payload["audio_format"] == "wav"
    assert payload["audio_duration"] == 30.0
    assert payload["vocal_language"] == "ko"
    assert payload["batch_size"] == 1
    assert payload["model"] == "acestep-v15-turbo"
    assert payload["use_random_seed"] is False
    assert payload["seed"] == 777
    assert "female lead vocal" in str(payload["prompt"])
    assert str(payload["lyrics"]).startswith("[Verse]")
    assert "vocal_gender" not in payload  # no invented upstream fields


async def test_provider_maps_task_failure(tmp_path):
    server = FakeAceStepServer(FIXTURE, fail_task=True, fail_message="CUDA out of memory")
    provider = _provider(server, tmp_path)
    with pytest.raises(GenerationProviderError) as excinfo:
        await provider.generate(_request())
    assert excinfo.value.error_code is ErrorCode.OUT_OF_MEMORY


async def test_provider_times_out_with_generation_timeout(tmp_path):
    server = FakeAceStepServer(FIXTURE, never_finish=True)
    # Duration-aware scaling is switched off here so the flat budget is
    # the one under test. An operator who deliberately tightens the
    # timeout must still get the timeout they asked for.
    provider = _provider(
        server,
        tmp_path,
        generation_timeout=0.05,
        timeout_base_seconds=0.0,
        timeout_multiplier=0.0,
    )
    with pytest.raises(GenerationProviderError) as excinfo:
        await provider.generate(_request())
    assert excinfo.value.error_code is ErrorCode.GENERATION_TIMEOUT


async def test_provider_refuses_uninitialized_server(tmp_path):
    server = FakeAceStepServer(FIXTURE, models_initialized=False)
    provider = _provider(server, tmp_path)
    with pytest.raises(GenerationProviderError) as excinfo:
        await provider.generate(_request())
    assert excinfo.value.error_code is ErrorCode.MODEL_LOAD_FAILED


async def test_provider_unreachable_server_maps_to_model_load_failed(tmp_path):
    import httpx

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    config = AceStepProviderConfig(base_url="http://down.test", output_dir=tmp_path)
    client = AceStepClient(config.base_url, transport=httpx.MockTransport(refuse))
    provider = AceStepProvider(config, client=client)
    with pytest.raises(GenerationProviderError) as excinfo:
        await provider.generate(_request())
    assert excinfo.value.error_code is ErrorCode.MODEL_LOAD_FAILED


class TestUpstreamErrorClassification:
    """A busy engine is not a broken one.

    ACE-Step answers HTTP 429 with "Server busy: queue is full" when its
    own queue is saturated (``release_task_route.py``). The same request
    submitted later succeeds, so flattening it into
    UNKNOWN_GENERATION_ERROR told the user their song had failed when
    nothing about it was wrong.
    """

    def test_a_full_provider_queue_is_its_own_code(self):
        exc = AceStepApiError("ACE-Step API returned HTTP 429", status_code=429)
        assert AceStepProvider._classify_api_error(exc) is ErrorCode.PROVIDER_BUSY

    def test_other_statuses_are_not_busy(self):
        exc = AceStepApiError("ACE-Step API returned HTTP 500", status_code=500)
        assert AceStepProvider._classify_api_error(exc) is not ErrorCode.PROVIDER_BUSY

    def test_classification_does_not_depend_on_wording(self):
        """Only the status is read, so the engine may reword freely."""
        exc = AceStepApiError("완전히 다른 문구", status_code=429)
        assert AceStepProvider._classify_api_error(exc) is ErrorCode.PROVIDER_BUSY

    def test_an_error_without_a_status_falls_back_to_the_message(self):
        assert (
            AceStepProvider._classify_api_error(RuntimeError("request timed out"))
            is ErrorCode.GENERATION_TIMEOUT
        )

    def test_an_unrecognised_failure_stays_unknown(self):
        assert (
            AceStepProvider._classify_api_error(RuntimeError("something odd"))
            is ErrorCode.UNKNOWN_GENERATION_ERROR
        )
