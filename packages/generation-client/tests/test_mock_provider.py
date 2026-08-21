from pathlib import Path

import pytest

from luber_generation_client import (
    GenerationProviderError,
    GenerationRequest,
    MockGenerationProvider,
)
from luber_generation_client.mock import MOCK_MODEL_NAME, MOCK_MODEL_VERSION, MOCK_PROVIDER_NAME
from luber_schemas import ErrorCode, VocalGender

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mock_generation.wav"


def _request(**overrides):
    defaults = dict(
        title="TEST SONG",
        prompt="Dreamy Korean indie pop",
        lyrics="[Verse]\n테스트 가사",
        vocal_gender=VocalGender.FEMALE,
        duration_seconds=30,
        language="ko",
    )
    defaults.update(overrides)
    return GenerationRequest(**defaults)


async def test_mock_provider_returns_real_fixture_audio():
    provider = MockGenerationProvider(FIXTURE)
    result = await provider.generate(_request(seed=42))

    # A rendering of the fixture rather than the fixture itself: the
    # provider honours the requested duration, and a double whose output
    # ignored the request would not satisfy the contract it claims to.
    assert result.audio_path.is_file()
    assert result.audio_path.suffix == ".wav"
    assert result.duration_seconds == pytest.approx(_request().duration_seconds, rel=0.01)
    assert result.sample_rate == 48000
    assert result.seed_used == 42
    # Honest self-identification — never masquerades as a real model.
    assert result.provider == MOCK_PROVIDER_NAME == "mock"
    assert result.model_name == MOCK_MODEL_NAME == "mock-generation-provider"
    assert result.model_version == MOCK_MODEL_VERSION == "phase1"


async def test_mock_provider_missing_fixture_fails_honestly(tmp_path):
    provider = MockGenerationProvider(tmp_path / "missing.wav")
    with pytest.raises(GenerationProviderError) as excinfo:
        await provider.generate(_request())
    assert excinfo.value.error_code is ErrorCode.MODEL_LOAD_FAILED


async def test_mock_provider_rejects_invalid_fixture(tmp_path):
    bogus = tmp_path / "bogus.wav"
    bogus.write_text("not audio")
    provider = MockGenerationProvider(bogus)
    with pytest.raises(GenerationProviderError) as excinfo:
        await provider.generate(_request())
    assert excinfo.value.error_code is ErrorCode.INVALID_AUDIO
