from pathlib import Path

import pytest
from pydantic import ValidationError

from luber_generation_client import (
    GenerationRequest,
    GenerationResult,
    MusicGenerationProvider,
)
from luber_schemas import VocalGender


def test_provider_is_abstract():
    with pytest.raises(TypeError):
        MusicGenerationProvider()  # type: ignore[abstract]


def test_request_validates_duration_bounds():
    with pytest.raises(ValidationError):
        GenerationRequest(
            title="x",
            prompt="y",
            vocal_gender=VocalGender.FEMALE,
            duration_seconds=9999,
        )


def test_request_defaults():
    req = GenerationRequest(
        title="Stay With Me",
        prompt="Emotional Korean pop ballad with warm piano",
        lyrics="[Verse 1]\n오늘도 너를 기다려",
        vocal_gender=VocalGender.FEMALE,
    )
    assert req.duration_seconds == 180
    assert req.seed is None
    assert "[Verse 1]" in req.lyrics  # section tags must be preserved


async def test_concrete_provider_satisfies_contract(tmp_path: Path):
    class _StubProvider(MusicGenerationProvider):
        name = "stub"

        async def generate(self, request: GenerationRequest) -> GenerationResult:
            return GenerationResult(
                audio_path=tmp_path / "out.wav",
                duration_seconds=1.0,
                sample_rate=48000,
                provider=self.name,
                model_name="stub-model",
                model_version="0",
            )

    provider = _StubProvider()
    result = await provider.generate(
        GenerationRequest(title="t", prompt="p", vocal_gender=VocalGender.INSTRUMENTAL)
    )
    assert result.sample_rate == 48000
    assert result.provider == "stub"
