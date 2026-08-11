"""MockGenerationProvider — CI-safe provider returning a real fixture WAV.

This provider reads a genuine, valid, non-silent PCM WAV fixture
(``tests/fixtures/mock_generation.wav``) and returns it through the
standard provider contract. It reports itself honestly as ``mock`` /
``mock-generation-provider`` — it never pretends to be a real AI model
run.
"""

from __future__ import annotations

import wave
from pathlib import Path

from luber_generation_client.errors import GenerationProviderError
from luber_generation_client.provider import (
    GenerationRequest,
    GenerationResult,
    MusicGenerationProvider,
)
from luber_schemas import ErrorCode

MOCK_PROVIDER_NAME = "mock"
MOCK_MODEL_NAME = "mock-generation-provider"
MOCK_MODEL_VERSION = "phase1"


class MockGenerationProvider(MusicGenerationProvider):
    """Returns the committed fixture WAV as the generation output."""

    name = MOCK_PROVIDER_NAME

    def __init__(self, fixture_path: Path) -> None:
        self._fixture_path = fixture_path

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        path = self._fixture_path
        if not path.is_file():
            raise GenerationProviderError(
                f"mock fixture WAV missing: {path}",
                error_code=ErrorCode.MODEL_LOAD_FAILED,
            )
        try:
            with wave.open(str(path), "rb") as wav:
                sample_rate = wav.getframerate()
                frames = wav.getnframes()
        except (wave.Error, EOFError) as exc:
            raise GenerationProviderError(
                f"mock fixture is not a valid WAV: {path}: {exc}",
                error_code=ErrorCode.INVALID_AUDIO,
            ) from exc
        if sample_rate <= 0 or frames <= 0:
            raise GenerationProviderError(
                f"mock fixture has no audio: {path}",
                error_code=ErrorCode.INVALID_AUDIO,
            )
        return GenerationResult(
            audio_path=path,
            duration_seconds=frames / sample_rate,
            sample_rate=sample_rate,
            seed_used=request.seed,
            provider=MOCK_PROVIDER_NAME,
            model_name=MOCK_MODEL_NAME,
            model_version=MOCK_MODEL_VERSION,
        )
