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

from luber_generation_client.editing import (
    AudioEditingProvider,
    AudioEditKind,
    AudioEditRequest,
)
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


class MockGenerationProvider(MusicGenerationProvider, AudioEditingProvider):
    """Returns the committed fixture WAV as the generation output.

    Implements the editing contract too, so tests can prove the worker
    routes an edit to :meth:`edit` and never to :meth:`generate`. It does
    not pretend to *perform* an edit: the returned audio is the same
    fixture, and every call is recorded on :attr:`edits` for assertions.
    A provider that cannot edit is exercised separately, because failing
    rather than falling back is itself a requirement.
    """

    name = MOCK_PROVIDER_NAME

    def __init__(self, fixture_path: Path) -> None:
        self._fixture_path = fixture_path
        #: Every edit this provider was asked to perform, in order.
        self.edits: list[AudioEditRequest] = []

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        return self._fixture_result(seed=request.seed)

    def _fixture_result(self, *, seed: int | None) -> GenerationResult:
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
            seed_used=seed,
            provider=MOCK_PROVIDER_NAME,
            model_name=MOCK_MODEL_NAME,
            model_version=MOCK_MODEL_VERSION,
        )

    # ── editing (Phase 13B) ────────────────────────────────────────

    def supports_edit(self, kind: AudioEditKind) -> bool:
        return kind is AudioEditKind.REGENERATE_RANGE

    def describe_edit(self, request: AudioEditRequest) -> dict[str, object]:
        return {
            "provider": MOCK_PROVIDER_NAME,
            "operation": "edit",
            "edit_kind": request.kind.value,
            "start_seconds": request.start_seconds,
            "end_seconds": request.end_seconds,
            "source_audio_bytes": request.source_audio.stat().st_size,
        }

    async def edit(self, request: AudioEditRequest) -> GenerationResult:
        """Record the edit and return the fixture.

        The recorded request is the point: it lets a test assert that the
        parent's real audio, and the measured range, reached the provider.
        """
        self.edits.append(request)
        # Deliberately not routed through ``generate``: an edit's canvas
        # can be shorter than ``DURATION_MIN``, and borrowing the
        # text-to-music request model would impose a bound that does not
        # apply to editing.
        return self._fixture_result(seed=request.seed)
