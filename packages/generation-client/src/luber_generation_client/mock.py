"""MockGenerationProvider — CI-safe provider returning a real fixture WAV.

This provider reads a genuine, valid, non-silent PCM WAV fixture
(``tests/fixtures/mock_generation.wav``) and returns it through the
standard provider contract. It reports itself honestly as ``mock`` /
``mock-generation-provider`` — it never pretends to be a real AI model
run.

**It honours the requested duration.** The fixture is two seconds long
and is repeated to approximately the length that was asked for. That is
not cosmetic: a double whose output has no relationship to the request
does not "satisfy the same interface contract as production providers",
and Phase 29 measures duration adherence — a two-second answer to a
thirty-second request is a failure the real product should catch and a
test double should not manufacture.

The repetition is audible as a seam every two seconds. That is fine and
deliberate: nothing here claims to be music, and every technical
property a test asserts on — level, spectrum, correlation, silence — is
unchanged by repeating identical content.
"""

from __future__ import annotations

import tempfile
import wave
from pathlib import Path

from luber_generation_client.audio_to_audio import (
    AudioToAudioProvider,
    AudioToAudioRequest,
)
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


def _wav_seconds(path: Path) -> float | None:
    """How long *path* is, or ``None`` if that cannot be read.

    Returning ``None`` rather than raising keeps a test that hands the
    double something odd failing on its own assertion instead of on a
    decode error from the stand-in.
    """
    try:
        with wave.open(str(path), "rb") as wav:
            rate = wav.getframerate()
            frames = wav.getnframes()
    except (OSError, wave.Error, EOFError):
        return None
    if rate <= 0 or frames <= 0:
        return None
    return frames / rate


class MockGenerationProvider(MusicGenerationProvider, AudioEditingProvider, AudioToAudioProvider):
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
        # Rendered files live for the life of the provider. The caller
        # copies what it needs; cleaning up earlier would pull the file
        # out from under it.
        self._rendered = tempfile.TemporaryDirectory(prefix="luber-mock-")
        #: Every edit this provider was asked to perform, in order.
        self.edits: list[AudioEditRequest] = []
        #: Every source-conditioned generation, in order. Recorded so a
        #: test can assert which provider method a request reached.
        self.covers: list[AudioToAudioRequest] = []

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        return self._fixture_result(
            seed=request.seed, duration_seconds=float(request.duration_seconds)
        )

    def _render(self, source: Path, sample_rate: int, frames: int, target_seconds: float) -> Path:
        """Repeat the fixture to the requested length.

        Whole frames only, and never longer than asked: a provider that
        overshot would be manufacturing the opposite failure.
        """
        wanted = round(target_seconds * sample_rate)
        if wanted <= 0 or wanted == frames:
            return source

        with wave.open(str(source), "rb") as wav:
            params = wav.getparams()
            payload = wav.readframes(frames)

        repeats = -(-wanted // frames)  # ceiling division
        frame_bytes = params.sampwidth * params.nchannels
        rendered = (payload * repeats)[: wanted * frame_bytes]

        destination = Path(self._rendered.name) / f"mock-{wanted}.wav"
        if not destination.is_file():
            with wave.open(str(destination), "wb") as out:
                out.setnchannels(params.nchannels)
                out.setsampwidth(params.sampwidth)
                out.setframerate(sample_rate)
                out.writeframes(rendered)
        return destination

    def _fixture_result(
        self, *, seed: int | None, duration_seconds: float | None = None
    ) -> GenerationResult:
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
        if duration_seconds is not None:
            path = self._render(path, sample_rate, frames, duration_seconds)
            with wave.open(str(path), "rb") as wav:
                frames = wav.getnframes()

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
        """Record the edit and return the fixture, at the source's length.

        The recorded request is the point: it lets a test assert that the
        parent's real audio, and the measured range, reached the provider.

        The length matters too. Replacing an interior span leaves the
        song the length it was, and a double that returned a fixed
        two-second clip would let a regression in that rule pass — the
        assertion it breaks would be about the double, not the code.
        """
        self.edits.append(request)
        # Deliberately not routed through ``generate``: an edit's canvas
        # can be shorter than ``DURATION_MIN``, and borrowing the
        # text-to-music request model would impose a bound that does not
        # apply to editing.
        return self._fixture_result(
            seed=request.seed, duration_seconds=_wav_seconds(request.source_audio)
        )

    # ── source-conditioned generation (Phase 13D) ──────────────────

    def supports_audio_to_audio(self) -> bool:
        return True

    def validated_adherence_range(self) -> tuple[float, float]:
        """Mirrors the real provider's calibrated band.

        The same band as ACE-Step so tests exercise the same rejection
        the product will hit, rather than a permissive stand-in.
        """
        return (0.75, 1.0)

    def describe_audio_to_audio(self, request: AudioToAudioRequest) -> dict[str, object]:
        return {
            "provider": MOCK_PROVIDER_NAME,
            "operation": "cover",
            "source_adherence": request.source_adherence,
            "source_audio_bytes": request.source_audio.stat().st_size,
        }

    async def create_from_audio(self, request: AudioToAudioRequest) -> GenerationResult:
        """Record the request and return the fixture.

        The recording is the point: it lets a test prove the source audio
        and the mapped adherence actually reached the provider.
        """
        low, high = self.validated_adherence_range()
        if not low <= request.source_adherence <= high:
            raise GenerationProviderError(
                f"source_adherence {request.source_adherence} outside {low}-{high}",
                error_code=ErrorCode.UNKNOWN_GENERATION_ERROR,
            )
        self.covers.append(request)
        # A cover is the same song again, so it is the same length again.
        return self._fixture_result(
            seed=request.seed, duration_seconds=_wav_seconds(request.source_audio)
        )
