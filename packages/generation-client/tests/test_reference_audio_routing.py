"""Reference audio from the generation row to the provider.

The ingestion tests prove a file can be stored safely. These prove the
stored reference actually reaches the engine, survives the trip through
the database and the worker unchanged, and — the part that matters most
— that a reference which cannot be honoured stops the generation instead
of quietly producing a song the user did not ask for.
"""

from __future__ import annotations

import json
import shutil
import uuid
import wave
from pathlib import Path

import pytest
from pydantic import ValidationError

from luber_audio_utils import LocalAudioStorage
from luber_database import GenerationRepository
from luber_generation_client import MockGenerationProvider
from luber_generation_client.provider import GenerationRequest, ReferenceAudioInput
from luber_generation_client.service import GenerationService
from luber_schemas import AssetType, ErrorCode, GenerationStatus, reference_storage_key

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mock_generation.wav"


def write_reference(path: Path, seconds: float = 2.0, rate: int = 48_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x01" * 2 * int(seconds * rate))
    return path


async def stored_reference(
    repository: GenerationRepository, storage: LocalAudioStorage, tmp_path: Path
):
    """A reference row whose object really exists in storage."""
    reference_id = uuid.uuid4()
    key = reference_storage_key(reference_id)
    await storage.put(key, write_reference(tmp_path / "ref.wav"))
    return await repository.create_reference_audio(
        reference_id=reference_id,
        storage_key=key,
        sha256="a" * 64,
        source_sha256="b" * 64,
        source_format="wav",
        duration_seconds=2.0,
        sample_rate=48_000,
        channels=2,
        file_size=1024,
        display_name="ref.wav",
    )


async def queued(repository: GenerationRepository, **extra):
    return await repository.create_generation(
        title="Reference routing",
        prompt="warm indie pop",
        lyrics="",
        vocal_gender="female",
        duration_requested=30,
        seed=None,
        language="en",
        instrumental=True,
        status=GenerationStatus.QUEUED.value,
        idempotency_key=None,
        **extra,
    )


class RecordingProvider(MockGenerationProvider):
    """Mock provider that remembers what it was asked to generate."""

    def __init__(self, fixture: Path, *, supports_reference: bool) -> None:
        super().__init__(fixture)
        self._supports_reference = supports_reference
        self.seen: list[GenerationRequest] = []

    @property
    def supports_reference_audio(self) -> bool:
        return self._supports_reference

    async def generate(self, request: GenerationRequest):
        self.seen.append(request)
        return await super().generate(request)


class TestContract:
    def test_a_reference_input_is_frozen(self):
        """Nothing downstream may retarget a reference mid-flight."""
        value = ReferenceAudioInput(
            reference_id=uuid.uuid4(),
            audio_path=Path("/tmp/x.wav"),
            duration_seconds=2.0,
            sha256="c" * 64,
        )
        with pytest.raises(ValidationError):
            value.audio_path = Path("/tmp/other.wav")  # type: ignore[misc]

    def test_a_request_without_a_reference_is_unchanged(self):
        """Backward compatibility: the field is optional and defaults off."""
        request = GenerationRequest(
            title="t", prompt="p", vocal_gender="female", duration_seconds=30
        )
        assert request.reference_audio is None

    def test_a_reference_survives_serialisation_unchanged(self):
        """The condition crosses the worker boundary as JSON."""
        original = GenerationRequest(
            title="t",
            prompt="p",
            vocal_gender="female",
            duration_seconds=30,
            reference_audio=ReferenceAudioInput(
                reference_id=uuid.uuid4(),
                audio_path=Path("/tmp/ref.wav"),
                duration_seconds=2.5,
                sha256="d" * 64,
            ),
        )
        restored = GenerationRequest.model_validate(json.loads(original.model_dump_json()))
        assert restored.reference_audio == original.reference_audio

    def test_providers_do_not_claim_reference_support_by_default(self):
        """Opt-in, so an unwired provider fails loudly rather than silently."""
        assert MockGenerationProvider(FIXTURE).supports_reference_audio is False


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg required for the delivery pipeline",
)
class TestRouting:
    async def test_the_provider_receives_the_reference(self, repository, tmp_path):
        storage = LocalAudioStorage(tmp_path / "store")
        reference = await stored_reference(repository, storage, tmp_path)
        generation = await queued(repository, reference_audio_id=reference.id)
        provider = RecordingProvider(FIXTURE, supports_reference=True)

        result = await GenerationService(repository, provider, storage).execute(
            generation.id, worker_id="w"
        )

        assert result is GenerationStatus.COMPLETED
        assert len(provider.seen) == 1
        seen = provider.seen[0].reference_audio
        assert seen is not None
        assert seen.reference_id == reference.id
        assert seen.sha256 == reference.sha256
        # A real, readable file — not just an identifier.
        assert seen.audio_path.is_file()

    async def test_a_generation_without_a_reference_still_works(self, repository, tmp_path):
        """The regression that matters: ordinary generation is untouched."""
        storage = LocalAudioStorage(tmp_path / "store")
        generation = await queued(repository)
        provider = RecordingProvider(FIXTURE, supports_reference=True)

        result = await GenerationService(repository, provider, storage).execute(
            generation.id, worker_id="w"
        )
        assert result is GenerationStatus.COMPLETED
        assert provider.seen[0].reference_audio is None

    async def test_provenance_records_which_reference_was_used(self, repository, tmp_path):
        storage = LocalAudioStorage(tmp_path / "store")
        reference = await stored_reference(repository, storage, tmp_path)
        generation = await queued(repository, reference_audio_id=reference.id)

        await GenerationService(
            repository, RecordingProvider(FIXTURE, supports_reference=True), storage
        ).execute(generation.id, worker_id="w")

        fetched = await repository.get_generation(generation.id)
        assert fetched.reference_audio_id == reference.id

    async def test_a_reference_never_becomes_an_audio_asset(self, repository, tmp_path):
        storage = LocalAudioStorage(tmp_path / "store")
        reference = await stored_reference(repository, storage, tmp_path)
        generation = await queued(repository, reference_audio_id=reference.id)

        await GenerationService(
            repository, RecordingProvider(FIXTURE, supports_reference=True), storage
        ).execute(generation.id, worker_id="w")

        assets = await repository.get_audio_assets(generation.id)
        assert {a.asset_type for a in assets} <= {
            AssetType.MASTER.value,
            AssetType.FINISHED_MASTER.value,
            AssetType.PREVIEW.value,
        }
        for asset in assets:
            assert not asset.storage_key.startswith("reference/")
            assert asset.sha256 != reference.sha256


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg required for the delivery pipeline",
)
class TestExplicitFailure:
    async def test_a_provider_without_support_refuses_the_generation(self, repository, tmp_path):
        """Never silently downgraded to an ordinary generation."""
        storage = LocalAudioStorage(tmp_path / "store")
        reference = await stored_reference(repository, storage, tmp_path)
        generation = await queued(repository, reference_audio_id=reference.id)
        provider = RecordingProvider(FIXTURE, supports_reference=False)

        result = await GenerationService(repository, provider, storage).execute(
            generation.id, worker_id="w"
        )

        assert result is GenerationStatus.FAILED
        assert provider.seen == [], "the provider must not be called at all"
        fetched = await repository.get_generation(generation.id)
        assert fetched.error_code == ErrorCode.REFERENCE_AUDIO_UNAVAILABLE.value
        assert not await repository.get_audio_assets(generation.id)

    async def test_a_missing_reference_row_fails_explicitly(self, repository, tmp_path):
        """The reference row vanished between queueing and running."""
        from sqlalchemy import delete

        from luber_database.models.generation import ReferenceAudio

        storage = LocalAudioStorage(tmp_path / "store")
        reference = await stored_reference(repository, storage, tmp_path)
        generation = await queued(repository, reference_audio_id=reference.id)

        # Delete the row out from under the queued generation, leaving its
        # link dangling. Production guards this with ON DELETE RESTRICT;
        # the point here is that the worker refuses even if a row is lost
        # some other way, rather than trusting the constraint.
        await repository._session.execute(
            delete(ReferenceAudio).where(ReferenceAudio.id == reference.id)
        )
        await repository._session.commit()

        provider = RecordingProvider(FIXTURE, supports_reference=True)
        result = await GenerationService(repository, provider, storage).execute(
            generation.id, worker_id="w"
        )

        assert result is GenerationStatus.FAILED
        assert provider.seen == [], "the provider must not be called without the reference"
        fetched = await repository.get_generation(generation.id)
        assert fetched.error_code == ErrorCode.REFERENCE_AUDIO_UNAVAILABLE.value

    async def test_a_reference_object_missing_from_storage_fails_explicitly(
        self, repository, tmp_path
    ):
        """The row survives but the bytes are gone."""
        storage = LocalAudioStorage(tmp_path / "store")
        reference = await stored_reference(repository, storage, tmp_path)
        await storage.delete(reference.storage_key)
        generation = await queued(repository, reference_audio_id=reference.id)
        provider = RecordingProvider(FIXTURE, supports_reference=True)

        result = await GenerationService(repository, provider, storage).execute(
            generation.id, worker_id="w"
        )

        assert result is GenerationStatus.FAILED
        assert provider.seen == []
        fetched = await repository.get_generation(generation.id)
        assert fetched.error_code == ErrorCode.REFERENCE_AUDIO_UNAVAILABLE.value
