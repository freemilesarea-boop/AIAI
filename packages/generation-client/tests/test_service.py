from pathlib import Path

from luber_audio_utils import LocalAudioStorage
from luber_generation_client import (
    GenerationRequest,
    GenerationResult,
    GenerationService,
    MockGenerationProvider,
    MusicGenerationProvider,
)
from luber_schemas import AssetType, ErrorCode, GenerationStatus

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mock_generation.wav"


class _RecordingRepository:
    """Wraps GenerationRepository to record the status transition order."""

    def __init__(self, inner):
        self._inner = inner
        self.status_history: list[str] = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def mark_started(self, generation_id, *, status):
        self.status_history.append(status)
        return await self._inner.mark_started(generation_id, status=status)

    async def update_status(self, generation_id, status):
        self.status_history.append(status)
        return await self._inner.update_status(generation_id, status)

    async def mark_completed(self, generation_id, **kwargs):
        self.status_history.append(kwargs["status"])
        return await self._inner.mark_completed(generation_id, **kwargs)

    async def mark_failed(self, generation_id, **kwargs):
        self.status_history.append(kwargs["status"])
        return await self._inner.mark_failed(generation_id, **kwargs)


class _ExplodingProvider(MusicGenerationProvider):
    name = "exploding"

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        raise RuntimeError("synthetic provider crash")


async def _create_generation(repository):
    return await repository.create_generation(
        title="TEST SONG",
        prompt="Dreamy Korean indie pop",
        lyrics="[Verse]\n테스트 가사",
        vocal_gender="female",
        duration_requested=30,
        status=GenerationStatus.QUEUED.value,
        language="ko",
    )


async def test_service_success_walks_full_lifecycle(repository, tmp_path):
    gen = await _create_generation(repository)
    await repository.create_job(
        gen.id, queue_name="luber:generation", status=GenerationStatus.QUEUED.value
    )
    recording = _RecordingRepository(repository)
    service = GenerationService(
        recording,  # type: ignore[arg-type]
        MockGenerationProvider(FIXTURE),
        LocalAudioStorage(tmp_path),
    )

    final = await service.execute(gen.id, worker_id="test-worker")

    assert final is GenerationStatus.COMPLETED
    assert recording.status_history == [
        GenerationStatus.STARTING.value,
        GenerationStatus.GENERATING.value,
        GenerationStatus.POST_PROCESSING.value,
        GenerationStatus.UPLOADING.value,
        GenerationStatus.COMPLETED.value,
    ]

    fetched = await repository.get_generation(gen.id)
    assert fetched.status == GenerationStatus.COMPLETED.value
    assert fetched.provider == "mock"
    assert fetched.model_name == "mock-generation-provider"
    assert fetched.model_version == "phase1"
    assert fetched.duration_actual and fetched.duration_actual > 0
    assert fetched.started_at is not None and fetched.completed_at is not None

    # Post-processing produces both delivery assets.
    assets = {a.asset_type: a for a in await repository.get_audio_assets(gen.id)}
    assert set(assets) == {AssetType.MASTER.value, AssetType.PREVIEW.value}

    master = assets[AssetType.MASTER.value]
    assert master.format == "wav"
    assert master.mime_type == "audio/wav"
    assert master.file_extension == "wav"
    assert master.sample_rate == 48000
    assert master.channels == 2
    assert master.bit_depth == 24
    assert len(master.sha256) == 64

    preview = assets[AssetType.PREVIEW.value]
    assert preview.format == "mp3"
    assert preview.mime_type == "audio/mpeg"
    assert preview.sample_rate == 48000
    assert preview.channels == 2
    assert preview.bitrate == 320000
    assert len(preview.sha256) == 64
    assert (tmp_path / preview.storage_key).is_file()
    # The stored file is a real copy of the WAV, not a fake path.
    stored = tmp_path / master.storage_key
    assert stored.is_file()
    assert stored.stat().st_size == master.file_size

    job = await repository.get_latest_job(gen.id)
    assert job.status == GenerationStatus.COMPLETED.value
    assert job.worker_id == "test-worker"


async def test_service_failure_marks_failed_with_error_code(repository, tmp_path):
    gen = await _create_generation(repository)
    await repository.create_job(
        gen.id, queue_name="luber:generation", status=GenerationStatus.QUEUED.value
    )
    service = GenerationService(repository, _ExplodingProvider(), LocalAudioStorage(tmp_path))

    final = await service.execute(gen.id, worker_id="test-worker")

    assert final is GenerationStatus.FAILED
    fetched = await repository.get_generation(gen.id)
    assert fetched.status == GenerationStatus.FAILED.value
    assert fetched.error_code == ErrorCode.UNKNOWN_GENERATION_ERROR.value
    assert "synthetic provider crash" in fetched.error_message
    job = await repository.get_latest_job(gen.id)
    assert job.status == GenerationStatus.FAILED.value
    assert await repository.get_audio_assets(gen.id) == []


async def test_service_missing_fixture_maps_to_model_load_failed(repository, tmp_path):
    gen = await _create_generation(repository)
    service = GenerationService(
        repository,
        MockGenerationProvider(tmp_path / "missing.wav"),
        LocalAudioStorage(tmp_path),
    )
    final = await service.execute(gen.id)
    assert final is GenerationStatus.FAILED
    fetched = await repository.get_generation(gen.id)
    assert fetched.error_code == ErrorCode.MODEL_LOAD_FAILED.value
