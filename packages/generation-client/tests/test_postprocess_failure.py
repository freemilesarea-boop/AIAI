"""Failure boundaries and idempotency around post-processing.

The rule these enforce: a generation is COMPLETED only when every
required delivery asset exists in storage and is recorded. Any failure
in normalization, encoding, or upload must leave the generation FAILED.
"""

import shutil
import uuid
from pathlib import Path

import pytest

from luber_audio_utils import (
    AudioStorageError,
    LocalAudioStorage,
    master_storage_key,
    preview_storage_key,
)
from luber_database import GenerationRepository
from luber_generation_client import GenerationService, MockGenerationProvider
from luber_generation_client.postprocess import produce_delivery_assets
from luber_schemas import AssetType, ErrorCode, GenerationStatus

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe required for audio post-processing tests",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mock_generation.wav"


async def _queued_generation(repository: GenerationRepository):
    return await repository.create_generation(
        title="Failure Boundary",
        prompt="test",
        lyrics="[Verse]\n테스트",
        vocal_gender="female",
        duration_requested=30,
        seed=None,
        language="ko",
        instrumental=False,
        status=GenerationStatus.QUEUED.value,
        idempotency_key=None,
    )


class ExplodingStorage(LocalAudioStorage):
    """Local storage that fails on the Nth put, simulating upload loss."""

    def __init__(self, base_dir: Path, fail_on_key_substring: str) -> None:
        super().__init__(base_dir)
        self._fail_on = fail_on_key_substring

    async def put(self, storage_key: str, source: Path) -> str:
        if self._fail_on in storage_key:
            raise AudioStorageError(f"simulated upload failure for {storage_key}")
        return await super().put(storage_key, source)


# ── 13. post-processing failure → not COMPLETED ───────────────────────


async def test_transcode_failure_leaves_generation_failed(repository, tmp_path):
    """Unreadable model output must fail the generation, not complete it."""
    gen = await _queued_generation(repository)
    broken = tmp_path / "broken.wav"
    # Valid WAV header so inspect_wav-level checks pass far enough to
    # reach the transcoder, but no decodable audio.
    broken.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 32)
    provider = MockGenerationProvider(broken)
    service = GenerationService(repository, provider, LocalAudioStorage(tmp_path / "store"))

    final = await service.execute(gen.id, worker_id="w")

    assert final is GenerationStatus.FAILED
    fetched = await repository.get_generation(gen.id)
    assert fetched.status == GenerationStatus.FAILED.value
    assert fetched.error_code in {
        ErrorCode.INVALID_AUDIO.value,
        ErrorCode.ENCODING_FAILED.value,
    }
    # No half-written assets recorded.
    assert await repository.get_audio_assets(gen.id) == []


# ── 14. upload/storage failure → not COMPLETED ────────────────────────


async def test_master_upload_failure_leaves_generation_failed(repository, tmp_path):
    gen = await _queued_generation(repository)
    service = GenerationService(
        repository,
        MockGenerationProvider(FIXTURE),
        ExplodingStorage(tmp_path / "store", "master"),
    )

    final = await service.execute(gen.id, worker_id="w")

    assert final is GenerationStatus.FAILED
    fetched = await repository.get_generation(gen.id)
    assert fetched.status == GenerationStatus.FAILED.value
    assert fetched.error_code == ErrorCode.UPLOAD_FAILED.value
    assert await repository.get_audio_assets(gen.id) == []


async def test_preview_upload_failure_also_blocks_completion(repository, tmp_path):
    """The preview is a required asset: losing it must fail the run."""
    gen = await _queued_generation(repository)
    service = GenerationService(
        repository,
        MockGenerationProvider(FIXTURE),
        ExplodingStorage(tmp_path / "store", "preview"),
    )

    final = await service.execute(gen.id, worker_id="w")

    assert final is GenerationStatus.FAILED
    fetched = await repository.get_generation(gen.id)
    assert fetched.status != GenerationStatus.COMPLETED.value
    # No asset rows are written unless both uploads succeeded.
    assert await repository.get_audio_assets(gen.id) == []


async def test_failed_run_leaves_no_temporary_files(repository, tmp_path):
    """A failed post-processing run must not leak intermediates."""
    gen = await _queued_generation(repository)
    store = tmp_path / "store"
    service = GenerationService(
        repository, MockGenerationProvider(FIXTURE), ExplodingStorage(store, "master")
    )

    await service.execute(gen.id, worker_id="w")

    leftovers = list(store.rglob("*")) if store.exists() else []
    assert [p for p in leftovers if p.is_file()] == []


# ── 15. retry must not duplicate assets ───────────────────────────────


async def test_rerunning_post_processing_does_not_duplicate_assets(repository, tmp_path):
    """Re-running a generation overwrites its assets in place."""
    gen = await _queued_generation(repository)
    storage = LocalAudioStorage(tmp_path / "store")
    service = GenerationService(repository, MockGenerationProvider(FIXTURE), storage)

    assert await service.execute(gen.id, worker_id="w1") is GenerationStatus.COMPLETED
    first = {a.asset_type: a for a in await repository.get_audio_assets(gen.id)}

    # A retry of the same generation (e.g. a redelivered queue message).
    assert await service.execute(gen.id, worker_id="w2") is GenerationStatus.COMPLETED
    second = {a.asset_type: a for a in await repository.get_audio_assets(gen.id)}

    assert len(second) == 2
    assert set(second) == {AssetType.MASTER.value, AssetType.PREVIEW.value}
    # Same rows updated in place, same deterministic keys.
    assert {a.id for a in first.values()} == {a.id for a in second.values()}
    assert first[AssetType.MASTER.value].storage_key == second[AssetType.MASTER.value].storage_key


async def test_repeated_post_processing_keeps_one_object_per_role(tmp_path):
    """Storage-level idempotency: keys are stable across attempts."""
    storage = LocalAudioStorage(tmp_path / "store")
    gid = uuid.uuid4()

    first = await produce_delivery_assets(gid, FIXTURE, storage)
    second = await produce_delivery_assets(gid, FIXTURE, storage)

    assert first.master.storage_key == second.master.storage_key == master_storage_key(gid)
    assert first.preview.storage_key == second.preview.storage_key == preview_storage_key(gid)
    generation_dir = (tmp_path / "store" / "audio" / str(gid)).resolve()
    assert sorted(p.name for p in generation_dir.iterdir()) == ["master.wav", "preview.mp3"]


# ── recorded digests describe the stored bytes ────────────────────────


async def test_recorded_sha256_matches_stored_objects(tmp_path):
    import hashlib

    storage = LocalAudioStorage(tmp_path / "store")
    gid = uuid.uuid4()

    produced = await produce_delivery_assets(gid, FIXTURE, storage)

    for asset in produced.assets:
        stored = await storage.open(asset.storage_key)
        assert hashlib.sha256(stored).hexdigest() == asset.sha256
        assert len(stored) == asset.file_size


async def test_master_and_preview_describe_the_same_audio(tmp_path):
    storage = LocalAudioStorage(tmp_path / "store")
    produced = await produce_delivery_assets(uuid.uuid4(), FIXTURE, storage)

    assert produced.master.sample_rate == produced.preview.sample_rate == 48000
    assert produced.master.channels == produced.preview.channels == 2
    assert abs(produced.master.duration - produced.preview.duration) < 0.1
    assert produced.master.bit_depth == 24
    assert produced.master.bitrate is None or produced.master.bitrate > 0
    assert produced.preview.bitrate == 320000
    assert produced.master.sha256 != produced.preview.sha256
