import uuid
from pathlib import Path

import pytest

from luber_audio_utils import AudioStorageError, LocalAudioStorage

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mock_generation.wav"


async def test_store_and_resolve_master_wav(tmp_path):
    storage = LocalAudioStorage(tmp_path)
    generation_id = uuid.uuid4()

    key = await storage.store_master_wav(generation_id, FIXTURE)

    assert key == f"audio/{generation_id}/master.wav"
    stored = storage.resolve_path(key)
    assert stored.is_file()
    assert stored.stat().st_size == FIXTURE.stat().st_size


async def test_store_missing_source_raises(tmp_path):
    storage = LocalAudioStorage(tmp_path)
    with pytest.raises(AudioStorageError, match="missing"):
        await storage.store_master_wav(uuid.uuid4(), tmp_path / "ghost.wav")


async def test_delete_generation_audio_is_idempotent(tmp_path):
    storage = LocalAudioStorage(tmp_path)
    generation_id = uuid.uuid4()
    key = await storage.store_master_wav(generation_id, FIXTURE)
    assert storage.resolve_path(key).is_file()

    await storage.delete_generation_audio(generation_id)
    assert not storage.resolve_path(key).exists()

    # Second delete of the same generation is a no-op, not an error.
    await storage.delete_generation_audio(generation_id)
