import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from luber_schemas import AssetType, GenerationStatus


async def _create(repository, **overrides):
    defaults = dict(
        title="TEST SONG",
        prompt="Dreamy Korean indie pop",
        lyrics="[Verse]\n테스트 가사",
        vocal_gender="female",
        duration_requested=30,
        status=GenerationStatus.QUEUED.value,
        language="ko",
    )
    defaults.update(overrides)
    return await repository.create_generation(**defaults)


async def test_create_and_get_generation(repository):
    created = await _create(repository, seed=1234)

    fetched = await repository.get_generation(created.id)
    assert fetched is not None
    assert fetched.title == "TEST SONG"
    assert fetched.lyrics.startswith("[Verse]")  # section tags preserved
    assert fetched.vocal_gender == "female"
    assert fetched.seed == 1234
    assert fetched.status == GenerationStatus.QUEUED.value
    assert fetched.audio_assets == []


async def test_get_missing_generation_returns_none(repository):
    assert await repository.get_generation(uuid.uuid4()) is None


async def test_list_generations_pagination(repository):
    for i in range(5):
        await _create(repository, title=f"Song {i}")
    items, total = await repository.list_generations(limit=2, offset=0)
    assert total == 5
    assert len(items) == 2


async def test_duplicate_idempotency_key_rejected_by_db_constraint(repository):
    """Race protection lives in the DB unique index, not app-level SELECT."""
    await _create(repository, idempotency_key="idem-race-1")
    with pytest.raises(IntegrityError):
        await _create(repository, idempotency_key="idem-race-1")
    # The repository stays usable after the constraint violation.
    survivor = await repository.get_by_idempotency_key("idem-race-1")
    assert survivor is not None


async def test_null_idempotency_keys_do_not_collide(repository):
    await _create(repository)
    await _create(repository)  # second NULL key must not violate uniqueness
    _, total = await repository.list_generations()
    assert total == 2


async def test_status_lifecycle_updates(repository):
    gen = await _create(repository)
    await repository.mark_started(gen.id, status=GenerationStatus.STARTING.value)
    await repository.update_status(gen.id, GenerationStatus.GENERATING.value)
    await repository.mark_completed(
        gen.id,
        status=GenerationStatus.COMPLETED.value,
        duration_actual=2.0,
        provider="mock",
        model_name="mock-generation-provider",
        model_version="phase1",
        seed=99,
    )
    fetched = await repository.get_generation(gen.id)
    assert fetched.status == GenerationStatus.COMPLETED.value
    assert fetched.started_at is not None
    assert fetched.completed_at is not None
    assert fetched.duration_actual == 2.0
    assert fetched.seed == 99


async def test_mark_failed_stores_error(repository):
    gen = await _create(repository)
    await repository.mark_failed(
        gen.id,
        status=GenerationStatus.FAILED.value,
        error_code="UNKNOWN_GENERATION_ERROR",
        error_message="boom",
    )
    fetched = await repository.get_generation(gen.id)
    assert fetched.status == GenerationStatus.FAILED.value
    assert fetched.error_code == "UNKNOWN_GENERATION_ERROR"
    assert fetched.error_message == "boom"


async def test_jobs_and_assets_roundtrip(repository):
    gen = await _create(repository)
    job = await repository.create_job(
        gen.id, queue_name="luber:generation", status=GenerationStatus.QUEUED.value
    )
    await repository.mark_job_started(
        job.id, status=GenerationStatus.STARTING.value, worker_id="w-1"
    )
    await repository.mark_job_finished(job.id, status=GenerationStatus.COMPLETED.value)

    latest = await repository.get_latest_job(gen.id)
    assert latest.attempt == 1
    assert latest.worker_id == "w-1"
    assert latest.finished_at is not None

    asset = await repository.create_audio_asset(
        gen.id,
        asset_type=AssetType.MASTER.value,
        format="wav",
        mime_type="audio/wav",
        file_extension="wav",
        sample_rate=48000,
        bit_depth=24,
        bitrate=None,
        channels=2,
        duration=2.0,
        storage_key=f"audio/{gen.id}/master.wav",
        sha256="a" * 64,
        file_size=384044,
    )
    assets = await repository.get_audio_assets(gen.id)
    assert [a.id for a in assets] == [asset.id]
    assert assets[0].sha256 == "a" * 64


async def test_delete_generation_removes_children(repository):
    gen = await _create(repository)
    await repository.create_job(
        gen.id, queue_name="luber:generation", status=GenerationStatus.QUEUED.value
    )
    await repository.create_audio_asset(
        gen.id,
        asset_type=AssetType.MASTER.value,
        format="wav",
        mime_type="audio/wav",
        file_extension="wav",
        sample_rate=48000,
        bit_depth=24,
        bitrate=None,
        channels=2,
        duration=2.0,
        storage_key=f"audio/{gen.id}/master.wav",
        sha256="b" * 64,
        file_size=1,
    )
    assert await repository.delete_generation(gen.id) is True
    assert await repository.get_generation(gen.id) is None
    assert await repository.get_audio_assets(gen.id) == []
    assert await repository.get_latest_job(gen.id) is None
    assert await repository.delete_generation(gen.id) is False
