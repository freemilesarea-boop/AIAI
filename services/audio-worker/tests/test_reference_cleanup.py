"""Reference cleanup: what it deletes, and everything it must not.

The dangerous failure here is asymmetric. Leaving an abandoned upload
behind costs disk; deleting one that is still in use destroys audio a
song depends on and provenance nobody can reconstruct. So most of these
tests assert survival rather than deletion.
"""

from __future__ import annotations

import uuid
import wave
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import update

from luber_audio_utils import AudioStorageError, LocalAudioStorage
from luber_audio_worker.reference_cleanup import cleanup_abandoned_references
from luber_database import GenerationRepository
from luber_database.models.generation import ReferenceAudio
from luber_schemas import GenerationStatus, reference_storage_key

GRACE_HOURS = 24
NOW = datetime(2026, 8, 15, 12, 0, tzinfo=UTC)


def wav(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(48_000)
        handle.writeframes(b"\x00\x01" * 4800)
    return path


async def make_reference(
    repository: GenerationRepository,
    storage: LocalAudioStorage,
    tmp_path: Path,
    *,
    age_hours: float,
    name: str = "ref.wav",
) -> ReferenceAudio:
    """A stored reference with a controlled creation time."""
    reference_id = uuid.uuid4()
    key = reference_storage_key(reference_id)
    await storage.put(key, wav(tmp_path / f"{reference_id}.wav"))
    row = await repository.create_reference_audio(
        reference_id=reference_id,
        storage_key=key,
        sha256=uuid.uuid4().hex * 2,
        source_sha256=uuid.uuid4().hex * 2,
        source_format="wav",
        duration_seconds=1.0,
        sample_rate=48_000,
        channels=2,
        file_size=64,
        display_name=name,
    )
    # created_at is server-generated, so age is applied afterwards.
    await repository._session.execute(
        update(ReferenceAudio)
        .where(ReferenceAudio.id == reference_id)
        .values(created_at=NOW - timedelta(hours=age_hours))
    )
    await repository._session.commit()
    return row


async def attach(
    repository: GenerationRepository, reference: ReferenceAudio, status: GenerationStatus
):
    return await repository.create_generation(
        title="Uses a reference",
        prompt="p",
        lyrics="",
        vocal_gender="instrumental",
        duration_requested=30,
        seed=None,
        language="en",
        instrumental=True,
        status=status.value,
        idempotency_key=None,
        reference_audio_id=reference.id,
    )


async def run(repository, storage, **kwargs):
    return await cleanup_abandoned_references(
        repository, storage, grace_hours=GRACE_HOURS, batch_size=100, now=NOW, **kwargs
    )


@pytest.fixture
def storage(tmp_path: Path) -> LocalAudioStorage:
    return LocalAudioStorage(tmp_path / "store")


class TestAgeEligibility:
    async def test_a_fresh_unused_reference_survives(self, repository, storage, tmp_path):
        """Uploaded an hour ago: the user may still be mid-flow."""
        reference = await make_reference(repository, storage, tmp_path, age_hours=1)
        report = await run(repository, storage)
        assert report.deleted == 0
        assert await repository.get_reference_audio(reference.id) is not None
        assert await storage.exists(reference.storage_key)

    async def test_an_old_unused_reference_is_deleted(self, repository, storage, tmp_path):
        reference = await make_reference(repository, storage, tmp_path, age_hours=48)
        report = await run(repository, storage)
        assert report.deleted == 1
        assert await repository.get_reference_audio(reference.id) is None
        assert not await storage.exists(reference.storage_key)

    async def test_the_cutoff_boundary_is_deterministic(self, repository, storage, tmp_path):
        """Exactly at the grace period is not yet past it."""
        exact = await make_reference(repository, storage, tmp_path, age_hours=GRACE_HOURS)
        just_past = await make_reference(
            repository, storage, tmp_path, age_hours=GRACE_HOURS + 0.001
        )
        await run(repository, storage)
        assert await repository.get_reference_audio(exact.id) is not None
        assert await repository.get_reference_audio(just_past.id) is None


class TestUsedReferencesSurvive:
    @pytest.mark.parametrize(
        "status",
        [
            GenerationStatus.QUEUED,
            GenerationStatus.GENERATING,
            GenerationStatus.POST_PROCESSING,
            GenerationStatus.COMPLETED,
            GenerationStatus.FAILED,
        ],
    )
    async def test_any_generation_protects_its_reference(
        self, repository, storage, tmp_path, status
    ):
        """Including FAILED — that is exactly when the input matters."""
        reference = await make_reference(repository, storage, tmp_path, age_hours=999)
        await attach(repository, reference, status)

        report = await run(repository, storage)

        assert report.deleted == 0
        assert await repository.get_reference_audio(reference.id) is not None
        assert await storage.exists(reference.storage_key)

    async def test_a_reference_shared_by_several_generations_survives(
        self, repository, storage, tmp_path
    ):
        """One upload, two songs — the result_count path does this."""
        reference = await make_reference(repository, storage, tmp_path, age_hours=999)
        await attach(repository, reference, GenerationStatus.COMPLETED)
        await attach(repository, reference, GenerationStatus.COMPLETED)

        assert (await run(repository, storage)).deleted == 0
        assert await repository.get_reference_audio(reference.id) is not None

    async def test_only_the_unused_one_is_removed(self, repository, storage, tmp_path):
        used = await make_reference(repository, storage, tmp_path, age_hours=999, name="used.wav")
        unused = await make_reference(
            repository, storage, tmp_path, age_hours=999, name="unused.wav"
        )
        await attach(repository, used, GenerationStatus.COMPLETED)

        report = await run(repository, storage)

        assert report.deleted == 1
        assert await repository.get_reference_audio(used.id) is not None
        assert await repository.get_reference_audio(unused.id) is None


class TestDryRun:
    async def test_dry_run_deletes_nothing(self, repository, storage, tmp_path):
        reference = await make_reference(repository, storage, tmp_path, age_hours=999)
        report = await run(repository, storage, dry_run=True)
        assert report.dry_run is True
        assert report.eligible == 1
        assert report.deleted == 0
        assert await repository.get_reference_audio(reference.id) is not None
        assert await storage.exists(reference.storage_key)

    async def test_dry_run_names_its_candidates(self, repository, storage, tmp_path):
        reference = await make_reference(repository, storage, tmp_path, age_hours=999)
        assert (await run(repository, storage, dry_run=True)).candidates == [str(reference.id)]


class TestConcurrencySafety:
    async def test_a_reference_attached_after_the_scan_is_not_deleted(
        self, repository, storage, tmp_path
    ):
        """The race this design exists to survive.

        A generation commits between the candidate scan and the delete.
        The condition lives inside the DELETE, so the statement matches
        zero rows rather than destroying audio a live generation needs.
        """
        reference = await make_reference(repository, storage, tmp_path, age_hours=999)
        candidates = await repository.find_abandoned_references(
            cutoff=NOW - timedelta(hours=GRACE_HOURS), limit=10
        )
        assert [c.id for c in candidates] == [reference.id]

        # The window: a generation attaches now.
        await attach(repository, reference, GenerationStatus.QUEUED)

        deleted = await repository.delete_reference_audio_if_unused(reference.id)

        assert deleted is False
        assert await repository.get_reference_audio(reference.id) is not None

    async def test_the_report_counts_the_loss_of_the_race(self, repository, storage, tmp_path):
        """A reference that becomes used mid-run is skipped, not failed."""
        reference = await make_reference(repository, storage, tmp_path, age_hours=999)

        original = repository.find_abandoned_references

        async def scan_then_attach(**kwargs):
            found = await original(**kwargs)
            await attach(repository, reference, GenerationStatus.QUEUED)
            return found

        repository.find_abandoned_references = scan_then_attach  # type: ignore[method-assign]
        try:
            report = await run(repository, storage)
        finally:
            repository.find_abandoned_references = original  # type: ignore[method-assign]

        assert report.skipped_referenced == 1
        assert report.deleted == 0
        assert await storage.exists(reference.storage_key)


class TestStorageFailure:
    async def test_a_storage_failure_is_reported_not_swallowed(self, repository, storage, tmp_path):
        reference = await make_reference(repository, storage, tmp_path, age_hours=999)

        async def explode(key: str) -> None:
            raise AudioStorageError(f"cannot delete {key}")

        storage.delete = explode  # type: ignore[method-assign]
        report = await run(repository, storage)

        # The row went first, so it is gone; the object leaked, and the
        # report says so rather than claiming a clean run.
        assert report.deleted == 1
        assert report.storage_failures == 1
        assert await repository.get_reference_audio(reference.id) is None

    async def test_one_failure_does_not_stop_the_others(self, repository, storage, tmp_path):
        first = await make_reference(repository, storage, tmp_path, age_hours=999, name="a.wav")
        second = await make_reference(repository, storage, tmp_path, age_hours=998, name="b.wav")
        original = storage.delete
        failed_key = first.storage_key

        async def flaky(key: str) -> None:
            if key == failed_key:
                raise AudioStorageError("nope")
            await original(key)

        storage.delete = flaky  # type: ignore[method-assign]
        report = await run(repository, storage)

        assert report.deleted == 2
        assert report.storage_failures == 1
        assert not await storage.exists(second.storage_key)

    async def test_a_missing_object_is_not_an_error(self, repository, storage, tmp_path):
        """Retrying a partially-completed run must be safe."""
        reference = await make_reference(repository, storage, tmp_path, age_hours=999)
        await storage.delete(reference.storage_key)

        report = await run(repository, storage)

        assert report.deleted == 1
        assert report.storage_failures == 0

    async def test_a_second_run_finds_nothing_left_to_do(self, repository, storage, tmp_path):
        await make_reference(repository, storage, tmp_path, age_hours=999)
        assert (await run(repository, storage)).deleted == 1
        second = await run(repository, storage)
        assert second.scanned == 0
        assert second.deleted == 0


class TestBounds:
    async def test_the_batch_limit_is_respected(self, repository, storage, tmp_path):
        for index in range(5):
            await make_reference(
                repository, storage, tmp_path, age_hours=999 - index, name=f"r{index}.wav"
            )
        report = await cleanup_abandoned_references(
            repository, storage, grace_hours=GRACE_HOURS, batch_size=2, now=NOW
        )
        assert report.scanned == 2
        assert report.deleted == 2

    async def test_the_backlog_drains_over_repeated_runs(self, repository, storage, tmp_path):
        for index in range(5):
            await make_reference(
                repository, storage, tmp_path, age_hours=999 - index, name=f"r{index}.wav"
            )
        total = 0
        for _ in range(3):
            total += (
                await cleanup_abandoned_references(
                    repository, storage, grace_hours=GRACE_HOURS, batch_size=2, now=NOW
                )
            ).deleted
        assert total == 5


class TestObservability:
    async def test_the_report_covers_every_required_counter(self, repository, storage, tmp_path):
        await make_reference(repository, storage, tmp_path, age_hours=999)
        report = await run(repository, storage)
        assert set(report.as_dict()) == {
            "scanned",
            "eligible",
            "deleted",
            "skipped_referenced",
            "storage_failures",
            "database_failures",
            "dry_run",
        }

    async def test_the_report_carries_no_filesystem_path(self, repository, storage, tmp_path):
        await make_reference(repository, storage, tmp_path, age_hours=999)
        report = await run(repository, storage, dry_run=True)
        serialised = repr(report.as_dict()) + repr(report.candidates)
        assert "/" not in serialised
        assert str(tmp_path) not in serialised
