"""What happens when a generation is stopped part-way through.

The queue cancels a running job whenever its worker is told to stop, and
cancellation arrives as ``asyncio.CancelledError`` — a ``BaseException``,
so an ``except Exception`` handler never sees it. That is how a row came
to sit at GENERATING with nothing running behind it: indistinguishable,
to an operator or a user, from a generation that is merely slow.

The other half is the retry that cancellation implies. A job re-run after
its work actually finished must not produce a second, different song in
place of the one already delivered.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from luber_audio_utils import LocalAudioStorage
from luber_generation_client import (
    GenerationRequest,
    GenerationResult,
    GenerationService,
    MockGenerationProvider,
    MusicGenerationProvider,
)
from luber_schemas import ErrorCode, GenerationStatus

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mock_generation.wav"


class _CancellingProvider(MusicGenerationProvider):
    """Stands in for a worker stopped mid-inference."""

    name = "cancelling"

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        raise asyncio.CancelledError()


class _CountingProvider(MusicGenerationProvider):
    """A real result, and a record of how many times it was asked for."""

    name = "counting"

    def __init__(self) -> None:
        self.calls = 0
        self._inner = MockGenerationProvider(FIXTURE)

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls += 1
        return await self._inner.generate(request)


async def _create(repository):
    generation = await repository.create_generation(
        title="TEST SONG",
        prompt="Dreamy Korean indie pop",
        lyrics="",
        vocal_gender="instrumental",
        duration_requested=30,
        status=GenerationStatus.QUEUED.value,
        language="ko",
    )
    await repository.create_job(
        generation.id, queue_name="luber:generation", status=GenerationStatus.QUEUED.value
    )
    return generation


class TestInterruption:
    async def test_cancellation_is_recorded_instead_of_left_mid_flight(self, repository, tmp_path):
        generation = await _create(repository)
        service = GenerationService(repository, _CancellingProvider(), LocalAudioStorage(tmp_path))

        with pytest.raises(asyncio.CancelledError):
            await service.execute(generation.id, worker_id="w1")

        row = await repository.get_generation(generation.id)
        assert row.status == GenerationStatus.FAILED.value
        assert row.error_code == ErrorCode.GENERATION_INTERRUPTED.value
        # The regression: anything still claiming to be running.
        assert row.status not in {
            GenerationStatus.GENERATING.value,
            GenerationStatus.STARTING.value,
            GenerationStatus.POST_PROCESSING.value,
            GenerationStatus.UPLOADING.value,
        }

    async def test_cancellation_still_propagates_so_the_queue_can_retry(self, repository, tmp_path):
        """Swallowing it would turn a retryable job into a lost one."""
        generation = await _create(repository)
        service = GenerationService(repository, _CancellingProvider(), LocalAudioStorage(tmp_path))

        with pytest.raises(asyncio.CancelledError):
            await service.execute(generation.id, worker_id="w1")

    async def test_the_job_row_records_the_interruption_too(self, repository, tmp_path):
        generation = await _create(repository)
        service = GenerationService(repository, _CancellingProvider(), LocalAudioStorage(tmp_path))

        with pytest.raises(asyncio.CancelledError):
            await service.execute(generation.id, worker_id="w1")

        job = await repository.get_latest_job(generation.id)
        assert job.status == GenerationStatus.FAILED.value
        assert job.error_code == ErrorCode.GENERATION_INTERRUPTED.value

    async def test_a_retry_recovers_an_interrupted_generation(self, repository, tmp_path):
        """The interruption must not be terminal when a worker returns."""
        generation = await _create(repository)
        storage = LocalAudioStorage(tmp_path)

        with pytest.raises(asyncio.CancelledError):
            await GenerationService(repository, _CancellingProvider(), storage).execute(
                generation.id, worker_id="w1"
            )

        final = await GenerationService(
            repository, MockGenerationProvider(FIXTURE), storage
        ).execute(generation.id, worker_id="w2")

        assert final is GenerationStatus.COMPLETED
        row = await repository.get_generation(generation.id)
        assert row.status == GenerationStatus.COMPLETED.value
        assert row.error_code is None


class TestDuplicateInvocation:
    async def test_a_completed_generation_is_not_run_again(self, repository, tmp_path):
        """A retry can arrive after the work finished. It must be a no-op.

        Re-running would hand the user different audio in place of the
        song they already have.
        """
        generation = await _create(repository)
        provider = _CountingProvider()
        storage = LocalAudioStorage(tmp_path)

        first = await GenerationService(repository, provider, storage).execute(
            generation.id, worker_id="w1"
        )
        before = await repository.get_audio_assets(generation.id)
        completed_at = (await repository.get_generation(generation.id)).completed_at

        second = await GenerationService(repository, provider, storage).execute(
            generation.id, worker_id="w2"
        )

        assert first is second is GenerationStatus.COMPLETED
        assert provider.calls == 1, "inference ran twice for one generation"
        after = await repository.get_audio_assets(generation.id)
        assert {a.storage_key for a in after} == {a.storage_key for a in before}
        assert (await repository.get_generation(generation.id)).completed_at == completed_at

    async def test_running_the_same_generation_twice_keeps_one_asset_per_role(
        self, repository, tmp_path
    ):
        """The interrupted-then-retried path, which does run twice."""
        generation = await _create(repository)
        storage = LocalAudioStorage(tmp_path)

        with pytest.raises(asyncio.CancelledError):
            await GenerationService(repository, _CancellingProvider(), storage).execute(
                generation.id, worker_id="w1"
            )
        await GenerationService(repository, MockGenerationProvider(FIXTURE), storage).execute(
            generation.id, worker_id="w2"
        )

        roles = [asset.asset_type for asset in await repository.get_audio_assets(generation.id)]
        assert len(roles) == len(set(roles)), f"duplicate asset roles: {roles}"
