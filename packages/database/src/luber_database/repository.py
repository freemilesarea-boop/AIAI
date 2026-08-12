"""GenerationRepository — the only place that touches generation ORM models.

API routes and services never manipulate ORM entities directly. Each
mutating method commits, so partially-applied lifecycles are never left
in an open transaction when a worker crashes mid-generation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from luber_database.models.generation import AudioAsset, Generation, GenerationJob


def _utcnow() -> datetime:
    return datetime.now(UTC)


class GenerationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ── generations ────────────────────────────────────────────────

    async def create_generation(
        self,
        *,
        title: str,
        prompt: str,
        lyrics: str,
        vocal_gender: str,
        duration_requested: int,
        status: str,
        seed: int | None = None,
        language: str | None = None,
        instrumental: bool = False,
        user_id: UUID | None = None,
        idempotency_key: str | None = None,
    ) -> Generation:
        """Insert a generation row.

        Raises ``sqlalchemy.exc.IntegrityError`` when ``idempotency_key``
        collides with an existing row — the DB unique index is the
        race-condition guard, not application-level SELECT-then-INSERT.
        """
        generation = Generation(
            title=title,
            prompt=prompt,
            lyrics=lyrics,
            vocal_gender=vocal_gender,
            duration_requested=duration_requested,
            seed=seed,
            language=language,
            instrumental=instrumental,
            status=status,
            user_id=user_id,
            idempotency_key=idempotency_key,
        )
        self._session.add(generation)
        try:
            await self._session.commit()
        except Exception:
            await self._session.rollback()
            raise
        await self._session.refresh(generation)
        return generation

    async def get_generation(self, generation_id: UUID) -> Generation | None:
        result = await self._session.execute(
            select(Generation)
            .options(selectinload(Generation.audio_assets))
            .where(Generation.id == generation_id)
        )
        return result.scalar_one_or_none()

    async def get_by_idempotency_key(self, idempotency_key: str) -> Generation | None:
        result = await self._session.execute(
            select(Generation)
            .options(selectinload(Generation.audio_assets))
            .where(Generation.idempotency_key == idempotency_key)
        )
        return result.scalar_one_or_none()

    async def list_generations(
        self, *, limit: int = 20, offset: int = 0
    ) -> tuple[list[Generation], int]:
        total = (await self._session.execute(select(func.count(Generation.id)))).scalar_one()
        result = await self._session.execute(
            select(Generation)
            .options(selectinload(Generation.audio_assets))
            .order_by(Generation.created_at.desc(), Generation.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all()), total

    async def update_status(self, generation_id: UUID, status: str) -> None:
        generation = await self._session.get(Generation, generation_id)
        if generation is None:
            raise LookupError(f"generation not found: {generation_id}")
        generation.status = status
        await self._session.commit()

    async def mark_started(self, generation_id: UUID, *, status: str) -> None:
        generation = await self._session.get(Generation, generation_id)
        if generation is None:
            raise LookupError(f"generation not found: {generation_id}")
        generation.status = status
        generation.started_at = _utcnow()
        await self._session.commit()

    async def mark_completed(
        self,
        generation_id: UUID,
        *,
        status: str,
        duration_actual: float,
        provider: str,
        model_name: str,
        model_version: str,
        seed: int | None,
    ) -> None:
        generation = await self._session.get(Generation, generation_id)
        if generation is None:
            raise LookupError(f"generation not found: {generation_id}")
        generation.status = status
        generation.duration_actual = duration_actual
        generation.provider = provider
        generation.model_name = model_name
        generation.model_version = model_version
        if seed is not None:
            generation.seed = seed
        generation.completed_at = _utcnow()
        await self._session.commit()

    async def mark_failed(
        self,
        generation_id: UUID,
        *,
        status: str,
        error_code: str,
        error_message: str,
    ) -> None:
        generation = await self._session.get(Generation, generation_id)
        if generation is None:
            raise LookupError(f"generation not found: {generation_id}")
        generation.status = status
        generation.error_code = error_code
        generation.error_message = error_message
        generation.completed_at = _utcnow()
        await self._session.commit()

    async def delete_generation(self, generation_id: UUID) -> bool:
        """Hard-delete a generation with its jobs and asset rows.

        Deletes children explicitly (portable across SQLite/PostgreSQL
        regardless of FK enforcement). Returns False when the row does
        not exist.
        """
        generation = await self._session.get(Generation, generation_id)
        if generation is None:
            return False
        for job in (
            (
                await self._session.execute(
                    select(GenerationJob).where(GenerationJob.generation_id == generation_id)
                )
            )
            .scalars()
            .all()
        ):
            await self._session.delete(job)
        for asset in (
            (
                await self._session.execute(
                    select(AudioAsset).where(AudioAsset.generation_id == generation_id)
                )
            )
            .scalars()
            .all()
        ):
            await self._session.delete(asset)
        await self._session.delete(generation)
        await self._session.commit()
        return True

    # ── jobs ───────────────────────────────────────────────────────

    async def create_job(
        self,
        generation_id: UUID,
        *,
        queue_name: str,
        status: str,
        max_attempts: int = 1,
    ) -> GenerationJob:
        job = GenerationJob(
            generation_id=generation_id,
            queue_name=queue_name,
            status=status,
            max_attempts=max_attempts,
        )
        self._session.add(job)
        await self._session.commit()
        await self._session.refresh(job)
        return job

    async def get_latest_job(self, generation_id: UUID) -> GenerationJob | None:
        result = await self._session.execute(
            select(GenerationJob)
            .where(GenerationJob.generation_id == generation_id)
            .order_by(GenerationJob.enqueued_at.desc(), GenerationJob.id.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def mark_job_started(self, job_id: UUID, *, status: str, worker_id: str | None) -> None:
        job = await self._session.get(GenerationJob, job_id)
        if job is None:
            raise LookupError(f"generation job not found: {job_id}")
        job.status = status
        job.attempt += 1
        job.worker_id = worker_id
        job.started_at = _utcnow()
        await self._session.commit()

    async def mark_job_finished(
        self,
        job_id: UUID,
        *,
        status: str,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        job = await self._session.get(GenerationJob, job_id)
        if job is None:
            raise LookupError(f"generation job not found: {job_id}")
        job.status = status
        job.error_code = error_code
        job.error_message = error_message
        job.finished_at = _utcnow()
        await self._session.commit()

    # ── audio assets ───────────────────────────────────────────────

    async def create_audio_asset(
        self,
        generation_id: UUID,
        *,
        asset_type: str,
        format: str,
        mime_type: str,
        file_extension: str,
        sample_rate: int,
        bit_depth: int | None,
        bitrate: int | None,
        channels: int,
        duration: float,
        storage_key: str,
        sha256: str,
        file_size: int,
    ) -> AudioAsset:
        """Record an audio asset, replacing any existing one of that type.

        A generation has at most one asset per role, so a retry updates
        the existing row in place rather than inserting a duplicate.
        This mirrors the storage layer, where the deterministic key means
        a retry overwrites the same object.
        """
        existing = await self._session.execute(
            select(AudioAsset).where(
                AudioAsset.generation_id == generation_id,
                AudioAsset.asset_type == asset_type,
            )
        )
        asset = existing.scalar_one_or_none()
        if asset is None:
            asset = AudioAsset(generation_id=generation_id, asset_type=asset_type)
            self._session.add(asset)

        asset.format = format
        asset.mime_type = mime_type
        asset.file_extension = file_extension
        asset.sample_rate = sample_rate
        asset.bit_depth = bit_depth
        asset.bitrate = bitrate
        asset.channels = channels
        asset.duration = duration
        asset.storage_key = storage_key
        asset.sha256 = sha256
        asset.file_size = file_size

        await self._session.commit()
        await self._session.refresh(asset)
        return asset

    async def get_audio_assets(self, generation_id: UUID) -> list[AudioAsset]:
        result = await self._session.execute(
            select(AudioAsset)
            .where(AudioAsset.generation_id == generation_id)
            .order_by(AudioAsset.created_at.asc())
        )
        return list(result.scalars().all())
