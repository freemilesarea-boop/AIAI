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

from luber_database.models.generation import (
    AudioAsset,
    Generation,
    GenerationJob,
    GenerationQA,
    LyricLineQA,
    Project,
)


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
        bpm: int | None = None,
        key_scale: str | None = None,
        time_signature: str | None = None,
        advisories: str | None = None,
        parent_generation_id: UUID | None = None,
        variation_label: str | None = None,
        generation_group_id: UUID | None = None,
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
            bpm=bpm,
            key_scale=key_scale,
            time_signature=time_signature,
            advisories=advisories,
            parent_generation_id=parent_generation_id,
            variation_label=variation_label,
            generation_group_id=generation_group_id,
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

    async def record_request_trace(self, generation_id: UUID, *, trace: str) -> None:
        """Store the provider request trace.

        Written *before* the provider runs, so a failed generation is as
        inspectable as a successful one.
        """
        generation = await self._session.get(Generation, generation_id)
        if generation is None:
            raise LookupError(f"generation not found: {generation_id}")
        generation.request_trace = trace
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

        Descendants are *kept* and re-pointed to ``NULL``. Deleting a take
        must never delete the takes made from it, and a child left
        pointing at a row that no longer exists is a lie the lineage view
        would then have to render. PostgreSQL would do this itself via
        ``ON DELETE SET NULL``; doing it here as well means SQLite (where
        the unit tests run, with FK enforcement off) reaches the same
        state instead of quietly diverging from production.
        """
        generation = await self._session.get(Generation, generation_id)
        if generation is None:
            return False
        for child in (
            (
                await self._session.execute(
                    select(Generation).where(Generation.parent_generation_id == generation_id)
                )
            )
            .scalars()
            .all()
        ):
            child.parent_generation_id = None
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

    # ── product metadata (Phase 12) ────────────────────────────────
    #
    # Presentation state only. Everything a generation *is* — prompt,
    # lyrics, seed, model, parameters — is a historical fact about a run
    # that already happened, and none of it is reachable from here.

    async def update_generation_metadata(
        self,
        generation_id: UUID,
        *,
        title: str | None = None,
        favorite: bool | None = None,
    ) -> Generation:
        """Rename and/or (un)favourite. ``None`` means "leave alone"."""
        generation = await self._session.get(Generation, generation_id)
        if generation is None:
            raise LookupError(f"generation not found: {generation_id}")
        if title is not None:
            generation.title = title
        if favorite is not None:
            generation.favorite = favorite
        await self._session.commit()
        await self._session.refresh(generation)
        return generation

    async def list_generations_in_group(self, group_id: UUID) -> list[Generation]:
        """Siblings from one CREATE, in the order they were created."""
        result = await self._session.execute(
            select(Generation)
            .options(selectinload(Generation.audio_assets))
            .where(Generation.generation_group_id == group_id)
            .order_by(Generation.created_at.asc(), Generation.id.asc())
        )
        return list(result.scalars().all())

    async def bulk_set_project(self, generation_ids: list[UUID], project_id: UUID | None) -> int:
        """File (or unfile) several generations in one transaction.

        Returns how many rows actually existed, so the caller can report
        the truth rather than echoing the request size back.
        """
        if not generation_ids:
            return 0
        rows = (
            (
                await self._session.execute(
                    select(Generation).where(Generation.id.in_(generation_ids))
                )
            )
            .scalars()
            .all()
        )
        for row in rows:
            row.project_id = project_id
        await self._session.commit()
        return len(rows)

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

    # ── Human QA (Phase 9) ────────────────────────────────────────────
    #
    # QA is written by a listener after the fact and is always an
    # upsert: re-reviewing a track corrects the record rather than
    # appending a second, conflicting opinion.

    async def get_generation_qa(self, generation_id: UUID) -> GenerationQA | None:
        result = await self._session.execute(
            select(GenerationQA).where(GenerationQA.generation_id == generation_id)
        )
        return result.scalar_one_or_none()

    async def upsert_generation_qa(
        self,
        generation_id: UUID,
        *,
        overall_rating: int | None = None,
        failure_tags: str | None = None,
        section_verdicts: str | None = None,
        notes: str | None = None,
        reviewer: str | None = None,
    ) -> GenerationQA:
        record = await self.get_generation_qa(generation_id)
        if record is None:
            record = GenerationQA(generation_id=generation_id)
            self._session.add(record)
        record.overall_rating = overall_rating
        record.failure_tags = failure_tags
        record.section_verdicts = section_verdicts
        record.notes = notes
        record.reviewer = reviewer
        record.updated_at = _utcnow()
        await self._session.commit()
        await self._session.refresh(record)
        return record

    async def get_lyric_line_qa(self, generation_id: UUID) -> list[LyricLineQA]:
        result = await self._session.execute(
            select(LyricLineQA)
            .where(LyricLineQA.generation_id == generation_id)
            .order_by(LyricLineQA.line_index.asc())
        )
        return list(result.scalars().all())

    async def replace_lyric_line_qa(
        self,
        generation_id: UUID,
        lines: list[dict[str, object]],
    ) -> list[LyricLineQA]:
        """Replace the whole line-QA set for one generation.

        Whole-set replacement rather than per-line patching: a review is
        a single pass over the track, and a partial write would leave
        lines from an older pass mixed in with the new one.
        """
        existing = {row.line_index: row for row in await self.get_lyric_line_qa(generation_id)}
        seen: set[int] = set()

        for line in lines:
            raw_index = line["line_index"]
            index = int(raw_index) if isinstance(raw_index, (int, str)) else 0
            seen.add(index)
            row = existing.get(index)
            if row is None:
                row = LyricLineQA(generation_id=generation_id, line_index=index)
                self._session.add(row)
            section_label = line.get("section_label")
            row.section_label = str(section_label) if section_label is not None else None
            row.line_text = str(line["line_text"])
            row.verdict = str(line["verdict"])
            note = line.get("note")
            row.note = str(note) if note is not None else None
            row.updated_at = _utcnow()

        for index, row in existing.items():
            if index not in seen:
                await self._session.delete(row)

        await self._session.commit()
        return await self.get_lyric_line_qa(generation_id)

    # ── Projects (Phase 11) ───────────────────────────────────────────

    async def create_project(self, *, name: str, user_id: UUID | None = None) -> Project:
        project = Project(name=name, user_id=user_id)
        self._session.add(project)
        await self._session.commit()
        await self._session.refresh(project)
        return project

    async def get_project(self, project_id: UUID) -> Project | None:
        return await self._session.get(Project, project_id)

    async def list_projects(self) -> list[Project]:
        result = await self._session.execute(select(Project).order_by(Project.created_at.desc()))
        return list(result.scalars().all())

    async def list_projects_with_counts(self) -> list[tuple[Project, int]]:
        """Every project with its song count, in one query.

        Counting per project in a loop is an N+1 that grows with the
        sidebar, and the sidebar is rendered on every visit to Projects.
        An outer join keeps projects with no songs, which are exactly the
        ones a new user has.
        """
        result = await self._session.execute(
            select(Project, func.count(Generation.id))
            .outerjoin(Generation, Generation.project_id == Project.id)
            .group_by(Project.id)
            .order_by(Project.created_at.desc())
        )
        return [(project, int(count)) for project, count in result.all()]

    async def rename_project(self, project_id: UUID, *, name: str) -> Project:
        project = await self.get_project(project_id)
        if project is None:
            raise LookupError(f"project not found: {project_id}")
        project.name = name
        project.updated_at = _utcnow()
        await self._session.commit()
        await self._session.refresh(project)
        return project

    async def delete_project(self, project_id: UUID) -> None:
        """Delete a project. Its generations survive, unfiled.

        The FK is ``ON DELETE SET NULL`` precisely so that removing a
        folder never removes the music inside it.
        """
        project = await self.get_project(project_id)
        if project is None:
            raise LookupError(f"project not found: {project_id}")
        await self._session.delete(project)
        await self._session.commit()

    async def count_project_generations(self, project_id: UUID) -> int:
        result = await self._session.execute(
            select(func.count(Generation.id)).where(Generation.project_id == project_id)
        )
        return int(result.scalar_one())

    async def set_generation_project(
        self, generation_id: UUID, project_id: UUID | None
    ) -> Generation:
        """File a generation under a project, or unfile it with ``None``."""
        generation = await self._session.get(Generation, generation_id)
        if generation is None:
            raise LookupError(f"generation not found: {generation_id}")
        generation.project_id = project_id
        await self._session.commit()
        await self._session.refresh(generation)
        return generation

    async def list_generations_for_project(self, project_id: UUID) -> list[Generation]:
        result = await self._session.execute(
            select(Generation)
            .options(selectinload(Generation.audio_assets))
            .where(Generation.project_id == project_id)
            .order_by(Generation.created_at.desc())
        )
        return list(result.scalars().all())

    async def list_children(self, generation_id: UUID) -> list[Generation]:
        """Generations created from *generation_id* (lineage, Phase 8)."""
        result = await self._session.execute(
            select(Generation)
            .options(selectinload(Generation.audio_assets))
            .where(Generation.parent_generation_id == generation_id)
            .order_by(Generation.created_at.asc())
        )
        return list(result.scalars().all())
