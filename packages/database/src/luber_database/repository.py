"""GenerationRepository — the only place that touches generation ORM models.

API routes and services never manipulate ORM entities directly. Each
mutating method commits, so partially-applied lifecycles are never left
in an open transaction when a worker crashes mid-generation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, TypeVar, cast
from uuid import UUID

from sqlalchemy import CursorResult, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from luber_database.errors import GenerationHasDescendantsError
from luber_database.models.generation import (
    AudioAsset,
    Generation,
    GenerationJob,
    GenerationQA,
    LyricLineQA,
    Project,
    ReferenceAudio,
)

#: Any model with an ``id`` and an owner column.
_Row = TypeVar("_Row")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class GenerationRepository:
    """Generation-domain access, scoped to one owner.

    The owner lives on the repository rather than being passed to each
    method. Seventeen methods each taking a ``user_id`` is seventeen
    chances for a route to forget one, and a forgotten filter is a
    silent cross-user read — the failure mode that looks like working
    software. Here, scoping is what the object *is*.

    ``owner=None`` means unscoped and is reserved for trusted callers
    with no session: the ARQ worker, which operates on a generation the
    authenticated API already established ownership for, and maintenance
    tooling. The API's request dependency always supplies an owner from
    the session, and a test pins that it cannot do otherwise.
    """

    def __init__(self, session: AsyncSession, owner: UUID | None = None) -> None:
        self._session = session
        self._owner = owner

    @property
    def owner(self) -> UUID | None:
        return self._owner

    def _owned(self, statement: Any, column: Any) -> Any:
        """Add the ownership predicate, when this repository has one."""
        if self._owner is None:
            return statement
        return statement.where(column == self._owner)

    async def _fetch_owned(self, model: type[_Row], row_id: UUID, column: Any) -> _Row | None:
        """Load a row only if this repository is allowed to see it.

        Replaces ``session.get`` for the three owned models. An unscoped
        repository behaves exactly as ``get`` did; a scoped one returns
        None for somebody else's row, so a mutation path cannot act on
        it even by accident.
        """
        # ``id`` is declared on Base, not on the TypeVar, so the lookup
        # column is read dynamically while the return type stays exact.
        identity = cast("Any", model).id
        result = await self._session.execute(
            self._owned(select(model).where(identity == row_id), column)
        )
        return cast("_Row | None", result.scalar_one_or_none())

    def _require_owner(self) -> UUID:
        """The owner a create must attribute the new row to.

        An unscoped repository creating product data is a bug: it would
        have to invent an owner, and the only value available would be
        the legacy anchor — which is how new data silently becomes
        historical data.
        """
        if self._owner is None:
            raise ValueError("this repository is unscoped; product rows need an explicit owner")
        return self._owner

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
        edit_kind: str | None = None,
        edit_start_seconds: float | None = None,
        edit_end_seconds: float | None = None,
        source_adherence: float | None = None,
        reference_audio_id: UUID | None = None,
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
            edit_kind=edit_kind,
            edit_start_seconds=edit_start_seconds,
            edit_end_seconds=edit_end_seconds,
            source_adherence=source_adherence,
            reference_audio_id=reference_audio_id,
            status=status,
            user_id=user_id if user_id is not None else self._require_owner(),
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
            self._owned(
                select(Generation)
                .options(selectinload(Generation.audio_assets))
                .where(Generation.id == generation_id),
                Generation.user_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_by_idempotency_key(self, idempotency_key: str) -> Generation | None:
        result = await self._session.execute(
            self._owned(
                select(Generation)
                .options(selectinload(Generation.audio_assets))
                .where(Generation.idempotency_key == idempotency_key),
                Generation.user_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_generations(
        self, *, limit: int = 20, offset: int = 0
    ) -> tuple[list[Generation], int]:
        # The total is scoped too. A correct-looking page with a global
        # count still tells the caller how much other people have.
        total = (
            await self._session.execute(
                self._owned(select(func.count(Generation.id)), Generation.user_id)
            )
        ).scalar_one()
        result = await self._session.execute(
            self._owned(
                select(Generation)
                .options(selectinload(Generation.audio_assets))
                .order_by(Generation.created_at.desc(), Generation.id.desc())
                .limit(limit)
                .offset(offset),
                Generation.user_id,
            )
        )
        return list(result.scalars().all()), total

    async def update_status(self, generation_id: UUID, status: str) -> None:
        generation = await self._fetch_owned(Generation, generation_id, Generation.user_id)
        if generation is None:
            raise LookupError(f"generation not found: {generation_id}")
        generation.status = status
        await self._session.commit()

    async def mark_started(self, generation_id: UUID, *, status: str) -> None:
        """Begin (or begin again) a run, clearing any earlier failure.

        A retry of an interrupted generation passes through here. Leaving
        the previous attempt's ``error_code`` in place would let a song
        that finished perfectly well arrive at the client still carrying
        the reason its *first* attempt stopped.
        """
        generation = await self._fetch_owned(Generation, generation_id, Generation.user_id)
        if generation is None:
            raise LookupError(f"generation not found: {generation_id}")
        generation.status = status
        generation.started_at = _utcnow()
        generation.error_code = None
        generation.error_message = None
        await self._session.commit()

    async def record_request_trace(self, generation_id: UUID, *, trace: str) -> None:
        """Store the provider request trace.

        Written *before* the provider runs, so a failed generation is as
        inspectable as a successful one.
        """
        generation = await self._fetch_owned(Generation, generation_id, Generation.user_id)
        if generation is None:
            raise LookupError(f"generation not found: {generation_id}")
        generation.request_trace = trace
        await self._session.commit()

    async def record_finishing_trace(self, generation_id: UUID, *, trace: str) -> None:
        """Store what the finishing engine decided.

        Written whether the engine acted, declined, or failed, so that a
        generation with no finished master can still say which of those
        three happened.
        """
        generation = await self._fetch_owned(Generation, generation_id, Generation.user_id)
        if generation is None:
            raise LookupError(f"generation not found: {generation_id}")
        generation.finishing_trace = trace
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
        generation = await self._fetch_owned(Generation, generation_id, Generation.user_id)
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
        generation = await self._fetch_owned(Generation, generation_id, Generation.user_id)
        if generation is None:
            raise LookupError(f"generation not found: {generation_id}")
        generation.status = status
        generation.error_code = error_code
        generation.error_message = error_message
        generation.completed_at = _utcnow()
        await self._session.commit()

    async def get_ancestry(self, generation_id: UUID, *, max_depth: int = 64) -> list[Generation]:
        """Ancestors from the immediate parent up to the root.

        One query per level rather than one per node, and the depth
        ceiling plus the visited set mean a self-parent or a cycle
        terminates instead of hanging a request. A parent that no longer
        exists simply ends the walk — legacy rows are not assumed to be
        well-formed.
        """
        chain: list[Generation] = []
        seen: set[UUID] = {generation_id}
        current = await self._fetch_owned(Generation, generation_id, Generation.user_id)
        while current is not None and current.parent_generation_id is not None:
            if current.parent_generation_id in seen or len(chain) >= max_depth:
                break
            seen.add(current.parent_generation_id)
            parent = await self._fetch_owned(
                Generation, current.parent_generation_id, Generation.user_id
            )
            if parent is None:
                break
            chain.append(parent)
            current = parent
        return chain

    async def get_descendants(
        self, generation_id: UUID, *, max_depth: int = 16, max_nodes: int = 200
    ) -> list[Generation]:
        """Everything derived from this generation, breadth-first.

        Bounded twice — by depth and by total nodes — so one pathological
        lineage cannot turn a page load into an unbounded scan. One query
        per level, not per node.
        """
        collected: list[Generation] = []
        seen: set[UUID] = {generation_id}
        frontier = [generation_id]
        depth = 0
        while frontier and depth < max_depth and len(collected) < max_nodes:
            result = await self._session.execute(
                self._owned(
                    select(Generation)
                    .where(Generation.parent_generation_id.in_(frontier))
                    .order_by(Generation.created_at),
                    Generation.user_id,
                )
            )
            children = [row for row in result.scalars().all() if row.id not in seen]
            if not children:
                break
            for child in children:
                if len(collected) >= max_nodes:
                    break
                seen.add(child.id)
                collected.append(child)
            frontier = [child.id for child in children]
            depth += 1
        return collected

    async def count_descendants(self, generation_id: UUID, *, max_depth: int = 64) -> int:
        """How many generations descend from this one, directly or not.

        Walks the parent links breadth-first with a visited set and a
        depth ceiling. Legacy data is not trusted to be a tree: a row that
        is its own parent, or a cycle introduced by some future path,
        would otherwise loop forever inside a delete request.
        """
        seen: set[UUID] = set()
        frontier = [generation_id]
        depth = 0
        while frontier and depth < max_depth:
            result = await self._session.execute(
                select(Generation.id).where(Generation.parent_generation_id.in_(frontier))
            )
            children = [row for row in result.scalars().all() if row not in seen]
            # A self-parent would put the row back in its own frontier;
            # excluding the origin keeps it from counting as its own child.
            children = [child for child in children if child != generation_id]
            if not children:
                break
            seen.update(children)
            frontier = children
            depth += 1
        return len(seen)

    async def delete_generation(self, generation_id: UUID) -> bool:
        """Hard-delete a generation with its jobs and asset rows.

        Deletes jobs and asset rows explicitly (portable across
        SQLite/PostgreSQL regardless of FK enforcement). Returns False
        when the row does not exist.

        **Refuses when anything was derived from this generation.** The
        previous behaviour re-pointed each child's ``parent_generation_id``
        to NULL, on the reasoning that deleting a take must not delete the
        takes made from it. That half is right; the result was not. A
        child keeps its ``edit_kind``, so nulling the link leaves a row
        that claims to be an extension of nothing — not a missing edge but
        a contradiction, which version history would draw as a root
        labelled "Extended".

        There is no third option that preserves provenance: cascading
        destroys the children, re-parenting invents a history that never
        happened, and nulling corrupts the record. Refusing is the only
        one that keeps the truth, and it is recoverable — the user deletes
        the derived versions first.

        Raises :class:`GenerationHasDescendantsError` rather than
        returning a flag, so a caller cannot mistake the refusal for
        "already gone".
        """
        generation = await self._fetch_owned(Generation, generation_id, Generation.user_id)
        if generation is None:
            return False

        # Database-authoritative, and checked before a single asset row
        # is touched so a refusal leaves nothing partially deleted.
        descendants = await self.count_descendants(generation_id)
        if descendants:
            raise GenerationHasDescendantsError(generation_id, descendants)

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
        generation = await self._fetch_owned(Generation, generation_id, Generation.user_id)
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
            self._owned(
                select(Generation)
                .options(selectinload(Generation.audio_assets))
                .where(Generation.generation_group_id == group_id)
                .order_by(Generation.created_at.asc(), Generation.id.asc()),
                Generation.user_id,
            )
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
                    # Scoped, so a bulk request naming somebody else's id
                    # simply does not see it. Nothing reports which ids
                    # were skipped: that would confirm they exist.
                    self._owned(
                        select(Generation).where(Generation.id.in_(generation_ids)),
                        Generation.user_id,
                    )
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

    # ── reference audio (inputs, not assets) ───────────────────────

    async def create_reference_audio(
        self,
        *,
        reference_id: UUID,
        storage_key: str,
        sha256: str,
        source_sha256: str,
        source_format: str,
        duration_seconds: float,
        sample_rate: int,
        channels: int,
        file_size: int,
        display_name: str | None,
        user_id: UUID | None = None,
    ) -> ReferenceAudio:
        """Record an uploaded reference track.

        Always an insert. References are content-addressed and immutable,
        and two users uploading the same bytes get two rows: sharing one
        would tie their lifecycles together for no benefit.
        """
        # The id is supplied rather than generated here: the caller already
        # built the storage key from it, and a row whose key names a
        # different id than the row itself is a reference that cannot be
        # reasoned about.
        reference = ReferenceAudio(
            id=reference_id,
            storage_key=storage_key,
            sha256=sha256,
            source_sha256=source_sha256,
            source_format=source_format,
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
            channels=channels,
            file_size=file_size,
            display_name=display_name,
            user_id=user_id if user_id is not None else self._require_owner(),
        )
        self._session.add(reference)
        await self._session.commit()
        await self._session.refresh(reference)
        return reference

    async def get_reference_audio(self, reference_id: UUID) -> ReferenceAudio | None:
        return await self._fetch_owned(ReferenceAudio, reference_id, ReferenceAudio.user_id)

    async def find_abandoned_references(
        self, *, cutoff: datetime, limit: int
    ) -> list[ReferenceAudio]:
        """References older than *cutoff* that no generation cites.

        Candidates only. Nothing here is safe to delete on the strength
        of this result alone — a generation can attach one microsecond
        later, which is why the delete re-checks atomically rather than
        trusting this list.
        """
        used = select(Generation.reference_audio_id).where(
            Generation.reference_audio_id.is_not(None)
        )
        result = await self._session.execute(
            select(ReferenceAudio)
            .where(ReferenceAudio.created_at < cutoff)
            .where(ReferenceAudio.id.not_in(used))
            .order_by(ReferenceAudio.created_at)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def delete_reference_audio_if_unused(self, reference_id: UUID) -> bool:
        """Delete a reference, but only if still nothing references it.

        The condition lives *inside* the DELETE rather than in a preceding
        SELECT, which is what makes this race-safe. A generation committed
        between the candidate scan and this call makes the NOT EXISTS
        false, the statement matches zero rows, and the caller learns the
        reference is in use instead of destroying provenance.

        PostgreSQL additionally refuses via ON DELETE RESTRICT. The
        condition is not redundant: SQLite runs the unit tests with
        foreign keys disabled, so without it the guarantee would exist
        only in production and never be exercised by a test.

        Returns True when a row was actually removed.
        """
        referenced = (
            select(Generation.id).where(Generation.reference_audio_id == reference_id).exists()
        )
        result = await self._session.execute(
            delete(ReferenceAudio).where(ReferenceAudio.id == reference_id).where(~referenced)
        )
        await self._session.commit()
        # CursorResult on a DML statement; the base Result protocol does
        # not declare rowcount, so it is read off the cursor explicitly.
        return bool(cast("CursorResult[Any]", result).rowcount)

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

    async def delete_audio_asset(self, generation_id: UUID, *, asset_type: str) -> bool:
        """Remove one asset row; ``True`` when a row was actually deleted.

        Needed so a retry can retract a finished master that the current
        engine version no longer produces. Without it a stale
        FINISHED_MASTER row would keep winning delivery selection and
        point at an object the new run never wrote.
        """
        existing = await self._session.execute(
            select(AudioAsset).where(
                AudioAsset.generation_id == generation_id,
                AudioAsset.asset_type == asset_type,
            )
        )
        asset = existing.scalar_one_or_none()
        if asset is None:
            return False
        await self._session.delete(asset)
        await self._session.commit()
        return True

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
        project = Project(
            name=name, user_id=user_id if user_id is not None else self._require_owner()
        )
        self._session.add(project)
        await self._session.commit()
        await self._session.refresh(project)
        return project

    async def get_project(self, project_id: UUID) -> Project | None:
        return await self._fetch_owned(Project, project_id, Project.user_id)

    async def list_projects(self) -> list[Project]:
        result = await self._session.execute(
            self._owned(select(Project).order_by(Project.created_at.desc()), Project.user_id)
        )
        return list(result.scalars().all())

    async def list_projects_with_counts(self) -> list[tuple[Project, int]]:
        """Every project with its song count, in one query.

        Counting per project in a loop is an N+1 that grows with the
        sidebar, and the sidebar is rendered on every visit to Projects.
        An outer join keeps projects with no songs, which are exactly the
        ones a new user has.
        """
        result = await self._session.execute(
            self._owned(
                select(Project, func.count(Generation.id))
                .outerjoin(Generation, Generation.project_id == Project.id)
                .group_by(Project.id)
                .order_by(Project.created_at.desc()),
                Project.user_id,
            )
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
        generation = await self._fetch_owned(Generation, generation_id, Generation.user_id)
        if generation is None:
            raise LookupError(f"generation not found: {generation_id}")
        generation.project_id = project_id
        await self._session.commit()
        await self._session.refresh(generation)
        return generation

    async def list_generations_for_project(self, project_id: UUID) -> list[Generation]:
        result = await self._session.execute(
            self._owned(
                select(Generation)
                .options(selectinload(Generation.audio_assets))
                .where(Generation.project_id == project_id)
                .order_by(Generation.created_at.desc()),
                Generation.user_id,
            )
        )
        return list(result.scalars().all())

    async def list_children(self, generation_id: UUID) -> list[Generation]:
        """Generations created from *generation_id* (lineage, Phase 8)."""
        result = await self._session.execute(
            self._owned(
                select(Generation)
                .options(selectinload(Generation.audio_assets))
                .where(Generation.parent_generation_id == generation_id)
                .order_by(Generation.created_at.asc()),
                Generation.user_id,
            )
        )
        return list(result.scalars().all())
