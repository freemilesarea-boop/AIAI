"""Generation domain tables: generations, generation_jobs, audio_assets.

Types are dialect-portable (``sa.Uuid`` with client-side defaults) so
the same models run on PostgreSQL in production and SQLite in unit
tests. Status/enum values are stored as text and owned by
``luber_schemas`` — renames are breaking changes.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    false,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from luber_database.base import Base


def _utcnow() -> datetime:
    """Timezone-aware insert-time default with sub-second resolution."""
    return datetime.now(UTC)


class ReferenceAudio(Base):
    """Audio a user supplied to steer a generation.

    Its own table, not an ``audio_assets`` role. Assets belong to the
    generation that produced them; a reference is an input that exists
    before any generation, may steer several, and must never be
    reachable through the routes that serve masters. Keeping it here
    means there is no asset row that could name it and no key shape that
    could collide with one.

    Rows are immutable once written: the bytes are content-addressed by
    ``sha256`` and the canonical file is never rewritten, so a stored
    generation can always say exactly what conditioned it.
    """

    __tablename__ = "reference_audio"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    #: Key under ``reference/``, never under ``audio/``.
    storage_key: Mapped[str] = mapped_column(Text, nullable=False)
    #: Digest of the canonical stored bytes.
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    #: Digest of what the user actually uploaded, before normalisation.
    #: Kept so provenance survives a change to the canonical format.
    source_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    #: Container the upload arrived in, for provenance only.
    source_format: Mapped[str] = mapped_column(String(10), nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    sample_rate: Mapped[int] = mapped_column(Integer, nullable=False)
    channels: Mapped[int] = mapped_column(Integer, nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    #: Free-text label from the upload, shown back to the user. Never a
    #: path and never used to build one.
    display_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), default=_utcnow
    )

    generations: Mapped[list[Generation]] = relationship(back_populates="reference_audio")


class Generation(Base):
    __tablename__ = "generations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)

    title: Mapped[str] = mapped_column(Text, nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    lyrics: Mapped[str] = mapped_column(Text, nullable=False, default="")
    vocal_gender: Mapped[str] = mapped_column(String(20), nullable=False)
    duration_requested: Mapped[int] = mapped_column(Integer, nullable=False)
    duration_actual: Mapped[float | None] = mapped_column(Float, nullable=True)
    seed: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    instrumental: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Advanced musical controls. NULL means "the user did not specify",
    # which is distinct from any particular value — the provider omits
    # the field entirely rather than sending a default of its own.
    bpm: Mapped[int | None] = mapped_column(Integer, nullable=True)
    key_scale: Mapped[str | None] = mapped_column(String(20), nullable=True)
    time_signature: Mapped[str | None] = mapped_column(String(8), nullable=True)

    #: JSON: exactly what was sent to the provider, minus credentials.
    #: NULL on rows created before the trace existed.
    request_trace: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: JSON: pre-flight advisories recorded at submission time.
    advisories: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: The reference track this generation was conditioned on, if any.
    #: RESTRICT rather than CASCADE: deleting a reference must not delete
    #: the songs made from it, and a generation that silently lost its
    #: provenance would be worse than a refused delete.
    reference_audio_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("reference_audio.id", ondelete="RESTRICT"), nullable=True, index=True
    )

    #: JSON: what the Phase 14 finishing engine decided — outcome, engine
    #: version, the digest of the raw master it read, and the plan. NULL
    #: means the engine never ran (the generation predates Phase 14B),
    #: which is a different fact from it having run and found nothing.
    finishing_trace: Mapped[str | None] = mapped_column(Text, nullable=True)

    #: Set when this generation was produced by "generate again" or by
    #: a variation. SET NULL on delete so removing an original does not
    #: remove what was made from it.
    parent_generation_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("generations.id", ondelete="SET NULL"), nullable=True, index=True
    )
    variation_label: Mapped[str | None] = mapped_column(String(50), nullable=True)

    #: Workspace this generation is filed under (Phase 11). SET NULL on
    #: delete: removing a project must never remove the music in it.
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("projects.id", ondelete="SET NULL"), nullable=True, index=True
    )

    #: Phase 12 product state. Server-side, because a favourite that only
    #: exists in one browser is not a favourite.
    favorite: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )

    #: Siblings produced by a single CREATE, so the user can compare
    #: alternatives. Purely LUBER application metadata — the provider is
    #: never told about it, and each sibling is an independent job with
    #: its own seed, status and asset. No foreign key: a group is the set
    #: of rows sharing this id, not a separate entity with a lifecycle.
    generation_group_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True, index=True)

    #: Reserved for generated cover art. Never written in Phase 12 — the
    #: UI falls back to the deterministic placeholder while this is NULL.
    #: A fabricated URL here would be worse than nothing.
    cover_art_url: Mapped[str | None] = mapped_column(Text, nullable=True)

    #: Phase 13B. NULL means an ordinary text-to-music generation; a
    #: value means this row was produced by editing its parent's audio.
    #:
    #: Deliberately *not* folded into ``variation_label``: that field is
    #: client-settable free text, so a caller could label a plain
    #: generation "extend" and make the two indistinguishable. The worker
    #: routes on this column, so it must not be forgeable from the API.
    edit_kind: Mapped[str | None] = mapped_column(String(30), nullable=True)

    #: The source range this edit regenerated, in seconds from the start
    #: of the parent's master. Measured from the audio itself, not copied
    #: from any requested duration. Stored because the request trace is
    #: explicitly best-effort diagnostics and may be absent, while "what
    #: range was edited" must stay answerable for any edited row.
    edit_start_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    edit_end_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    #: Phase 13D. How closely a COVER was asked to follow its source,
    #: 0-1, higher being closer. NULL for every other kind of generation.
    #:
    #: Its own column rather than a reuse of ``edit_start_seconds``: that
    #: field is documented as seconds into the source, and storing a ratio
    #: in it would make the recorded provenance untrue. The worker routes
    #: on this value, so it also cannot live in ``request_trace``, which is
    #: best-effort diagnostics and may legitimately be absent.
    source_adherence: Mapped[float | None] = mapped_column(Float, nullable=True)

    status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    provider: Mapped[str | None] = mapped_column(String(50), nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Unique index (NULLs excluded) is the DB-level idempotency guarantee.
    idempotency_key: Mapped[str | None] = mapped_column(
        String(200), nullable=True, unique=True, index=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    error_code: Mapped[str | None] = mapped_column(String(50), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    jobs: Mapped[list[GenerationJob]] = relationship(back_populates="generation")
    audio_assets: Mapped[list[AudioAsset]] = relationship(back_populates="generation")
    #: Human QA (Phase 9). Absent until somebody listens.
    qa: Mapped[GenerationQA | None] = relationship(
        back_populates="generation", uselist=False, cascade="all, delete-orphan"
    )
    lyric_line_qa: Mapped[list[LyricLineQA]] = relationship(
        back_populates="generation", cascade="all, delete-orphan"
    )
    project: Mapped[Project | None] = relationship(back_populates="generations")
    reference_audio: Mapped[ReferenceAudio | None] = relationship(back_populates="generations")


class GenerationJob(Base):
    __tablename__ = "generation_jobs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    generation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("generations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    queue_name: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    enqueued_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(50), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    generation: Mapped[Generation] = relationship(back_populates="jobs")


class AudioAsset(Base):
    __tablename__ = "audio_assets"
    # One asset per role per generation. This is the DB-level guarantee
    # that re-running post-processing (a retry) updates a generation's
    # master/preview instead of accumulating duplicates.
    #
    # Named to match migration 0003. Left unnamed, autogenerate invents a
    # different name from the one in the database and `alembic check`
    # reports permanent phantom drift, which trains everyone to ignore it.
    __table_args__ = (
        UniqueConstraint(
            "generation_id", "asset_type", name="uq_audio_assets_generation_id_asset_type"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    generation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("generations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    asset_type: Mapped[str] = mapped_column(String(20), nullable=False)
    format: Mapped[str] = mapped_column(String(10), nullable=False)
    # Delivery metadata: what the client is told this object is. Serving
    # code uses these instead of guessing from the storage key.
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    file_extension: Mapped[str] = mapped_column(String(10), nullable=False)
    sample_rate: Mapped[int] = mapped_column(Integer, nullable=False)
    bit_depth: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: Bits per second for compressed formats; NULL for PCM.
    bitrate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    channels: Mapped[int] = mapped_column(Integer, nullable=False)
    duration: Mapped[float] = mapped_column(Float, nullable=False)
    storage_key: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    generation: Mapped[Generation] = relationship(back_populates="audio_assets")


class GenerationQA(Base):
    """One human's verdict on one generation.

    Phase 9. Everything the automated stack can measure lives on
    ``Generation``; this is the part only a listener can supply. The
    absence of a row means nobody has listened yet — which is different
    from a bad score and must stay distinguishable from one.
    """

    __tablename__ = "generation_qa"
    # One record per generation; re-reviewing updates it in place rather
    # than accumulating conflicting verdicts.
    __table_args__ = (UniqueConstraint("generation_id"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    generation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("generations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    #: 1-10 triage. NULL = not yet rated.
    overall_rating: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: JSON list of failure tags, e.g. ["KOREAN_LINE_OMISSION"].
    failure_tags: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: JSON object mapping section name to verdict, for the full-song view.
    section_verdicts: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewer: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    generation: Mapped[Generation] = relationship(back_populates="qa")


class LyricLineQA(Base):
    """What a listener heard happen to one submitted lyric line.

    The Korean failure that matters most is whole lines being skipped.
    No automatic detector exists in this stack, so this records the
    human answer per line and keeps the submitted text alongside it —
    snapshotted, so the record stays readable if the generation's lyrics
    are later edited.
    """

    __tablename__ = "lyric_line_qa"
    __table_args__ = (UniqueConstraint("generation_id", "line_index"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    generation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("generations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    #: Position in the submitted sheet, 0-based, section tags excluded.
    line_index: Mapped[int] = mapped_column(Integer, nullable=False)
    section_label: Mapped[str | None] = mapped_column(String(50), nullable=True)
    line_text: Mapped[str] = mapped_column(Text, nullable=False)
    #: COMPLETE | PARTIAL | SKIPPED | DUPLICATED | UNKNOWN
    verdict: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    generation: Mapped[Generation] = relationship(back_populates="lyric_line_qa")


class Project(Base):
    """A workspace grouping for generations.

    Deliberately minimal: a name and the generations filed under it.
    Collaboration, sharing and permissions are later phases, and
    modelling them now would be guessing at requirements.
    """

    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    #: Reserved for the authentication phase, like ``Generation.user_id``.
    user_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)
    # A client-side default as well as the server one. Projects are
    # ordered by creation, and SQLite's CURRENT_TIMESTAMP has one-second
    # resolution — two projects made in the same second tie, and the
    # order the user sees then depends on the query plan. The Python
    # default has microsecond resolution and applies on every ORM insert;
    # the server default stays for anything writing raw SQL.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        server_default=func.now(),
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, server_default=func.now()
    )

    generations: Mapped[list[Generation]] = relationship(back_populates="project")
