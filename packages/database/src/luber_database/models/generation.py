"""Generation domain tables: generations, generation_jobs, audio_assets.

Types are dialect-portable (``sa.Uuid`` with client-side defaults) so
the same models run on PostgreSQL in production and SQLite in unit
tests. Status/enum values are stored as text and owned by
``luber_schemas`` — renames are breaking changes.
"""

from __future__ import annotations

import uuid
from datetime import datetime

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
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from luber_database.base import Base


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

    #: Set when this generation was produced by "generate again" or by
    #: a variation. SET NULL on delete so removing an original does not
    #: remove what was made from it.
    parent_generation_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("generations.id", ondelete="SET NULL"), nullable=True, index=True
    )
    variation_label: Mapped[str | None] = mapped_column(String(50), nullable=True)

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
    __table_args__ = (UniqueConstraint("generation_id", "asset_type"),)

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
