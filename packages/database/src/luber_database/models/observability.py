"""Phase 30 tables: what analytics may see, and what it concluded.

Two tables, and the first one's job is to be a wall.

`inference_observations` is a projection of `generations` holding only
privacy-safe facts. It exists because the source table carries `prompt`,
`lyrics` and `title` as ordinary columns, and `request_trace` holds the
full original prompt and lyrics inside its JSON. Analytics that queried
`generations` directly would be one careless `SELECT *` from putting
somebody's lyrics on a dashboard. Here there is no column a prompt could
occupy, so the guarantee is structural rather than remembered.

`inference_incidents` is what the regression engine concluded, kept so a
detector run every few minutes updates one row instead of appending a
thousand.

Neither table has a foreign key to `generations`. That is deliberate:
deleting a generation must not delete the record that it happened, and
an analytics row that vanished when a user removed a song would make
last month's counts change retroactively. `generation_id` is a
pseudonymous handle, not a relationship.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from luber_database.base import Base


class InferenceObservationRow(Base):
    """One generation, reduced to what may be counted.

    Flat on purpose: every column is a scalar an index can cover. The
    alternative — a JSON blob parsed per row per query — is what makes
    analytics slow enough that people stop running it.
    """

    __tablename__ = "inference_observations"
    __table_args__ = (
        # The three shapes every query has. Time first because every
        # question is bounded by a window; the other two because
        # "provider by time" and "revision by time" are what a
        # regression run asks for on every evaluation.
        #
        # Nothing else is indexed. An index per dimension would slow
        # ingestion for queries that scan a window's rows anyway.
        Index("ix_inference_observations_occurred_at", "occurred_at"),
        Index("ix_inference_observations_provider_time", "provider", "occurred_at"),
        Index(
            "ix_inference_observations_revision_time",
            "provider_revision",
            "occurred_at",
        ),
    )

    #: The generation this describes, and the primary key. Being the key
    #: is what makes ingestion idempotent: ingesting the same generation
    #: twice updates one row rather than adding a second.
    generation_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)

    #: When the generation started — not when it was observed. Trend
    #: analysis asks what the system was doing at a time, and ingestion
    #: time would compress a backfilled week into one afternoon.
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    #: Digest of the request. A drilldown handle, never a grouping
    #: dimension — grouping by it would produce one bucket per request.
    request_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)

    provider: Mapped[str] = mapped_column(String(50), nullable=False, default="UNKNOWN")
    model_name: Mapped[str] = mapped_column(String(100), nullable=False, default="UNKNOWN")
    model_version: Mapped[str] = mapped_column(String(50), nullable=False, default="UNKNOWN")
    provider_revision: Mapped[str] = mapped_column(String(160), nullable=False, default="UNKNOWN")
    #: Written only by incremental ingestion, where the ingesting process
    #: is the one that produced the generation. Backfill writes UNKNOWN,
    #: because a backfill today cannot know last week's commit.
    luber_revision: Mapped[str] = mapped_column(String(64), nullable=False, default="UNKNOWN")

    task_type: Mapped[str] = mapped_column(String(32), nullable=False, default="UNKNOWN")
    requested_duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    duration_bucket: Mapped[str] = mapped_column(String(16), nullable=False, default="UNKNOWN")
    #: Explicit request metadata only. Never inferred from prompt text.
    language: Mapped[str] = mapped_column(String(16), nullable=False, default="UNKNOWN")
    instrumental: Mapped[str] = mapped_column(String(8), nullable=False, default="UNKNOWN")
    bpm_requested: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    key_requested: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    reference_conditioned: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    qc_policy: Mapped[str] = mapped_column(String(40), nullable=False, default="UNKNOWN")
    qc_schema_version: Mapped[str] = mapped_column(String(40), nullable=False, default="UNKNOWN")
    qc_engine_version: Mapped[str] = mapped_column(String(40), nullable=False, default="UNKNOWN")
    retry_policy_version: Mapped[str] = mapped_column(String(40), nullable=False, default="UNKNOWN")
    finishing_version: Mapped[str] = mapped_column(String(40), nullable=False, default="UNKNOWN")

    generation_status: Mapped[str] = mapped_column(String(20), nullable=False, default="UNKNOWN")
    generation_failure_code: Mapped[str | None] = mapped_column(String(50), nullable=True)
    #: False for generations predating Phase 29. Their retry counts are
    #: unknown, not zero, and every candidate-derived rate excludes them.
    qc_data_available: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    qc_outcome: Mapped[str | None] = mapped_column(String(40), nullable=True)
    finishing_outcome: Mapped[str | None] = mapped_column(String(40), nullable=True)

    candidate_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    provider_call_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quality_retry_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    selected_on_attempt: Mapped[int | None] = mapped_column(Integer, nullable=True)
    first_candidate_accepted: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    retry_exhausted: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    candidate_rejections: Mapped[int | None] = mapped_column(Integer, nullable=True)

    provider_latency_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    qc_latency_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    #: Post-processing, finishing, encoding and upload together. Named
    #: for what it covers: Phase 22 measures no timing of its own, and a
    #: column called "finishing" that held four stages would be a lie.
    delivery_latency_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_latency_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    #: JSON arrays. Small and bounded — a generation has a handful of
    #: findings — so a child table would cost a join on every query to
    #: normalise something nothing queries independently.
    critical_findings: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    soft_findings: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    data_quality_issues: Mapped[str] = mapped_column(Text, nullable=False, default="[]")

    schema_version: Mapped[str] = mapped_column(String(64), nullable=False)


class InferenceIncidentRow(Base):
    """One logical regression, over its whole life.

    Updated in place by successive detector runs rather than appended to,
    which is the difference between an operator seeing three problems and
    seeing three hundred rows describing three problems.
    """

    __tablename__ = "inference_incidents"
    __table_args__ = (
        Index("ix_inference_incidents_status_seen", "status", "last_seen"),
        Index("ix_inference_incidents_created", "created_at"),
    )

    #: The fingerprint: a hash of finding type, category, metric and
    #: segment. Being the key is what makes deduplication structural —
    #: the same regression cannot become two rows.
    incident_id: Mapped[str] = mapped_column(String(32), primary_key=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    #: The worst it ever was. Separate from `severity` so a recovering
    #: incident still records how bad it got.
    peak_severity: Mapped[str] = mapped_column(String(16), nullable=False)

    finding_type: Mapped[str] = mapped_column(String(64), nullable=False)
    category: Mapped[str] = mapped_column(String(20), nullable=False)
    metric: Mapped[str] = mapped_column(String(64), nullable=False)

    provider: Mapped[str | None] = mapped_column(String(50), nullable=True)
    provider_revision: Mapped[str | None] = mapped_column(String(160), nullable=True)
    #: JSON: the dimension filters this incident is scoped to.
    segment: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

    first_seen: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_seen: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    occurrence_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    consecutive_clean: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    #: JSON: the windows compared, and the evidence timeline. Bounded by
    #: the incident policy's evidence limit so a long-running incident
    #: does not grow a column nobody can load.
    baseline_window: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    current_window: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    evidence: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    recommendations: Mapped[str] = mapped_column(Text, nullable=False, default="[]")

    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    acknowledged_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    dismissed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    dismissed_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    #: Never cleared. The record of why something was ignored is exactly
    #: what the next person needs when it comes back.
    dismissal_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    incident_policy_version: Mapped[str] = mapped_column(String(40), nullable=False)


__all__ = ["InferenceIncidentRow", "InferenceObservationRow"]
