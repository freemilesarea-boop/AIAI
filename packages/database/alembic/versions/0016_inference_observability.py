"""the inference observability projection

Phase 30 answers "is inference quality getting worse over time?", and it
cannot answer it from the `generations` table directly.

Not because the data is missing — it is all there, in the row and in the
Phase 29 candidate trace — but because that table also holds `prompt`,
`lyrics` and `title` as ordinary columns, and `request_trace` carries the
full original prompt and lyrics inside its JSON. An analytics layer with
a query path to that table is one careless `SELECT *` away from putting
somebody's lyrics on a dashboard or into an exported report.

So `inference_observations` is a projection: one row per generation,
holding counts, latencies, outcome codes and low-cardinality dimensions,
and holding no field a prompt could occupy. The privacy guarantee stops
being a rule somebody has to remember and becomes a fact about the
schema.

`inference_incidents` holds what the regression engine concluded. Keyed
on a fingerprint of the finding rather than on a timestamp, so a
detector running every few minutes updates one row instead of appending
one per run — the difference between an operator seeing three problems
and seeing three hundred rows describing three problems.

Neither table references `generations`. Deleting a song must not delete
the record that it was generated: an analytics row that vanished with a
user's deletion would silently change last month's counts.
`generation_id` is a pseudonymous handle here, not a relationship.

Both tables are additive and empty on creation. Nothing reads them until
something has been ingested, and existing behaviour is untouched.

Revision ID: 0016
Revises: 0015
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "inference_observations",
        # The primary key is the generation, which is what makes both
        # backfill and incremental ingestion idempotent for free:
        # ingesting the same generation twice updates one row.
        sa.Column("generation_id", sa.Uuid(), nullable=False),
        # The generation's start, not the observation's write time. Trend
        # analysis asks what the system was doing at a moment, and a
        # backfill stamped with its own clock would compress a week of
        # history into one afternoon.
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("request_sha256", sa.String(length=64), nullable=True),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("model_name", sa.String(length=100), nullable=False),
        sa.Column("model_version", sa.String(length=50), nullable=False),
        sa.Column("provider_revision", sa.String(length=160), nullable=False),
        sa.Column("luber_revision", sa.String(length=64), nullable=False),
        sa.Column("task_type", sa.String(length=32), nullable=False),
        sa.Column("requested_duration_seconds", sa.Float(), nullable=True),
        sa.Column("duration_bucket", sa.String(length=16), nullable=False),
        sa.Column("language", sa.String(length=16), nullable=False),
        sa.Column("instrumental", sa.String(length=8), nullable=False),
        sa.Column("bpm_requested", sa.Boolean(), nullable=False),
        sa.Column("key_requested", sa.Boolean(), nullable=False),
        sa.Column("reference_conditioned", sa.Boolean(), nullable=False),
        sa.Column("qc_policy", sa.String(length=40), nullable=False),
        sa.Column("qc_schema_version", sa.String(length=40), nullable=False),
        sa.Column("qc_engine_version", sa.String(length=40), nullable=False),
        sa.Column("retry_policy_version", sa.String(length=40), nullable=False),
        sa.Column("finishing_version", sa.String(length=40), nullable=False),
        sa.Column("generation_status", sa.String(length=20), nullable=False),
        sa.Column("generation_failure_code", sa.String(length=50), nullable=True),
        # False for generations that predate Phase 29 (commit 460642e).
        # Their retry counts are unknown rather than zero, and every
        # candidate-derived rate excludes them rather than averaging them
        # in as flawless.
        sa.Column("qc_data_available", sa.Boolean(), nullable=False),
        sa.Column("qc_outcome", sa.String(length=40), nullable=True),
        sa.Column("finishing_outcome", sa.String(length=40), nullable=True),
        sa.Column("candidate_count", sa.Integer(), nullable=True),
        sa.Column("provider_call_count", sa.Integer(), nullable=True),
        sa.Column("quality_retry_count", sa.Integer(), nullable=True),
        sa.Column("selected_on_attempt", sa.Integer(), nullable=True),
        sa.Column("first_candidate_accepted", sa.Boolean(), nullable=True),
        sa.Column("retry_exhausted", sa.Boolean(), nullable=True),
        sa.Column("candidate_rejections", sa.Integer(), nullable=True),
        sa.Column("provider_latency_seconds", sa.Float(), nullable=True),
        sa.Column("qc_latency_seconds", sa.Float(), nullable=True),
        # Post-processing, finishing, encoding and upload together.
        # Named for what it covers rather than for finishing alone:
        # Phase 22 measures no timing of its own, so a column called
        # "finishing_latency" would be four stages wearing one name.
        sa.Column("delivery_latency_seconds", sa.Float(), nullable=True),
        sa.Column("total_latency_seconds", sa.Float(), nullable=True),
        # Small bounded JSON arrays. A child table would cost a join on
        # every analytics query to normalise something nothing queries
        # on its own.
        sa.Column("critical_findings", sa.Text(), nullable=False),
        sa.Column("soft_findings", sa.Text(), nullable=False),
        sa.Column("data_quality_issues", sa.Text(), nullable=False),
        sa.Column("schema_version", sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint("generation_id"),
    )
    # Three indexes, matching the only three shapes a query takes: a
    # window, a window per provider, a window per revision. Indexing
    # every dimension would slow ingestion to speed up queries that scan
    # the window's rows anyway.
    op.create_index(
        "ix_inference_observations_occurred_at",
        "inference_observations",
        ["occurred_at"],
    )
    op.create_index(
        "ix_inference_observations_provider_time",
        "inference_observations",
        ["provider", "occurred_at"],
    )
    op.create_index(
        "ix_inference_observations_revision_time",
        "inference_observations",
        ["provider_revision", "occurred_at"],
    )

    op.create_table(
        "inference_incidents",
        # The fingerprint: finding type, category, metric and segment,
        # hashed. Deriving the key from what the problem *is* — never
        # from what it currently measures — is what stops a detector run
        # from minting a new incident every few minutes.
        sa.Column("incident_id", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("peak_severity", sa.String(length=16), nullable=False),
        sa.Column("finding_type", sa.String(length=64), nullable=False),
        sa.Column("category", sa.String(length=20), nullable=False),
        sa.Column("metric", sa.String(length=64), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=True),
        sa.Column("provider_revision", sa.String(length=160), nullable=True),
        sa.Column("segment", sa.Text(), nullable=False),
        sa.Column("first_seen", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_seen", sa.DateTime(timezone=True), nullable=True),
        sa.Column("occurrence_count", sa.Integer(), nullable=False),
        sa.Column("consecutive_clean", sa.Integer(), nullable=False),
        sa.Column("baseline_window", sa.Text(), nullable=False),
        sa.Column("current_window", sa.Text(), nullable=False),
        sa.Column("evidence", sa.Text(), nullable=False),
        sa.Column("recommendations", sa.Text(), nullable=False),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("acknowledged_by", sa.String(length=100), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dismissed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dismissed_by", sa.String(length=100), nullable=True),
        # Never cleared. Why something was ignored is exactly what the
        # next person needs when it comes back.
        sa.Column("dismissal_reason", sa.Text(), nullable=True),
        sa.Column("incident_policy_version", sa.String(length=40), nullable=False),
        sa.PrimaryKeyConstraint("incident_id"),
    )
    op.create_index(
        "ix_inference_incidents_status_seen",
        "inference_incidents",
        ["status", "last_seen"],
    )
    op.create_index("ix_inference_incidents_created", "inference_incidents", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_inference_incidents_created", table_name="inference_incidents")
    op.drop_index("ix_inference_incidents_status_seen", table_name="inference_incidents")
    op.drop_table("inference_incidents")
    op.drop_index("ix_inference_observations_revision_time", table_name="inference_observations")
    op.drop_index("ix_inference_observations_provider_time", table_name="inference_observations")
    op.drop_index("ix_inference_observations_occurred_at", table_name="inference_observations")
    op.drop_table("inference_observations")
