"""provider circuits and their transition history

Phase 31 stops calling a provider that is not answering, and needs the
decision to be shared and to survive a restart.

Shared, because `max_jobs = 1` per generation worker means several
worker processes is how this scales. A circuit held in each process's
memory would let worker A give up on a provider while worker B kept
calling it for another four minutes — and the operator would see a
circuit that was open and traffic that never stopped.

Surviving a restart, because a deploy in the middle of a provider outage
would otherwise reset every circuit to closed and send the whole queue
back at a provider that is still down.

`provider_circuits` holds current state, one row per ``provider:task``,
rewritten under a compare-and-set on ``revision``. Two workers crossing
the failure threshold at the same instant both try to write; one lands,
the other re-reads and finds the circuit already open, which is the
right answer rather than a race to resolve.

`provider_circuit_transitions` is append-only. The state row is
overwritten on every change, so without this an operator could see that
a circuit is open but never that it has opened eleven times today — and
a manual override's reason would be lost the moment the policy moved the
state again.

Identity is provider *and* task type. One circuit per provider would let
a broken cover endpoint take text-to-music offline with it, which is the
breaker doing more damage than the fault it is reacting to.

Neither table references another. A circuit is about configuration — a
provider name, a task type — not about a row, and a foreign key to
something that is not a table would be an invention.

Both are additive and empty on creation. Nothing reads them until a
provider fails, and existing behaviour is untouched.

Revision ID: 0017
Revises: 0016
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0017"
down_revision = "0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "provider_circuits",
        # ``provider:task_type``. The identity itself is the key: a
        # surrogate id would let one circuit exist as two rows, and two
        # rows would disagree.
        sa.Column("circuit_key", sa.String(length=160), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("task_type", sa.String(length=40), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        # AUTOMATIC or MANUAL. A circuit a human pinned is not moved by
        # the policy — they had a reason the evidence does not contain.
        sa.Column("control", sa.String(length=16), nullable=False),
        # The rolling window of counted outcomes, as JSON. Bounded by
        # the state machine before it is written: a busy provider must
        # not grow a column that is loaded on every routing decision.
        sa.Column("window", sa.Text(), nullable=False),
        sa.Column("consecutive_failures", sa.Integer(), nullable=False),
        sa.Column("consecutive_successes", sa.Integer(), nullable=False),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=True),
        # When a probe becomes permissible. NULL under a manual open: a
        # human released that, and a clock does not.
        sa.Column("open_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("consecutive_opens", sa.Integer(), nullable=False),
        sa.Column("open_reason", sa.Text(), nullable=True),
        sa.Column("open_evidence", sa.Text(), nullable=False),
        # Probe slots held, as JSON {token: expiry}. Small by
        # construction — probe concurrency is 1 by default — and leased,
        # so a worker that dies mid-probe does not hold the only slot
        # forever and wedge the circuit in HALF_OPEN.
        sa.Column("probes", sa.Text(), nullable=False),
        sa.Column("probe_successes", sa.Integer(), nullable=False),
        sa.Column("last_failure_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_failure_category", sa.String(length=48), nullable=True),
        sa.Column("last_success_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_transition_at", sa.DateTime(timezone=True), nullable=True),
        # Recorded, not keyed on. A model revision changes under a
        # running deployment, and keying the circuit on it would wipe
        # the evidence at exactly the moment a bad rollout needed it.
        sa.Column("last_provider_revision", sa.String(length=96), nullable=True),
        sa.Column("manual_reason", sa.Text(), nullable=True),
        sa.Column("manual_operator", sa.String(length=100), nullable=True),
        sa.Column("manual_at", sa.DateTime(timezone=True), nullable=True),
        # The compare-and-set token. This column is the whole
        # multi-worker coordination mechanism.
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("circuit_policy_version", sa.String(length=40), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("circuit_key"),
    )
    # One index, for the only query an operator runs: which circuits are
    # not closed, most recently changed first.
    op.create_index(
        "ix_provider_circuits_state",
        "provider_circuits",
        ["state", "last_transition_at"],
    )

    op.create_table(
        "provider_circuit_transitions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("circuit_key", sa.String(length=160), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("task_type", sa.String(length=40), nullable=False),
        sa.Column("previous_state", sa.String(length=16), nullable=False),
        sa.Column("current_state", sa.String(length=16), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        # False for an operator action. Without the distinction a manual
        # override is indistinguishable from the policy having decided,
        # which is exactly what an audit needs to tell apart.
        sa.Column("automatic", sa.Boolean(), nullable=False),
        sa.Column("operator", sa.String(length=100), nullable=True),
        sa.Column("evidence", sa.Text(), nullable=False),
        sa.Column("latency_seconds", sa.Float(), nullable=True),
        sa.Column("circuit_policy_version", sa.String(length=40), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_provider_circuit_transitions_key_at",
        "provider_circuit_transitions",
        ["circuit_key", "occurred_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_provider_circuit_transitions_key_at",
        table_name="provider_circuit_transitions",
    )
    op.drop_table("provider_circuit_transitions")
    op.drop_index("ix_provider_circuits_state", table_name="provider_circuits")
    op.drop_table("provider_circuits")
