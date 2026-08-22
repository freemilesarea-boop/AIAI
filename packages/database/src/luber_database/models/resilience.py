"""Phase 31 tables: circuit state, and the history of how it got there.

Two tables, and the split is the usual one — current state that gets
overwritten, and an append-only record of every change.

`provider_circuits` is one row per circuit identity, rewritten under a
compare-and-set on `revision`. That column is the whole coordination
mechanism: several generation workers share a provider, and a circuit
held in each process's memory would let worker A give up while worker B
kept calling for another four minutes.

`provider_circuit_transitions` is append-only. An operator asking "has
this been flapping all week" needs the history, and a table that only
held current state could not answer. It is also what makes a manual
override auditable: the reason and the operator survive the next
automatic transition that overwrites the state row.

Neither table references anything. A circuit is about a provider name
and a task type, both of which are configuration rather than rows, and
a foreign key to something that does not exist as a table would be an
invention.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
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


class ProviderCircuitRow(Base):
    """One circuit's current state.

    Keyed by `circuit_key` — ``provider:task_type`` — rather than by a
    surrogate id, because that string *is* the identity and a generated
    key would let the same circuit exist twice.
    """

    __tablename__ = "provider_circuits"
    __table_args__ = (
        # The operator query: which circuits are not closed, worst first.
        Index("ix_provider_circuits_state", "state", "last_transition_at"),
    )

    #: ``provider:task_type``. The identity, and the primary key: two
    #: rows for one circuit is the failure this prevents.
    circuit_key: Mapped[str] = mapped_column(String(160), primary_key=True)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    task_type: Mapped[str] = mapped_column(String(40), nullable=False)

    state: Mapped[str] = mapped_column(String(16), nullable=False)
    #: AUTOMATIC or MANUAL. A manual pin is not moved by the policy.
    control: Mapped[str] = mapped_column(String(16), nullable=False)

    #: The rolling window of counted outcomes, as JSON. Bounded by the
    #: circuit itself before it is written — a busy provider must not
    #: grow a column somebody has to load on every routing decision.
    window: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    consecutive_failures: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    consecutive_successes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    opened_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    #: When a probe becomes permissible. NULL under a manual open: a
    #: human released it, and a clock does not.
    open_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    consecutive_opens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    open_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    open_evidence: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

    #: Probe slots currently held, as JSON ``{token: expiry}``. Small by
    #: construction: the policy's probe concurrency is 1 by default.
    probes: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    probe_successes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    last_failure_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_failure_category: Mapped[str | None] = mapped_column(String(48), nullable=True)
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_transition_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    #: Recorded rather than keyed on: a revision changes under a running
    #: deployment, and keying the circuit on it would wipe the evidence
    #: at exactly the moment a bad rollout needed it.
    last_provider_revision: Mapped[str | None] = mapped_column(String(96), nullable=True)

    manual_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    manual_operator: Mapped[str | None] = mapped_column(String(100), nullable=True)
    manual_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    #: Compare-and-set token. Every write requires the revision it read;
    #: two workers crossing the failure threshold at the same instant
    #: means one write lands and the other re-reads to find the circuit
    #: already open, which is the right answer rather than a conflict.
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    circuit_policy_version: Mapped[str] = mapped_column(String(40), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ProviderCircuitTransitionRow(Base):
    """One state change, kept forever.

    Append-only. The state row is overwritten on every transition, so
    without this an operator could see that a circuit is open and never
    that it has opened eleven times today — which is the more useful
    fact.
    """

    __tablename__ = "provider_circuit_transitions"
    __table_args__ = (
        Index("ix_provider_circuit_transitions_key_at", "circuit_key", "occurred_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    circuit_key: Mapped[str] = mapped_column(String(160), nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    task_type: Mapped[str] = mapped_column(String(40), nullable=False)

    previous_state: Mapped[str] = mapped_column(String(16), nullable=False)
    current_state: Mapped[str] = mapped_column(String(16), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    reason: Mapped[str] = mapped_column(Text, nullable=False)
    #: False for an operator action. The distinction is what makes a
    #: manual override auditable rather than indistinguishable from the
    #: policy having decided.
    automatic: Mapped[bool] = mapped_column(nullable=False, default=True)
    operator: Mapped[str | None] = mapped_column(String(100), nullable=True)
    evidence: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

    #: Latency of the attempt that triggered this, where there was one.
    latency_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    circuit_policy_version: Mapped[str] = mapped_column(String(40), nullable=False)


__all__ = ["ProviderCircuitRow", "ProviderCircuitTransitionRow"]
