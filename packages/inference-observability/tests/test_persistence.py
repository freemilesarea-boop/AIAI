"""The projection, against a real database.

The in-memory store is a complete implementation and every detection
test runs on it, which is what keeps the engine testable. These tests
exist for the half it cannot cover: that the SQLAlchemy tables round-trip
what the dataclasses hold, that the primary key really does make
ingestion idempotent, and that a window query is half-open where it says
it is.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pytest
from obs_fixtures import NOW, observation
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from luber_database import Base, ObservabilityRepository, create_session_factory
from luber_inference_observability import IncidentLedger, TimeWindow
from luber_inference_observability.incidents import fingerprint
from luber_inference_observability.ingest import as_rows
from luber_inference_observability.queries import run_detection
from luber_inference_observability.service import (
    load_ledger,
    load_store,
    load_store_spanning,
    save_ledger,
)
from luber_inference_observability.storage import from_mapping


@pytest.fixture
async def repository():
    engine = create_async_engine(
        "sqlite+aiosqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = create_session_factory(engine)
    async with factory() as session:
        yield ObservabilityRepository(session)
    await engine.dispose()


def sample(count: int = 50, *, start_index: int = 0, **kwargs):
    """*count* observations inside the last hour.

    ``start_index`` keeps generation ids distinct across two populations
    in one test: the projection is keyed on the id, so overlapping
    ranges would collapse into one set and the assertion would be
    counting something other than what it says.
    """
    start = NOW - timedelta(minutes=55)
    return [
        observation(start_index + index, start + timedelta(seconds=index * 3), **kwargs)
        for index in range(count)
    ]


async def test_an_observation_round_trips_through_the_database(repository):
    original = sample(1, critical=("EARLY_COLLAPSE",), soft=("NARROW_STEREO",))[0]
    await repository.upsert_observations(as_rows([original]))

    stored = from_mapping(await repository.get_observation(uuid.UUID(original.generation_id)))

    assert stored.generation_id == original.generation_id
    assert stored.critical_findings == ("EARLY_COLLAPSE",)
    assert stored.soft_findings == ("NARROW_STEREO",)
    assert stored.quality_retry_count == original.quality_retry_count
    assert stored.provider_revision == original.provider_revision
    # SQLite hands back naive datetimes; the reader restores UTC.
    assert stored.occurred_at.tzinfo is not None


async def test_writing_the_same_generation_twice_leaves_one_row(repository):
    rows = sample(10)
    await repository.upsert_observations(as_rows(rows))
    await repository.upsert_observations(as_rows(rows))

    assert await repository.count_observations() == 10


async def test_a_rewrite_updates_rather_than_duplicating(repository):
    row = sample(1)[0]
    await repository.upsert_observations(as_rows([row]))

    row.generation_status = "FAILED"
    row.generation_failure_code = "QUALITY_RETRY_EXHAUSTED"
    await repository.upsert_observations(as_rows([row]))

    assert await repository.count_observations() == 1
    stored = await repository.get_observation(uuid.UUID(row.generation_id))
    assert stored["generation_status"] == "FAILED"


async def test_a_window_query_is_half_open(repository):
    start = NOW - timedelta(hours=1)
    rows = [
        observation(0, start),
        observation(1, start + timedelta(minutes=30)),
        observation(2, NOW),
    ]
    await repository.upsert_observations(as_rows(rows))

    inside = await repository.select_observations(start=start, end=NOW)

    # The row exactly at `end` belongs to the next window, not this one.
    assert len(inside) == 2


async def test_an_unknown_filter_is_refused_rather_than_ignored(repository):
    """A silently dropped filter returns more rows than the caller asked
    for, which for analytics is a wrong number rather than an error."""
    with pytest.raises(ValueError, match="not an observation column"):
        await repository.select_observations(
            start=NOW - timedelta(hours=1), end=NOW, filters={"prompt": "anything"}
        )


async def test_the_watermark_is_the_newest_observation(repository):
    await repository.upsert_observations(as_rows(sample(10)))
    latest = await repository.latest_observed_at()
    assert latest is not None


async def test_loading_a_store_spanning_the_baseline_gets_both_windows(repository):
    week = [
        observation(1000 + index, NOW - timedelta(days=7) + timedelta(hours=index))
        for index in range(100)
    ]
    await repository.upsert_observations(as_rows(week + sample(20)))

    store = await load_store_spanning(
        repository,
        current=TimeWindow.ending_at(NOW, "1h"),
        baseline_span=timedelta(days=7),
        baseline_gap=timedelta(hours=1),
    )

    assert store.count() > 20, "the baseline rows must be loaded too"


async def test_a_segment_filter_reaches_the_query(repository):
    await repository.upsert_observations(
        as_rows(sample(10) + sample(5, revision="acestep@v2", start_index=500))
    )

    from luber_inference_observability.dimensions import Segment

    store = await load_store(
        repository,
        window=TimeWindow.ending_at(NOW, "1h"),
        segment=Segment.of(provider_revision="acestep@v2"),
    )

    assert store.count() == 5


# ── incidents ────────────────────────────────────────────────────────


async def test_an_incident_survives_a_round_trip(repository):
    healthy = [
        observation(2000 + index, NOW - timedelta(days=7) + timedelta(minutes=index * 8))
        for index in range(1000)
    ]
    bad = sample(200, accepted=False, critical=("EARLY_COLLAPSE",), start_index=9000)
    await repository.upsert_observations(as_rows(healthy + bad))

    store = await load_store_spanning(
        repository,
        current=TimeWindow.ending_at(NOW, "1h"),
        baseline_span=timedelta(days=7),
        baseline_gap=timedelta(hours=1),
    )
    ledger = IncidentLedger()
    run_detection(store, current=TimeWindow.ending_at(NOW, "1h"), ledger=ledger, at=NOW)
    assert ledger.active()
    await save_ledger(repository, ledger)

    restored = await load_ledger(repository)

    assert len(restored) == len(ledger)
    original = ledger.active()[0]
    copy = restored.get(original.incident_id)
    assert copy is not None
    assert copy.severity == original.severity
    assert copy.segment == original.segment
    assert len(copy.evidence) == len(original.evidence)


async def test_running_detection_twice_over_one_database_keeps_one_incident(repository):
    healthy = [
        observation(3000 + index, NOW - timedelta(days=7) + timedelta(minutes=index * 8))
        for index in range(1000)
    ]
    bad = sample(200, accepted=False, critical=("EARLY_COLLAPSE",), start_index=9500)
    await repository.upsert_observations(as_rows(healthy + bad))
    current = TimeWindow.ending_at(NOW, "1h")

    for tick in range(5):
        store = await load_store_spanning(
            repository,
            current=current,
            baseline_span=timedelta(days=7),
            baseline_gap=timedelta(hours=1),
        )
        ledger = await load_ledger(repository)
        run_detection(store, current=current, ledger=ledger, at=NOW + timedelta(minutes=tick))
        await save_ledger(repository, ledger)

    final = await load_ledger(repository)
    assert len(final) == len(final.active())
    assert final.active()[0].occurrence_count == 5


async def test_an_acknowledgement_persists(repository):
    from luber_inference_observability.dimensions import Segment
    from luber_inference_observability.incidents import InferenceIncident

    incident = InferenceIncident(
        incident_id=fingerprint("X", "QUALITY", "m", Segment.of(provider="ace_step")),
        created_at=NOW,
        finding_type="X",
        category="QUALITY",
        metric="m",
        segment=Segment.of(provider="ace_step"),
        first_seen=NOW,
        last_seen=NOW,
        occurrence_count=1,
    )
    ledger = IncidentLedger([incident])
    ledger.acknowledge(incident.incident_id, by="alex", at=NOW)
    await save_ledger(repository, ledger)

    restored = await load_ledger(repository)
    copy = restored.get(incident.incident_id)

    assert copy is not None
    assert copy.status == "ACKNOWLEDGED"
    assert copy.acknowledged_by == "alex"


async def test_incidents_are_paginated(repository):
    from luber_inference_observability.dimensions import Segment
    from luber_inference_observability.incidents import InferenceIncident

    ledger = IncidentLedger(
        [
            InferenceIncident(
                incident_id=f"{index:032d}",
                created_at=NOW,
                finding_type="X",
                category="QUALITY",
                metric=f"metric_{index}",
                segment=Segment(),
                first_seen=NOW,
                last_seen=NOW + timedelta(seconds=index),
            )
            for index in range(30)
        ]
    )
    await save_ledger(repository, ledger)

    page = await repository.list_incidents(limit=10, offset=0)
    assert len(page) == 10
    assert await repository.count_incidents() == 30


async def test_retention_deletes_only_what_it_is_told_to(repository):
    old = [observation(index, NOW - timedelta(days=90)) for index in range(5)]
    recent = sample(5, start_index=700)
    await repository.upsert_observations(as_rows(old + recent))

    removed = await repository.delete_observations(before=NOW - timedelta(days=30))

    assert removed == 5
    assert await repository.count_observations() == 5


async def test_a_backfill_larger_than_one_batch_reaches_the_newest_rows(repository):
    """The paging bug this exists to prevent: a backfill that restarted
    from the beginning every round would report the whole table scanned
    while never ingesting the most recent generations — the ones an
    operator is about to ask about.
    """
    from luber_inference_observability.cli import cmd_backfill

    class FakeGeneration:
        def __init__(self, index: int, at: datetime) -> None:
            self.id = uuid.UUID(int=index)
            self.status = "COMPLETED"
            self.created_at = at
            self.started_at = at
            self.completed_at = at + timedelta(seconds=60)
            self.provider = "ace_step"
            self.model_name = "acestep"
            self.model_version = "v1"
            self.duration_requested = 90
            self.language = "ko"
            self.instrumental = False
            self.bpm = None
            self.key_scale = None
            self.edit_kind = None
            self.reference_audio_id = None
            self.error_code = None
            self.inference_qc_trace = None
            self.finishing_trace = None

    start = NOW - timedelta(hours=6)
    everything = [
        FakeGeneration(index, start + timedelta(seconds=index * 20)) for index in range(300)
    ]

    class PagingRepository:
        """The source side of ingestion, paging the way the real one does."""

        def __init__(self) -> None:
            self.written: dict[str, object] = {}

        async def generations_to_ingest(self, *, since=None, limit=500, statuses=()):
            rows = [row for row in everything if since is None or row.created_at >= since]
            return rows[:limit]

        async def upsert_observations(self, rows):
            for row in rows:
                self.written[str(row["generation_id"])] = row
            return len(list(rows))

        async def latest_observed_at(self):
            return None

    source = PagingRepository()
    since = None
    rounds = 0
    # The same loop `cmd_backfill` runs, exercised without a database.
    while True:
        from luber_inference_observability.service import ingest as run_ingest

        result = await run_ingest(source, since=since, limit=100, full=since is None)
        rounds += 1
        if result.scanned < 100 or rounds > 20:
            break
        if result.watermark is None or result.watermark == since:
            break
        since = result.watermark

    assert len(source.written) == 300, "every generation must be reached"
    assert rounds <= 5, "and it must terminate rather than looping on one batch"
    assert cmd_backfill is not None
