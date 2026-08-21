"""A synthetic observability store, and the console that serves it.

The rows here are built through the real ingestion path — a `Generation`
with a real Phase 29 trace, projected by the real projector — rather
than written directly into the analytics table. A hand-written
observation would let the read model pass against a shape nothing emits,
and the first place that would show up is production data rendering as
blanks.

Every generation carries a prompt, lyrics and a title that are
deliberately unmistakable strings. The privacy tests search for them in
every response; a leak that produced them anywhere fails loudly rather
than looking like ordinary text.

No audio, no model, no network, no GPU.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine

from luber_api.main import create_app
from luber_database import Base, ObservabilityRepository, create_session_factory
from luber_database.models.generation import Generation
from luber_database.models.user import User
from luber_inference_observability.incidents import IncidentLedger
from luber_inference_observability.queries import run_detection
from luber_inference_observability.service import (
    ingest,
    load_store_spanning,
    save_ledger,
)
from luber_inference_observability.windows import TimeWindow

OPERATOR_TOKEN = "test-operator-token-0123456789"

#: The moment every fixture is anchored to, fixed once at import.
#:
#: Real "now" rather than a hard-coded date, because the console's
#: windows are relative — "the last hour", "the last seven days" — and a
#: fixture anchored to a fixed past would drift out of every window the
#: API actually asks for. Computed once so all the rows in one run share
#: an origin and a test cannot straddle a second boundary.
NOW = datetime.now(UTC)

OWNER = uuid.UUID("11111111-1111-4111-8111-111111111111")

#: Strings that must never appear in any console response. Distinctive
#: on purpose: a substring search for "prompt" would match a field name,
#: while these can only come from user content.
SECRET_PROMPT = "ZZPROMPTZZ dreamy Korean indie pop about a late train"
SECRET_LYRICS = "ZZLYRICSZZ [Verse] 오늘 밤 너를 생각해"
SECRET_TITLE = "ZZTITLEZZ Midnight Window"


def qc_trace(
    *,
    retries: int = 0,
    critical: str | None = None,
    exhausted: bool = False,
    provider_seconds: float = 40.0,
    qc_seconds: float = 1.2,
) -> str:
    """A Phase 29 trace in the exact shape the controller writes."""
    attempts: list[dict[str, Any]] = []
    for index in range(retries + 1):
        last = index == retries
        findings: list[dict[str, Any]] = []
        if critical and (not last or exhausted):
            findings.append({"code": critical, "severity": "CRITICAL", "detail": "synthetic"})
        if last and not exhausted:
            findings.append({"code": "NARROW_STEREO", "severity": "INFO", "detail": "synthetic"})
        attempts.append(
            {
                "candidate_id": f"cand_{index:02d}",
                "attempt_index": index,
                "attribution": "USER_REQUEST" if index == 0 else "QUALITY_RETRY",
                "status": "REJECTED" if (not last or exhausted) else "ELIGIBLE",
                "selection_status": "SELECTED" if (last and not exhausted) else "NOT_SELECTED",
                "seed": 1000 + index,
                "duration_seconds": 200.0,
                "retry_reason": None if index == 0 else f"{critical}: retried",
                "not_selected_reason": None if (last and not exhausted) else "rejected",
                "findings": findings,
                "provider_seconds": provider_seconds,
                "qc_seconds": qc_seconds,
            }
        )
    return json.dumps(
        {
            "qc_schema_version": "luber-inference-qc/1",
            "qc_engine_version": "qc-v1",
            "candidate_selection_version": "sel-v1",
            "retry_policy_version": "retry-v1",
            "request_sha256": "d" * 64,
            "base_seed": 1000,
            "policy": {"name": "STANDARD"},
            "attempts": attempts,
            "selected_candidate_id": None if exhausted else f"cand_{retries:02d}",
            "outcome": "RETRY_EXHAUSTED" if exhausted else "SELECTED",
            "outcome_detail": "synthetic",
            "finishing_outcome": "FINISHED",
            "exhausted": exhausted,
            "timings": {
                "candidate_phase_seconds": (provider_seconds + qc_seconds) * (retries + 1),
                "provider_seconds": provider_seconds * (retries + 1),
                "qc_seconds": qc_seconds * (retries + 1),
            },
            "budget": {
                "provider_calls_used": retries + 1,
                "candidates_generated": retries + 1,
                "retry_rounds": retries,
                "elapsed_seconds": (provider_seconds + qc_seconds) * (retries + 1),
                "maximum_total_provider_calls": 3,
                "maximum_candidate_count": 3,
                "maximum_retry_rounds": 2,
            },
        }
    )


def generation(
    *,
    at: datetime,
    status: str = "COMPLETED",
    revision: str = "v1",
    duration: int = 200,
    trace: str | None = None,
    error_code: str | None = None,
    edit_kind: str | None = None,
) -> Generation:
    """One generation row, carrying content the privacy tests hunt for."""
    return Generation(
        id=uuid.uuid4(),
        user_id=OWNER,
        title=SECRET_TITLE,
        prompt=SECRET_PROMPT,
        lyrics=SECRET_LYRICS,
        vocal_gender="female",
        duration_requested=duration,
        language="ko",
        instrumental=False,
        status=status,
        provider="ace_step",
        model_name="acestep",
        model_version=revision,
        edit_kind=edit_kind,
        error_code=error_code,
        created_at=at,
        started_at=at,
        completed_at=at + timedelta(seconds=60) if status != "CANCELLED" else None,
        inference_qc_trace=trace,
        # The trace that *does* contain the prompt. Present so the
        # privacy tests are testing against the real hazard rather than
        # a sanitised fixture.
        request_trace=json.dumps(
            {"original_prompt": SECRET_PROMPT, "original_lyrics": SECRET_LYRICS}
        ),
    )


def healthy_history(count: int = 300) -> list[Generation]:
    """A week of ordinary traffic, for a baseline to be built from."""
    start = NOW - timedelta(days=7)
    rows: list[Generation] = []
    for index in range(count):
        at = start + timedelta(seconds=index * 1900)
        rows.append(
            generation(
                at=at,
                trace=qc_trace(retries=1 if index % 40 == 0 else 0, critical="EARLY_COLLAPSE"),
            )
        )
    return rows


def regressed_window(count: int = 120) -> list[Generation]:
    """An hour where early collapse is everywhere."""
    start = NOW - timedelta(minutes=55)
    rows: list[Generation] = []
    for index in range(count):
        at = start + timedelta(seconds=index * 25)
        bad = index % 10 < 4
        rows.append(
            generation(
                at=at,
                trace=qc_trace(
                    retries=1 if bad else 0,
                    critical="EARLY_COLLAPSE" if bad else None,
                ),
            )
        )
    return rows


# `ops_environment` is the training console's fixture, reused rather than
# redefined: both consoles are gated by the same switch and the same
# token, and a second fixture that set the same variables would be a
# second place for them to drift apart.


@pytest.fixture
async def inference_app(ops_environment: Any, tmp_path: Any) -> Any:
    """The console, over a SQLite database seeded through real ingestion."""
    application = create_app()
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/inference-test.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = create_session_factory(engine)
    application.state.db_engine = engine
    application.state.session_factory = factory

    async with factory() as session:
        session.add(
            User(
                id=OWNER,
                email="operator@example.com",
                password_hash="x",
                display_name="Operator",
            )
        )
        for row in healthy_history() + regressed_window():
            session.add(row)
        await session.commit()

        repository = ObservabilityRepository(session)
        # Ingested through the real projector, in batches, exactly as a
        # backfill would.
        for _ in range(10):
            result = await ingest(repository, full=True, limit=500)
            if result.scanned == 0:
                break

        # One detector pass, so the console has incidents to render.
        window = TimeWindow.ending_at(NOW, "1h")
        store = await load_store_spanning(
            repository,
            current=window,
            baseline_span=timedelta(days=7),
            baseline_gap=timedelta(hours=1),
        )
        ledger = IncidentLedger()
        run_detection(store, current=window, ledger=ledger, at=NOW)
        await save_ledger(repository, ledger)

    yield application
    await engine.dispose()


@pytest.fixture
async def inference_client(inference_app: FastAPI) -> Any:
    """A client that carries the operator token on every request."""
    async with AsyncClient(
        transport=ASGITransport(app=inference_app),
        base_url="http://ops.test",
        headers={"X-Luber-Operator-Token": OPERATOR_TOKEN},
    ) as client:
        yield client


@pytest.fixture
async def empty_app(ops_environment: Any, tmp_path: Any) -> Any:
    """The console with nothing ingested. The empty state has to work."""
    application = create_app()
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/empty-test.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    application.state.db_engine = engine
    application.state.session_factory = create_session_factory(engine)
    yield application
    await engine.dispose()


@pytest.fixture
async def empty_client(empty_app: FastAPI) -> Any:
    async with AsyncClient(
        transport=ASGITransport(app=empty_app),
        base_url="http://ops.test",
        headers={"X-Luber-Operator-Token": OPERATOR_TOKEN},
    ) as client:
        yield client


SECRETS = (SECRET_PROMPT, SECRET_LYRICS, SECRET_TITLE, "ZZPROMPTZZ", "ZZLYRICSZZ", "ZZTITLEZZ")


def assert_no_user_content(payload: Any) -> None:
    """Nothing a user wrote may appear anywhere in a response."""
    rendered = json.dumps(payload, default=str)
    for secret in SECRETS:
        assert secret not in rendered, f"user content leaked into a console response: {secret}"
    for forbidden in ("original_prompt", "original_lyrics", "request_trace", "user_id"):
        assert forbidden not in rendered, f"a forbidden field reached a response: {forbidden}"
