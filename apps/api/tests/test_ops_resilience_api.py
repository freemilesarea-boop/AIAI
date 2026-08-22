"""The circuit view: what an operator can see, and what nobody can do.

Three things are proved here. That the panel reports circuit state
honestly, including the parts that are unavailable. That it carries
nothing a user wrote and no credential. And that it cannot change
anything — the console is read-only by construction, not by a route
nobody has called yet.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from ops_fixtures import OPERATOR_TOKEN
from sqlalchemy.ext.asyncio import create_async_engine

from luber_api.main import create_app
from luber_api.settings import get_settings
from luber_database import Base, ResilienceRepository, create_session_factory
from luber_provider_resilience import (
    CircuitIdentity,
    CircuitRecord,
    CircuitState,
    ControlMode,
    DurableCircuitStore,
    Outcome,
)

BASE = "/v1/ops/resilience"
NOW = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)

READ_PATHS = ("/circuits", "/transitions", "/readiness", "/policy")


def _open_circuit(provider: str, task: str) -> CircuitRecord:
    """A circuit that opened five minutes ago, as the worker would leave it."""
    return CircuitRecord(
        identity=CircuitIdentity(provider, task),
        state=CircuitState.OPEN.value,
        window=[
            Outcome(at=NOW - timedelta(seconds=index), succeeded=False, category="PROVIDER_TIMEOUT")
            for index in range(5)
        ],
        consecutive_failures=5,
        opened_at=NOW - timedelta(minutes=5),
        open_until=NOW + timedelta(seconds=30),
        open_reason="5 consecutive failures (PROVIDER_TIMEOUT)",
        consecutive_opens=1,
        last_failure_at=NOW,
        last_failure_category="PROVIDER_TIMEOUT",
        revision=1,
    )


@pytest.fixture
async def resilience_app(ops_environment: Any, tmp_path: Any, monkeypatch: Any) -> Any:
    """The console over a database with one provider and one open circuit.

    `ace_step` rather than `mock`, because the readiness view reports on
    routable providers and a deployment configured with a test double
    has none — which is correct, and would make this fixture prove
    nothing.
    """
    monkeypatch.setenv("GENERATION_PROVIDER", "ace_step")
    monkeypatch.setenv("PROVIDER_RESILIENCE_ENABLED", "true")
    monkeypatch.setenv("ACE_STEP_MODEL", "acestep-v15-turbo")
    get_settings.cache_clear()

    application = create_app()
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/resilience-test.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = create_session_factory(engine)
    application.state.db_engine = engine
    application.state.session_factory = factory

    store = DurableCircuitStore(ResilienceRepository(factory))
    await store.save(_open_circuit("ace_step", "REFERENCE_CONDITIONED"), expected_revision=0)
    await store.save(
        CircuitRecord(identity=CircuitIdentity("ace_step", "TEXT_TO_MUSIC"), revision=1),
        expected_revision=0,
    )
    await ResilienceRepository(factory).record_transition(
        {
            "circuit_key": "ace_step:REFERENCE_CONDITIONED",
            "provider": "ace_step",
            "task_type": "REFERENCE_CONDITIONED",
            "previous_state": CircuitState.CLOSED.value,
            "current_state": CircuitState.OPEN.value,
            "occurred_at": NOW - timedelta(minutes=5),
            "reason": "5 consecutive failures (PROVIDER_TIMEOUT)",
            "automatic": True,
            "operator": None,
            "evidence": {"rule": "consecutive_failures"},
            "circuit_policy_version": "1.0.0",
        }
    )

    yield application
    await engine.dispose()
    get_settings.cache_clear()


@pytest.fixture
async def resilience_client(resilience_app: Any) -> Any:
    async with AsyncClient(
        transport=ASGITransport(app=resilience_app),
        base_url="http://ops.test",
        headers={"X-Luber-Operator-Token": OPERATOR_TOKEN},
    ) as client:
        yield client


# ── the gate ─────────────────────────────────────────────────────────


async def test_every_route_refuses_without_the_operator_token(resilience_app):
    async with AsyncClient(
        transport=ASGITransport(app=resilience_app), base_url="http://ops.test"
    ) as anonymous:
        for path in READ_PATHS:
            assert (await anonymous.get(f"{BASE}{path}")).status_code in (401, 403), path


async def test_the_view_is_absent_when_the_console_is_switched_off(app):
    """404, not 403. A deployment without the console has never heard
    of it."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://ops.test") as client:
        assert (await client.get(f"{BASE}/circuits")).status_code == 404


async def test_nothing_here_can_change_a_circuit(resilience_client):
    """Read-only by construction.

    An override belongs in a tool that works during an incident, and
    this console is refused in production. Every mutating verb is
    unroutable rather than merely unimplemented.
    """
    for path in (*READ_PATHS, "/circuits/ace_step:TEXT_TO_MUSIC"):
        for method in ("POST", "PUT", "PATCH", "DELETE"):
            response = await resilience_client.request(
                method, f"{BASE}{path}", json={"operator": "someone"}
            )
            assert response.status_code in (404, 405), f"{method} {path} accepted a mutation"


# ── what it shows ────────────────────────────────────────────────────


async def test_the_circuit_list_reports_state_and_evidence(resilience_client):
    payload = (await resilience_client.get(f"{BASE}/circuits")).json()

    by_key = {item["circuit_key"]: item for item in payload["circuits"]}
    assert set(by_key) == {"ace_step:REFERENCE_CONDITIONED", "ace_step:TEXT_TO_MUSIC"}

    broken = by_key["ace_step:REFERENCE_CONDITIONED"]
    assert broken["state"] == CircuitState.OPEN.value
    assert broken["consecutive_failures"] == 5
    assert broken["last_failure_category"] == "PROVIDER_TIMEOUT"
    assert broken["failure_rate"] == 1.0
    assert broken["open_until"] is not None
    assert payload["circuit_policy_version"]


async def test_a_rate_is_never_reported_without_something_to_divide(resilience_client):
    """An untouched circuit reports `null`, not 0%.

    Zero would read as "nothing is failing" when the truth is "nothing
    has been measured", and those look identical on a dashboard.
    """
    payload = (await resilience_client.get(f"{BASE}/circuits")).json()
    quiet = next(
        item for item in payload["circuits"] if item["circuit_key"] == "ace_step:TEXT_TO_MUSIC"
    )
    assert quiet["sample_count"] == 0
    assert quiet["failure_rate"] is None


async def test_readiness_names_the_capability_that_is_down(resilience_client):
    payload = (await resilience_client.get(f"{BASE}/readiness")).json()

    statuses = {item["capability"]: item["status"] for item in payload["capabilities"]}
    assert statuses["TEXT_TO_MUSIC"] == "AVAILABLE"
    assert statuses["REFERENCE_CONDITIONED"] == "UNAVAILABLE"
    assert payload["generation_available"] is True
    assert payload["degraded"] is True
    assert "REFERENCE_CONDITIONED" in payload["summary"]
    assert payload["metrics"]["circuits_open"] == 1


async def test_the_policy_says_whether_failover_is_actually_possible(resilience_client):
    """A single-provider deployment cannot fail over, whatever the mode
    is set to, and the console says so rather than implying redundancy."""
    payload = (await resilience_client.get(f"{BASE}/policy")).json()

    assert payload["resilience_enabled"] is True
    assert payload["routable_providers"] == ["ace_step"]
    assert payload["failover_possible"] is False
    assert payload["circuit_policy"]["consecutive_failure_threshold"] == 5


async def test_the_transition_log_is_the_audit_trail(resilience_client):
    payload = (await resilience_client.get(f"{BASE}/transitions")).json()

    assert len(payload["transitions"]) == 1
    entry = payload["transitions"][0]
    assert entry["previous_state"] == CircuitState.CLOSED.value
    assert entry["current_state"] == CircuitState.OPEN.value
    assert entry["automatic"] is True
    assert entry["operator"] is None


async def test_a_circuit_for_a_provider_nobody_configures_is_named_not_hidden(
    resilience_client, resilience_app
):
    """A decommissioned provider's open circuit still exists.

    Filtering it out would make an open circuit invisible; naming it
    lets an operator delete it deliberately.
    """
    factory = resilience_app.state.session_factory
    await DurableCircuitStore(ResilienceRepository(factory)).save(
        _open_circuit("retired_provider", "TEXT_TO_MUSIC"), expected_revision=0
    )

    payload = (await resilience_client.get(f"{BASE}/circuits")).json()
    assert payload["unconfigured_providers"] == ["retired_provider"]


# ── privacy ──────────────────────────────────────────────────────────


async def test_no_route_returns_user_content_or_a_credential(resilience_client, monkeypatch):
    """Structural, not filtered: there is nowhere in these models for a
    prompt or a key to go. Asserted anyway, because a later field could
    change that and this is where it would be caught.
    """
    monkeypatch.setenv("ACE_STEP_API_KEY", "sk-test-do-not-leak-0123456789")
    get_settings.cache_clear()

    forbidden = ("sk-test-do-not-leak", "Bearer", "password", "prompt", "lyrics")
    for path in READ_PATHS:
        body = json.dumps((await resilience_client.get(f"{BASE}{path}")).json())
        for needle in forbidden:
            assert needle.lower() not in body.lower(), f"{path} leaked {needle!r}"


async def test_a_manual_override_records_who_did_it_without_recording_why_privately(
    resilience_client, resilience_app
):
    """Operator identity is shown; the reason is an operator's sentence.

    Both are operator-authored, so neither is user content — but this is
    the one place a human string reaches the console, and it is worth a
    test that says so out loud.
    """
    factory = resilience_app.state.session_factory
    await ResilienceRepository(factory).record_transition(
        {
            "circuit_key": "ace_step:TEXT_TO_MUSIC",
            "provider": "ace_step",
            "task_type": "TEXT_TO_MUSIC",
            "previous_state": CircuitState.CLOSED.value,
            "current_state": CircuitState.OPEN.value,
            "occurred_at": NOW,
            "reason": "draining for a model swap",
            "automatic": False,
            "operator": "oncall",
            "evidence": {},
            "circuit_policy_version": "1.0.0",
        }
    )

    payload = (await resilience_client.get(f"{BASE}/transitions?limit=5")).json()
    manual = next(item for item in payload["transitions"] if not item["automatic"])
    assert manual["operator"] == "oncall"
    assert manual["reason"] == "draining for a model swap"


async def test_a_manually_controlled_circuit_says_so(resilience_client, resilience_app):
    factory = resilience_app.state.session_factory
    await DurableCircuitStore(ResilienceRepository(factory)).save(
        CircuitRecord(
            identity=CircuitIdentity("ace_step", "COVER"),
            state=CircuitState.OPEN.value,
            control=ControlMode.MANUAL.value,
            manual_operator="oncall",
            manual_reason="draining for a model swap",
            revision=1,
        ),
        expected_revision=0,
    )

    payload = (await resilience_client.get(f"{BASE}/circuits")).json()
    manual = next(item for item in payload["circuits"] if item["task_type"] == "COVER")
    assert manual["control"] == ControlMode.MANUAL.value
    assert manual["manual_operator"] == "oncall"

    readiness = (await resilience_client.get(f"{BASE}/readiness")).json()
    assert readiness["metrics"]["circuits_manual"] == 1
