"""The one thing that must not be wrong: who can reach the console.

Everything else in Phase 28 is a display problem. This is the file that
decides whether training internals — dataset identities, checkpoint
digests, worker hosts, trainer logs — are visible to anyone who can
reach the API, so it asserts the boundary from four independent
directions rather than once.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from ops_fixtures import OPERATOR_TOKEN, Scenario, make_context

from luber_api.main import create_app
from luber_api.settings import get_settings

OVERVIEW = "/v1/ops/training/overview"


async def _client(app: FastAPI, headers: dict[str, str] | None = None) -> AsyncClient:
    return AsyncClient(
        transport=ASGITransport(app=app), base_url="http://ops.test", headers=headers or {}
    )


def _operator_paths(app: FastAPI) -> list[str]:
    """Every operator path this app actually serves.

    Read from the generated OpenAPI document rather than by walking
    `app.routes`: included routers are wrapped, so a naive walk finds
    nothing and an assertion that "no operator route exists" would pass
    on an app that serves all of them.
    """
    return sorted(path for path in app.openapi()["paths"] if path.startswith("/v1/ops/"))


async def test_console_absent_when_not_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Off by default, and off reads as absent rather than forbidden.

    404 rather than 403 on purpose: a 403 would confirm to an anonymous
    prober that this deployment has a training console worth attacking.
    """
    monkeypatch.delenv("OPS_CONSOLE_ENABLED", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "test")
    get_settings.cache_clear()
    try:
        app = create_app()
        async with await _client(app) as client:
            response = await client.get(OVERVIEW)
        assert response.status_code == 404
        # And not merely unreachable: no operator path is served at all.
        assert _operator_paths(app) == []
    finally:
        get_settings.cache_clear()


async def test_console_is_never_mounted_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enabling it in production does not enable it.

    The router is not registered at all, so the refusal is structural
    rather than a check inside a route that a later edit could weaken.
    """
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("OPS_CONSOLE_ENABLED", "true")
    monkeypatch.setenv("OPS_OPERATOR_TOKEN", OPERATOR_TOKEN)
    get_settings.cache_clear()
    try:
        app = create_app()
        assert _operator_paths(app) == []
        async with await _client(app, {"X-Luber-Operator-Token": OPERATOR_TOKEN}) as client:
            response = await client.get(OVERVIEW)
        assert response.status_code == 404
    finally:
        get_settings.cache_clear()


async def test_enabled_without_a_token_fails_closed(
    monkeypatch: pytest.MonkeyPatch, scenario: Scenario
) -> None:
    """A half configuration serves nothing.

    Enabled with no token is the shape that leaves a console open, so it
    refuses with 503 rather than falling back to no authentication.
    """
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("OPS_CONSOLE_ENABLED", "true")
    monkeypatch.delenv("OPS_OPERATOR_TOKEN", raising=False)
    get_settings.cache_clear()
    try:
        app = create_app()
        app.state.ops_context = make_context(scenario)
        async with await _client(app) as client:
            response = await client.get(OVERVIEW)
        assert response.status_code == 503
        assert "operator token" in response.json()["detail"]
    finally:
        get_settings.cache_clear()


async def test_missing_and_wrong_tokens_are_rejected(ops_app: FastAPI) -> None:
    async with await _client(ops_app) as client:
        assert (await client.get(OVERVIEW)).status_code == 401
    async with await _client(ops_app, {"X-Luber-Operator-Token": "wrong"}) as client:
        assert (await client.get(OVERVIEW)).status_code == 401


async def test_a_product_session_grants_nothing(ops_app: FastAPI, scenario: Scenario) -> None:
    """A logged-in customer is exactly as far out as an anonymous one.

    The console does not consult the session cookie at all, so there is
    no product account — new, old, or otherwise — that reaching it
    depends on.
    """
    async with await _client(ops_app, {"Cookie": "luber_session=whatever"}) as client:
        response = await client.get(OVERVIEW)
    assert response.status_code == 401


async def test_every_operator_route_is_gated(ops_app: FastAPI) -> None:
    """Not one endpoint, all of them.

    Enumerated from the app's own routing table rather than a list
    written here: a route added later is covered by having been added.
    """
    paths = _operator_paths(ops_app)
    assert len(paths) >= 15, f"expected the whole operator surface, found {paths}"

    async with await _client(ops_app) as client:
        for path in paths:
            concrete = (
                path.replace("{run_id}", "run_x")
                .replace("{experiment_id}", "exp_x")
                .replace("{worker_id}", "wrk_x")
                .replace("{checkpoint_id}", "ckpt_x")
                .replace("{evaluation_id}", "eval_x")
            )
            for method in ("GET", "POST"):
                response = await client.request(method, concrete)
                # 401 for a gated route; 405 where that verb does not
                # exist. Never 200, and never a body.
                assert response.status_code in {401, 405}, f"{method} {concrete}"


async def test_unsafe_requests_from_a_foreign_origin_are_refused(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    run_id = scenario.run_ids["draft"]
    response = await ops_client.post(
        f"/v1/ops/training/runs/{run_id}/actions/cancel",
        headers={"Origin": "https://not-ours.example"},
    )
    assert response.status_code == 403


async def test_reads_are_allowed_with_the_token(ops_client: AsyncClient) -> None:
    response = await ops_client.get(OVERVIEW)
    assert response.status_code == 200
    assert response.json()["registry_present"] is True


async def test_consumer_routes_are_untouched(ops_app: FastAPI) -> None:
    """Mounting the console changes nothing about the product API.

    Asserted by asking the product API a question it has always
    answered, from the same process that is serving the console.
    """
    async with await _client(ops_app) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "luber-api"}


def test_no_operator_model_carries_a_secret_field() -> None:
    """A credential has nowhere to go, by construction.

    Stronger than remembering to strip one: a field that does not exist
    cannot be populated by a later change.
    """
    from luber_api.ops import schemas

    forbidden = ("ssh_key", "credential_ref", "token", "password", "secret", "private_key")
    offenders: list[str] = []
    for name in dir(schemas):
        model = getattr(schemas, name)
        fields: dict[str, Any] | None = getattr(model, "model_fields", None)
        if not isinstance(fields, dict):
            continue
        offenders.extend(
            f"{name}.{field}"
            for field in fields
            if any(word in field.lower() for word in forbidden)
        )
    assert offenders == []
