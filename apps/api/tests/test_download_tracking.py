"""What counts as a download, and what a failed count must not cost.

Two properties, and the second is the one that would hurt if it broke.

**A download is a save, not a play.** The in-page player streams the same
URL without `download=true`, and an audio element fetches it in ranges.
Counting those would make the number measure HTTP behaviour rather than
how many songs people took away — and it is the second question an
operator is actually asking.

**Recording must never cost a customer their file.** The metrics write
happens after every authorisation check has passed and the bytes are
about to be delivered. If that write fails — a locked table, a migration
half-applied — the customer still gets the song. An undercounted
statistic is a smaller problem than a paid download that returned 500.
"""

from __future__ import annotations

import uuid

import pytest
from admin_fixtures import signed_up_client
from fastapi import FastAPI
from httpx import AsyncClient
from plan_fixtures import set_plan
from sqlalchemy import func, select

from luber_database.models.admin import DownloadEvent
from luber_schemas.enums import UserRole
from luber_schemas.plans import PlanId

PAYLOAD = {
    "title": "다운로드 대상",
    "prompt": "잔잔한 피아노",
    "lyrics": "",
    "vocal_gender": "instrumental",
    "duration": 30,
    "language": "ko",
    "instrumental": True,
}


async def _a_song(client: AsyncClient) -> str:
    response = await client.post("/v1/generations", json=PAYLOAD)
    assert response.status_code == 202, response.text
    return str(response.json()["generation_id"])


async def _events(app: FastAPI) -> int:
    factory = app.state.session_factory
    async with factory() as session:
        return int((await session.execute(select(func.count(DownloadEvent.id)))).scalar_one())


async def test_a_save_is_recorded(app: FastAPI, client: AsyncClient) -> None:
    generation_id = await _a_song(client)

    response = await client.get(f"/v1/generations/{generation_id}/audio?download=true")

    assert response.status_code == 200
    assert await _events(app) == 1


async def test_streaming_for_the_player_is_not_a_download(
    app: FastAPI, client: AsyncClient
) -> None:
    """The player hits the same URL without the flag, repeatedly."""
    generation_id = await _a_song(client)

    for _ in range(3):
        assert (await client.get(f"/v1/generations/{generation_id}/audio")).status_code == 200

    assert await _events(app) == 0


async def test_the_recorded_row_carries_the_plan_in_force(
    app: FastAPI, client: AsyncClient
) -> None:
    """Denormalised on purpose: "was this download allowed" stays
    answerable after the account changes tier."""
    generation_id = await _a_song(client)
    await client.get(f"/v1/generations/{generation_id}/audio?download=true")

    factory = app.state.session_factory
    async with factory() as session:
        row = (await session.execute(select(DownloadEvent))).scalar_one()

    assert row.plan_id == PlanId.BASIC.value
    assert row.asset_kind == "master"
    assert str(row.user_id) == client.user_id  # type: ignore[attr-defined]


async def test_a_refused_download_is_not_recorded(app: FastAPI, client: AsyncClient) -> None:
    """A Free account is refused before anything is delivered."""
    generation_id = await _a_song(client)
    await set_plan(app, client.user_id, PlanId.FREE)  # type: ignore[attr-defined]

    response = await client.get(f"/v1/generations/{generation_id}/audio?download=true")

    assert response.status_code == 402
    assert await _events(app) == 0


async def test_another_accounts_download_is_not_recorded(
    app: FastAPI, client: AsyncClient, client_b: AsyncClient
) -> None:
    generation_id = await _a_song(client)

    response = await client_b.get(f"/v1/generations/{generation_id}/audio?download=true")

    assert response.status_code == 404
    assert await _events(app) == 0


async def test_a_failed_recording_does_not_break_the_download(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The property that matters.

    A customer who paid for downloads gets their file even when the
    metrics table cannot be written.
    """
    from luber_api.routes import generations as module

    generation_id = await _a_song(client)

    class _Unwritable:
        """Stands in for a table that cannot be written."""

        def __init__(self, **_: object) -> None:
            raise RuntimeError("download_events is unavailable")

    monkeypatch.setattr(module, "DownloadEvent", _Unwritable)

    response = await client.get(f"/v1/generations/{generation_id}/audio?download=true")

    assert response.status_code == 200, "the customer still gets their song"
    assert len(response.content) > 0
    assert await _events(app) == 0


async def test_the_count_reaches_the_console(app: FastAPI, client: AsyncClient) -> None:
    generation_id = await _a_song(client)
    await client.get(f"/v1/generations/{generation_id}/audio?download=true")

    operator = await signed_up_client(app, "downloads-admin@example.com", UserRole.ADMIN)
    try:
        body = (await operator.get("/v1/admin/analytics/downloads")).json()
        detail = (
            await operator.get(f"/v1/admin/users/{client.user_id}")  # type: ignore[attr-defined]
        ).json()
    finally:
        await operator.aclose()

    assert body["total"] == 1
    assert detail["activity"]["downloads"] == 1


async def test_a_download_of_an_unknown_generation_records_nothing(
    app: FastAPI, client: AsyncClient
) -> None:
    response = await client.get(f"/v1/generations/{uuid.uuid4()}/audio?download=true")

    assert response.status_code == 404
    assert await _events(app) == 0
