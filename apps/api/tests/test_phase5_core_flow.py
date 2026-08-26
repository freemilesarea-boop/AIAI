"""The core loop, walked as a user walks it.

Create → generation accepted → completion → the song is in *my* library
without a save step → playable → downloadable. Then the same walk from a
second account, to show the two libraries never meet.

The ownership matrix is covered adversarially elsewhere; this asks a
different question — does the product loop close at all, and does a
successful generation end up in the right library by itself.
"""

from __future__ import annotations

import pytest

PAYLOAD = {
    "title": "첫 곡",
    "prompt": "따뜻한 일렉트릭 피아노와 부드러운 드럼",
    "lyrics": "",
    "vocal_gender": "instrumental",
    "duration": 30,
    "language": "ko",
    "instrumental": True,
}


async def create(http, title: str) -> str:
    response = await http.post("/v1/generations", json={**PAYLOAD, "title": title})
    assert response.status_code == 202, response.text
    return str(response.json()["generation_id"])


async def library(http) -> list[dict]:
    response = await http.get("/v1/generations?limit=50")
    assert response.status_code == 200
    return list(response.json()["items"])


@pytest.mark.asyncio
async def test_a_generation_lands_in_its_creators_library_by_itself(client):
    """No save step. If BOORDA made it, it is already in the library."""
    before = await library(client)
    assert before == []

    generation_id = await create(client, "첫 곡")

    after = await library(client)
    assert [item["id"] for item in after] == [generation_id]
    assert after[0]["title"] == "첫 곡"


@pytest.mark.asyncio
async def test_the_submission_is_immediately_addressable(client):
    """202 hands back an id the client can poll at once, not later."""
    generation_id = await create(client, "폴링 대상")
    detail = await client.get(f"/v1/generations/{generation_id}")
    assert detail.status_code == 200
    assert detail.json()["id"] == generation_id
    # A status the UI can translate, present from the first read.
    assert detail.json()["status"]


@pytest.mark.asyncio
async def test_the_newest_song_is_first(client):
    await create(client, "먼저 만든 곡")
    second = await create(client, "나중에 만든 곡")
    items = await library(client)
    assert items[0]["id"] == second, "library defaults to newest first"


@pytest.mark.asyncio
async def test_resubmitting_one_key_does_not_make_a_second_song(client):
    """A refresh that replays the request must not double-charge the user."""
    key = "phase5-idempotency"
    first = await client.post("/v1/generations", json=PAYLOAD, headers={"Idempotency-Key": key})
    second = await client.post("/v1/generations", json=PAYLOAD, headers={"Idempotency-Key": key})
    assert first.status_code == 202 and second.status_code in (200, 202)
    assert first.json()["generation_id"] == second.json()["generation_id"]
    assert len(await library(client)) == 1


@pytest.mark.asyncio
async def test_two_accounts_walk_the_same_loop_and_never_meet(client, client_b):
    a1 = await create(client, "A의 곡")
    b1 = await create(client_b, "B의 곡")

    assert [item["title"] for item in await library(client)] == ["A의 곡"]
    assert [item["title"] for item in await library(client_b)] == ["B의 곡"]

    # B knows A's id and it buys nothing, on every step of the loop.
    assert (await client_b.get(f"/v1/generations/{a1}")).status_code == 404
    assert (await client_b.get(f"/v1/generations/{a1}/audio")).status_code == 404
    assert (await client_b.get(f"/v1/generations/{a1}/audio?download=true")).status_code == 404
    # And the reverse.
    assert (await client.get(f"/v1/generations/{b1}")).status_code == 404


@pytest.mark.asyncio
async def test_audio_and_download_require_a_session(anon_client, client):
    generation_id = await create(client, "비공개 곡")
    for path in (
        f"/v1/generations/{generation_id}",
        f"/v1/generations/{generation_id}/audio",
        f"/v1/generations/{generation_id}/audio?download=true",
        "/v1/generations",
    ):
        response = await anon_client.get(path)
        assert response.status_code == 401, f"{path} served a guest"


@pytest.mark.asyncio
async def test_a_brand_new_account_sees_an_empty_library_not_an_error(client):
    response = await client.get("/v1/generations?limit=50")
    assert response.status_code == 200
    body = response.json()
    assert body["items"] == []
    assert body.get("total") == 0


#: Everything a client-facing response must never contain: how the
#: server finds the bytes, and anything that would let a client try to
#: fetch them without going through the authenticated route.
FORBIDDEN_IN_RESPONSES = (
    "storage_key",
    "/Users/",
    "/var/",
    "/tmp/",
    "s3://",
    "https://s3",
    ".amazonaws.com",
    "aws_access_key",
    "aws_secret",
    "AKIA",
    "X-Amz-Signature",
    "presigned",
)


@pytest.mark.asyncio
async def test_no_client_facing_response_reveals_where_the_bytes_live(client):
    """Audio is addressed by generation id, and by nothing else.

    The server resolves the storage key itself, from a generation whose
    ownership it has already checked. A client never needs the key, so a
    client never receives it — not the key, not a path, not a bucket, not
    a credential, not a signed URL sitting in a JSON body.
    """
    generation_id = await create(client, "노출 검사")
    for path in (
        f"/v1/generations/{generation_id}",
        "/v1/generations",
        f"/v1/generations/{generation_id}/lineage",
        "/v1/projects",
    ):
        response = await client.get(path)
        assert response.status_code == 200, path
        body = response.text
        for leak in FORBIDDEN_IN_RESPONSES:
            assert leak not in body, f"{path} exposed {leak!r}"


@pytest.mark.asyncio
async def test_the_asset_still_carries_what_a_player_needs(client):
    """Removing the key must not remove the facts playback depends on."""
    generation_id = await create(client, "재생 정보")
    body = (await client.get(f"/v1/generations/{generation_id}")).json()
    assets = body.get("audio_assets") or []
    assert assets, "a completed generation should carry at least one asset"
    for asset in assets:
        for field in ("asset_type", "format", "mime_type", "duration", "file_size"):
            assert field in asset, f"asset lost {field}"
        assert "storage_key" not in asset
