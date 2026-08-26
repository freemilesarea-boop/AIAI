"""Authorized delivery of MASTER and PREVIEW assets."""

import hashlib

import pytest
from asset_fixtures import asset_storage_keys

CREATE_PAYLOAD = {
    "title": "Midnight Window",
    "prompt": "Dreamy Korean indie pop",
    "lyrics": "[Verse]\n오늘 밤 너를 생각해",
    "vocal_gender": "female",
    "duration": 30,
    "language": "ko",
}


async def _completed(client) -> dict:
    created = (await client.post("/v1/generations", json=CREATE_PAYLOAD)).json()
    body = (await client.get(f"/v1/generations/{created['generation_id']}")).json()
    assert body["status"] == "COMPLETED"
    return body


# ── 11/12. authorized WAV and MP3 download ────────────────────────────


async def test_authorized_master_wav_download(client):
    generation = await _completed(client)

    resp = await client.get(
        f"/v1/generations/{generation['id']}/audio",
        params={"asset": "master", "download": "true"},
    )

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.headers["content-disposition"].startswith("attachment")
    # Human-readable name, RFC 5987 encoded because of the spaces.
    assert "LUBER%20-%20Midnight%20Window.wav" in resp.headers["content-disposition"]
    assert int(resp.headers["content-length"]) == len(resp.content)
    assert resp.content[:4] == b"RIFF" and resp.content[8:12] == b"WAVE"


async def test_authorized_preview_mp3_download(client):
    generation = await _completed(client)

    resp = await client.get(
        f"/v1/generations/{generation['id']}/audio",
        params={"asset": "preview", "download": "true"},
    )

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    assert "LUBER%20-%20Midnight%20Window.mp3" in resp.headers["content-disposition"]
    assert len(resp.content) > 0
    # MPEG frame sync or ID3 header.
    assert resp.content[0] == 0xFF or resp.content[:3] == b"ID3"


async def test_master_is_the_default_asset(client):
    """Phase 3 clients that omit ?asset still get the master."""
    generation = await _completed(client)
    resp = await client.get(f"/v1/generations/{generation['id']}/audio")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"


async def test_preview_plays_inline_by_default(client):
    generation = await _completed(client)
    resp = await client.get(
        f"/v1/generations/{generation['id']}/audio", params={"asset": "preview"}
    )
    assert resp.headers["content-disposition"].startswith("inline")


# ── 5/6. delivered bytes match the recorded digests ───────────────────


def delivered_asset(generation: dict, asset: str) -> dict:
    """The row the API will serve for ``?asset=``.

    Mirrors the server's delivery selector rather than assuming a type:
    a test that hardcoded "MASTER" would keep passing while the endpoint
    served something else entirely.
    """
    assets = generation["audio_assets"]
    if asset == "preview":
        return next(a for a in assets if a["asset_type"] == "PREVIEW")
    for wanted in ("FINISHED_MASTER", "MASTER"):
        match = next((a for a in assets if a["asset_type"] == wanted), None)
        if match is not None:
            return match
    raise AssertionError("generation has no master asset")


@pytest.mark.parametrize("asset", ["master", "preview"])
async def test_delivered_bytes_match_recorded_sha256(client, asset):
    generation = await _completed(client)
    record = delivered_asset(generation, asset)

    resp = await client.get(f"/v1/generations/{generation['id']}/audio", params={"asset": asset})

    assert hashlib.sha256(resp.content).hexdigest() == record["sha256"]
    assert len(resp.content) == record["file_size"]


async def test_master_download_serves_the_finished_master_when_one_exists(client, app):
    """Phase 14B: `?asset=master` means "the master you should have".

    The raw master stays available as its own asset, but it is not what
    a listener is served once the engine has improved on it.
    """
    generation = await _completed(client)
    kinds = {a["asset_type"] for a in generation["audio_assets"]}
    if "FINISHED_MASTER" not in kinds:
        pytest.skip("finishing took no action on this fixture")

    finished = next(a for a in generation["audio_assets"] if a["asset_type"] == "FINISHED_MASTER")
    raw = next(a for a in generation["audio_assets"] if a["asset_type"] == "MASTER")
    keys = await asset_storage_keys(app, generation["id"])
    assert keys["FINISHED_MASTER"] != keys["MASTER"]

    resp = await client.get(f"/v1/generations/{generation['id']}/audio", params={"asset": "master"})
    assert hashlib.sha256(resp.content).hexdigest() == finished["sha256"]
    assert hashlib.sha256(resp.content).hexdigest() != raw["sha256"]


async def test_master_and_preview_are_distinct_objects(client):
    generation = await _completed(client)
    master = await client.get(
        f"/v1/generations/{generation['id']}/audio", params={"asset": "master"}
    )
    preview = await client.get(
        f"/v1/generations/{generation['id']}/audio", params={"asset": "preview"}
    )
    assert master.content != preview.content
    # The preview is compressed, so materially smaller than 24-bit PCM.
    assert len(preview.content) < len(master.content)


# ── 2/4. recorded asset metadata ──────────────────────────────────────


async def test_master_metadata_matches_production_contract(client):
    generation = await _completed(client)
    master = next(a for a in generation["audio_assets"] if a["asset_type"] == "MASTER")

    assert master["format"] == "wav"
    assert master["mime_type"] == "audio/wav"
    assert master["file_extension"] == "wav"
    assert master["sample_rate"] == 48000
    assert master["channels"] == 2
    assert master["bit_depth"] == 24
    assert master["duration"] > 0
    assert master["file_size"] > 0
    assert len(master["sha256"]) == 64
    assert master["created_at"]


async def test_preview_metadata_matches_contract(client):
    generation = await _completed(client)
    preview = next(a for a in generation["audio_assets"] if a["asset_type"] == "PREVIEW")

    assert preview["format"] == "mp3"
    assert preview["mime_type"] == "audio/mpeg"
    assert preview["file_extension"] == "mp3"
    assert preview["sample_rate"] == 48000
    assert preview["channels"] == 2
    assert preview["bitrate"] == 320000
    assert preview["duration"] > 0
    assert len(preview["sha256"]) == 64


# ── 10. unauthorized download rejection ───────────────────────────────


async def test_a_stranger_cannot_download_another_users_audio(client, client_b):
    """The property the old X-User-Id policy tried to provide, now real.

    Identity comes from the session and nowhere else, so there is no
    header to set and no value a caller can choose.
    """
    generation = await _completed(client)
    response = await client_b.get(f"/v1/generations/{generation['id']}/audio")
    assert response.status_code == 404


async def test_anonymous_download_is_refused_before_ownership_matters(anon_client, client):
    generation = await _completed(client)
    response = await anon_client.get(f"/v1/generations/{generation['id']}/audio")
    assert response.status_code == 401


async def test_unauthorized_response_is_indistinguishable_from_missing(client, client_b):
    """Ownership must not leak whether a generation exists."""
    import uuid as _uuid

    generation = await _completed(client)
    forbidden = await client_b.get(f"/v1/generations/{generation['id']}/audio")
    missing = await client_b.get(f"/v1/generations/{_uuid.uuid4()}/audio")

    assert forbidden.status_code == missing.status_code == 404
    assert forbidden.json() == missing.json()


async def test_the_x_user_id_header_cannot_grant_access(client, client_b):
    """Spoofing the owner's id must change nothing.

    The header is no longer read anywhere in product authorization; this
    pins that it stays that way.
    """
    generation = await _completed(client)
    spoofed = await client_b.get(
        f"/v1/generations/{generation['id']}/audio",
        headers={"X-User-Id": str(client.user_id)},
    )
    assert spoofed.status_code == 404


async def test_a_malformed_user_header_changes_nothing(client):
    """It is ignored, so even a hostile value is inert."""
    generation = await _completed(client)
    response = await client.get(
        f"/v1/generations/{generation['id']}/audio",
        headers={"X-User-Id": "not-a-uuid'; DROP TABLE generations;--"},
    )
    assert response.status_code == 200
