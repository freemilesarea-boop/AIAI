"""Authorized delivery of MASTER and PREVIEW assets."""

import hashlib
import uuid

import pytest

from luber_api.routes.generations import build_download_filename, caller_may_access

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


@pytest.mark.parametrize(("asset", "asset_type"), [("master", "MASTER"), ("preview", "PREVIEW")])
async def test_delivered_bytes_match_recorded_sha256(client, asset, asset_type):
    generation = await _completed(client)
    record = next(a for a in generation["audio_assets"] if a["asset_type"] == asset_type)

    resp = await client.get(f"/v1/generations/{generation['id']}/audio", params={"asset": asset})

    assert hashlib.sha256(resp.content).hexdigest() == record["sha256"]
    assert len(resp.content) == record["file_size"]


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


async def test_owned_generation_rejects_anonymous_download(client, app):
    """Once a generation has an owner, a stranger cannot fetch its audio."""
    from luber_database import GenerationRepository

    generation = await _completed(client)
    owner = uuid.uuid4()
    async with app.state.session_factory() as session:
        repo = GenerationRepository(session)
        row = await repo.get_generation(uuid.UUID(generation["id"]))
        row.user_id = owner
        await session.commit()

    anonymous = await client.get(f"/v1/generations/{generation['id']}/audio")
    assert anonymous.status_code == 404

    wrong_user = await client.get(
        f"/v1/generations/{generation['id']}/audio",
        headers={"X-User-Id": str(uuid.uuid4())},
    )
    assert wrong_user.status_code == 404

    owner_request = await client.get(
        f"/v1/generations/{generation['id']}/audio",
        headers={"X-User-Id": str(owner)},
    )
    assert owner_request.status_code == 200
    assert owner_request.headers["content-type"] == "audio/wav"


async def test_unauthorized_response_is_indistinguishable_from_missing(client, app):
    """Ownership must not leak whether a generation exists."""
    from luber_database import GenerationRepository

    generation = await _completed(client)
    async with app.state.session_factory() as session:
        repo = GenerationRepository(session)
        row = await repo.get_generation(uuid.UUID(generation["id"]))
        row.user_id = uuid.uuid4()
        await session.commit()

    forbidden = await client.get(f"/v1/generations/{generation['id']}/audio")
    missing = await client.get(f"/v1/generations/{uuid.uuid4()}/audio")

    assert forbidden.status_code == missing.status_code == 404
    assert forbidden.json() == missing.json()


async def test_malformed_user_header_is_treated_as_anonymous(client, app):
    from luber_database import GenerationRepository

    generation = await _completed(client)
    async with app.state.session_factory() as session:
        repo = GenerationRepository(session)
        row = await repo.get_generation(uuid.UUID(generation["id"]))
        row.user_id = uuid.uuid4()
        await session.commit()

    resp = await client.get(
        f"/v1/generations/{generation['id']}/audio",
        headers={"X-User-Id": "not-a-uuid'; DROP TABLE generations;--"},
    )
    assert resp.status_code == 404


def test_caller_may_access_policy():
    class _Gen:
        user_id = None

    unowned = _Gen()
    assert caller_may_access(unowned, None) is True
    assert caller_may_access(unowned, uuid.uuid4()) is True

    owner = uuid.uuid4()
    owned = _Gen()
    owned.user_id = owner
    assert caller_may_access(owned, owner) is True
    assert caller_may_access(owned, None) is False
    assert caller_may_access(owned, uuid.uuid4()) is False


# ── missing assets / bad input ────────────────────────────────────────


async def test_missing_generation_returns_404(client):
    resp = await client.get(f"/v1/generations/{uuid.uuid4()}/audio")
    assert resp.status_code == 404


async def test_unknown_asset_kind_is_rejected(client):
    generation = await _completed(client)
    resp = await client.get(f"/v1/generations/{generation['id']}/audio", params={"asset": "stem"})
    assert resp.status_code == 422


async def test_generation_without_assets_returns_404(client, app):
    from luber_database import GenerationRepository
    from luber_schemas import GenerationStatus

    async with app.state.session_factory() as session:
        repo = GenerationRepository(session)
        generation = await repo.create_generation(
            title="No Audio",
            prompt="p",
            lyrics="",
            vocal_gender="female",
            duration_requested=30,
            seed=None,
            language="ko",
            instrumental=False,
            status=GenerationStatus.QUEUED.value,
            idempotency_key=None,
        )
        generation_id = generation.id

    for asset in ("master", "preview"):
        resp = await client.get(f"/v1/generations/{generation_id}/audio", params={"asset": asset})
        assert resp.status_code == 404


async def test_mime_extension_mismatch_is_refused(client, app):
    """A tampered asset row must not be served under a wrong media type."""
    from luber_database import GenerationRepository

    generation = await _completed(client)
    async with app.state.session_factory() as session:
        repo = GenerationRepository(session)
        row = await repo.get_generation(uuid.UUID(generation["id"]))
        master = next(a for a in row.audio_assets if a.asset_type == "MASTER")
        master.mime_type = "text/html"
        await session.commit()

    resp = await client.get(f"/v1/generations/{generation['id']}/audio")
    assert resp.status_code == 404


# ── no path or infrastructure leakage ─────────────────────────────────


async def test_responses_never_leak_filesystem_paths(client, tmp_path):
    generation = await _completed(client)
    gid = generation["id"]

    responses = [
        await client.get(f"/v1/generations/{gid}/audio", params={"asset": "master"}),
        await client.get(f"/v1/generations/{gid}/audio", params={"asset": "preview"}),
        await client.get(f"/v1/generations/{uuid.uuid4()}/audio"),
        await client.get(f"/v1/generations/{gid}"),
    ]
    for resp in responses:
        headers = " ".join(f"{k}: {v}" for k, v in resp.headers.items())
        assert str(tmp_path) not in headers
        assert "/Users/" not in headers
        if resp.headers.get("content-type", "").startswith("application/json"):
            assert "/Users/" not in resp.text
            assert str(tmp_path) not in resp.text


async def test_storage_keys_stay_relative(client):
    generation = await _completed(client)
    for asset in generation["audio_assets"]:
        assert asset["storage_key"].startswith("audio/")
        assert not asset["storage_key"].startswith("/")
        assert ".." not in asset["storage_key"]


# ── filename derivation ───────────────────────────────────────────────


def test_download_filename_uses_the_asset_extension():
    gid = uuid.uuid4()
    assert build_download_filename("Midnight Window", gid, "wav") == "LUBER - Midnight Window.wav"
    assert build_download_filename("Midnight Window", gid, "mp3") == "LUBER - Midnight Window.mp3"


@pytest.mark.parametrize(
    "hostile", ["../../etc/passwd", "/absolute/path", 'a"; rm -rf /', "\n\rInjected", "..\\..\\win"]
)
def test_download_filename_neutralizes_hostile_titles(hostile):
    name = build_download_filename(hostile, uuid.uuid4(), "mp3")
    assert "/" not in name and "\\" not in name and ".." not in name
    assert "\n" not in name and "\r" not in name and '"' not in name
    assert name.endswith(".mp3")


def test_download_filename_neutralizes_hostile_extension():
    name = build_download_filename("Song", uuid.uuid4(), "../../evil")
    assert name == "LUBER - Song.evil"


def test_download_filename_keeps_a_korean_title_readable():
    """A Korean title must survive to the user's downloads folder.

    Phase 3 slugged to ASCII, which turned every Korean track into
    ``luber-track-1a2b3c4d`` — unusable for this product's main audience.
    Starlette emits ``filename*=utf-8''`` for these, which browsers decode.
    """
    gid = uuid.uuid4()
    assert build_download_filename("오늘 밤", gid, "wav") == "LUBER - 오늘 밤.wav"


def test_download_filename_falls_back_when_a_title_sanitises_to_nothing():
    gid = uuid.uuid4()
    assert build_download_filename("///", gid, "wav") == f"LUBER - track-{gid.hex[:8]}.wav"
