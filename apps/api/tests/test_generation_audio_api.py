"""MASTER audio delivery endpoint tests.

The app fixture runs the real GenerationService inline, so a completed
generation here has a genuine stored MASTER WAV on disk — these assert
against real bytes, not stubs.
"""

import hashlib
import uuid

import pytest

from luber_api.routes.generations import build_download_filename
from luber_audio_utils import AudioStorageError, LocalAudioStorage

CREATE_PAYLOAD = {
    "title": "Midnight Window",
    "prompt": "Dreamy Korean indie pop",
    "lyrics": "[Verse]\n오늘 밤 너를 생각해",
    "vocal_gender": "female",
    "duration": 30,
    "language": "ko",
}


async def _completed_generation(client) -> dict:
    created = (await client.post("/v1/generations", json=CREATE_PAYLOAD)).json()
    body = (await client.get(f"/v1/generations/{created['generation_id']}")).json()
    assert body["status"] == "COMPLETED"
    return body


async def test_serves_master_audio_with_correct_headers(client):
    generation = await _completed_generation(client)

    resp = await client.get(f"/v1/generations/{generation['id']}/audio")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert int(resp.headers["content-length"]) == len(resp.content)
    assert resp.headers["content-disposition"].startswith("inline")
    assert resp.content[:4] == b"RIFF"
    assert resp.content[8:12] == b"WAVE"


async def test_served_bytes_match_master_sha256(client):
    """The browser must receive exactly the master the backend recorded.

    Which master that is depends on whether finishing acted, so the row
    is resolved the way the endpoint resolves it.
    """
    generation = await _completed_generation(client)
    assets = generation["audio_assets"]
    master = next(
        (a for a in assets if a["asset_type"] == "FINISHED_MASTER"),
        next(a for a in assets if a["asset_type"] == "MASTER"),
    )

    resp = await client.get(f"/v1/generations/{generation['id']}/audio")

    assert hashlib.sha256(resp.content).hexdigest() == master["sha256"]
    assert len(resp.content) == master["file_size"]


async def test_download_flag_sets_attachment_disposition(client):
    generation = await _completed_generation(client)

    resp = await client.get(
        f"/v1/generations/{generation['id']}/audio", params={"download": "true"}
    )

    disposition = resp.headers["content-disposition"]
    assert disposition.startswith("attachment")
    assert "LUBER%20-%20Midnight%20Window.wav" in disposition


async def test_missing_generation_returns_404(client):
    resp = await client.get(f"/v1/generations/{uuid.uuid4()}/audio")
    assert resp.status_code == 404


async def test_generation_without_master_returns_404(client, app):
    """A generation that has not produced a MASTER yet is a clean 404."""
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

    resp = await client.get(f"/v1/generations/{generation_id}/audio")
    assert resp.status_code == 404


async def test_response_never_leaks_filesystem_paths(client, tmp_path):
    """No response body or header may expose a local absolute path."""
    generation = await _completed_generation(client)
    gid = generation["id"]

    audio = await client.get(f"/v1/generations/{gid}/audio")
    missing = await client.get(f"/v1/generations/{uuid.uuid4()}/audio")
    detail = await client.get(f"/v1/generations/{gid}")

    leaked = str(tmp_path)
    for resp in (missing, detail):
        assert leaked not in resp.text
        assert "/Users/" not in resp.text
        assert "storage_key" not in resp.text or "audio/" in resp.text
    for resp in (audio, missing, detail):
        joined = " ".join(f"{k}: {v}" for k, v in resp.headers.items())
        assert leaked not in joined
        assert "/Users/" not in joined


async def test_detail_response_exposes_no_absolute_path(client):
    generation = await _completed_generation(client)
    master = next(a for a in generation["audio_assets"] if a["asset_type"] == "MASTER")
    # storage_key is a relative key, never a filesystem location.
    assert master["storage_key"].startswith("audio/")
    assert not master["storage_key"].startswith("/")


# ── storage boundary: path traversal ──────────────────────────────────


@pytest.mark.parametrize(
    "malicious_key",
    [
        "../../../../etc/passwd",
        "audio/../../../etc/passwd",
        "/etc/passwd",
        "audio/../../secrets.env",
    ],
)
def test_resolve_path_rejects_traversal(tmp_path, malicious_key):
    storage = LocalAudioStorage(tmp_path / "store")
    with pytest.raises(AudioStorageError, match="escapes storage root"):
        storage.resolve_path(malicious_key)


def test_resolve_path_allows_normal_key(tmp_path):
    storage = LocalAudioStorage(tmp_path / "store")
    key = f"audio/{uuid.uuid4()}/master.wav"
    resolved = storage.resolve_path(key)
    assert str(resolved).startswith(str((tmp_path / "store").resolve()))


# ── download filename safety ──────────────────────────────────────────


def test_download_filename_is_human_readable():
    gid = uuid.uuid4()
    assert build_download_filename("Midnight Window", gid) == "LUBER - Midnight Window.wav"
    # Surrounding and repeated whitespace is collapsed, punctuation kept.
    assert build_download_filename("  Hello   World!! ", gid) == "LUBER - Hello World!!.wav"


@pytest.mark.parametrize(
    "hostile_title",
    ["../../etc/passwd", "..\\..\\windows", "/absolute/path", 'a"; rm -rf /', "\n\rInjected"],
)
def test_download_filename_strips_path_and_control_characters(hostile_title):
    name = build_download_filename(hostile_title, uuid.uuid4())
    assert "/" not in name.removesuffix(".wav")
    assert "\\" not in name
    assert ".." not in name
    assert "\n" not in name and "\r" not in name
    assert '"' not in name
    assert name.endswith(".wav")


def test_download_filename_keeps_unicode_titles():
    gid = uuid.uuid4()
    assert build_download_filename("오늘 밤", gid) == "LUBER - 오늘 밤.wav"


def test_download_filename_is_length_bounded():
    name = build_download_filename("word " * 200, uuid.uuid4())
    title_part = name.removesuffix(".wav").removeprefix("LUBER - ")
    assert len(title_part) <= 60
