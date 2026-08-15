"""Generation endpoint tests, including the full Phase 1 integration flow.

The app fixture wires the InlineGenerationRunner, so POST drives the
real GenerationService + MockGenerationProvider + LocalAudioStorage
against the committed fixture WAV — no fabricated results.
"""

import asyncio
import hashlib
import uuid

CREATE_PAYLOAD = {
    "title": "TEST SONG",
    "prompt": "Dreamy Korean indie pop",
    "lyrics": "[Verse]\n테스트 가사",
    "vocal_gender": "female",
    "duration": 30,
    "language": "ko",
}


async def test_post_creates_generation(client):
    resp = await client.post("/v1/generations", json=CREATE_PAYLOAD)
    assert resp.status_code == 202
    body = resp.json()
    uuid.UUID(body["generation_id"])  # valid UUID
    assert body["status"] == "QUEUED"


async def test_post_validation_rejects_bad_payload(client):
    bad = dict(CREATE_PAYLOAD, duration=5)  # below 10s minimum
    assert (await client.post("/v1/generations", json=bad)).status_code == 422
    bad = dict(CREATE_PAYLOAD, title="")
    assert (await client.post("/v1/generations", json=bad)).status_code == 422
    bad = dict(CREATE_PAYLOAD, vocal_gender="robot")
    assert (await client.post("/v1/generations", json=bad)).status_code == 422


async def test_get_generation_by_id(client):
    created = (await client.post("/v1/generations", json=CREATE_PAYLOAD)).json()
    resp = await client.get(f"/v1/generations/{created['generation_id']}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["title"] == "TEST SONG"
    assert body["lyrics"] == "[Verse]\n테스트 가사"  # Korean + section tag preserved
    assert body["vocal_gender"] == "female"
    assert body["duration_requested"] == 30
    assert body["language"] == "ko"


async def test_get_missing_generation_404(client):
    resp = await client.get(f"/v1/generations/{uuid.uuid4()}")
    assert resp.status_code == 404


async def test_list_generations_with_pagination(client):
    for i in range(3):
        await client.post("/v1/generations", json=dict(CREATE_PAYLOAD, title=f"Song {i}"))
    resp = await client.get("/v1/generations", params={"limit": 2, "offset": 0})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 3
    assert body["limit"] == 2
    assert body["offset"] == 0
    assert len(body["items"]) == 2


async def test_delete_generation(client, tmp_path):
    created = (await client.post("/v1/generations", json=CREATE_PAYLOAD)).json()
    generation_id = created["generation_id"]

    # Completed generation has a stored master WAV on disk.
    detail = (await client.get(f"/v1/generations/{generation_id}")).json()
    assert detail["status"] == "COMPLETED"
    storage_key = detail["audio_assets"][0]["storage_key"]
    stored = tmp_path / "audio-store" / storage_key
    assert stored.is_file()

    resp = await client.delete(f"/v1/generations/{generation_id}")
    assert resp.status_code == 204
    assert (await client.get(f"/v1/generations/{generation_id}")).status_code == 404
    assert not stored.exists()  # local file lifecycle: removed with the row

    assert (await client.delete(f"/v1/generations/{generation_id}")).status_code == 404


async def test_idempotency_key_returns_same_generation(client):
    key = f"idem-{uuid.uuid4()}"
    first = await client.post(
        "/v1/generations", json=CREATE_PAYLOAD, headers={"Idempotency-Key": key}
    )
    second = await client.post(
        "/v1/generations", json=CREATE_PAYLOAD, headers={"Idempotency-Key": key}
    )
    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["generation_id"] == second.json()["generation_id"]

    listing = (await client.get("/v1/generations")).json()
    assert listing["total"] == 1  # no duplicate row was created


async def test_concurrent_same_idempotency_key_creates_one_generation(client):
    key = f"idem-race-{uuid.uuid4()}"

    async def submit():
        return await client.post(
            "/v1/generations", json=CREATE_PAYLOAD, headers={"Idempotency-Key": key}
        )

    responses = await asyncio.gather(submit(), submit())
    ids = {r.json()["generation_id"] for r in responses}
    assert all(r.status_code == 202 for r in responses)
    assert len(ids) == 1
    assert (await client.get("/v1/generations")).json()["total"] == 1


async def test_oversized_idempotency_key_rejected(client):
    resp = await client.post(
        "/v1/generations", json=CREATE_PAYLOAD, headers={"Idempotency-Key": "x" * 201}
    )
    assert resp.status_code == 422


async def test_instrumental_vocal_gender_sets_flag(client):
    payload = dict(CREATE_PAYLOAD, vocal_gender="instrumental", instrumental=False)
    created = (await client.post("/v1/generations", json=payload)).json()
    detail = (await client.get(f"/v1/generations/{created['generation_id']}")).json()
    assert detail["instrumental"] is True


async def test_full_generation_flow_produces_master_wav(client, tmp_path):
    """Phase 1 acceptance: POST → job → mock provider → real WAV asset → GET."""
    resp = await client.post(
        "/v1/generations",
        json=CREATE_PAYLOAD,
        headers={"Idempotency-Key": f"flow-{uuid.uuid4()}"},
    )
    assert resp.status_code == 202
    generation_id = resp.json()["generation_id"]

    detail = (await client.get(f"/v1/generations/{generation_id}")).json()

    # Lifecycle reached COMPLETED with honest mock provenance.
    assert detail["status"] == "COMPLETED"
    assert detail["provider"] == "mock"
    assert detail["model_name"] == "mock-generation-provider"
    assert detail["model_version"] == "phase1"
    assert detail["duration_actual"] > 0
    assert detail["started_at"] is not None
    assert detail["completed_at"] is not None
    assert detail["error_code"] is None

    # Both delivery assets exist, each with verifiable bytes on disk.
    assets = {a["asset_type"]: a for a in detail["audio_assets"]}
    assert {"MASTER", "PREVIEW"} <= set(assets)
    assert set(assets) <= {"MASTER", "FINISHED_MASTER", "PREVIEW"}

    master = assets["MASTER"]
    assert master["format"] == "wav"
    assert master["mime_type"] == "audio/wav"
    assert master["sample_rate"] == 48000
    assert master["channels"] == 2
    # Post-processing normalizes to the 24-bit production master format.
    assert master["bit_depth"] == 24
    assert master["duration"] > 0
    assert master["storage_key"] == f"audio/{generation_id}/master.wav"

    preview = assets["PREVIEW"]
    assert preview["format"] == "mp3"
    assert preview["mime_type"] == "audio/mpeg"
    assert preview["bitrate"] == 320000
    assert preview["storage_key"] == f"audio/{generation_id}/preview.mp3"

    for asset in (master, preview):
        stored = tmp_path / "audio-store" / asset["storage_key"]
        assert stored.is_file()
        assert stored.stat().st_size == asset["file_size"] > 0
        # Stored SHA256 matches the actual stored bytes.
        assert hashlib.sha256(stored.read_bytes()).hexdigest() == asset["sha256"]
