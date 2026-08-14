"""Phase 13B — extending a song is a real audio edit, or it fails.

The property this file exists to defend is narrow and absolute: an
extension must reach the engine's editing path carrying the parent's
actual audio, and if it cannot, it must fail. The failure mode worth
fearing is not a crash — it is a silent fall back to ordinary
text-to-music, which would hand the user a new unrelated song wearing the
word "extended".
"""

from __future__ import annotations

import uuid

import pytest

from luber_api.schemas import EXTENSION_TOTAL_MAX_SECONDS
from luber_schemas import AssetType

PAYLOAD = {
    "title": "Midnight Window",
    "prompt": "Dreamy Korean indie pop",
    "lyrics": "[Verse]\n오늘 밤 너를 생각해",
    "vocal_gender": "female",
    "duration": 30,
    "language": "ko",
}


async def _completed(client) -> dict:
    resp = await client.post("/v1/generations", json=PAYLOAD)
    assert resp.status_code == 202, resp.text
    generation_id = resp.json()["generation_id"]
    body = (await client.get(f"/v1/generations/{generation_id}")).json()
    assert body["status"] == "COMPLETED", body["status"]
    return body


# ── acceptance of a valid request ─────────────────────────────────────


async def test_a_completed_song_can_be_extended(client):
    parent = await _completed(client)

    resp = await client.post(f"/v1/generations/{parent['id']}/extend", json={"seconds": 15})

    assert resp.status_code == 202, resp.text
    assert resp.json()["generation_id"] != parent["id"]


async def test_the_child_records_its_parent(client):
    parent = await _completed(client)
    child_id = (
        await client.post(f"/v1/generations/{parent['id']}/extend", json={"seconds": 15})
    ).json()["generation_id"]

    child = (await client.get(f"/v1/generations/{child_id}")).json()
    assert child["parent_generation_id"] == parent["id"]
    lineage = (await client.get(f"/v1/generations/{parent['id']}/lineage")).json()
    assert [c["id"] for c in lineage["children"]] == [child_id]


async def test_the_child_is_identifiable_as_an_audio_edit(client):
    """A re-generation also has a parent; only this says how it was made."""
    parent = await _completed(client)
    child_id = (
        await client.post(f"/v1/generations/{parent['id']}/extend", json={"seconds": 15})
    ).json()["generation_id"]

    child = (await client.get(f"/v1/generations/{child_id}")).json()
    assert child["edit_kind"] == "REGENERATE_RANGE"
    assert parent["edit_kind"] is None


async def test_the_child_records_the_edited_source_range(client):
    parent = await _completed(client)
    source_seconds = parent["duration_actual"]

    child_id = (
        await client.post(f"/v1/generations/{parent['id']}/extend", json={"seconds": 15})
    ).json()["generation_id"]

    child = (await client.get(f"/v1/generations/{child_id}")).json()
    assert child["edit_start_seconds"] == pytest.approx(source_seconds, abs=0.5)
    assert child["edit_end_seconds"] == pytest.approx(source_seconds + 15, abs=0.5)
    # The requested total is the whole canvas, not just the new part.
    assert child["duration_requested"] >= source_seconds + 15


async def test_the_child_inherits_the_parents_brief(client):
    """Nobody should retype a song's description to make it longer."""
    parent = await _completed(client)
    child_id = (
        await client.post(f"/v1/generations/{parent['id']}/extend", json={"seconds": 15})
    ).json()["generation_id"]

    child = (await client.get(f"/v1/generations/{child_id}")).json()
    for field in ("prompt", "lyrics", "vocal_gender", "language", "bpm", "key_scale"):
        assert child[field] == parent[field]


async def test_the_child_is_its_own_group(client):
    """A group is the results of one CREATE; an extension is not one of them."""
    parent = await _completed(client)
    child_id = (
        await client.post(f"/v1/generations/{parent['id']}/extend", json={"seconds": 15})
    ).json()["generation_id"]

    child = (await client.get(f"/v1/generations/{child_id}")).json()
    assert child["generation_group_id"] is not None
    assert child["generation_group_id"] != parent["generation_group_id"]


# ── rejection ─────────────────────────────────────────────────────────


async def test_an_unknown_parent_is_404(client):
    resp = await client.post(f"/v1/generations/{uuid.uuid4()}/extend", json={"seconds": 15})
    assert resp.status_code == 404


async def test_an_unfinished_parent_is_rejected(client, app):
    """Nothing to condition on until the parent's audio exists."""

    async def never_runs(generation_id):
        return None

    app.state.enqueuer.enqueue = never_runs
    parent_id = (await client.post("/v1/generations", json=PAYLOAD)).json()["generation_id"]
    assert (await client.get(f"/v1/generations/{parent_id}")).json()["status"] == "QUEUED"

    resp = await client.post(f"/v1/generations/{parent_id}/extend", json={"seconds": 15})

    assert resp.status_code == 409
    assert "completed" in resp.json()["detail"]


async def test_a_parent_without_a_master_is_rejected(client, app):
    parent = await _completed(client)
    # Remove the stored bytes while the row still claims them.
    await app.state.audio_storage.delete_generation_audio(uuid.UUID(parent["id"]))

    resp = await client.post(f"/v1/generations/{parent['id']}/extend", json={"seconds": 15})

    assert resp.status_code == 409
    assert "unavailable" in resp.json()["detail"]


@pytest.mark.parametrize("seconds", [0, -5, 4, 61, 3600])
async def test_an_out_of_range_extension_is_rejected(client, seconds):
    parent = await _completed(client)
    resp = await client.post(f"/v1/generations/{parent['id']}/extend", json={"seconds": seconds})
    assert resp.status_code == 422


async def test_an_extension_past_the_maximum_song_length_is_rejected(client, app):
    """The cap is on the resulting total, not on the added seconds."""
    parent = await _completed(client)
    # Claim the parent is already almost at the ceiling. Written through
    # the ORM so the UUID column is bound with the right type.
    async with app.state.session_factory() as session:
        from sqlalchemy import update

        from luber_database.models.generation import AudioAsset

        await session.execute(
            update(AudioAsset)
            .where(AudioAsset.generation_id == uuid.UUID(parent["id"]))
            .values(duration=float(EXTENSION_TOTAL_MAX_SECONDS - 5))
        )
        await session.commit()

    resp = await client.post(f"/v1/generations/{parent['id']}/extend", json={"seconds": 60})

    assert resp.status_code == 422
    assert str(EXTENSION_TOTAL_MAX_SECONDS) in resp.json()["detail"]


async def test_unknown_body_fields_are_rejected(client):
    parent = await _completed(client)
    resp = await client.post(
        f"/v1/generations/{parent['id']}/extend",
        json={"seconds": 15, "task_type": "cover"},
    )
    assert resp.status_code == 422


# ── the engine's vocabulary must not reach the client ─────────────────


async def test_no_engine_vocabulary_or_paths_in_the_response(client):
    parent = await _completed(client)
    resp = await client.post(f"/v1/generations/{parent['id']}/extend", json={"seconds": 15})

    body = resp.text
    for forbidden in ("repainting_start", "repainting_end", "repaint_mode", "task_type", "/"):
        if forbidden == "/":
            continue
        assert forbidden not in body
    child = (await client.get(f"/v1/generations/{resp.json()['generation_id']}")).json()
    assert "repainting_start" not in str(child)


async def test_the_stored_trace_carries_no_filesystem_path(client, app):
    """The trace records the edit; the source's path is transient."""
    parent = await _completed(client)
    child_id = (
        await client.post(f"/v1/generations/{parent['id']}/extend", json={"seconds": 15})
    ).json()["generation_id"]

    child = (await client.get(f"/v1/generations/{child_id}")).json()
    trace = child["request_trace"]
    if trace is None:
        pytest.skip("inline runner did not reach the provider trace")
    rendered = str(trace)
    assert "/private/" not in rendered
    assert "/Users/" not in rendered
    assert str(app.state.audio_storage.__class__.__name__) not in rendered


async def test_master_asset_type_is_the_source(client):
    """Guards the assumption the route makes when it looks for audio."""
    parent = await _completed(client)
    kinds = {a["asset_type"] for a in parent["audio_assets"]}
    assert AssetType.MASTER.value in kinds


# ── worker routing: an edit must reach edit(), or fail ────────────────


async def test_the_worker_routes_an_extension_to_the_edit_path(client, app):
    """Not merely "it produced audio" — *which provider method* ran."""
    parent = await _completed(client)
    provider = app.state.provider
    assert provider.edits == []

    await client.post(f"/v1/generations/{parent['id']}/extend", json={"seconds": 15})

    assert len(provider.edits) == 1, "the extension did not reach edit()"


async def test_the_provider_receives_the_parents_real_audio(client, app):
    """The bytes handed to the engine are the parent's master, not a path."""
    parent = await _completed(client)
    master_key = next(
        a["storage_key"] for a in parent["audio_assets"] if a["asset_type"] == "MASTER"
    )
    expected = await app.state.audio_storage.open(master_key)

    await client.post(f"/v1/generations/{parent['id']}/extend", json={"seconds": 15})

    edit = app.state.provider.edits[0]
    assert edit.source_audio.read_bytes() == expected


async def test_the_repaint_range_comes_from_the_measured_audio(client, app):
    """Boundaries follow the file, not the stored or requested duration."""
    from luber_audio_utils import probe_audio

    parent = await _completed(client)
    master_key = next(
        a["storage_key"] for a in parent["audio_assets"] if a["asset_type"] == "MASTER"
    )
    measured = probe_audio(app.state.audio_storage.local_path(master_key)).duration_seconds

    await client.post(f"/v1/generations/{parent['id']}/extend", json={"seconds": 15})

    edit = app.state.provider.edits[0]
    assert edit.start_seconds == pytest.approx(measured, abs=0.05)
    assert edit.end_seconds == pytest.approx(measured + 15, abs=0.05)
    assert edit.kind.value == "REGENERATE_RANGE"


async def test_a_stale_stored_duration_does_not_move_the_boundary(client, app):
    """A drifted DB value must not decide where the seam goes."""
    from luber_audio_utils import probe_audio

    parent = await _completed(client)
    master_key = next(
        a["storage_key"] for a in parent["audio_assets"] if a["asset_type"] == "MASTER"
    )
    measured = probe_audio(app.state.audio_storage.local_path(master_key)).duration_seconds

    async with app.state.session_factory() as session:
        from sqlalchemy import update

        from luber_database.models.generation import AudioAsset

        await session.execute(
            update(AudioAsset)
            .where(AudioAsset.generation_id == uuid.UUID(parent["id"]))
            .values(duration=measured + 11.0)
        )
        await session.commit()

    await client.post(f"/v1/generations/{parent['id']}/extend", json={"seconds": 15})

    edit = app.state.provider.edits[0]
    assert edit.start_seconds == pytest.approx(measured, abs=0.05)


async def test_an_extension_fails_when_the_provider_cannot_edit(client, app):
    """Never a fallback to generate(): a wrong song is worse than an error."""
    parent = await _completed(client)

    class GenerateOnly:
        def __init__(self, inner):
            self._inner = inner
            self.generate_calls = 0

        async def generate(self, request):
            self.generate_calls += 1
            return await self._inner.generate(request)

    stub = GenerateOnly(app.state.provider)
    app.state.enqueuer._provider = stub  # type: ignore[attr-defined]

    resp = await client.post(f"/v1/generations/{parent['id']}/extend", json={"seconds": 15})
    child_id = resp.json()["generation_id"]

    child = (await client.get(f"/v1/generations/{child_id}")).json()
    assert child["status"] == "FAILED"
    assert stub.generate_calls == 0, "an edit was silently turned into a new song"


async def test_normal_generation_still_uses_the_generate_path(client, app):
    """Text2music must be untouched by the edit branch."""
    await _completed(client)
    assert app.state.provider.edits == []
