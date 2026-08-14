"""Phase 13C — replacing an interior span is real inpainting, or it fails.

Extension could be checked loosely: the song got longer, so something
happened. Interior replacement cannot. A full regeneration of the same
length would also "work" by every shallow measure, so the tests here pin
the things that distinguish an edit from a re-run — the range the engine
receives, the length it must keep, and the fact that a request which
would preserve nothing is refused rather than quietly turned into a new
song.
"""

from __future__ import annotations

import uuid

import pytest

from luber_api.schemas import MIN_PRESERVED_SECONDS, MIN_REPLACE_SECONDS

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
    body = (await client.get(f"/v1/generations/{resp.json()['generation_id']}")).json()
    assert body["status"] == "COMPLETED"
    return body


async def _replace(client, parent_id: str, **body):
    return await client.post(f"/v1/generations/{parent_id}/replace-range", json=body)


#: The committed fixture is 2.0s, so the interior span used throughout is
#: 0.5-1.5s: a full second replaced with a full second preserved, which is
#: exactly the minimum the validator allows. Ranges are written relative
#: to these names rather than hard-coded, so a longer fixture would not
#: silently invalidate the suite.
SPAN_START = 0.5
SPAN_END = 1.5


# ── acceptance ────────────────────────────────────────────────────────


async def test_an_interior_span_can_be_replaced(client):
    parent = await _completed(client)
    resp = await _replace(client, parent["id"], start_seconds=SPAN_START, end_seconds=SPAN_END)
    assert resp.status_code == 202, resp.text
    assert resp.json()["generation_id"] != parent["id"]


async def test_the_child_keeps_the_parents_length(client):
    """An extension makes the song longer; a replacement must not."""
    parent = await _completed(client)
    child_id = (
        await _replace(client, parent["id"], start_seconds=SPAN_START, end_seconds=SPAN_END)
    ).json()["generation_id"]

    child = (await client.get(f"/v1/generations/{child_id}")).json()
    assert child["duration_requested"] == pytest.approx(parent["duration_actual"], abs=1)
    assert child["duration_actual"] == pytest.approx(parent["duration_actual"], abs=0.5)


async def test_the_child_records_the_replaced_span(client):
    parent = await _completed(client)
    child_id = (
        await _replace(client, parent["id"], start_seconds=SPAN_START, end_seconds=SPAN_END)
    ).json()["generation_id"]

    child = (await client.get(f"/v1/generations/{child_id}")).json()
    assert child["edit_kind"] == "REPLACE_RANGE"
    assert child["edit_start_seconds"] == pytest.approx(SPAN_START)
    assert child["edit_end_seconds"] == pytest.approx(SPAN_END)
    assert child["parent_generation_id"] == parent["id"]


async def test_replacement_is_distinguishable_from_extension(client):
    """Both are edits of the same parent; lineage must tell them apart."""
    parent = await _completed(client)
    extended = (
        await client.post(f"/v1/generations/{parent['id']}/extend", json={"seconds": 15})
    ).json()["generation_id"]
    replaced = (
        await _replace(client, parent["id"], start_seconds=SPAN_START, end_seconds=SPAN_END)
    ).json()["generation_id"]

    kinds = {
        gid: (await client.get(f"/v1/generations/{gid}")).json()["edit_kind"]
        for gid in (extended, replaced)
    }
    assert kinds[extended] == "EXTEND"
    assert kinds[replaced] == "REPLACE_RANGE"

    lineage = (await client.get(f"/v1/generations/{parent['id']}/lineage")).json()
    assert {c["id"] for c in lineage["children"]} == {extended, replaced}


async def test_the_parent_row_is_not_mutated(client):
    parent = await _completed(client)
    before = dict(parent)

    await _replace(
        client, parent["id"], start_seconds=SPAN_START, end_seconds=SPAN_END, prompt="new brief"
    )

    after = (await client.get(f"/v1/generations/{parent['id']}")).json()
    for field in ("prompt", "lyrics", "duration_actual", "edit_kind", "seed"):
        assert after[field] == before[field]


async def test_an_optional_prompt_conditions_the_child_only(client):
    parent = await _completed(client)
    child_id = (
        await _replace(
            client,
            parent["id"],
            start_seconds=SPAN_START,
            end_seconds=SPAN_END,
            prompt="sparse piano outro",
        )
    ).json()["generation_id"]

    child = (await client.get(f"/v1/generations/{child_id}")).json()
    assert child["prompt"] == "sparse piano outro"
    # Lyrics are never re-cut: LUBER has no lyric-to-time alignment.
    assert child["lyrics"] == parent["lyrics"]


async def test_the_parents_brief_is_inherited_when_none_is_given(client):
    parent = await _completed(client)
    child_id = (
        await _replace(client, parent["id"], start_seconds=SPAN_START, end_seconds=SPAN_END)
    ).json()["generation_id"]
    child = (await client.get(f"/v1/generations/{child_id}")).json()
    assert child["prompt"] == parent["prompt"]


# ── rejection ─────────────────────────────────────────────────────────


async def test_an_unknown_parent_is_404(client):
    resp = await _replace(client, str(uuid.uuid4()), start_seconds=1, end_seconds=5)
    assert resp.status_code == 404


async def test_a_negative_start_is_rejected(client):
    parent = await _completed(client)
    assert (
        await _replace(client, parent["id"], start_seconds=-1, end_seconds=1.5)
    ).status_code == 422


async def test_an_end_at_or_before_the_start_is_rejected(client):
    parent = await _completed(client)
    assert (await _replace(client, parent["id"], start_seconds=1, end_seconds=1)).status_code == 422
    assert (
        await _replace(client, parent["id"], start_seconds=1.5, end_seconds=1)
    ).status_code == 422


async def test_a_span_shorter_than_the_crossfade_is_rejected(client):
    """Below ~1s the whole span is boundary blend, so nothing is replaced."""
    parent = await _completed(client)
    resp = await _replace(client, parent["id"], start_seconds=0.5, end_seconds=1.0)
    assert resp.status_code == 422
    assert str(MIN_REPLACE_SECONDS) in resp.text


async def test_a_range_past_the_end_of_the_song_is_rejected(client):
    """That would be an extension wearing the wrong name."""
    parent = await _completed(client)
    resp = await _replace(client, parent["id"], start_seconds=1.0, end_seconds=9.0)
    assert resp.status_code == 422
    assert "ends after the song does" in resp.json()["detail"]


async def test_replacing_the_whole_song_is_rejected(client):
    """Preserving nothing is a regeneration, not an edit."""
    parent = await _completed(client)
    resp = await _replace(client, parent["id"], start_seconds=0, end_seconds=2.0)
    assert resp.status_code == 422
    assert str(MIN_PRESERVED_SECONDS) in resp.text


async def test_an_unfinished_parent_is_rejected(client, app):
    async def never_runs(generation_id):
        return None

    app.state.enqueuer.enqueue = never_runs
    parent_id = (await client.post("/v1/generations", json=PAYLOAD)).json()["generation_id"]

    resp = await _replace(client, parent_id, start_seconds=SPAN_START, end_seconds=SPAN_END)
    assert resp.status_code == 409


async def test_a_parent_whose_audio_is_gone_is_rejected(client, app):
    parent = await _completed(client)
    await app.state.audio_storage.delete_generation_audio(uuid.UUID(parent["id"]))

    resp = await _replace(client, parent["id"], start_seconds=SPAN_START, end_seconds=SPAN_END)
    assert resp.status_code == 409


async def test_engine_fields_cannot_be_smuggled_in(client):
    parent = await _completed(client)
    resp = await _replace(
        client, parent["id"], start_seconds=SPAN_START, end_seconds=SPAN_END, task_type="cover"
    )
    assert resp.status_code == 422


# ── the engine receives the right edit ────────────────────────────────


async def test_the_worker_routes_to_edit_not_generate(client, app):
    parent = await _completed(client)
    assert app.state.provider.edits == []

    await _replace(client, parent["id"], start_seconds=SPAN_START, end_seconds=SPAN_END)

    assert len(app.state.provider.edits) == 1


async def test_the_provider_receives_the_interior_range_verbatim(client, app):
    """The user's own times, not re-derived from any measurement."""
    parent = await _completed(client)

    await _replace(client, parent["id"], start_seconds=SPAN_START, end_seconds=SPAN_END)

    edit = app.state.provider.edits[0]
    assert edit.start_seconds == pytest.approx(SPAN_START)
    assert edit.end_seconds == pytest.approx(SPAN_END)
    assert edit.kind.value == "REGENERATE_RANGE"


async def test_the_provider_receives_the_parents_audio(client, app):
    parent = await _completed(client)
    master_key = next(
        a["storage_key"] for a in parent["audio_assets"] if a["asset_type"] == "MASTER"
    )
    expected = await app.state.audio_storage.open(master_key)

    await _replace(client, parent["id"], start_seconds=SPAN_START, end_seconds=SPAN_END)

    assert app.state.provider.edits[0].source_audio.read_bytes() == expected


async def test_a_replacement_never_falls_back_to_generation(client, app):
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

    resp = await _replace(client, parent["id"], start_seconds=SPAN_START, end_seconds=SPAN_END)
    child = (await client.get(f"/v1/generations/{resp.json()['generation_id']}")).json()

    assert child["status"] == "FAILED"
    assert stub.generate_calls == 0


async def test_no_paths_or_engine_words_reach_the_client(client):
    parent = await _completed(client)
    resp = await _replace(client, parent["id"], start_seconds=SPAN_START, end_seconds=SPAN_END)
    child = (await client.get(f"/v1/generations/{resp.json()['generation_id']}")).json()

    rendered = str(child)
    for forbidden in (
        "repainting_start",
        "repaint_mode",
        "chunk_mask_mode",
        "/Users/",
        "/private/",
    ):
        assert forbidden not in rendered


# ── the other paths still work ────────────────────────────────────────


async def test_extend_still_works(client, app):
    parent = await _completed(client)
    resp = await client.post(f"/v1/generations/{parent['id']}/extend", json={"seconds": 15})
    assert resp.status_code == 202
    child = (await client.get(f"/v1/generations/{resp.json()['generation_id']}")).json()
    assert child["edit_kind"] == "EXTEND"
    assert child["duration_requested"] > parent["duration_actual"]


async def test_text2music_still_takes_the_generate_path(client, app):
    await _completed(client)
    assert app.state.provider.edits == []
