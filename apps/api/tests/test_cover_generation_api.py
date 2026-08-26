"""Phase 13D-2 — Create Cover is source-conditioned, or it fails.

Cover sits between the two operations already shipped and must not be
confused with either. It is not text-to-music: the source has to reach the
engine. It is not repaint: nothing is preserved, so it must never be
routed through the edit path, which would claim a guarantee this operation
does not provide.

The other thing pinned here is the strength mapping. The product labels
describe *transformation* and the engine parameter measures *adherence*,
so they run in opposite directions. A silent flip would leave both labels
lying while every test that only checked "a value was sent" still passed.
"""

from __future__ import annotations

import uuid

import pytest

from luber_api.schemas import COVER_STRENGTH_TO_ADHERENCE, CoverStrength

PAYLOAD = {
    "title": "Midnight Window",
    "prompt": "Dreamy Korean indie pop",
    "lyrics": "[Verse]\n오늘 밤 너를 생각해",
    "vocal_gender": "female",
    "duration": 30,
    "language": "ko",
}

STYLE = "modern synth pop with glossy production"


async def _completed(client) -> dict:
    resp = await client.post("/v1/generations", json=PAYLOAD)
    assert resp.status_code == 202, resp.text
    body = (await client.get(f"/v1/generations/{resp.json()['generation_id']}")).json()
    assert body["status"] == "COMPLETED"
    return body


async def _cover(client, parent_id: str, **body):
    body.setdefault("prompt", STYLE)
    return await client.post(f"/v1/generations/{parent_id}/cover", json=body)


# ── the strength mapping, in both directions ──────────────────────────


def test_more_transformation_means_less_source_adherence():
    """The product label and the engine dial run opposite ways.

    If this ever reads the other way round, "Strong" would produce the
    result closest to the original and the UI would be lying twice.
    """
    subtle = COVER_STRENGTH_TO_ADHERENCE[CoverStrength.SUBTLE]
    strong = COVER_STRENGTH_TO_ADHERENCE[CoverStrength.STRONG]
    assert strong < subtle


def test_every_preset_sits_inside_the_calibrated_band():
    """Nothing reachable from the product was left uncalibrated."""
    from luber_generation_client.ace_step.provider import (
        COVER_ADHERENCE_MAX,
        COVER_ADHERENCE_MIN,
    )

    for preset, adherence in COVER_STRENGTH_TO_ADHERENCE.items():
        assert COVER_ADHERENCE_MIN <= adherence <= COVER_ADHERENCE_MAX, preset


def test_only_measured_values_are_offered():
    """0.50 measured as indistinguishable from an unrelated song."""
    assert set(COVER_STRENGTH_TO_ADHERENCE.values()) == {1.0, 0.75}


# ── acceptance ────────────────────────────────────────────────────────


async def _master_key(app, generation_id: str) -> str:
    """The master's storage key, read from the database.

    Server-side tests that need the stored object still need the key; it
    is no longer in the API response, and the database is where it lives.
    """
    from sqlalchemy import select

    from luber_database.models.generation import AudioAsset

    async with app.state.session_factory() as session:
        result = await session.execute(
            select(AudioAsset.storage_key).where(
                AudioAsset.generation_id == uuid.UUID(str(generation_id)),
                AudioAsset.asset_type == "MASTER",
            )
        )
        key = result.scalars().first()
    assert key, "the completed generation has no MASTER asset"
    return str(key)


async def test_a_completed_song_can_be_covered(client):
    parent = await _completed(client)
    resp = await _cover(client, parent["id"])
    assert resp.status_code == 202, resp.text
    assert resp.json()["generation_id"] != parent["id"]


async def test_the_child_records_cover_provenance(client):
    parent = await _completed(client)
    child_id = (await _cover(client, parent["id"])).json()["generation_id"]

    child = (await client.get(f"/v1/generations/{child_id}")).json()
    assert child["edit_kind"] == "COVER"
    assert child["parent_generation_id"] == parent["id"]
    # A cover has no time range — it regenerates everything.
    assert child["edit_start_seconds"] is None
    assert child["edit_end_seconds"] is None


async def test_the_child_records_the_adherence_it_will_use(client):
    parent = await _completed(client)
    child_id = (await _cover(client, parent["id"], strength="strong")).json()["generation_id"]

    child = (await client.get(f"/v1/generations/{child_id}")).json()
    assert child["source_adherence"] == pytest.approx(0.75)


async def test_the_target_style_becomes_the_childs_brief(client):
    parent = await _completed(client)
    child_id = (await _cover(client, parent["id"], prompt="warm contemporary R&B")).json()[
        "generation_id"
    ]

    child = (await client.get(f"/v1/generations/{child_id}")).json()
    assert child["prompt"] == "warm contemporary R&B"
    assert child["prompt"] != parent["prompt"]


async def test_lyrics_are_inherited_not_editable(client):
    """LUBER has no lyric-to-time alignment, so it does not offer one."""
    parent = await _completed(client)
    child_id = (await _cover(client, parent["id"])).json()["generation_id"]

    child = (await client.get(f"/v1/generations/{child_id}")).json()
    assert child["lyrics"] == parent["lyrics"]


async def test_the_cover_keeps_the_sources_length(client):
    parent = await _completed(client)
    child_id = (await _cover(client, parent["id"])).json()["generation_id"]

    child = (await client.get(f"/v1/generations/{child_id}")).json()
    assert child["duration_actual"] == pytest.approx(parent["duration_actual"], abs=0.5)


async def test_the_parent_is_not_mutated(client):
    parent = await _completed(client)
    await _cover(client, parent["id"], prompt="something else entirely")
    after = (await client.get(f"/v1/generations/{parent['id']}")).json()
    assert after["prompt"] == parent["prompt"]
    assert after["edit_kind"] is None
    assert after["source_adherence"] is None


async def test_a_cover_is_distinguishable_from_the_other_derivations(client):
    parent = await _completed(client)
    extended = (
        await client.post(f"/v1/generations/{parent['id']}/extend", json={"seconds": 15})
    ).json()["generation_id"]
    replaced = (
        await client.post(
            f"/v1/generations/{parent['id']}/replace-range",
            json={"start_seconds": 0.5, "end_seconds": 1.5},
        )
    ).json()["generation_id"]
    covered = (await _cover(client, parent["id"])).json()["generation_id"]

    kinds = {}
    for gid in (extended, replaced, covered):
        kinds[gid] = (await client.get(f"/v1/generations/{gid}")).json()["edit_kind"]
    assert kinds[extended] == "EXTEND"
    assert kinds[replaced] == "REPLACE_RANGE"
    assert kinds[covered] == "COVER"


# ── rejection ─────────────────────────────────────────────────────────


async def test_an_unknown_parent_is_404(client):
    assert (await _cover(client, str(uuid.uuid4()))).status_code == 404


async def test_an_unfinished_parent_is_rejected(client, app):
    async def never_runs(generation_id):
        return None

    app.state.enqueuer.enqueue = never_runs
    parent_id = (await client.post("/v1/generations", json=PAYLOAD)).json()["generation_id"]
    resp = await _cover(client, parent_id)
    assert resp.status_code == 409
    assert "covered" in resp.json()["detail"]


async def test_a_parent_whose_audio_is_gone_is_rejected(client, app):
    parent = await _completed(client)
    await app.state.audio_storage.delete_generation_audio(uuid.UUID(parent["id"]))
    assert (await _cover(client, parent["id"])).status_code == 409


async def test_an_unknown_strength_is_rejected(client):
    parent = await _completed(client)
    assert (await _cover(client, parent["id"], strength="extreme")).status_code == 422


async def test_a_blank_style_is_rejected(client):
    parent = await _completed(client)
    assert (await _cover(client, parent["id"], prompt="   ")).status_code == 422


async def test_engine_controls_cannot_be_smuggled_in(client):
    parent = await _completed(client)
    for field, value in [
        ("task_type", "cover"),
        ("audio_cover_strength", 0.1),
        ("cover_noise_strength", 1.0),
        ("src_audio", "/etc/passwd"),
        ("thinking", True),
    ]:
        resp = await _cover(client, parent["id"], **{field: value})
        assert resp.status_code == 422, field


# ── what reaches the provider ─────────────────────────────────────────


async def test_the_worker_routes_a_cover_to_the_audio_to_audio_path(client, app):
    parent = await _completed(client)
    provider = app.state.provider
    assert provider.covers == [] and provider.edits == []

    await _cover(client, parent["id"])

    assert len(provider.covers) == 1, "the cover did not reach create_from_audio()"
    assert provider.edits == [], "a cover must never be routed through the edit path"


async def test_the_provider_receives_the_sources_real_audio(client, app):
    parent = await _completed(client)
    # The raw MASTER row's bytes, read from the database rather than from
    # the API: the key is no longer serialised to clients, and the
    # delivery endpoint may resolve to a finished master, which is not
    # what the provider is handed.
    expected = await app.state.audio_storage.open(await _master_key(app, parent["id"]))

    await _cover(client, parent["id"])

    assert app.state.provider.covers[0].source_audio.read_bytes() == expected


async def test_the_mapped_adherence_reaches_the_provider(client, app):
    parent = await _completed(client)

    await _cover(client, parent["id"], strength="strong")

    assert app.state.provider.covers[0].source_adherence == pytest.approx(0.75)


async def test_subtle_reaches_the_provider_as_the_highest_adherence(client, app):
    parent = await _completed(client)
    await _cover(client, parent["id"], strength="subtle")
    assert app.state.provider.covers[0].source_adherence == pytest.approx(1.0)


async def test_the_provider_receives_the_target_style(client, app):
    parent = await _completed(client)
    await _cover(client, parent["id"], prompt="dreamy indie pop with spacious guitars")
    assert app.state.provider.covers[0].prompt == "dreamy indie pop with spacious guitars"


async def test_a_cover_never_falls_back_to_text2music(client, app):
    """A source-less song in place of a cover would be the worst outcome."""
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

    resp = await _cover(client, parent["id"])
    child = (await client.get(f"/v1/generations/{resp.json()['generation_id']}")).json()

    assert child["status"] == "FAILED"
    assert stub.generate_calls == 0


async def test_no_paths_or_engine_words_reach_the_client(client):
    parent = await _completed(client)
    resp = await _cover(client, parent["id"])
    child = (await client.get(f"/v1/generations/{resp.json()['generation_id']}")).json()

    rendered = str(child)
    for forbidden in (
        "audio_cover_strength",
        "cover_noise_strength",
        "task_type",
        "src_audio",
        "/Users/",
        "/private/",
    ):
        assert forbidden not in rendered


# ── the other operations still work ───────────────────────────────────


async def test_text2music_is_untouched(client, app):
    await _completed(client)
    assert app.state.provider.covers == []
    assert app.state.provider.edits == []


async def test_extend_still_routes_to_the_edit_path(client, app):
    parent = await _completed(client)
    await client.post(f"/v1/generations/{parent['id']}/extend", json={"seconds": 15})
    assert len(app.state.provider.edits) == 1
    assert app.state.provider.covers == []


async def test_replace_range_still_routes_to_the_edit_path(client, app):
    parent = await _completed(client)
    await client.post(
        f"/v1/generations/{parent['id']}/replace-range",
        json={"start_seconds": 0.5, "end_seconds": 1.5},
    )
    assert len(app.state.provider.edits) == 1
    assert app.state.provider.covers == []
