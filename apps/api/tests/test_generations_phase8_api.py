"""Phase 8 API surface: advanced controls, advisories, lineage, trace.

The app fixture wires the InlineGenerationRunner, so a POST runs the
real GenerationService against the fixture WAV. Generations therefore
reach COMPLETED inside the request, which lets these tests assert on
persisted state rather than on intent.
"""

from __future__ import annotations

import json
import uuid

from luber_database import GenerationRepository
from luber_database.models.generation import Generation
from luber_schemas import GenerationStatus

LEGACY_PAYLOAD = {
    "title": "PHASE 7 SONG",
    "prompt": "Dreamy Korean indie pop",
    "lyrics": "[Verse]\n오늘도 너를 기다려\n조용한 창가에 앉아\n"
    "[Chorus]\n다시 만날 그날까지\n나는 여기 있을게",
    "vocal_gender": "female",
    "duration": 30,
    "language": "ko",
}


async def _insert_generation(app, **overrides) -> Generation:
    """Insert a row directly, for states the API cannot create itself."""
    async with app.state.session_factory() as session:
        repository = GenerationRepository(session)
        defaults = dict(
            title="DIRECT ROW",
            prompt="p",
            lyrics="",
            vocal_gender="female",
            duration_requested=30,
            status=GenerationStatus.COMPLETED.value,
        )
        defaults.update(overrides)
        return await repository.create_generation(**defaults)


async def _set_column(app, generation_id, **columns) -> None:
    async with app.state.session_factory() as session:
        row = await session.get(Generation, generation_id)
        for name, value in columns.items():
            setattr(row, name, value)
        await session.commit()


# ── Legacy requests keep working ──────────────────────────────────────


async def test_legacy_request_without_advanced_controls_still_succeeds(client):
    resp = await client.post("/v1/generations", json=LEGACY_PAYLOAD)
    assert resp.status_code == 202

    detail = (await client.get(f"/v1/generations/{resp.json()['generation_id']}")).json()
    assert detail["status"] == "COMPLETED"
    assert detail["bpm"] is None
    assert detail["key_scale"] is None
    assert detail["time_signature"] is None
    assert detail["parent_generation_id"] is None
    assert detail["variation_label"] is None


async def test_legacy_row_predating_phase8_serializes_with_nulls(client, app):
    # A Phase 3-7 row: advisories and trace were never written.
    row = await _insert_generation(app, title="OLD ROW")
    detail = (await client.get(f"/v1/generations/{row.id}")).json()
    assert detail["advisories"] == []
    assert detail["request_trace"] is None
    assert detail["bpm"] is None


# ── Advanced control propagation ──────────────────────────────────────


async def test_bpm_is_stored_and_returned(client):
    created = await client.post("/v1/generations", json=dict(LEGACY_PAYLOAD, bpm=128))
    detail = (await client.get(f"/v1/generations/{created.json()['generation_id']}")).json()
    assert detail["bpm"] == 128


async def test_key_scale_is_stored_and_returned(client):
    created = await client.post("/v1/generations", json=dict(LEGACY_PAYLOAD, key_scale="F# minor"))
    detail = (await client.get(f"/v1/generations/{created.json()['generation_id']}")).json()
    assert detail["key_scale"] == "F# minor"


async def test_time_signature_is_stored_and_returned(client):
    created = await client.post("/v1/generations", json=dict(LEGACY_PAYLOAD, time_signature="3"))
    detail = (await client.get(f"/v1/generations/{created.json()['generation_id']}")).json()
    assert detail["time_signature"] == "3"


async def test_all_three_controls_together(client):
    created = await client.post(
        "/v1/generations",
        json=dict(LEGACY_PAYLOAD, bpm=92, key_scale="Bb major", time_signature="6"),
    )
    assert created.status_code == 202
    detail = (await client.get(f"/v1/generations/{created.json()['generation_id']}")).json()
    assert (detail["bpm"], detail["key_scale"], detail["time_signature"]) == (92, "Bb major", "6")


async def test_empty_string_controls_are_treated_as_unset(client):
    created = await client.post(
        "/v1/generations", json=dict(LEGACY_PAYLOAD, key_scale="", time_signature="")
    )
    assert created.status_code == 202
    detail = (await client.get(f"/v1/generations/{created.json()['generation_id']}")).json()
    assert detail["key_scale"] is None
    assert detail["time_signature"] is None


async def test_out_of_range_bpm_is_rejected(client):
    for bpm in (29, 301, -1):
        resp = await client.post("/v1/generations", json=dict(LEGACY_PAYLOAD, bpm=bpm))
        assert resp.status_code == 422, bpm


async def test_unsupported_key_scale_is_rejected(client):
    for key_scale in ("H minor", "C dorian", "C♯ major"):
        resp = await client.post("/v1/generations", json=dict(LEGACY_PAYLOAD, key_scale=key_scale))
        assert resp.status_code == 422, key_scale


async def test_unsupported_time_signature_is_rejected(client):
    for value in ("4/4", "5", "seven"):
        resp = await client.post("/v1/generations", json=dict(LEGACY_PAYLOAD, time_signature=value))
        assert resp.status_code == 422, value


# ── Advisories ────────────────────────────────────────────────────────


async def test_advisories_are_returned_on_create(client):
    dense = "\n".join("가나다라마바사아자차카타파하" for _ in range(12))
    resp = await client.post("/v1/generations", json=dict(LEGACY_PAYLOAD, lyrics=dense))
    assert resp.status_code == 202
    codes = {a["code"] for a in resp.json()["advisories"]}
    assert "LYRICS_DENSE_FOR_DURATION" in codes


async def test_advisories_never_block_the_generation(client):
    # Every heuristic firing at once must still produce a real track.
    hostile = "[Drop]\n" + "\n".join("가나다라마바사아자차카타파하" for _ in range(20))
    resp = await client.post(
        "/v1/generations", json=dict(LEGACY_PAYLOAD, lyrics=hostile, language="en")
    )
    assert resp.status_code == 202
    assert len(resp.json()["advisories"]) >= 3

    detail = (await client.get(f"/v1/generations/{resp.json()['generation_id']}")).json()
    assert detail["status"] == "COMPLETED"
    assert detail["audio_assets"]


async def test_advisories_do_not_alter_the_submitted_lyrics(client):
    hostile = "[Drop]\n  오늘 밤 lonely midnight feeling  \n" + "가" * 60
    created = await client.post("/v1/generations", json=dict(LEGACY_PAYLOAD, lyrics=hostile))
    detail = (await client.get(f"/v1/generations/{created.json()['generation_id']}")).json()
    # Byte-for-byte, including the leading/trailing spaces.
    assert detail["lyrics"] == hostile


async def test_advisories_are_persisted_and_reread(client):
    dense = "\n".join("가나다라마바사아자차카타파하" for _ in range(12))
    created = await client.post("/v1/generations", json=dict(LEGACY_PAYLOAD, lyrics=dense))
    detail = (await client.get(f"/v1/generations/{created.json()['generation_id']}")).json()
    codes = {a["code"] for a in detail["advisories"]}
    assert "LYRICS_DENSE_FOR_DURATION" in codes


async def test_clean_request_records_an_empty_advisory_list(client):
    created = await client.post("/v1/generations", json=LEGACY_PAYLOAD)
    assert created.json()["advisories"] == []
    detail = (await client.get(f"/v1/generations/{created.json()['generation_id']}")).json()
    assert detail["advisories"] == []


async def test_malformed_persisted_advisories_degrade_to_empty(client, app):
    created = await client.post("/v1/generations", json=LEGACY_PAYLOAD)
    generation_id = created.json()["generation_id"]
    await _set_column(app, uuid.UUID(generation_id), advisories="{not json at all")

    resp = await client.get(f"/v1/generations/{generation_id}")
    assert resp.status_code == 200  # a corrupt diagnostic is not a 500
    assert resp.json()["advisories"] == []


async def test_advisories_stored_as_a_non_list_degrade_to_empty(client, app):
    created = await client.post("/v1/generations", json=LEGACY_PAYLOAD)
    generation_id = created.json()["generation_id"]
    await _set_column(app, uuid.UUID(generation_id), advisories='{"code": "X"}')
    assert (await client.get(f"/v1/generations/{generation_id}")).json()["advisories"] == []


async def test_partially_malformed_advisories_keep_the_readable_entries(client, app):
    created = await client.post("/v1/generations", json=LEGACY_PAYLOAD)
    generation_id = created.json()["generation_id"]
    await _set_column(
        app,
        uuid.UUID(generation_id),
        advisories=json.dumps(
            [
                {"code": "GOOD", "level": "info", "message": "kept", "detail": {}},
                "not an object",
                {"missing": "fields"},
            ]
        ),
    )
    advisories = (await client.get(f"/v1/generations/{generation_id}")).json()["advisories"]
    assert [a["code"] for a in advisories] == ["GOOD"]


async def test_malformed_request_trace_degrades_to_null(client, app):
    created = await client.post("/v1/generations", json=LEGACY_PAYLOAD)
    generation_id = created.json()["generation_id"]
    await _set_column(app, uuid.UUID(generation_id), request_trace="]]not json[[")

    resp = await client.get(f"/v1/generations/{generation_id}")
    assert resp.status_code == 200
    assert resp.json()["request_trace"] is None


async def test_list_endpoint_decodes_advisories_too(client):
    dense = "\n".join("가나다라마바사아자차카타파하" for _ in range(12))
    await client.post("/v1/generations", json=dict(LEGACY_PAYLOAD, lyrics=dense))
    items = (await client.get("/v1/generations")).json()["items"]
    assert any(a["code"] == "LYRICS_DENSE_FOR_DURATION" for i in items for a in i["advisories"])


# ── Idempotency ───────────────────────────────────────────────────────


async def test_idempotent_replay_returns_the_same_advisories(client):
    dense = "\n".join("가나다라마바사아자차카타파하" for _ in range(12))
    payload = dict(LEGACY_PAYLOAD, lyrics=dense)
    headers = {"Idempotency-Key": "phase8-replay-key"}

    first = await client.post("/v1/generations", json=payload, headers=headers)
    second = await client.post("/v1/generations", json=payload, headers=headers)

    assert first.json()["generation_id"] == second.json()["generation_id"]
    assert first.json()["advisories"] == second.json()["advisories"]
    assert second.json()["advisories"]  # not silently emptied on replay


async def test_idempotent_replay_of_a_clean_request_returns_no_advisories(client):
    headers = {"Idempotency-Key": "phase8-clean-key"}
    first = await client.post("/v1/generations", json=LEGACY_PAYLOAD, headers=headers)
    second = await client.post("/v1/generations", json=LEGACY_PAYLOAD, headers=headers)
    assert first.json()["advisories"] == second.json()["advisories"] == []


# ── Lineage ───────────────────────────────────────────────────────────


async def test_generation_can_declare_a_parent(client):
    parent_id = (await client.post("/v1/generations", json=LEGACY_PAYLOAD)).json()["generation_id"]

    child = await client.post(
        "/v1/generations", json=dict(LEGACY_PAYLOAD, parent_generation_id=parent_id)
    )
    assert child.status_code == 202

    detail = (await client.get(f"/v1/generations/{child.json()['generation_id']}")).json()
    assert detail["parent_generation_id"] == parent_id


async def test_variation_label_is_stored_alongside_its_parent(client):
    parent_id = (await client.post("/v1/generations", json=LEGACY_PAYLOAD)).json()["generation_id"]
    child = await client.post(
        "/v1/generations",
        json=dict(LEGACY_PAYLOAD, parent_generation_id=parent_id, variation_label="take 2"),
    )
    detail = (await client.get(f"/v1/generations/{child.json()['generation_id']}")).json()
    assert detail["variation_label"] == "take 2"


async def test_unknown_parent_is_rejected(client):
    resp = await client.post(
        "/v1/generations", json=dict(LEGACY_PAYLOAD, parent_generation_id=str(uuid.uuid4()))
    )
    assert resp.status_code == 422
    assert "parent" in resp.json()["detail"]


async def test_malformed_parent_id_is_rejected(client):
    resp = await client.post(
        "/v1/generations", json=dict(LEGACY_PAYLOAD, parent_generation_id="not-a-uuid")
    )
    assert resp.status_code == 422


async def test_parent_owned_by_someone_else_is_rejected(client, app):
    owned = await _insert_generation(app, title="SOMEONE ELSE", user_id=uuid.uuid4())

    resp = await client.post(
        "/v1/generations", json=dict(LEGACY_PAYLOAD, parent_generation_id=str(owned.id))
    )
    assert resp.status_code == 422


async def test_inaccessible_parent_is_indistinguishable_from_a_missing_one(client, app):
    # Lineage must not become an existence oracle for other users' work.
    owned = await _insert_generation(app, title="SOMEONE ELSE", user_id=uuid.uuid4())

    forbidden = await client.post(
        "/v1/generations", json=dict(LEGACY_PAYLOAD, parent_generation_id=str(owned.id))
    )
    missing = await client.post(
        "/v1/generations", json=dict(LEGACY_PAYLOAD, parent_generation_id=str(uuid.uuid4()))
    )
    assert forbidden.status_code == missing.status_code == 422
    assert forbidden.json() == missing.json()


async def test_the_owner_may_use_their_own_generation_as_a_parent(client, app):
    owner = uuid.uuid4()
    owned = await _insert_generation(app, title="MINE", user_id=owner)

    resp = await client.post(
        "/v1/generations",
        json=dict(LEGACY_PAYLOAD, parent_generation_id=str(owned.id)),
        headers={"X-User-Id": str(owner)},
    )
    assert resp.status_code == 202


async def test_variation_label_without_a_parent_is_rejected(client):
    resp = await client.post("/v1/generations", json=dict(LEGACY_PAYLOAD, variation_label="take 2"))
    assert resp.status_code == 422
    assert "variation_label" in json.dumps(resp.json())


async def test_blank_variation_label_without_a_parent_is_accepted_as_absent(client):
    resp = await client.post("/v1/generations", json=dict(LEGACY_PAYLOAD, variation_label="   "))
    assert resp.status_code == 202
    detail = (await client.get(f"/v1/generations/{resp.json()['generation_id']}")).json()
    assert detail["variation_label"] is None


async def test_deleting_a_parent_leaves_the_child_intact(client):
    """Deleting an original must not delete what was made from it.

    Scope note: this asserts the child *survives*, which is what this
    suite can prove. The ``ON DELETE SET NULL`` that blanks the child's
    ``parent_generation_id`` is enforced by the database, and SQLite
    ignores foreign keys unless ``PRAGMA foreign_keys=ON`` — so the
    NULL-ing itself is verified against PostgreSQL in the migration
    check (see docs/PHASE8_ADVANCED_CONTROLS.md), not here.
    """
    parent_id = (await client.post("/v1/generations", json=LEGACY_PAYLOAD)).json()["generation_id"]
    child_id = (
        await client.post(
            "/v1/generations", json=dict(LEGACY_PAYLOAD, parent_generation_id=parent_id)
        )
    ).json()["generation_id"]

    assert (await client.delete(f"/v1/generations/{parent_id}")).status_code == 204

    detail = (await client.get(f"/v1/generations/{child_id}")).json()
    assert detail["status"] == "COMPLETED"
    assert detail["audio_assets"]  # the derived track is still deliverable


# ── Request trace over the API ────────────────────────────────────────


async def test_request_trace_is_absent_for_the_mock_provider(client):
    # MockGenerationProvider does not implement describe_request, so the
    # trace is genuinely "not recorded" rather than an invented default.
    created = await client.post("/v1/generations", json=LEGACY_PAYLOAD)
    detail = (await client.get(f"/v1/generations/{created.json()['generation_id']}")).json()
    assert detail["request_trace"] is None


async def test_request_trace_is_exposed_as_structured_json(client, app):
    created = await client.post("/v1/generations", json=LEGACY_PAYLOAD)
    generation_id = created.json()["generation_id"]
    await _set_column(
        app,
        uuid.UUID(generation_id),
        request_trace=json.dumps({"provider": "ace_step", "payload": {"bpm": 128}}),
    )
    trace = (await client.get(f"/v1/generations/{generation_id}")).json()["request_trace"]
    assert trace == {"provider": "ace_step", "payload": {"bpm": 128}}


# ── Preflight endpoint ────────────────────────────────────────────────


async def test_preflight_returns_advisories_without_creating_anything(client):
    before = (await client.get("/v1/generations")).json()["total"]
    dense = "\n".join("가나다라마바사아자차카타파하" for _ in range(12))

    resp = await client.post(
        "/v1/generations/preflight", json={"lyrics": dense, "duration": 30, "language": "ko"}
    )
    assert resp.status_code == 200
    assert any(a["code"] == "LYRICS_DENSE_FOR_DURATION" for a in resp.json()["advisories"])

    after = (await client.get("/v1/generations")).json()["total"]
    assert after == before  # side-effect free


async def test_preflight_reports_parsed_structure(client):
    resp = await client.post(
        "/v1/generations/preflight",
        json={"lyrics": "[Verse 1]\na\nb\n[Drop]\nc", "duration": 60},
    )
    sections = resp.json()["sections"]
    assert [s["label"] for s in sections] == ["Verse 1", "Drop"]
    assert sections[0]["kind"] == "verse"
    assert sections[0]["index"] == 1
    assert sections[0]["line_count"] == 2
    assert sections[1]["kind"] is None
    assert sections[1]["recognised"] is False


async def test_preflight_reports_untagged_lyrics_as_preamble(client):
    resp = await client.post(
        "/v1/generations/preflight", json={"lyrics": "plain line\nanother", "duration": 60}
    )
    body = resp.json()
    assert body["sections"] == []
    assert body["preamble_line_count"] == 2


async def test_preflight_agrees_with_what_create_records(client):
    # One implementation of the heuristics, so the editor cannot show a
    # different verdict from the one that gets stored.
    dense = "\n".join("가나다라마바사아자차카타파하" for _ in range(12))
    payload = dict(LEGACY_PAYLOAD, lyrics=dense)

    previewed = await client.post(
        "/v1/generations/preflight",
        json={
            "lyrics": dense,
            "duration": payload["duration"],
            "language": payload["language"],
            "instrumental": False,
        },
    )
    created = await client.post("/v1/generations", json=payload)
    assert previewed.json()["advisories"] == created.json()["advisories"]


async def test_preflight_reports_estimated_syllables(client):
    resp = await client.post(
        "/v1/generations/preflight", json={"lyrics": "안녕하세요", "duration": 30}
    )
    assert resp.json()["estimated_syllables"] == 5


async def test_preflight_on_empty_lyrics_is_quiet(client):
    resp = await client.post("/v1/generations/preflight", json={"lyrics": "", "duration": 30})
    assert resp.status_code == 200
    assert resp.json() == {
        "advisories": [],
        "sections": [],
        "preamble_line_count": 0,
        "estimated_syllables": 0,
    }


async def test_preflight_respects_the_instrumental_flag(client):
    dense = "\n".join("가나다라마바사아자차카타파하" for _ in range(20))
    resp = await client.post(
        "/v1/generations/preflight",
        json={"lyrics": dense, "duration": 30, "instrumental": True},
    )
    codes = {a["code"] for a in resp.json()["advisories"]}
    assert "LYRICS_DENSE_FOR_DURATION" not in codes


async def test_preflight_validates_duration_like_the_create_endpoint(client):
    resp = await client.post("/v1/generations/preflight", json={"lyrics": "x", "duration": 5})
    assert resp.status_code == 422
