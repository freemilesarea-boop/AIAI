"""Phase 12 — song management, favourites, groups and bulk actions.

The properties defended here are the ones a user would notice being
broken:

- A song can be renamed and favourited; **nothing else** about it can be
  edited, because everything else is a record of a run that happened.
- Deleting a take never deletes the takes made from it, and never leaves
  a child pointing at a row that is gone.
- Asking for two songs produces two genuinely independent generations —
  two rows, two jobs, two seeds, two statuses — not one provider call
  with a batch size.
- One of them failing does not fail the other.
"""

from __future__ import annotations

import uuid

import pytest

PAYLOAD = {
    "title": "Midnight Window",
    "prompt": "Dreamy Korean indie pop",
    "lyrics": "[Verse]\n오늘 밤 너를 생각해",
    "vocal_gender": "female",
    "duration": 30,
    "language": "ko",
}


async def _create(client, **overrides) -> dict:
    resp = await client.post("/v1/generations", json=dict(PAYLOAD, **overrides))
    assert resp.status_code == 202, resp.text
    return resp.json()


async def _generation(client, **overrides) -> str:
    return (await _create(client, **overrides))["generation_id"]


async def _project(client, name: str = "EP") -> str:
    resp = await client.post("/v1/projects", json={"name": name})
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


# ── 2. song management: rename ────────────────────────────────────────


async def test_rename_changes_only_the_title(client):
    generation_id = await _generation(client)
    before = (await client.get(f"/v1/generations/{generation_id}")).json()

    resp = await client.patch(f"/v1/generations/{generation_id}", json={"title": "Renamed Track"})

    assert resp.status_code == 200, resp.text
    after = resp.json()
    assert after["title"] == "Renamed Track"
    for field in ("prompt", "lyrics", "seed", "duration_requested", "vocal_gender", "language"):
        assert after[field] == before[field]


async def test_rename_persists(client):
    generation_id = await _generation(client)
    await client.patch(f"/v1/generations/{generation_id}", json={"title": "Kept"})
    reread = (await client.get(f"/v1/generations/{generation_id}")).json()
    assert reread["title"] == "Kept"


async def test_a_blank_title_is_rejected(client):
    generation_id = await _generation(client)
    assert (
        await client.patch(f"/v1/generations/{generation_id}", json={"title": "   "})
    ).status_code == 422


async def test_titles_are_trimmed(client):
    generation_id = await _generation(client)
    body = (
        await client.patch(f"/v1/generations/{generation_id}", json={"title": "  Spaced  "})
    ).json()
    assert body["title"] == "Spaced"


async def test_an_empty_patch_is_rejected(client):
    """ "Update nothing" is a client bug, not a no-op worth accepting."""
    generation_id = await _generation(client)
    assert (await client.patch(f"/v1/generations/{generation_id}", json={})).status_code == 422


async def test_patching_an_unknown_generation_is_404(client):
    resp = await client.patch(f"/v1/generations/{uuid.uuid4()}", json={"title": "x"})
    assert resp.status_code == 404


# ── 12. provenance is immutable ───────────────────────────────────────


@pytest.mark.parametrize(
    "field,value",
    [
        ("prompt", "something else"),
        ("lyrics", "different words"),
        ("seed", 1234),
        ("model_name", "not-ace-step"),
        ("provider", "someone-else"),
        ("bpm", 128),
        ("duration_requested", 240),
        ("status", "COMPLETED"),
        ("request_trace", {}),
        ("created_at", "2020-01-01T00:00:00Z"),
    ],
)
async def test_provenance_fields_cannot_be_patched(client, field, value):
    """Rejected loudly, not dropped silently.

    A caller that sends ``prompt`` believes it is editing the prompt. A
    204 that quietly ignored it would leave the library describing audio
    it does not match.
    """
    generation_id = await _generation(client)
    resp = await client.patch(
        f"/v1/generations/{generation_id}", json={"title": "New", field: value}
    )
    assert resp.status_code == 422, resp.text

    unchanged = (await client.get(f"/v1/generations/{generation_id}")).json()
    assert unchanged["title"] == PAYLOAD["title"]


# ── 3. favourites ─────────────────────────────────────────────────────


async def test_a_generation_starts_unfavourited(client):
    generation_id = await _generation(client)
    assert (await client.get(f"/v1/generations/{generation_id}")).json()["favorite"] is False


async def test_favourite_toggles_and_persists(client):
    generation_id = await _generation(client)

    assert (await client.patch(f"/v1/generations/{generation_id}", json={"favorite": True})).json()[
        "favorite"
    ] is True
    assert (await client.get(f"/v1/generations/{generation_id}")).json()["favorite"] is True

    assert (
        await client.patch(f"/v1/generations/{generation_id}", json={"favorite": False})
    ).json()["favorite"] is False
    assert (await client.get(f"/v1/generations/{generation_id}")).json()["favorite"] is False


async def test_favourite_survives_a_rename(client):
    generation_id = await _generation(client)
    await client.patch(f"/v1/generations/{generation_id}", json={"favorite": True})
    body = (await client.patch(f"/v1/generations/{generation_id}", json={"title": "R"})).json()
    assert body["favorite"] is True


async def test_favourite_appears_in_the_list(client):
    generation_id = await _generation(client)
    await client.patch(f"/v1/generations/{generation_id}", json={"favorite": True})
    items = (await client.get("/v1/generations")).json()["items"]
    assert [g["favorite"] for g in items if g["id"] == generation_id] == [True]


# ── 19. cover art foundation ──────────────────────────────────────────


async def test_cover_art_url_is_present_and_null(client):
    """The field exists so the client can branch on it, and is never faked."""
    generation_id = await _generation(client)
    body = (await client.get(f"/v1/generations/{generation_id}")).json()
    assert "cover_art_url" in body
    assert body["cover_art_url"] is None


# ── 2. deletion and lineage consistency ───────────────────────────────


async def test_deleting_a_parent_is_refused_while_it_has_children(client):
    """Phase 17 replaced the old behaviour, which nulled the child's link.

    That kept the child alive but left it claiming an ``edit_kind`` while
    descending from nothing — a contradiction version history would draw
    as a root labelled "Extended". Refusing is the only option that keeps
    the record true, and it is recoverable: delete the derived version
    first.
    """
    parent_id = await _generation(client)
    child_id = await _generation(client, parent_generation_id=parent_id, title="Take 2")

    refusal = await client.delete(f"/v1/generations/{parent_id}")
    assert refusal.status_code == 409
    detail = refusal.json()["detail"]
    assert detail["code"] == "GENERATION_HAS_DERIVED_VERSIONS"
    assert detail["derived_count"] == 1

    # Both rows survive untouched, and the link is intact.
    assert (await client.get(f"/v1/generations/{parent_id}")).status_code == 200
    child = (await client.get(f"/v1/generations/{child_id}")).json()
    assert child["parent_generation_id"] == parent_id


async def test_deleting_a_child_leaves_the_parent_alone(client):
    parent_id = await _generation(client)
    child_id = await _generation(client, parent_generation_id=parent_id, title="Take 2")

    await client.delete(f"/v1/generations/{child_id}")

    assert (await client.get(f"/v1/generations/{parent_id}")).status_code == 200
    lineage = (await client.get(f"/v1/generations/{parent_id}/lineage")).json()
    assert lineage["children"] == []


async def test_delete_removes_the_audio_after_the_database(client, app):
    """DB state first, bytes second — never the other way around.

    If storage were cleared first and the DB write then failed, the row
    would survive pointing at audio that no longer exists.
    """
    generation_id = await _generation(client)
    storage = app.state.audio_storage
    observed: list[bool] = []

    original = storage.delete_generation_audio

    async def spy(gid):
        # By the time bytes are touched, the row must already be gone.
        resp = await client.get(f"/v1/generations/{gid}")
        observed.append(resp.status_code == 404)
        return await original(gid)

    storage.delete_generation_audio = spy
    assert (await client.delete(f"/v1/generations/{generation_id}")).status_code == 204
    assert observed == [True]


async def test_deleting_an_unknown_generation_is_404(client):
    assert (await client.delete(f"/v1/generations/{uuid.uuid4()}")).status_code == 404


# ── 6/7. two-result generation ────────────────────────────────────────


async def test_a_single_result_is_still_the_default(client):
    """Existing callers that never heard of result_count are unaffected."""
    body = await _create(client)
    assert len(body["generations"]) == 1
    assert body["generations"][0]["generation_id"] == body["generation_id"]


async def test_two_results_create_two_independent_generations(client):
    body = await _create(client, result_count=2)

    ids = [g["generation_id"] for g in body["generations"]]
    assert len(ids) == 2
    assert ids[0] != ids[1]
    # Independent rows, each individually readable.
    for generation_id in ids:
        assert (await client.get(f"/v1/generations/{generation_id}")).status_code == 200


async def test_two_results_share_one_group(client):
    body = await _create(client, result_count=2)
    group_id = body["generation_group_id"]
    assert group_id is not None

    for entry in body["generations"]:
        row = (await client.get(f"/v1/generations/{entry['generation_id']}")).json()
        assert row["generation_group_id"] == group_id


async def test_the_group_can_be_read_back(client):
    body = await _create(client, result_count=2)
    listed = (await client.get(f"/v1/generations/groups/{body['generation_group_id']}")).json()
    assert listed["total"] == 2
    assert {g["id"] for g in listed["items"]} == {e["generation_id"] for e in body["generations"]}


async def test_an_unknown_group_is_404(client):
    assert (await client.get(f"/v1/generations/groups/{uuid.uuid4()}")).status_code == 404


async def test_each_result_gets_its_own_job(client):
    """Two songs means two queue jobs, not one job producing two files."""
    body = await _create(client, result_count=2)
    for entry in body["generations"]:
        row = (await client.get(f"/v1/generations/{entry['generation_id']}")).json()
        # The inline runner drives each job to completion independently.
        assert row["status"] in {"QUEUED", "COMPLETED"}


async def test_more_than_two_results_is_rejected(client):
    resp = await client.post("/v1/generations", json=dict(PAYLOAD, result_count=3))
    assert resp.status_code == 422


async def test_zero_results_is_rejected(client):
    resp = await client.post("/v1/generations", json=dict(PAYLOAD, result_count=0))
    assert resp.status_code == 422


# ── 5. seed workflow ──────────────────────────────────────────────────


async def test_a_fixed_seed_is_recorded_on_a_single_result(client):
    generation_id = await _generation(client, seed=4242)
    assert (await client.get(f"/v1/generations/{generation_id}")).json()["seed"] == 4242


async def test_a_fixed_seed_applies_to_the_first_result_only(client):
    """Two identical seeds would be two identical songs.

    The point of asking for two results is comparison, so the pinned seed
    anchors the first and the engine chooses the second.
    """
    body = await _create(client, result_count=2, seed=4242)
    assert body["generations"][0]["seed"] == 4242
    # Explicitly unpinned, not merely a different number: the engine
    # picks, and the seed it used is recorded when the run completes.
    assert body["generations"][1]["seed"] is None

    stored = [
        (await client.get(f"/v1/generations/{g['generation_id']}")).json()
        for g in body["generations"]
    ]
    assert stored[0]["seed"] == 4242


async def test_no_seed_means_the_engine_chooses(client):
    body = await _create(client, result_count=2)
    assert [g["seed"] for g in body["generations"]] == [None, None]


# ── idempotency across a group ────────────────────────────────────────


async def test_replaying_a_two_result_submission_returns_both(client):
    """A retry must not lose the second song, or the client resubmits it."""
    headers = {"Idempotency-Key": "phase12-two-results"}
    first = await client.post(
        "/v1/generations", json=dict(PAYLOAD, result_count=2), headers=headers
    )
    second = await client.post(
        "/v1/generations", json=dict(PAYLOAD, result_count=2), headers=headers
    )

    assert second.status_code == 202
    assert second.json()["generation_group_id"] == first.json()["generation_group_id"]
    assert {g["generation_id"] for g in second.json()["generations"]} == {
        g["generation_id"] for g in first.json()["generations"]
    }


async def test_a_replay_creates_no_extra_rows(client):
    headers = {"Idempotency-Key": "phase12-no-extras"}
    await client.post("/v1/generations", json=dict(PAYLOAD, result_count=2), headers=headers)
    await client.post("/v1/generations", json=dict(PAYLOAD, result_count=2), headers=headers)
    assert (await client.get("/v1/generations")).json()["total"] == 2


# ── 6. partial failure ────────────────────────────────────────────────


async def test_one_failed_sibling_does_not_fail_the_other(client, app):
    """A queue that rejects the second job must not lose the first."""
    enqueuer = app.state.enqueuer
    original = enqueuer.enqueue
    calls = {"n": 0}

    async def flaky(generation_id):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("queue unavailable")
        return await original(generation_id)

    enqueuer.enqueue = flaky
    body = await _create(client, result_count=2)

    statuses = [g["status"] for g in body["generations"]]
    assert statuses[1] == "FAILED"
    assert statuses[0] != "FAILED"

    # The whole request is still a success: one song is real.
    rows = (await client.get(f"/v1/generations/groups/{body['generation_group_id']}")).json()
    assert rows["total"] == 2
    assert {r["status"] for r in rows["items"]} == {"COMPLETED", "FAILED"}


async def test_a_failed_sibling_records_why(client, app):
    enqueuer = app.state.enqueuer
    original = enqueuer.enqueue
    calls = {"n": 0}

    async def flaky(generation_id):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("queue unavailable")
        return await original(generation_id)

    enqueuer.enqueue = flaky
    body = await _create(client, result_count=2)
    failed_id = body["generations"][1]["generation_id"]
    row = (await client.get(f"/v1/generations/{failed_id}")).json()
    assert row["error_code"] == "QUEUE_FAILED"


async def test_every_sibling_failing_is_a_503(client, app):
    """No result at all is an outage, not a partial success."""

    async def always_fails(generation_id):
        raise RuntimeError("queue unavailable")

    app.state.enqueuer.enqueue = always_fails
    resp = await client.post("/v1/generations", json=dict(PAYLOAD, result_count=2))
    assert resp.status_code == 503
    assert resp.json()["detail"] == "QUEUE_FAILED"


# ── 15. bulk actions ──────────────────────────────────────────────────


async def test_bulk_delete_removes_every_listed_song(client):
    ids = [await _generation(client), await _generation(client)]
    resp = await client.post("/v1/generations/bulk-delete", json={"ids": ids})

    assert resp.status_code == 200
    assert resp.json()["affected"] == 2
    for generation_id in ids:
        assert (await client.get(f"/v1/generations/{generation_id}")).status_code == 404


async def test_bulk_delete_reports_what_actually_existed(client):
    """A stale selection is normal; it must not abort the whole action."""
    real = await _generation(client)
    resp = await client.post("/v1/generations/bulk-delete", json={"ids": [real, str(uuid.uuid4())]})
    assert resp.json()["affected"] == 1
    assert (await client.get(f"/v1/generations/{real}")).status_code == 404


async def test_bulk_delete_needs_at_least_one_id(client):
    assert (await client.post("/v1/generations/bulk-delete", json={"ids": []})).status_code == 422


async def test_bulk_assign_files_every_song(client):
    project_id = await _project(client)
    ids = [await _generation(client), await _generation(client)]

    resp = await client.post(
        "/v1/generations/bulk-project", json={"ids": ids, "project_id": project_id}
    )

    assert resp.json()["affected"] == 2
    filed = (await client.get(f"/v1/projects/{project_id}/generations")).json()
    assert {g["id"] for g in filed["items"]} == set(ids)


async def test_bulk_unassign_returns_songs_to_unfiled(client):
    project_id = await _project(client)
    ids = [await _generation(client)]
    await client.post("/v1/generations/bulk-project", json={"ids": ids, "project_id": project_id})

    await client.post("/v1/generations/bulk-project", json={"ids": ids, "project_id": None})

    assert (await client.get(f"/v1/projects/{project_id}/generations")).json()["total"] == 0
    assert (await client.get(f"/v1/generations/{ids[0]}")).status_code == 200


async def test_bulk_assign_to_an_unknown_project_is_rejected(client):
    ids = [await _generation(client)]
    resp = await client.post(
        "/v1/generations/bulk-project", json={"ids": ids, "project_id": str(uuid.uuid4())}
    )
    assert resp.status_code == 422


# ── 13. project workflow ──────────────────────────────────────────────


async def test_assign_returns_the_updated_generation(client):
    """Phase 11 wrapped one row in a list; Phase 12 returns the row."""
    project_id = await _project(client)
    generation_id = await _generation(client)

    body = (
        await client.put(
            f"/v1/generations/{generation_id}/project", json={"project_id": project_id}
        )
    ).json()

    assert body["id"] == generation_id
    assert body["project_id"] == project_id


async def test_deleting_a_project_with_songs_keeps_the_songs(client):
    project_id = await _project(client)
    generation_id = await _generation(client)
    await client.put(f"/v1/generations/{generation_id}/project", json={"project_id": project_id})

    assert (await client.delete(f"/v1/projects/{project_id}")).status_code == 204

    assert (await client.get(f"/v1/generations/{generation_id}")).status_code == 200


async def test_project_counts_are_reported(client):
    project_id = await _project(client)
    generation_id = await _generation(client)
    await client.put(f"/v1/generations/{generation_id}/project", json={"project_id": project_id})

    listed = (await client.get("/v1/projects")).json()["items"]
    assert [p["generation_count"] for p in listed if p["id"] == project_id] == [1]
