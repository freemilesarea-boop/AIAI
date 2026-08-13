"""Projects and lineage.

A project is a folder. The properties worth defending:

- **Deleting a folder never deletes the music in it.** The generations
  survive and become unfiled.
- Filing is reversible — `project_id: null` puts a generation back.
- Lineage reports what actually happened: a child is a *re-generation*
  that recorded its origin, not a mutation of its parent's audio.
"""

from __future__ import annotations

import uuid

PAYLOAD = {
    "title": "Filed Song",
    "prompt": "Dreamy Korean indie pop",
    "lyrics": "[Verse]\n오늘도 너를 기다려",
    "vocal_gender": "female",
    "duration": 30,
    "language": "ko",
}


async def _generation(client, **overrides) -> str:
    resp = await client.post("/v1/generations", json=dict(PAYLOAD, **overrides))
    assert resp.status_code == 202, resp.text
    return resp.json()["generation_id"]


async def _project(client, name: str = "My Project") -> str:
    resp = await client.post("/v1/projects", json={"name": name})
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


# ── CRUD ──────────────────────────────────────────────────────────────


async def test_create_and_read_a_project(client):
    project_id = await _project(client, "Summer EP")
    body = (await client.get(f"/v1/projects/{project_id}")).json()
    assert body["name"] == "Summer EP"
    assert body["generation_count"] == 0


async def test_project_names_are_trimmed_and_blank_is_rejected(client):
    resp = await client.post("/v1/projects", json={"name": "  Spaced  "})
    assert resp.json()["name"] == "Spaced"
    assert (await client.post("/v1/projects", json={"name": "   "})).status_code == 422
    assert (await client.post("/v1/projects", json={"name": ""})).status_code == 422


async def test_projects_are_listed_newest_first(client):
    first = await _project(client, "First")
    second = await _project(client, "Second")
    items = (await client.get("/v1/projects")).json()["items"]
    assert [i["id"] for i in items][:2] == [second, first]


async def test_rename_a_project(client):
    project_id = await _project(client, "Old")
    resp = await client.patch(f"/v1/projects/{project_id}", json={"name": "New"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "New"
    assert (await client.get(f"/v1/projects/{project_id}")).json()["name"] == "New"


async def test_missing_project_is_404(client):
    missing = uuid.uuid4()
    assert (await client.get(f"/v1/projects/{missing}")).status_code == 404
    assert (await client.patch(f"/v1/projects/{missing}", json={"name": "x"})).status_code == 404
    assert (await client.delete(f"/v1/projects/{missing}")).status_code == 404


# ── Membership ────────────────────────────────────────────────────────


async def test_assign_a_generation_to_a_project(client):
    project_id = await _project(client)
    generation_id = await _generation(client)

    resp = await client.put(
        f"/v1/generations/{generation_id}/project", json={"project_id": project_id}
    )
    assert resp.status_code == 200

    listed = (await client.get(f"/v1/projects/{project_id}/generations")).json()
    assert [i["id"] for i in listed["items"]] == [generation_id]
    assert (await client.get(f"/v1/projects/{project_id}")).json()["generation_count"] == 1


async def test_remove_a_generation_from_a_project(client):
    project_id = await _project(client)
    generation_id = await _generation(client)
    await client.put(f"/v1/generations/{generation_id}/project", json={"project_id": project_id})

    resp = await client.put(f"/v1/generations/{generation_id}/project", json={"project_id": None})
    assert resp.status_code == 200

    listed = (await client.get(f"/v1/projects/{project_id}/generations")).json()
    assert listed["items"] == []
    # The generation itself is untouched.
    assert (await client.get(f"/v1/generations/{generation_id}")).status_code == 200


async def test_moving_between_projects_leaves_only_one_home(client):
    first, second = await _project(client, "A"), await _project(client, "B")
    generation_id = await _generation(client)

    await client.put(f"/v1/generations/{generation_id}/project", json={"project_id": first})
    await client.put(f"/v1/generations/{generation_id}/project", json={"project_id": second})

    assert (await client.get(f"/v1/projects/{first}/generations")).json()["items"] == []
    assert len((await client.get(f"/v1/projects/{second}/generations")).json()["items"]) == 1


async def test_assigning_to_an_unknown_project_is_rejected(client):
    generation_id = await _generation(client)
    resp = await client.put(
        f"/v1/generations/{generation_id}/project", json={"project_id": str(uuid.uuid4())}
    )
    assert resp.status_code == 422


async def test_assigning_an_unknown_generation_is_404(client):
    project_id = await _project(client)
    resp = await client.put(
        f"/v1/generations/{uuid.uuid4()}/project", json={"project_id": project_id}
    )
    assert resp.status_code == 404


async def test_deleting_a_project_keeps_the_music(client):
    """The property that matters most: a folder is not a shredder."""
    project_id = await _project(client)
    generation_id = await _generation(client)
    await client.put(f"/v1/generations/{generation_id}/project", json={"project_id": project_id})

    assert (await client.delete(f"/v1/projects/{project_id}")).status_code == 204

    detail = await client.get(f"/v1/generations/{generation_id}")
    assert detail.status_code == 200
    assert detail.json()["status"] == "COMPLETED"
    assert detail.json()["audio_assets"]


async def test_generations_for_an_unknown_project_is_404(client):
    assert (await client.get(f"/v1/projects/{uuid.uuid4()}/generations")).status_code == 404


# ── Lineage ───────────────────────────────────────────────────────────


async def test_lineage_of_a_root_generation_is_empty(client):
    generation_id = await _generation(client)
    body = (await client.get(f"/v1/generations/{generation_id}/lineage")).json()
    assert body["parent"] is None
    assert body["children"] == []


async def test_lineage_reports_parent_and_children(client):
    root = await _generation(client, title="Original")
    child_a = await _generation(client, title="Take 2", parent_generation_id=root)
    child_b = await _generation(client, title="Take 3", parent_generation_id=root)

    root_view = (await client.get(f"/v1/generations/{root}/lineage")).json()
    assert root_view["parent"] is None
    assert {c["id"] for c in root_view["children"]} == {child_a, child_b}

    child_view = (await client.get(f"/v1/generations/{child_a}/lineage")).json()
    assert child_view["parent"]["id"] == root
    assert child_view["children"] == []


async def test_lineage_children_are_oldest_first(client):
    root = await _generation(client, title="Original")
    first = await _generation(client, title="A", parent_generation_id=root)
    second = await _generation(client, title="B", parent_generation_id=root)
    children = (await client.get(f"/v1/generations/{root}/lineage")).json()["children"]
    assert [c["id"] for c in children] == [first, second]


async def test_lineage_of_a_missing_generation_is_404(client):
    assert (await client.get(f"/v1/generations/{uuid.uuid4()}/lineage")).status_code == 404


async def test_lineage_survives_the_parent_being_deleted(client):
    root = await _generation(client, title="Original")
    child = await _generation(client, title="Take 2", parent_generation_id=root)
    await client.delete(f"/v1/generations/{root}")

    # The child is still readable; its lineage simply no longer resolves
    # a parent. Nothing 500s.
    body = (await client.get(f"/v1/generations/{child}/lineage")).json()
    assert body["children"] == []
