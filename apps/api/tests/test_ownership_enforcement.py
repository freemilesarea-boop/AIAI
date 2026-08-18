"""Adversarial ownership: what one user can learn about another.

The suite is written from the attacker's side. Every test asks a
question a malicious client would ask — can I read this, download it,
edit it, delete it, name it as a parent, add it to my project — and
requires the answer to be indistinguishable from "no such thing".

Two properties are load-bearing and are asserted repeatedly rather than
once:

* **401 for anonymous, 404 for cross-user.** A 403 would confirm that a
  UUID belongs to somebody, which is the fact being protected.
* **Identity comes only from the session.** ``X-User-Id`` is no longer
  read anywhere; several tests set it hostilely to prove it is inert.
"""

from __future__ import annotations

import uuid

import pytest

from luber_schemas import LEGACY_OWNER_ID

PAYLOAD = {
    "title": "Song",
    "prompt": "bright synth pop",
    "lyrics": "",
    "vocal_gender": "instrumental",
    "duration": 30,
    "language": "en",
    "instrumental": True,
}


async def make_generation(http) -> str:
    response = await http.post("/v1/generations", json=PAYLOAD)
    assert response.status_code == 202, response.text
    return str(response.json()["generation_id"])


async def make_project(http, name="Album") -> str:
    response = await http.post("/v1/projects", json={"name": name})
    assert response.status_code in (200, 201), response.text
    return str(response.json()["id"])


# ── anonymous ─────────────────────────────────────────────────────────


class TestAnonymousIsRefused:
    """No session, no product. 401 before ownership is even consulted."""

    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("get", "/v1/generations"),
            ("post", "/v1/generations"),
            ("post", "/v1/generations/preflight"),
            ("get", "/v1/generations/{id}"),
            ("patch", "/v1/generations/{id}"),
            ("delete", "/v1/generations/{id}"),
            ("get", "/v1/generations/{id}/audio"),
            ("get", "/v1/generations/{id}/lineage"),
            ("post", "/v1/generations/{id}/extend"),
            ("post", "/v1/generations/{id}/replace-range"),
            ("post", "/v1/generations/{id}/cover"),
            ("put", "/v1/generations/{id}/project"),
            ("get", "/v1/generations/{id}/qa"),
            ("put", "/v1/generations/{id}/qa"),
            ("get", "/v1/generations/{id}/longform-qa"),
            ("post", "/v1/generations/bulk-delete"),
            ("post", "/v1/generations/bulk-project"),
            ("get", "/v1/generations/groups/{id}"),
            ("get", "/v1/projects"),
            ("post", "/v1/projects"),
            ("get", "/v1/projects/{id}"),
            ("patch", "/v1/projects/{id}"),
            ("delete", "/v1/projects/{id}"),
            ("get", "/v1/projects/{id}/generations"),
            ("post", "/v1/reference-audio"),
            ("get", "/v1/reference-audio/limits"),
        ],
    )
    async def test_every_product_operation_requires_a_session(self, anon_client, method, path):
        url = path.replace("{id}", str(uuid.uuid4()))
        # GET and DELETE take no body in httpx; the others need one to
        # reach the auth dependency rather than failing validation first.
        call = getattr(anon_client, method)
        response = await (call(url) if method in {"get", "delete"} else call(url, json={}))
        assert response.status_code == 401, f"{method.upper()} {path} → {response.status_code}"

    async def test_health_and_ready_stay_public(self, anon_client):
        """A probe has no cookie; an auth failure would look like an outage."""
        assert (await anon_client.get("/health")).status_code == 200
        assert (await anon_client.get("/ready")).status_code in (200, 503)


# ── cross-user ────────────────────────────────────────────────────────


class TestUserACannotReachUserB:
    async def test_b_generation_detail_is_absent(self, client, client_b):
        theirs = await make_generation(client_b)
        assert (await client.get(f"/v1/generations/{theirs}")).status_code == 404

    async def test_b_audio_is_absent(self, client, client_b):
        theirs = await make_generation(client_b)
        assert (await client.get(f"/v1/generations/{theirs}/audio")).status_code == 404

    async def test_b_lineage_is_absent(self, client, client_b):
        theirs = await make_generation(client_b)
        assert (await client.get(f"/v1/generations/{theirs}/lineage")).status_code == 404

    async def test_b_generation_cannot_be_renamed(self, client, client_b):
        theirs = await make_generation(client_b)
        response = await client.patch(f"/v1/generations/{theirs}", json={"title": "mine now"})
        assert response.status_code == 404

    async def test_b_generation_cannot_be_extended(self, client, client_b):
        theirs = await make_generation(client_b)
        response = await client.post(f"/v1/generations/{theirs}/extend", json={"seconds": 15})
        assert response.status_code == 404

    async def test_b_generation_cannot_be_replaced(self, client, client_b):
        theirs = await make_generation(client_b)
        response = await client.post(
            f"/v1/generations/{theirs}/replace-range",
            json={"start_seconds": 0.5, "end_seconds": 1.5},
        )
        assert response.status_code == 404

    async def test_b_generation_cannot_be_covered(self, client, client_b):
        theirs = await make_generation(client_b)
        response = await client.post(f"/v1/generations/{theirs}/cover", json={"prompt": "acoustic"})
        assert response.status_code == 404

    async def test_b_generation_cannot_be_deleted(self, client, client_b):
        theirs = await make_generation(client_b)
        assert (await client.delete(f"/v1/generations/{theirs}")).status_code == 404
        # And it is still there for its owner.
        assert (await client_b.get(f"/v1/generations/{theirs}")).status_code == 200

    async def test_b_project_is_absent(self, client, client_b):
        theirs = await make_project(client_b)
        assert (await client.get(f"/v1/projects/{theirs}")).status_code == 404
        assert (await client.patch(f"/v1/projects/{theirs}", json={"name": "x"})).status_code == 404
        assert (await client.delete(f"/v1/projects/{theirs}")).status_code == 404

    async def test_b_generation_cannot_be_filed_into_an_a_project(self, client, client_b):
        """The association both sides must own."""
        mine = await make_project(client)
        theirs = await make_generation(client_b)
        response = await client.put(f"/v1/generations/{theirs}/project", json={"project_id": mine})
        assert response.status_code == 404

    async def test_an_a_generation_cannot_be_filed_into_a_b_project(self, client, client_b):
        mine = await make_generation(client)
        theirs = await make_project(client_b)
        response = await client.put(f"/v1/generations/{mine}/project", json={"project_id": theirs})
        assert response.status_code in (404, 422)

    async def test_the_inverse_holds_for_b(self, client, client_b):
        """Not a special case for one account."""
        mine = await make_generation(client)
        assert (await client_b.get(f"/v1/generations/{mine}")).status_code == 404
        assert (await client_b.get(f"/v1/generations/{mine}/audio")).status_code == 404
        assert (await client_b.delete(f"/v1/generations/{mine}")).status_code == 404


class TestIndistinguishability:
    async def test_a_foreign_id_looks_exactly_like_a_missing_one(self, client, client_b):
        theirs = await make_generation(client_b)
        foreign = await client.get(f"/v1/generations/{theirs}")
        missing = await client.get(f"/v1/generations/{uuid.uuid4()}")
        assert foreign.status_code == missing.status_code == 404
        assert foreign.json() == missing.json()

    async def test_nothing_about_the_foreign_song_is_returned(self, client, client_b):
        response = await client_b.post("/v1/generations", json=dict(PAYLOAD, title="SECRET NAME"))
        theirs = str(response.json()["generation_id"])
        body = (await client.get(f"/v1/generations/{theirs}")).text
        assert "SECRET NAME" not in body


# ── listing and totals ────────────────────────────────────────────────


class TestListingIsScoped:
    async def test_a_fresh_account_sees_nothing(self, client):
        body = (await client.get("/v1/generations")).json()
        assert body["items"] == []
        assert body["total"] == 0

    async def test_the_total_does_not_count_other_users(self, client, client_b):
        """A correct page with a global count still leaks corpus size."""
        for _ in range(2):
            await make_generation(client_b)
        await make_generation(client)

        body = (await client.get("/v1/generations")).json()
        assert body["total"] == 1
        assert len(body["items"]) == 1

    async def test_projects_are_scoped(self, client, client_b):
        await make_project(client_b, name="Theirs")
        body = (await client.get("/v1/projects")).json()
        names = [p["name"] for p in (body["items"] if isinstance(body, dict) else body)]
        assert "Theirs" not in names


# ── spoofing ──────────────────────────────────────────────────────────


class TestHeaderSpoofingIsInert:
    async def test_claiming_another_users_id_grants_nothing(self, client, client_b):
        theirs = await make_generation(client_b)
        response = await client.get(
            f"/v1/generations/{theirs}", headers={"X-User-Id": str(client_b.user_id)}
        )
        assert response.status_code == 404

    async def test_claiming_another_users_id_does_not_change_the_library(self, client, client_b):
        await make_generation(client_b)
        body = (
            await client.get("/v1/generations", headers={"X-User-Id": str(client_b.user_id)})
        ).json()
        assert body["total"] == 0

    async def test_a_created_row_belongs_to_the_session_not_the_header(self, client, client_b, app):
        """Creation attribution comes from the session alone."""
        from luber_database import GenerationRepository

        response = await client.post(
            "/v1/generations", json=PAYLOAD, headers={"X-User-Id": str(client_b.user_id)}
        )
        generation_id = uuid.UUID(response.json()["generation_id"])
        async with app.state.session_factory() as session:
            row = await GenerationRepository(session, owner=None).get_generation(generation_id)
        assert str(row.user_id) == client.user_id


# ── legacy corpus ─────────────────────────────────────────────────────


class TestLegacyCorpusIsInvisible:
    """The 55 historical generations belong to an account nobody can use."""

    async def test_a_legacy_generation_is_not_reachable(self, client, app):
        from luber_database import GenerationRepository

        async with app.state.session_factory() as session:
            legacy = await GenerationRepository(session, owner=LEGACY_OWNER_ID).create_generation(
                title="HISTORICAL",
                prompt="p",
                lyrics="",
                vocal_gender="instrumental",
                duration_requested=30,
                status="COMPLETED",
            )
        assert (await client.get(f"/v1/generations/{legacy.id}")).status_code == 404
        assert (await client.get(f"/v1/generations/{legacy.id}/audio")).status_code == 404
        assert (await client.get(f"/v1/generations/{legacy.id}/lineage")).status_code == 404

    async def test_legacy_rows_never_appear_in_a_library(self, client, app):
        from luber_database import GenerationRepository

        async with app.state.session_factory() as session:
            await GenerationRepository(session, owner=LEGACY_OWNER_ID).create_generation(
                title="HISTORICAL",
                prompt="p",
                lyrics="",
                vocal_gender="instrumental",
                duration_requested=30,
                status="COMPLETED",
            )
        body = (await client.get("/v1/generations")).json()
        assert body["total"] == 0
        assert body["items"] == []


# ── bulk operations ───────────────────────────────────────────────────


class TestBulkIsOwnerSafe:
    async def test_a_mixed_batch_deletes_only_the_callers_rows(self, client, client_b):
        mine = await make_generation(client)
        theirs = await make_generation(client_b)

        response = await client.post(
            "/v1/generations/bulk-delete",
            json={"ids": [mine, theirs, str(uuid.uuid4())]},
        )
        assert response.status_code == 200

        assert (await client.get(f"/v1/generations/{mine}")).status_code == 404
        # Untouched, and still theirs.
        assert (await client_b.get(f"/v1/generations/{theirs}")).status_code == 200

    async def test_the_response_does_not_say_which_ids_were_foreign(self, client, client_b):
        """Reporting them would confirm those ids exist."""
        theirs = await make_generation(client_b)
        response = await client.post("/v1/generations/bulk-delete", json={"ids": [theirs]})
        assert theirs not in response.text


# ── creation attribution ──────────────────────────────────────────────


class TestCreationOwnership:
    async def test_a_new_generation_belongs_to_its_creator(self, client, app):
        from luber_database import GenerationRepository

        generation_id = uuid.UUID(await make_generation(client))
        async with app.state.session_factory() as session:
            row = await GenerationRepository(session, owner=None).get_generation(generation_id)
        assert str(row.user_id) == client.user_id

    async def test_it_is_never_attributed_to_the_legacy_anchor(self, client, app):
        """The Part 2 bridge must be unreachable from the product."""
        from luber_database import GenerationRepository

        generation_id = uuid.UUID(await make_generation(client))
        async with app.state.session_factory() as session:
            row = await GenerationRepository(session, owner=None).get_generation(generation_id)
        assert row.user_id != LEGACY_OWNER_ID

    async def test_a_project_belongs_to_its_creator(self, client, app):
        from luber_database import GenerationRepository

        project_id = uuid.UUID(await make_project(client))
        async with app.state.session_factory() as session:
            project = await GenerationRepository(session, owner=None).get_project(project_id)
        assert str(project.user_id) == client.user_id

    async def test_a_descendant_belongs_to_the_actor(self, client, app):
        """Lineage children take the session user, not the parent row."""
        from luber_database import GenerationRepository

        parent = await make_generation(client)
        response = await client.post(
            "/v1/generations", json=dict(PAYLOAD, parent_generation_id=parent)
        )
        assert response.status_code == 202
        child_id = uuid.UUID(response.json()["generation_id"])
        async with app.state.session_factory() as session:
            child = await GenerationRepository(session, owner=None).get_generation(child_id)
        assert str(child.user_id) == client.user_id


# ── CSRF ──────────────────────────────────────────────────────────────


class TestOriginOnProductMutations:
    async def test_a_foreign_origin_cannot_mutate(self, client):
        response = await client.post(
            "/v1/generations", json=PAYLOAD, headers={"Origin": "http://evil.example"}
        )
        assert response.status_code == 403

    async def test_a_foreign_origin_cannot_delete(self, client):
        mine = await make_generation(client)
        response = await client.delete(
            f"/v1/generations/{mine}", headers={"Origin": "http://evil.example"}
        )
        assert response.status_code == 403

    async def test_reads_are_not_blocked_by_origin(self, client):
        """A cross-origin GET changes nothing and must not break."""
        response = await client.get("/v1/generations", headers={"Origin": "http://evil.example"})
        assert response.status_code == 200

    async def test_the_products_own_origin_works(self, client):
        response = await client.post(
            "/v1/generations", json=PAYLOAD, headers={"Origin": "http://localhost:3000"}
        )
        assert response.status_code == 202
