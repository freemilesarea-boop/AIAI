"""The internal legacy owner, and the doors it must not open.

Pre-authentication data had to belong to somebody, and the somebody must
not be a person. This anchor holds the 55 generations that predate
authentication so that no future signup inherits them.

Its whole security property is negative: it exists, it owns rows, and it
can never become an account. Both routes into an account are tested
here, because "obviously it can't" is how an ownership anchor quietly
turns into a login with everything attached to it.
"""

from __future__ import annotations

import uuid

from luber_api.security import hash_password
from luber_api.session import SESSION_COOKIE_NAME
from luber_database import AuthRepository

LEGACY_OWNER_ID = uuid.UUID("e3c4d3cd-d86f-52f2-91b7-2b97f5011653")
LEGACY_OWNER_EMAIL = "legacy-system@internal.luber"
PASSWORD = "correct horse battery staple"


async def make_anchor(app):
    """The anchor as the migration creates it: no password hash."""
    async with app.state.session_factory() as session:
        repository = AuthRepository(session)
        user = await repository.create_user(
            email=LEGACY_OWNER_EMAIL,
            password_hash=None,  # type: ignore[arg-type]
            display_name="Legacy system data (pre-authentication)",
        )
        return user


class TestDeterministicIdentity:
    def test_the_uuid_is_derived_from_the_email_not_invented(self):
        """Every database that runs the migration gets the same anchor.

        A per-install UUID would make the migration non-idempotent and
        the owner unidentifiable across environments.
        """
        assert uuid.uuid5(uuid.NAMESPACE_DNS, LEGACY_OWNER_EMAIL) == LEGACY_OWNER_ID

    def test_the_migration_hardcodes_that_exact_value(self):
        """The literal in the migration must match the derivation."""
        from pathlib import Path

        migration = (
            Path(__file__).resolve().parents[3]
            / "packages/database/alembic/versions/0014_legacy_ownership.py"
        ).read_text()
        assert str(LEGACY_OWNER_ID) in migration
        assert LEGACY_OWNER_EMAIL in migration

    def test_the_address_cannot_receive_mail(self):
        """`.luber` is not a real TLD, so it can never be verified."""
        assert LEGACY_OWNER_EMAIL.endswith(".luber")


class TestTheAnchorCannotBecomeAnAccount:
    async def test_it_has_no_password_hash(self, app):
        user = await make_anchor(app)
        assert user.password_hash is None

    async def test_login_refuses_it(self, app, client):
        """A NULL hash is rejected before any verification runs."""
        await make_anchor(app)
        response = await client.post(
            "/v1/auth/login", json={"email": LEGACY_OWNER_EMAIL, "password": PASSWORD}
        )
        assert response.status_code == 401
        assert SESSION_COOKIE_NAME not in response.cookies

    async def test_login_refuses_it_indistinguishably(self, app, client):
        """It must not stand out as a special account under probing."""
        await make_anchor(app)
        anchor = await client.post(
            "/v1/auth/login", json={"email": LEGACY_OWNER_EMAIL, "password": PASSWORD}
        )
        stranger = await client.post(
            "/v1/auth/login", json={"email": "nobody@example.com", "password": PASSWORD}
        )
        assert anchor.status_code == stranger.status_code
        assert anchor.json() == stranger.json()

    async def test_signup_cannot_claim_it(self, app, client):
        """The unique email is what stops a takeover.

        Signing up with the anchor's address must not attach a password
        to the row that owns every legacy generation.
        """
        await make_anchor(app)
        response = await client.post(
            "/v1/auth/signup",
            json={"email": LEGACY_OWNER_EMAIL, "password": PASSWORD},
        )
        assert response.status_code == 409
        assert SESSION_COOKIE_NAME not in response.cookies

    async def test_a_failed_takeover_leaves_the_anchor_unusable(self, app, client):
        """The row must still have no password after the attempt."""
        await make_anchor(app)
        await client.post(
            "/v1/auth/signup", json={"email": LEGACY_OWNER_EMAIL, "password": PASSWORD}
        )
        async with app.state.session_factory() as session:
            user = await AuthRepository(session).get_user_by_email(LEGACY_OWNER_EMAIL)
        assert user is not None
        assert user.password_hash is None

    async def test_case_variation_cannot_claim_it_either(self, app, client):
        """Signup normalises, so a capitalised address hits the same row."""
        await make_anchor(app)
        response = await client.post(
            "/v1/auth/signup",
            json={"email": "Legacy-System@Internal.Luber", "password": PASSWORD},
        )
        assert response.status_code == 409

    async def test_it_never_holds_a_session(self, app, client):
        await make_anchor(app)
        await client.post(
            "/v1/auth/login", json={"email": LEGACY_OWNER_EMAIL, "password": PASSWORD}
        )
        async with app.state.session_factory() as session:
            assert await AuthRepository(session).count_sessions(LEGACY_OWNER_ID) == 0


class TestNormalAuthStillWorks:
    """The anchor must not have made ordinary accounts harder to use."""

    async def test_a_normal_signup_succeeds_alongside_the_anchor(self, app, client):
        await make_anchor(app)
        response = await client.post(
            "/v1/auth/signup", json={"email": "person@example.com", "password": PASSWORD}
        )
        assert response.status_code == 201
        assert SESSION_COOKIE_NAME in response.cookies

    async def test_a_normal_login_and_me_still_work(self, app, client):
        await make_anchor(app)
        await client.post(
            "/v1/auth/signup", json={"email": "person@example.com", "password": PASSWORD}
        )
        await client.post("/v1/auth/logout")
        login = await client.post(
            "/v1/auth/login", json={"email": "person@example.com", "password": PASSWORD}
        )
        assert login.status_code == 200
        assert (await client.get("/v1/auth/me")).json()["email"] == "person@example.com"

    async def test_a_real_account_is_unaffected_by_the_anchor_existing(self, app, client):
        """Two rows, one with a hash and one without, must not interfere."""
        await make_anchor(app)
        async with app.state.session_factory() as session:
            repository = AuthRepository(session)
            await repository.create_user(
                email="real@example.com", password_hash=hash_password(PASSWORD)
            )
        response = await client.post(
            "/v1/auth/login", json={"email": "real@example.com", "password": PASSWORD}
        )
        assert response.status_code == 200


class TestOwnershipModel:
    def test_assets_inherit_rather_than_duplicate_an_owner(self):
        """A second owner column is a second thing that can disagree.

        An audio asset belongs to whoever owns its generation. Adding
        ``user_id`` here would create a value that can drift from the
        generation's and then has to be reconciled.
        """
        from luber_database.models.generation import AudioAsset

        assert "user_id" not in AudioAsset.__table__.columns
        assert "generation_id" in AudioAsset.__table__.columns

    def test_the_three_owned_tables_require_an_owner(self):
        from luber_database.models.generation import Generation, Project, ReferenceAudio

        for model in (Generation, Project, ReferenceAudio):
            column = model.__table__.columns["user_id"]
            assert not column.nullable, f"{model.__tablename__}.user_id is nullable"
            assert column.index, f"{model.__tablename__}.user_id is not indexed"

    def test_owned_tables_point_at_users(self):
        from luber_database.models.generation import Generation, Project, ReferenceAudio

        for model in (Generation, Project, ReferenceAudio):
            targets = {
                fk.column.table.name for fk in model.__table__.columns["user_id"].foreign_keys
            }
            assert targets == {"users"}, f"{model.__tablename__}: {targets}"
