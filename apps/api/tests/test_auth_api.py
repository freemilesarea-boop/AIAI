"""Authentication core: what it accepts, and what it refuses.

Most of the value here is in the refusals. An auth system that logs the
right person in is easy; one that also declines to say whether an email
is registered, declines to be authenticated by a forgeable header, and
declines to keep working after logout is the part worth testing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from luber_api.security import (
    MIN_PASSWORD_LENGTH,
    hash_password,
    hash_session_token,
    normalise_email,
    verify_password,
)
from luber_api.session import SESSION_COOKIE_NAME
from luber_database import AuthRepository

PASSWORD = "correct horse battery staple"


async def signup(client, email="a@example.com", password=PASSWORD, **extra):
    return await client.post(
        "/v1/auth/signup", json={"email": email, "password": password, **extra}
    )


class TestPasswordHashing:
    def test_the_hash_is_not_the_password(self):
        stored = hash_password(PASSWORD)
        assert PASSWORD not in stored
        assert stored != PASSWORD

    def test_it_is_argon2id(self):
        assert hash_password(PASSWORD).startswith("$argon2id$")

    def test_the_same_password_hashes_differently_each_time(self):
        """Distinct salts: two users with one password must not match."""
        assert hash_password(PASSWORD) != hash_password(PASSWORD)

    def test_the_right_password_verifies(self):
        assert verify_password(PASSWORD, hash_password(PASSWORD))

    def test_the_wrong_password_does_not(self):
        assert not verify_password("something else entirely", hash_password(PASSWORD))

    def test_a_corrupt_stored_hash_fails_the_login_rather_than_the_request(self):
        assert verify_password(PASSWORD, "not-a-hash") is False

    def test_a_long_passphrase_is_not_truncated(self):
        """Silent truncation makes a long password weaker than it looks."""
        long_one = "a" * 200 + "TAIL"
        stored = hash_password(long_one)
        assert verify_password(long_one, stored)
        assert not verify_password("a" * 200 + "XXXX", stored)


class TestEmailNormalisation:
    def test_case_and_surrounding_space_are_ignored(self):
        assert normalise_email("  Person@Example.COM ") == "person@example.com"

    def test_plus_tags_and_dots_are_preserved(self):
        """Provider-specific rewriting merges addresses users keep apart."""
        assert normalise_email("a.b+tag@example.com") == "a.b+tag@example.com"


class TestSignup:
    async def test_it_creates_an_account_and_a_session(self, client):
        response = await signup(client)
        assert response.status_code == 201
        assert response.json()["email"] == "a@example.com"
        assert SESSION_COOKIE_NAME in response.cookies

    async def test_the_password_hash_never_reaches_the_client(self, client):
        response = await signup(client)
        assert "password" not in response.text.lower()
        assert "argon2" not in response.text

    async def test_the_email_is_stored_normalised(self, client):
        await signup(client, email="MiXeD@Example.com")
        response = await client.post(
            "/v1/auth/login", json={"email": "mixed@example.com", "password": PASSWORD}
        )
        assert response.status_code == 200

    async def test_a_duplicate_email_is_refused(self, client):
        await signup(client)
        assert (await signup(client)).status_code == 409

    async def test_a_duplicate_differing_only_by_case_is_refused(self, client):
        await signup(client, email="dup@example.com")
        assert (await signup(client, email="DUP@example.com")).status_code == 409

    async def test_a_malformed_email_is_refused(self, client):
        assert (await signup(client, email="not-an-email")).status_code == 422

    async def test_a_short_password_is_refused(self, client):
        response = await signup(client, password="a" * (MIN_PASSWORD_LENGTH - 1))
        assert response.status_code == 422

    async def test_a_display_name_is_optional(self, client):
        assert (await signup(client)).json()["display_name"] is None

    async def test_a_display_name_is_kept_when_given(self, client):
        response = await signup(client, display_name="Jin")
        assert response.json()["display_name"] == "Jin"


class TestLogin:
    async def test_correct_credentials_succeed(self, client):
        await signup(client)
        response = await client.post(
            "/v1/auth/login", json={"email": "a@example.com", "password": PASSWORD}
        )
        assert response.status_code == 200
        assert SESSION_COOKIE_NAME in response.cookies

    async def test_a_wrong_password_is_refused(self, client):
        await signup(client)
        response = await client.post(
            "/v1/auth/login", json={"email": "a@example.com", "password": "wrong password!!"}
        )
        assert response.status_code == 401

    async def test_an_unknown_email_is_refused_identically(self, client):
        """The account-enumeration defence.

        A different status or message here would turn login into a
        membership oracle for any address someone cares to try.
        """
        await signup(client)
        wrong_password = await client.post(
            "/v1/auth/login", json={"email": "a@example.com", "password": "wrong password!!"}
        )
        unknown_email = await client.post(
            "/v1/auth/login", json={"email": "nobody@example.com", "password": PASSWORD}
        )
        assert wrong_password.status_code == unknown_email.status_code == 401
        assert wrong_password.json() == unknown_email.json()

    async def test_logging_in_again_issues_a_different_session(self, client):
        await signup(client)
        first = (
            await client.post(
                "/v1/auth/login", json={"email": "a@example.com", "password": PASSWORD}
            )
        ).cookies[SESSION_COOKIE_NAME]
        second = (
            await client.post(
                "/v1/auth/login", json={"email": "a@example.com", "password": PASSWORD}
            )
        ).cookies[SESSION_COOKIE_NAME]
        assert first != second


class TestSessionCookie:
    async def test_it_is_httponly_and_lax_and_rooted(self, client):
        response = await signup(client)
        header = response.headers["set-cookie"].lower()
        assert "httponly" in header
        assert "samesite=lax" in header
        assert "path=/" in header

    async def test_it_is_not_secure_on_a_development_origin(self, client):
        """A Secure cookie is dropped over plain HTTP, so login would
        appear to work and every later request would be anonymous."""
        assert "secure" not in response_cookie(await signup(client))

    async def test_it_sets_no_domain(self, client):
        assert "domain=" not in response_cookie(await signup(client))

    async def test_it_carries_an_expiry(self, client):
        assert "max-age=" in response_cookie(await signup(client))

    async def test_the_cookie_value_is_not_the_stored_value(self, app, client):
        """A database dump must not contain usable sessions."""
        token = (await signup(client)).cookies[SESSION_COOKIE_NAME]
        async with app.state.session_factory() as session:
            repository = AuthRepository(session)
            assert (
                await repository.get_session_user(hash_session_token(token), now=datetime.now(UTC))
                is not None
            )
            # The raw token itself is not what is stored.
            assert await repository.get_session_user(token, now=datetime.now(UTC)) is None


def response_cookie(response) -> str:
    return response.headers["set-cookie"].lower()


class TestMe:
    async def test_it_returns_the_signed_in_user(self, client):
        await signup(client)
        response = await client.get("/v1/auth/me")
        assert response.status_code == 200
        assert response.json()["email"] == "a@example.com"

    async def test_an_anonymous_caller_gets_401(self, client):
        assert (await client.get("/v1/auth/me")).status_code == 401

    async def test_a_made_up_cookie_gets_401(self, client):
        client.cookies.set(SESSION_COOKIE_NAME, "not-a-real-token")
        assert (await client.get("/v1/auth/me")).status_code == 401

    async def test_an_x_user_id_header_cannot_authenticate(self, app, client):
        """The pre-auth placeholder must never produce a user.

        X-User-Id is forgeable by anyone. It still exists for
        unauthenticated product flows until Part 3 removes it, so this
        test pins the boundary: it does not reach the auth dependency.
        """
        created = await signup(client)
        user_id = created.json()["id"]
        client.cookies.clear()
        response = await client.get("/v1/auth/me", headers={"X-User-Id": user_id})
        assert response.status_code == 401

    async def test_an_expired_session_stops_authenticating(self, app, client):
        token = (await signup(client)).cookies[SESSION_COOKIE_NAME]
        async with app.state.session_factory() as session:
            repository = AuthRepository(session)
            user = await repository.get_session_user(
                hash_session_token(token), now=datetime.now(UTC)
            )
            assert user is not None
            # Expiry is evaluated server-side, so moving the clock
            # forward is enough — no cookie change required.
            assert (
                await repository.get_session_user(
                    hash_session_token(token), now=datetime.now(UTC) + timedelta(days=365)
                )
                is None
            )


class TestLogout:
    async def test_it_ends_the_session_for_good(self, client):
        await signup(client)
        assert (await client.get("/v1/auth/me")).status_code == 200
        assert (await client.post("/v1/auth/logout")).status_code == 204
        assert (await client.get("/v1/auth/me")).status_code == 401

    async def test_a_retained_token_is_useless_afterwards(self, client):
        """Logout destroys the server row, not just the client's copy."""
        token = (await signup(client)).cookies[SESSION_COOKIE_NAME]
        await client.post("/v1/auth/logout")
        client.cookies.set(SESSION_COOKIE_NAME, token)
        assert (await client.get("/v1/auth/me")).status_code == 401

    async def test_logging_out_twice_is_harmless(self, client):
        await signup(client)
        await client.post("/v1/auth/logout")
        assert (await client.post("/v1/auth/logout")).status_code == 204

    async def test_an_anonymous_logout_is_harmless(self, client):
        assert (await client.post("/v1/auth/logout")).status_code == 204


class TestSessionCleanup:
    async def test_expired_sessions_are_removed_and_live_ones_kept(self, app, client):
        await signup(client, email="live@example.com")
        async with app.state.session_factory() as session:
            repository = AuthRepository(session)
            user = await repository.get_user_by_email("live@example.com")
            await repository.create_session(
                token_hash="e" * 64,
                user_id=user.id,
                expires_at=datetime.now(UTC) - timedelta(hours=1),
            )
            removed = await repository.delete_expired_sessions(now=datetime.now(UTC))
            assert removed == 1
            # The live session from signup survives.
            assert await repository.count_sessions(user.id) == 1

    async def test_cleanup_never_touches_users(self, app, client):
        await signup(client, email="kept@example.com")
        async with app.state.session_factory() as session:
            repository = AuthRepository(session)
            await repository.delete_expired_sessions(now=datetime.now(UTC) + timedelta(days=999))
            assert await repository.get_user_by_email("kept@example.com") is not None


class TestRateLimiting:
    async def test_repeated_failures_are_eventually_refused(self, client, app):
        """Bounded, so credential stuffing costs something."""
        await signup(client, email="target@example.com")
        limit = (
            app.state.settings.auth_rate_limit_attempts if hasattr(app.state, "settings") else 10
        )
        statuses = []
        for _ in range(limit + 3):
            response = await client.post(
                "/v1/auth/login",
                json={"email": "target@example.com", "password": "wrong password!!"},
            )
            statuses.append(response.status_code)
        assert 429 in statuses, f"never rate limited: {statuses}"

    async def test_a_refusal_says_when_to_come_back(self, client):
        for _ in range(30):
            response = await client.post(
                "/v1/auth/login", json={"email": "x@example.com", "password": "wrong password!!"}
            )
            if response.status_code == 429:
                assert "retry-after" in {k.lower() for k in response.headers}
                return
        pytest.fail("rate limit never triggered")


class TestOriginValidation:
    """CSRF defence in depth behind SameSite=Lax.

    Lax already keeps the cookie off cross-site POSTs, but that is a
    browser behaviour. This check is ours, and it is the one that fails
    closed if a future cookie attribute is loosened.
    """

    async def test_a_foreign_origin_is_refused_on_an_unsafe_method(self, client):
        await signup(client)
        response = await client.post(
            "/v1/auth/login",
            json={"email": "a@example.com", "password": PASSWORD},
            headers={"Origin": "http://evil.example"},
        )
        assert response.status_code == 403

    async def test_the_products_own_origin_is_accepted(self, client, app):
        await signup(client)
        allowed = app.state.settings.cors_origins[0] if hasattr(app.state, "settings") else None
        origin = allowed or "http://localhost:3000"
        response = await client.post(
            "/v1/auth/login",
            json={"email": "a@example.com", "password": PASSWORD},
            headers={"Origin": origin},
        )
        assert response.status_code == 200

    async def test_a_request_with_no_origin_is_allowed(self, client):
        """curl, the tests and server-to-server callers send none, and
        they carry no ambient cookie for an attacker to ride."""
        await signup(client)
        response = await client.post(
            "/v1/auth/login", json={"email": "a@example.com", "password": PASSWORD}
        )
        assert response.status_code == 200

    async def test_a_safe_method_is_never_blocked_by_origin(self, client):
        """A cross-origin GET cannot change state, and blocking it would
        break legitimate reads without buying anything."""
        await signup(client)
        response = await client.get("/v1/auth/me", headers={"Origin": "http://evil.example"})
        assert response.status_code == 200
