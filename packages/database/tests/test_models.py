from luber_database import Base
from luber_database.models import User


def test_users_table_registered():
    assert "users" in Base.metadata.tables


def test_user_columns():
    # password_hash added by Phase 20A. The set is asserted exactly so a
    # column cannot appear here unnoticed — anything on this table is a
    # candidate for leaking through an API response.
    # `deleted_at` added by migration 0020 for account closure. It is
    # server-side state and must not reach a client — verified by
    # `public_user`, which builds `UserResponse` field by field rather
    # than from the row, so a new column cannot leak by default.
    #
    # `role` added by migration 0022 for the operator console. Unlike
    # `deleted_at` this one *is* returned by `/v1/auth/me`, deliberately:
    # the web app needs it to decide whether to draw a link to `/admin`.
    # It carries no authority — every `/v1/admin/*` request is checked
    # server-side against this same column on the session's own row, so a
    # browser that lies about it gets a nav item and 403s behind it.
    cols = {c.name for c in User.__table__.columns}
    assert cols == {
        "id",
        "email",
        "password_hash",
        "display_name",
        "created_at",
        "deleted_at",
        "role",
    }


def test_password_hash_is_nullable():
    """A user without a usable hash cannot be logged into.

    Part 2 relies on this: the ownership anchor for legacy rows is an
    account with no password rather than one with a password nobody
    knows.
    """
    assert User.__table__.columns["password_hash"].nullable


def test_session_columns():
    from luber_database.models.user import Session

    cols = {c.name for c in Session.__table__.columns}
    assert cols == {"id", "token_hash", "user_id", "created_at", "expires_at"}


def test_sessions_store_a_digest_not_a_token():
    """The column is named and sized for a SHA-256 hex digest."""
    from luber_database.models.user import Session

    token_hash = Session.__table__.columns["token_hash"]
    assert token_hash.type.length == 64
    assert token_hash.unique


def test_email_has_unique_index():
    users_table = Base.metadata.tables["users"]
    indexes = {idx.name: idx for idx in users_table.indexes}
    assert "ix_users_email" in indexes
    assert indexes["ix_users_email"].unique
