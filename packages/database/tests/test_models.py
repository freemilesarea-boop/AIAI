from luber_database import Base
from luber_database.models import User


def test_users_table_registered():
    assert "users" in Base.metadata.tables


def test_user_columns():
    cols = {c.name for c in User.__table__.columns}
    assert cols == {"id", "email", "display_name", "created_at"}


def test_email_has_unique_index():
    users_table = Base.metadata.tables["users"]
    indexes = {idx.name: idx for idx in users_table.indexes}
    assert "ix_users_email" in indexes
    assert indexes["ix_users_email"].unique
