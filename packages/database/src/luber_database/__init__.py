"""Database layer for LUBER MUSIC AI.

Schema changes go through version-controlled Alembic migrations only
(``packages/database/alembic``). Destructive raw SQL against shared
databases is forbidden.
"""

from luber_database.auth_repository import AuthRepository
from luber_database.base import Base
from luber_database.engine import create_async_engine_from_url, create_session_factory
from luber_database.errors import GenerationHasDescendantsError
from luber_database.observability_repository import ObservabilityRepository
from luber_database.repository import GenerationRepository

__all__ = [
    "AuthRepository",
    "Base",
    "GenerationHasDescendantsError",
    "GenerationRepository",
    "ObservabilityRepository",
    "create_async_engine_from_url",
    "create_session_factory",
]
