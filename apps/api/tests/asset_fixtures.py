"""Where a generation's bytes are stored, for tests that need the object.

`storage_key` is not part of the API: audio is addressed by generation id
and the server resolves the location itself. Server-side tests that need
the stored file — to probe its duration, to compare its bytes, to assert
it was deleted — still need the key, and the database is where it lives.

Kept in a distinctly named module rather than `conftest` for the reason
`conftest` itself records: several packages have one, and importing
`from conftest import ...` resolves to whichever the rootdir happened to
put on `sys.path`.
"""

from __future__ import annotations

import uuid
from typing import Any


async def asset_storage_keys(app: Any, generation_id: Any) -> dict[str, str]:
    """Storage keys for one generation's assets, keyed by asset type."""
    from sqlalchemy import select

    from luber_database.models.generation import AudioAsset

    async with app.state.session_factory() as session:
        rows = await session.execute(
            select(AudioAsset.asset_type, AudioAsset.storage_key).where(
                AudioAsset.generation_id == uuid.UUID(str(generation_id))
            )
        )
        return {str(asset_type): str(key) for asset_type, key in rows.all()}
