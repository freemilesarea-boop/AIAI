"""Plan assignment for API tests.

In a module with a distinct name rather than in `conftest`, following the
convention this repository already uses for `asset_fixtures` and
`ops_fixtures`: a whole-repository pytest run puts several packages'
`conftest` modules on the same import path, and `from conftest import x`
resolves to whichever one was imported first. That failure only appears
in the full suite, never when a file is run on its own, which is the
worst way for it to appear.
"""

from __future__ import annotations

import uuid

from fastapi import FastAPI

from luber_database.allowance_repository import AllowanceRepository
from luber_schemas.plans import PlanId

#: The plan the ordinary client fixtures sign up on.
#:
#: Basic rather than Free, because most of the API suite tests
#: generation and download *mechanics* and would otherwise be asserting
#: billing refusals by accident. Free behaviour — no downloads, twenty
#: songs — is covered explicitly in `test_plans_api.py`, where it is the
#: subject rather than the backdrop.
FIXTURE_PLAN = PlanId.BASIC


async def set_plan(app: FastAPI, user_id: str, plan: PlanId) -> None:
    """Put an account on a plan, the way an operator script would.

    There is no endpoint for this on purpose, so tests that need a tier
    reach the repository directly rather than exercising a route the
    product does not have.
    """
    factory = app.state.session_factory
    async with factory() as session:
        await AllowanceRepository(session, uuid.UUID(user_id)).set_plan(plan)


__all__ = ["FIXTURE_PLAN", "set_plan"]
