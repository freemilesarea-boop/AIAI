"""What the plans are, and what this account has left.

Two endpoints and nothing more.

``GET /v1/plans`` is the catalogue: the same four tiers the pricing page
shows, served from `luber_schemas.plans` so the browser renders exactly
what the server will enforce. It needs no session — pricing is public
information, and requiring a login to read it would only mean the
marketing page had to hardcode its own copy of the numbers.

``GET /v1/account/entitlement`` is the caller's own standing: plan,
period, three counts, three booleans. It is authenticated and scoped, and
it reports only the account making the request — there is no user id
parameter to supply, so there is nothing to tamper with.

There is deliberately no endpoint that *sets* a plan. Until a payment
provider exists, any such route would be a way to take Creator for
nothing, whatever it was called. Development assigns plans through
``scripts/ops/set_plan.py``, which needs shell access to the server.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from luber_api.dependencies import get_allowance
from luber_api.settings import get_settings
from luber_database.allowance_repository import AllowanceRepository
from luber_schemas.plans import RECOMMENDED_PLAN, Plan, ordered_plans

router = APIRouter(prefix="/v1", tags=["plans"])


class PlanResponse(BaseModel):
    """One tier as the pricing page needs it."""

    plan_id: str
    display_name: str
    monthly_price_krw: int
    monthly_generation_limit: int
    download_mp3: bool
    download_wav: bool
    commercial_use: bool
    priority_level: int
    lab_access: bool
    #: Presentation only. Highlights a column; grants nothing.
    recommended: bool = False


class PlanCatalogueResponse(BaseModel):
    plans: list[PlanResponse]
    #: Whether a payment provider is configured on this deployment. The
    #: pricing page renders its subscribe controls from this, so it must
    #: reflect the server rather than a constant — hardcoding it false
    #: hides the CTA on a deployment that can take payments, and
    #: hardcoding it true offers a checkout that opens nothing.
    checkout_available: bool = False


class EntitlementResponse(BaseModel):
    """What this account may do right now.

    The minimum the frontend needs to render honestly. No subscription
    row id, no ledger, nothing about how any of it is stored.
    """

    plan: PlanResponse
    period_start: str
    period_end: str
    generation_limit: int
    generation_used: int
    generation_remaining: int = Field(ge=0)
    download_mp3: bool
    download_wav: bool
    commercial_use: bool


def _plan_payload(plan: Plan) -> PlanResponse:
    return PlanResponse(**plan.to_dict(), recommended=plan.plan_id is RECOMMENDED_PLAN)


@router.get("/plans", response_model=PlanCatalogueResponse)
async def list_plans() -> PlanCatalogueResponse:
    return PlanCatalogueResponse(
        plans=[_plan_payload(plan) for plan in ordered_plans()],
        checkout_available=get_settings().billing_available(),
    )


@router.get("/account/entitlement", response_model=EntitlementResponse)
async def get_entitlement(
    allowance: Annotated[AllowanceRepository, Depends(get_allowance)],
) -> EntitlementResponse:
    entitlement = await allowance.entitlement()
    payload = entitlement.to_dict()
    payload["plan"] = _plan_payload(entitlement.plan)
    return EntitlementResponse(**payload)


__all__ = ["EntitlementResponse", "PlanCatalogueResponse", "PlanResponse", "router"]
