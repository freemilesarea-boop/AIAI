"""BOORDA's plans, defined once.

Every price, every allowance and every entitlement in the product comes
from this file. Nothing else may carry the number 19900, or 200, or the
fact that Free cannot download — a figure duplicated into a component is
a figure that will disagree with the server the first time it changes,
and the disagreement will be in the customer's favour or ours, never
neither.

The API serves these to the browser, so the frontend renders what the
backend will enforce rather than its own copy of the rules.

**Allowance is counted in songs, not credits.** One successfully
completed song is one unit, and the product says so in those words: a
user should never have to convert between a currency we invented and the
thing they wanted. The internal type is nonetheless a *cost per
generation* rather than a bare counter, so a later model that genuinely
costs more can charge more without a migration. For V1 every generation
costs exactly one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

#: What one successful generation costs against an allowance, today.
#:
#: A named constant rather than a literal `1` at the call site, because
#: the day a premium model costs two this is the thing that changes and
#: the call sites are already asking the right question.
STANDARD_GENERATION_COST = 1


class PlanId(StrEnum):
    """Stable internal identity. Never a display label.

    These strings live in the database and in API responses. Renaming
    "Basic" in the UI must not require a migration, which is exactly why
    the tier is `basic` here and `Basic` only in `display_name`.
    """

    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    CREATOR = "creator"


@dataclass(frozen=True)
class Plan:
    """One tier, and everything the product is allowed to decide from it."""

    plan_id: PlanId
    display_name: str
    monthly_price_krw: int
    #: Successful songs per allowance period. Failed generations never
    #: count against it — see the allowance ledger.
    monthly_generation_limit: int
    download_mp3: bool
    download_wav: bool
    #: Whether the plan's terms permit commercial use. Metadata only:
    #: this phase ships no licence document and the product must not
    #: claim more than the policy actually grants.
    commercial_use: bool
    #: Queue standing. Nothing consumes it yet; it exists so the value is
    #: configured in one place when a scheduler eventually reads it.
    priority_level: int
    #: Whether the tier may reach experimental models in LAB. Nothing in
    #: LAB is selectable yet, so nothing enforces this today.
    lab_access: bool

    @property
    def can_download(self) -> bool:
        return self.download_mp3 or self.download_wav

    @property
    def is_paid(self) -> bool:
        return self.monthly_price_krw > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id.value,
            "display_name": self.display_name,
            "monthly_price_krw": self.monthly_price_krw,
            "monthly_generation_limit": self.monthly_generation_limit,
            "download_mp3": self.download_mp3,
            "download_wav": self.download_wav,
            "commercial_use": self.commercial_use,
            "priority_level": self.priority_level,
            "lab_access": self.lab_access,
        }


#: The V1 tiers, in the order the pricing page shows them.
PLANS: dict[PlanId, Plan] = {
    PlanId.FREE: Plan(
        plan_id=PlanId.FREE,
        display_name="Free",
        monthly_price_krw=0,
        monthly_generation_limit=20,
        download_mp3=False,
        download_wav=False,
        commercial_use=False,
        priority_level=0,
        lab_access=False,
    ),
    PlanId.BASIC: Plan(
        plan_id=PlanId.BASIC,
        display_name="Basic",
        monthly_price_krw=19_900,
        monthly_generation_limit=200,
        download_mp3=True,
        download_wav=True,
        commercial_use=True,
        priority_level=1,
        lab_access=False,
    ),
    PlanId.PRO: Plan(
        plan_id=PlanId.PRO,
        display_name="Pro",
        monthly_price_krw=29_900,
        monthly_generation_limit=500,
        download_mp3=True,
        download_wav=True,
        commercial_use=True,
        priority_level=2,
        lab_access=True,
    ),
    PlanId.CREATOR: Plan(
        plan_id=PlanId.CREATOR,
        display_name="Creator",
        monthly_price_krw=49_900,
        monthly_generation_limit=1_000,
        download_mp3=True,
        download_wav=True,
        commercial_use=True,
        priority_level=3,
        lab_access=True,
    ),
}

#: Display order for the pricing page.
PLAN_ORDER: tuple[PlanId, ...] = (PlanId.FREE, PlanId.BASIC, PlanId.PRO, PlanId.CREATOR)

#: The tier the pricing page highlights. Presentation only — it grants
#: nothing and is not a discount.
RECOMMENDED_PLAN: PlanId = PlanId.PRO

#: What an account resolves to when it has no subscription row: every
#: user who existed before plans did, and every user who signs up.
DEFAULT_PLAN: PlanId = PlanId.FREE


def plan_for(plan_id: str | PlanId | None) -> Plan:
    """The plan for an id, falling back to Free.

    An unknown or missing id resolves to Free rather than raising. The
    caller is usually resolving a user's entitlement mid-request, and the
    safe failure there is the least-privileged tier — never an error page,
    and never a more generous plan than the one they pay for.
    """
    if plan_id is None:
        return PLANS[DEFAULT_PLAN]
    try:
        return PLANS[PlanId(str(plan_id))]
    except ValueError:
        return PLANS[DEFAULT_PLAN]


def ordered_plans() -> list[Plan]:
    return [PLANS[plan_id] for plan_id in PLAN_ORDER]


__all__ = [
    "DEFAULT_PLAN",
    "PLANS",
    "PLAN_ORDER",
    "RECOMMENDED_PLAN",
    "STANDARD_GENERATION_COST",
    "Plan",
    "PlanId",
    "ordered_plans",
    "plan_for",
]
