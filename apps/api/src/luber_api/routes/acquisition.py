"""Recording where a visit came from.

One public endpoint, because anonymous visitors are the entire point:
by definition nobody has signed in yet when the interesting part
happens. Public *ingestion*, though — nothing here reports anything, and
every read is behind `/v1/admin/acquisition`.

What that costs and how it is bounded:

**It writes at most one row pair per browser per visit**, and is rate
limited per visitor cookie and per address, so a script pointed at it
inflates its own visitor's session count and nothing else.

**It stores nothing it was not asked for.** The body carries campaign
parameters and a referrer; the sanitiser drops anything matching the
sensitive denylist before classification, and the landing path is
stripped of its query string entirely. No IP address, no user agent, no
fingerprint of any kind is recorded.

**A failure here is silent.** Analytics that breaks the page it measures
is worse than no analytics, so every error path returns 204 and the
visitor never knows. The one thing the endpoint owes the browser is a
cookie.
"""

from __future__ import annotations

import logging
import uuid
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Request, Response, status
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from luber_api.dependencies import get_redis, get_session_factory
from luber_api.rate_limit import RateLimitExceeded, enforce_rate_limit
from luber_api.settings import ApiSettings, get_settings
from luber_database.acquisition_repository import AcquisitionRepository
from luber_schemas.acquisition import (
    CAMPAIGN_PARAMS,
    CLICK_IDS,
    classify,
    is_self_referral,
    normalise_path,
    referrer_host,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/acquisition", tags=["acquisition"])

#: The first-party cookie naming this browser. Same shape as the session
#: cookie, and deliberately not readable by scripts.
VISITOR_COOKIE_NAME = "boorda_visitor"

#: How long a visitor stays the same visitor.
#:
#: 400 days is the ceiling Chrome enforces on cookie lifetime, so asking
#: for more would be asking for something no browser grants. This is a
#: cookie lifetime, not a data retention period — retention is a policy
#: BOORDA has not set. See docs/ACQUISITION_ANALYTICS.md.
VISITOR_COOKIE_MAX_AGE = 400 * 24 * 60 * 60

#: Visits one browser may record in an hour. Generous for a person
#: opening a few tabs; low enough that a loop cannot fill the table.
VISIT_RATE_LIMIT = 60
VISIT_RATE_WINDOW_SECONDS = 60 * 60

#: Paths that are not acquisition. The console, the operator tools and
#: anything under the API are navigation by people already here.
IGNORED_PREFIXES = ("/admin", "/ops", "/api", "/_next")


class VisitRequest(BaseModel):
    """What the browser may say about a visit.

    Everything is optional and nothing is trusted. There is no user id
    here and no place to put one: a visit says which *browser*, and the
    account it belongs to is decided only on the server's own signup
    path.
    """

    model_config = {"extra": "forbid"}

    path: str | None = Field(default=None, max_length=2048)
    referrer: str | None = Field(default=None, max_length=2048)
    #: Campaign and click parameters, as the browser saw them. Filtered
    #: through the allowlist and denylist before anything is stored.
    params: dict[str, str] | None = None


def _visitor_cookie(request: Request) -> UUID | None:
    """The visitor id this browser already carries, if it is a real one."""
    raw = request.cookies.get(VISITOR_COOKIE_NAME)
    if not raw:
        return None
    try:
        return UUID(raw)
    except ValueError:
        # A hand-edited or corrupted cookie is treated as absent, and
        # replaced below. Nothing is authorised by this value, so a bad
        # one is a nuisance rather than a risk.
        return None


def set_visitor_cookie(response: Response, visitor_key: UUID, *, settings: ApiSettings) -> None:
    """Attach the first-party visitor cookie.

    `HttpOnly` because nothing in the browser needs to read it — the
    server manages it end to end — and a value scripts cannot reach is
    one an XSS bug cannot exfiltrate. `SameSite=Lax` so it survives the
    click that brings someone here from a campaign link, which is the
    only navigation that matters for attribution.
    """
    response.set_cookie(
        VISITOR_COOKIE_NAME,
        str(visitor_key),
        max_age=VISITOR_COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=settings.is_production,
        path="/",
    )


@router.post("/visit", status_code=status.HTTP_204_NO_CONTENT)
async def record_visit(
    payload: VisitRequest,
    request: Request,
    response: Response,
    redis: Annotated[Redis, Depends(get_redis)],
    settings: Annotated[ApiSettings, Depends(get_settings)],
) -> Response:
    """Record one arrival. Always answers 204.

    The response carries a `Set-Cookie` and nothing else — no body, no
    identifier, no confirmation of what was stored. A caller learns
    nothing about our data from calling this.
    """
    visitor_key = _visitor_cookie(request) or uuid.uuid4()
    set_visitor_cookie(response, visitor_key, settings=settings)

    landing = normalise_path(payload.path)
    if landing.startswith(IGNORED_PREFIXES):
        # Operator navigation and asset requests are not acquisition.
        return Response(status_code=status.HTTP_204_NO_CONTENT, headers=dict(response.headers))

    try:
        await enforce_rate_limit(
            redis,
            key=f"acquisition:visit:{visitor_key}",
            limit=VISIT_RATE_LIMIT,
            window_seconds=VISIT_RATE_WINDOW_SECONDS,
        )
    except RateLimitExceeded:
        return Response(status_code=status.HTTP_204_NO_CONTENT, headers=dict(response.headers))
    except Exception:
        # A rate limiter that is down must not stop the page working.
        logger.warning("rate limiting unavailable for acquisition", exc_info=True)

    host = referrer_host(payload.referrer)
    attribution = classify(payload.params, payload.referrer)

    try:
        factory = get_session_factory(request)
        async with factory() as session:
            await AcquisitionRepository(session).record_visit(
                visitor_key=visitor_key,
                attribution=attribution,
                landing_path=landing,
                referrer_host=None if is_self_referral(host) else host,
            )
    except Exception:
        # Deliberately swallowed. See the module docstring: a visitor
        # whose page breaks because analytics failed is a real cost, and
        # a missing row is not.
        logger.warning("could not record acquisition visit", exc_info=True)

    return Response(status_code=status.HTTP_204_NO_CONTENT, headers=dict(response.headers))


__all__ = [
    "CAMPAIGN_PARAMS",
    "CLICK_IDS",
    "VISITOR_COOKIE_MAX_AGE",
    "VISITOR_COOKIE_NAME",
    "VisitRequest",
    "router",
    "set_visitor_cookie",
]
