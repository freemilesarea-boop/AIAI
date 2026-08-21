"""Who may reach the training console, and why the answer is not a role.

The console shows and moves things a product account must never touch:
private dataset identities, checkpoint digests, worker hosts, trainer
logs, and actions that spend money on rented hardware. So the first
question is what stops an ordinary logged-in user from reaching it.

**Not a role, because there is no role.** `User` has an id, an email, a
password hash and a display name. Adding `is_admin` here would invent a
privilege model the product has not designed, put it on the same table
every signup writes to, and make the difference between an operator and
a customer one boolean that one bug can flip. Step 2 of the phase brief
is explicit that a fake admin flag is the wrong answer.

**The deployment, because a deployment cannot be escalated into.** The
console is off unless the process was started with it on, refused
outright when the environment is production, and gated behind a shared
operator token even then. A user cannot register their way past any of
those: there is no request that turns the console on.

Three consequences worth stating.

*Disabled reads as absent.* A console that is off answers 404, not 403.
A 403 would confirm to an anonymous prober that this deployment has a
training console worth attacking; 404 tells them the path does not
exist here, which is also true.

*Production is refused, not gated.* `create_app` does not mount the
router when the environment is production, and this dependency refuses
a second time in case something else mounts it. Two independent checks
because the cost of the mistake is a public training console.

*A missing token fails closed.* Enabled with no token configured is a
misconfiguration, not a convenience, and it is the shape that leaves a
console open. It answers 503 with the reason rather than serving.
"""

from __future__ import annotations

import hmac
import logging
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from luber_api.settings import ApiSettings, get_settings

logger = logging.getLogger(__name__)

#: Carried in a header rather than a query string: query strings land in
#: proxy logs, browser history and referrers, and a shared secret that
#: leaks into a log line is a secret that has to be rotated.
OPERATOR_TOKEN_HEADER = "X-Luber-Operator-Token"

#: What the browser is told when the console is not available. Identical
#: for "switched off" and "wrong deployment" so the reply does not
#: describe the configuration of a host to someone who cannot use it.
_ABSENT_DETAIL = "Not found."


def console_available(settings: ApiSettings) -> bool:
    """Whether this process may serve the operator console at all.

    Production is excluded structurally rather than by policy: the
    console has no production story yet — no operator role, no audit of
    who acted, no per-operator identity — and shipping it to production
    behind a shared token would be all of those problems at once.
    """
    return settings.ops_console_enabled and not settings.is_production


def require_operator(
    request: Request,
    settings: Annotated[ApiSettings, Depends(get_settings)],
    token: Annotated[str | None, Header(alias=OPERATOR_TOKEN_HEADER)] = None,
) -> None:
    """The one gate every operator route sits behind.

    Applied at the router so a route added later is protected by having
    been added, not by somebody remembering to protect it.
    """
    if not console_available(settings):
        raise HTTPException(status.HTTP_404_NOT_FOUND, _ABSENT_DETAIL)

    expected = settings.ops_operator_token
    if not expected:
        # Deliberately loud and deliberately unhelpful to a stranger:
        # the operator sees the reason in the log, the caller sees that
        # the console is not serving.
        logger.error(
            "operator console is enabled but OPS_OPERATOR_TOKEN is unset; refusing to serve"
        )
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The operator console is enabled but no operator token is configured.",
        )

    # Constant time. A token check that returned early on the first
    # wrong byte would leak the token one byte at a time to a patient
    # caller.
    if token is None or not hmac.compare_digest(token, expected):
        logger.warning(
            "rejected operator console request to %s: %s",
            request.url.path,
            "missing token" if token is None else "token mismatch",
        )
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "A valid operator token is required.",
            headers={"WWW-Authenticate": OPERATOR_TOKEN_HEADER},
        )


def enforce_operator_origin(
    request: Request,
    settings: Annotated[ApiSettings, Depends(get_settings)],
) -> None:
    """Reject an unsafe operator request from an origin we do not serve.

    The console's mutations are driven from a browser, so the same
    reasoning as the product's CSRF layer applies — with one difference
    that makes it weaker here and worth stating: the operator token
    travels in a header, and a cross-site form post cannot set headers,
    so CSRF against these routes is already hard. This is the second
    lock, not the first.

    A missing ``Origin`` is allowed for the same reason as elsewhere:
    the CLI and the tests do not send one, and they carry no ambient
    credential for an attacker to ride.
    """
    if request.method in {"GET", "HEAD", "OPTIONS", "TRACE"}:
        return
    origin = request.headers.get("origin")
    if origin is None:
        return
    if origin not in settings.cors_origins:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Request origin is not permitted.",
        )
