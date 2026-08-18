"""A fixed-window rate limiter for the authentication endpoints.

Small on purpose. The job is to make credential stuffing and signup
flooding expensive, not to be an abuse platform — and a limiter with its
own storage, sliding windows and rule engine is a system that itself
needs operating.

Redis holds a counter per key per window, expiring on its own. That
places the limit on the server rather than in the browser, which is the
only place it means anything.
"""

from __future__ import annotations

from redis.asyncio import Redis


class RateLimitExceeded(Exception):
    """Too many attempts inside the window."""

    def __init__(self, retry_after_seconds: int) -> None:
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"rate limit exceeded; retry in {retry_after_seconds}s")


async def enforce_rate_limit(redis: Redis, *, key: str, limit: int, window_seconds: int) -> int:
    """Count this attempt; raise once the window's allowance is spent.

    Fixed window rather than sliding: one Redis key, one INCR, one TTL.
    A caller can in principle get up to twice the limit across a window
    boundary, which is an acceptable looseness for a control whose job
    is to turn thousands of guesses per minute into a handful.

    **If Redis is unavailable this allows the request.** That is a
    deliberate choice and it is the uncomfortable one, so it is stated
    rather than buried: the alternative is that a Redis outage locks
    every existing user out of their account. Passwords are still
    Argon2id and sessions are still server-side, so the loss is the
    brute-force ceiling, not authentication itself. The outage is
    already loud — ``/ready`` reports it and generation submission
    fails with 503 — so this does not fail silently at the system level.
    """
    try:
        attempts = await redis.incr(key)
        if attempts == 1:
            await redis.expire(key, window_seconds)
        if attempts > limit:
            ttl = await redis.ttl(key)
            raise RateLimitExceeded(ttl if ttl and ttl > 0 else window_seconds)
        return int(attempts)
    except RateLimitExceeded:
        raise
    except Exception:
        return 0


async def reset_rate_limit(redis: Redis, *, key: str) -> None:
    """Forget the attempts recorded under a key.

    Called after a successful login so that a user who mistyped their
    password twice and then got it right does not carry the penalty for
    the rest of the window. Failures should cost; success should clear.
    """
    try:
        await redis.delete(key)
    except Exception:
        return
