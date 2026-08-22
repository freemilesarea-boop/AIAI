"""Building a resilience gate from settings, or deciding not to.

One function, and most of it is refusals.

A deployment with resilience switched off gets ``None`` and the
pre-Phase-31 path: the configured provider is called directly. A
deployment with it on but only one provider still gets a circuit —
refusing fast when that provider is down is the whole value with one
provider, and it does not require a second.

What this will not do is invent a provider. `provider_from_settings`
knows two names, and one of them is `mock`: a test double that returns a
committed fixture. Registering it as a fallback would give a deployment
a "second provider" that answers every request with the same two seconds
of audio, and the first time a circuit opened, users would silently
receive it.
"""

from __future__ import annotations

import logging
from typing import Any

from luber_generation_client.factory import provider_from_settings
from luber_generation_client.provider import MusicGenerationProvider
from luber_generation_client.resilience import ResilienceGate
from luber_provider_resilience import (
    CircuitPolicy,
    DurableCircuitStore,
    FailoverMode,
    ProviderProfile,
    profile_provider,
)

logger = logging.getLogger(__name__)

#: Provider names that must never be routed to in a running deployment.
#:
#: `mock` returns a committed fixture. It is the right thing for tests
#: and the worst possible fallback: an outage would deliver the same
#: two seconds of audio to everybody, successfully, with nothing
#: anywhere saying so.
_NEVER_ROUTABLE: frozenset[str] = frozenset({"mock"})


def build_resilience_gate(
    settings: Any,
    *,
    repository: Any,
    provider: MusicGenerationProvider | None = None,
) -> ResilienceGate | None:
    """The gate for this deployment, or ``None`` when it is switched off.

    ``provider`` is the already-built provider, so the common path does
    not construct a second one — a provider holds an HTTP client and a
    duplicate would double the connection pool for no reason.
    """
    if not getattr(settings, "provider_resilience_enabled", False):
        return None

    name = settings.generation_provider
    primary = provider if provider is not None else provider_from_settings(settings)
    providers: dict[str, MusicGenerationProvider] = {name: primary}

    mode = str(getattr(settings, "provider_failover_mode", "DISABLED")).upper()
    if mode not in {item.value for item in FailoverMode}:
        raise ValueError(
            f"unknown failover mode {mode!r}. Known: "
            + ", ".join(sorted(item.value for item in FailoverMode))
        )

    preference = tuple(
        item.strip()
        for item in str(getattr(settings, "provider_preference", "")).split(",")
        if item.strip()
    )

    routable = {key: value for key, value in providers.items() if key not in _NEVER_ROUTABLE}
    if not routable:
        # Every configured provider is a test double. Rather than route
        # to one, the gate stands down and the service calls the
        # provider directly — which is what a test expects anyway.
        logger.info(
            "provider resilience is enabled but no routable provider is configured; "
            "the circuit is not engaged",
            extra={"configured_provider": name},
        )
        return None

    if mode != FailoverMode.DISABLED.value and len(routable) < 2:
        # Enabled with nowhere to go. Said once, plainly, rather than
        # left as a setting that looks active and never does anything.
        logger.warning(
            "failover mode is %s but only one routable provider is configured; "
            "no request can be moved. The circuit breaker still applies.",
            mode,
            extra={"providers": sorted(routable)},
        )

    revisions = {
        key: str(getattr(settings, "ace_step_model", "UNKNOWN")) if key == "ace_step" else "UNKNOWN"
        for key in routable
    }

    return ResilienceGate.build(
        routable,
        store=DurableCircuitStore(repository),
        failover=mode,
        preference=preference,
        circuit_policy=CircuitPolicy(),
        maximum_providers=int(getattr(settings, "provider_maximum_per_generation", 2)),
        revisions=revisions,
    )


async def profiles_from_settings(settings: Any) -> list[ProviderProfile]:
    """The configured providers' capability profiles, and nothing else.

    For readers that need to know what a deployment *could* do without
    being able to make it do anything — the readiness view, chiefly.
    Capabilities are read off the provider object rather than from a
    table of names, so a provider that gains an input cannot be
    misreported by a list somebody forgot to update.

    The provider is built and then closed. It holds an HTTP client and
    this is a read: leaving one open per call would leak a connection
    pool into a process whose job is to answer a question.
    """
    name = settings.generation_provider
    if name in _NEVER_ROUTABLE:
        return []
    provider = provider_from_settings(settings)
    try:
        revision = (
            str(getattr(settings, "ace_step_model", "UNKNOWN")) if name == "ace_step" else "UNKNOWN"
        )
        return [profile_provider(provider, name=name, revision=revision)]
    finally:
        closer = getattr(provider, "close", None)
        if closer is not None:
            await closer()


__all__ = ["build_resilience_gate", "profiles_from_settings"]
