"""Where the generation service meets the resilience layer.

The seam, and nothing else. `luber_provider_resilience` decides *which*
provider and whether the request may go there at all; it has never heard
of `GenerationRequest`, `ErrorCode` or ARQ. This module supplies exactly
that knowledge and keeps it out of both.

Four translations happen here.

**Requests into needs.** The router matches capabilities against what a
request requires, and a `Generation` row has to be turned into that
statement of requirements. Reading it wrong is how a reference-
conditioned request ends up at a provider that will drop the reference.

**Exceptions into categories.** Only this package knows that
`AceStepApiError` carries a status code, and that a 429 means "later"
while a 401 means "your key is wrong". Both currently arrive as
`MODEL_LOAD_FAILED`, so classifying here recovers a distinction the
error code has already lost — without changing what the user-facing
failure reports.

**Refusals into provider failures.** When the router refuses, Phase 29's
controller has to hear about it in the vocabulary it already speaks, so
the existing budget and retry rules apply unchanged. A refusal is a
`ProviderCallFailed`, and whether it is retryable is the router's answer
rather than a guess.

**Providers into profiles.** Capabilities are read off the provider
objects themselves — `supports_reference_audio`, `supports_edit(kind)`
— rather than from configuration somebody has to keep in sync.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from luber_generation_client.errors import GenerationProviderError
from luber_generation_client.provider import MusicGenerationProvider
from luber_inference_qc.controller import ProviderCallFailed
from luber_provider_resilience import (
    AttemptRecord,
    CircuitPolicy,
    FailoverMode,
    RequestNeeds,
    ResilienceManager,
    RoutingDecision,
    RoutingOutcome,
    RoutingPolicy,
    profile_provider,
)
from luber_provider_resilience.classification import (
    FailureCategory,
    classify,
    is_retryable,
)
from luber_schemas import EditKind, ErrorCode

logger = logging.getLogger(__name__)

#: Router outcomes that a further attempt could plausibly change.
#:
#: A circuit that is open now may be half-open in thirty seconds, and a
#: taken probe slot may be free. A missing capability will not appear,
#: and an explicit provider will not become a different one — retrying
#: those spends a budget to be told the same thing.
_RETRYABLE_REFUSALS: frozenset[str] = frozenset(
    {
        RoutingOutcome.PROVIDER_UNAVAILABLE_CIRCUIT_OPEN.value,
        RoutingOutcome.PROBE_SLOTS_TAKEN.value,
    }
)

#: What a refusal surfaces to the user as. `PROVIDER_BUSY` is honest for
#: an open circuit — the provider is not taking this request — and it is
#: an existing code, so no client learns a new one.
_REFUSAL_ERROR_CODE: dict[str, ErrorCode] = {
    RoutingOutcome.PROVIDER_UNAVAILABLE_CIRCUIT_OPEN.value: ErrorCode.PROVIDER_BUSY,
    RoutingOutcome.PROBE_SLOTS_TAKEN.value: ErrorCode.PROVIDER_BUSY,
    RoutingOutcome.NO_EQUIVALENT_PROVIDER.value: ErrorCode.MODEL_LOAD_FAILED,
    RoutingOutcome.EXPLICIT_PROVIDER_UNAVAILABLE.value: ErrorCode.MODEL_LOAD_FAILED,
    RoutingOutcome.PROVIDER_NOT_CONFIGURED.value: ErrorCode.MODEL_LOAD_FAILED,
    RoutingOutcome.FAILOVER_BUDGET_EXHAUSTED.value: ErrorCode.PROVIDER_BUSY,
}


def needs_for(generation: Any) -> RequestNeeds:
    """What this generation requires of a provider.

    Read from the routing columns rather than from anything a client
    supplies: `edit_kind` is what the worker dispatches on and
    `reference_audio_id` is a foreign key, so neither can be forged into
    a claim that a cheaper provider would do.
    """
    edit_kind = getattr(generation, "edit_kind", None)
    has_reference = getattr(generation, "reference_audio_id", None) is not None

    if edit_kind == EditKind.COVER.value:
        task = "COVER"
    elif edit_kind == EditKind.EXTEND.value:
        task = "EXTEND"
    elif edit_kind == EditKind.REPLACE_RANGE.value:
        task = "REPLACE_RANGE"
    elif edit_kind is not None:
        task = str(edit_kind)
    elif has_reference:
        task = "REFERENCE_CONDITIONED"
    else:
        task = "TEXT_TO_MUSIC"

    duration = getattr(generation, "duration_requested", None)
    return RequestNeeds(
        task_type=task,
        duration_seconds=None if duration is None else float(duration),
        has_reference=has_reference,
        has_lyrics=bool(getattr(generation, "lyrics", "")),
        instrumental=getattr(generation, "instrumental", None),
        has_musical_controls=bool(
            getattr(generation, "bpm", None) or getattr(generation, "key_scale", None)
        ),
        # Nothing in the product lets a caller name a provider today.
        # Left explicit rather than dropped: the router already refuses
        # to substitute for an explicit choice, and the day the product
        # offers one this is the line that wires it up.
        requested_provider=getattr(generation, "requested_provider", None),
    )


def classify_exception(exc: BaseException) -> tuple[str, int | None]:
    """The resilience category for a provider exception, and its status.

    The status code is recovered where the client kept one. By the time
    only an `ErrorCode` remains, a 429 and a 401 are both
    `MODEL_LOAD_FAILED`, and the difference between "try again shortly"
    and "nothing will work until somebody fixes the key" is gone.
    """
    status = _status_code_of(exc)
    timed_out = isinstance(exc, httpx.TimeoutException)
    transport = isinstance(exc, httpx.TransportError) and not timed_out
    error_code = exc.error_code.value if isinstance(exc, GenerationProviderError) else None
    category = classify(
        error_code=error_code,
        status_code=status,
        timed_out=timed_out,
        transport_error=transport,
    )
    return category, status


def _status_code_of(exc: BaseException) -> int | None:
    """Dig a status code out of whatever the provider raised."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    if isinstance(code, int):
        return code
    cause = exc.__cause__
    if cause is not None and cause is not exc:
        return _status_code_of(cause)
    return None


def refusal_to_failure(decision: RoutingDecision) -> ProviderCallFailed:
    """A routing refusal, in the vocabulary Phase 29 already speaks.

    Reusing `ProviderCallFailed` rather than inventing an exception is
    what keeps the budget in one place: the controller counts the
    attempt, applies its own retry rules, and never learns that a new
    layer exists.
    """
    code = _REFUSAL_ERROR_CODE.get(decision.outcome, ErrorCode.PROVIDER_BUSY)
    return ProviderCallFailed(
        decision.reason or decision.outcome,
        retryable=decision.outcome in _RETRYABLE_REFUSALS,
        error_code=code.value,
    )


@dataclass
class GateResult:
    """What the gate did for one generation, for the durable trace."""

    decisions: list[dict[str, Any]] = field(default_factory=list)
    attempts: list[AttemptRecord] = field(default_factory=list)
    providers_attempted: list[str] = field(default_factory=list)
    failovers: int = 0

    def to_dict(self) -> dict[str, Any]:
        from luber_provider_resilience.versions import version_block

        return {
            **version_block(),
            "decisions": self.decisions,
            "attempts": [item.to_dict() for item in self.attempts],
            "providers_attempted": list(self.providers_attempted),
            "provider_failovers": self.failovers,
            "narrative": self.narrative(),
        }

    def narrative(self) -> list[str]:
        """The routing story, in order, in sentences."""
        lines = [item.get("explanation", "") for item in self.decisions]
        for attempt in self.attempts:
            if attempt.outcome == "SUCCEEDED":
                lines.append(
                    f"{attempt.provider} succeeded"
                    + (" (recovery probe)" if attempt.was_probe else "")
                )
            else:
                lines.append(
                    f"{attempt.provider} failed: {attempt.category}; circuit "
                    f"{attempt.circuit_before} → {attempt.circuit_after}"
                )
        return [line for line in lines if line]


class ResilienceGate:
    """Chooses a provider per attempt and records how it went.

    Holds the provider objects; the router holds only their profiles.
    That split is what lets the resilience package stay free of the
    generation client while still routing to something callable.
    """

    def __init__(
        self,
        providers: dict[str, MusicGenerationProvider],
        *,
        manager: ResilienceManager,
    ) -> None:
        self._providers = dict(providers)
        self.manager = manager

    @classmethod
    def build(
        cls,
        providers: dict[str, MusicGenerationProvider],
        *,
        store: Any,
        failover: str = FailoverMode.DISABLED.value,
        preference: Sequence[str] = (),
        circuit_policy: CircuitPolicy | None = None,
        maximum_providers: int = 2,
        revisions: dict[str, str] | None = None,
    ) -> ResilienceGate:
        """Profile every provider and wire up a manager."""
        marks = revisions or {}
        profiles = [
            profile_provider(provider, name=name, revision=marks.get(name, "UNKNOWN"))
            for name, provider in providers.items()
        ]
        manager = ResilienceManager(
            profiles,
            store=store,
            routing_policy=RoutingPolicy(
                failover=failover,
                preference=tuple(preference),
                maximum_providers_per_generation=maximum_providers,
            ),
            circuit_policy=circuit_policy,
        )
        return cls(providers, manager=manager)

    def provider_for(self, name: str) -> MusicGenerationProvider:
        return self._providers[name]

    async def route(self, needs: RequestNeeds, *, attempted: Sequence[str]) -> RoutingDecision:
        decision = await self.manager.route(needs, attempted=attempted)
        logger.info(
            "provider routing decision",
            extra={
                "routing_outcome": decision.outcome,
                "selected_provider": decision.selected,
                "circuit_state": decision.circuit_state,
                "fallback_used": decision.fallback_used,
                "is_probe": decision.probe_token is not None,
            },
        )
        return decision

    async def record_success(
        self, decision: RoutingDecision, *, latency_seconds: float, attempt: int
    ) -> AttemptRecord:
        return await self.manager.record(
            decision, succeeded=True, latency_seconds=latency_seconds, attempt=attempt
        )

    async def record_failure(
        self,
        decision: RoutingDecision,
        exc: BaseException,
        *,
        latency_seconds: float,
        attempt: int,
    ) -> tuple[AttemptRecord, str]:
        category, status = classify_exception(exc)
        record = await self.manager.record(
            decision,
            succeeded=False,
            error_code=(exc.error_code.value if isinstance(exc, GenerationProviderError) else None),
            status_code=status,
            timed_out=category == FailureCategory.PROVIDER_TIMEOUT.value,
            latency_seconds=latency_seconds,
            attempt=attempt,
        )
        return record, category

    async def abandon(self, decision: RoutingDecision) -> None:
        """Release a probe slot without recording an outcome."""
        await self.manager.abandon(decision)


class Stopwatch:
    """Elapsed seconds, so a timing is not assembled from two clocks."""

    def __init__(self) -> None:
        self._started = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self._started


def failure_from_category(category: str, exc: BaseException) -> ProviderCallFailed:
    """A provider exception as Phase 29's controller expects it.

    Retryability comes from the resilience category rather than from the
    error code alone, which is the point: an auth failure reproduces on
    every attempt and must not burn the budget, and until now it was
    indistinguishable from a transport error.
    """
    code = (
        exc.error_code.value
        if isinstance(exc, GenerationProviderError)
        else ErrorCode.UNKNOWN_GENERATION_ERROR.value
    )
    return ProviderCallFailed(str(exc), retryable=is_retryable(category), error_code=code)


__all__ = [
    "GateResult",
    "ResilienceGate",
    "Stopwatch",
    "classify_exception",
    "failure_from_category",
    "needs_for",
    "refusal_to_failure",
]
