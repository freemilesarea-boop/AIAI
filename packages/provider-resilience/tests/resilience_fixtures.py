"""Providers that fail in one specific way, and a clock that does not tick.

Two pieces of test infrastructure, and both exist to make a device full
of timeouts testable without waiting.

**Deterministic providers.** Each one fails in exactly one way — times
out, refuses with 429, rejects the credentials — so a test that says
"a rate limit does not open the circuit as hard as a crash" is testing
that and not whichever failure a general-purpose fake happened to
produce. None of them is registered anywhere a deployment could reach:
`build_resilience_gate` refuses to route to a test double, and these do
not appear in `provider_from_settings` at all.

**A clock the test holds.** Cooldowns are thirty seconds and probe
leases are five minutes. Sleeping through those would make the suite
take an hour and still be flaky; advancing a clock the code was handed
makes the same assertions exact.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from luber_generation_client.errors import GenerationProviderError
from luber_generation_client.provider import (
    GenerationRequest,
    GenerationResult,
    MusicGenerationProvider,
)
from luber_provider_resilience import Capability, ProviderProfile
from luber_schemas import ErrorCode

#: A fixed origin. Every window in these tests is built from it, so a
#: verdict never depends on when the suite ran.
T0 = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)

ALL_CAPABILITIES = frozenset({item.value for item in Capability})

#: A provider that cannot take a reference track or a cover. The shape
#: that makes "failover is only safe when the request survives it"
#: testable rather than asserted.
TEXT_ONLY_CAPABILITIES = ALL_CAPABILITIES - {
    Capability.REFERENCE_CONDITIONED.value,
    Capability.COVER.value,
}


class Clock:
    """A clock the test advances. Never sleeps."""

    def __init__(self, start: datetime = T0) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: Any) -> datetime:
        self.now += timedelta(**kwargs)
        return self.now


class _Recorded(MusicGenerationProvider):
    """Base for the doubles: counts calls, never touches a network.

    `supports_reference_audio` is a **property**, matching
    `MusicGenerationProvider`. A double that made it a method would let
    the capability profiler pass here while reading every real provider
    as unable to take a reference track — which is exactly what happened
    before this shape was corrected.
    """

    name = "synthetic"

    def __init__(self, name: str = "synthetic") -> None:
        self.name = name
        self.calls: list[int | None] = []

    @property
    def supports_reference_audio(self) -> bool:
        return True

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        raise NotImplementedError


class HealthyProvider(_Recorded):
    """Answers every time, immediately."""

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request.seed)
        return GenerationResult(
            audio_path=Path("/dev/null"),
            duration_seconds=float(request.duration_seconds),
            sample_rate=48_000,
            seed_used=request.seed,
            provider=self.name,
            model_name=self.name,
            model_version="test",
        )


class TimeoutProvider(_Recorded):
    """Never answers in time."""

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request.seed)
        raise GenerationProviderError(
            "the engine did not answer", error_code=ErrorCode.GENERATION_TIMEOUT
        )


class FailingProvider(_Recorded):
    """Answers, badly, with a 500."""

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request.seed)
        raise _WithStatus("the engine returned HTTP 500", 500)


class RateLimitedProvider(_Recorded):
    """Working correctly, and declining."""

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request.seed)
        raise _WithStatus("slow down", 429)


class AuthFailingProvider(_Recorded):
    """Credentials rejected. Reproduces until somebody fixes it."""

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request.seed)
        raise _WithStatus("invalid api key", 401)


class SlowRecoveryProvider(_Recorded):
    """Fails until told to stop, then answers.

    For the recovery path: the circuit opens, cools down, probes, and
    the probe finds a provider that has come back.
    """

    def __init__(self, name: str = "slow_recovery") -> None:
        super().__init__(name)
        self.healthy = False

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request.seed)
        if not self.healthy:
            raise GenerationProviderError(
                "still starting up", error_code=ErrorCode.GENERATION_TIMEOUT
            )
        return GenerationResult(
            audio_path=Path("/dev/null"),
            duration_seconds=float(request.duration_seconds),
            sample_rate=48_000,
            seed_used=request.seed,
            provider=self.name,
            model_name=self.name,
            model_version="test",
        )


class _WithStatus(GenerationProviderError):
    """A provider error that kept its HTTP status.

    The real ACE-Step client collapses 429 and 401 into
    `MODEL_LOAD_FAILED`, losing the difference between "try later" and
    "your key is wrong". This double keeps the status so the
    classification that recovers that distinction can be tested.
    """

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message, error_code=ErrorCode.MODEL_LOAD_FAILED)
        self.status_code = status_code


def profile(name: str, capabilities: frozenset[str] = ALL_CAPABILITIES) -> ProviderProfile:
    return ProviderProfile(name=name, capabilities=capabilities, revision="test")
