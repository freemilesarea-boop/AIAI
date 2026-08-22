"""Which provider serves this request, and why that one.

One place, deliberately. Fallback decisions scattered through workers
are how a system ends up with three different answers to "what happens
when the provider is down", each written by whoever was on call that
week, none of them recorded anywhere a reader can find.

The router answers a single question — *which provider* — and never
*where it runs*. A future remote-generation executor plugs in underneath
the provider this chose. Conflating them would make circuit evidence
about a broken GPU host look like evidence about a model, and the first
confusing outage would be one where a bad machine took a healthy
provider offline.

Four rules the routing follows.

**A closed circuit is a precondition, not a preference.** An open
circuit means the request does not go there, and the caller is told so
in a typed refusal rather than by a timeout.

**An explicit choice is honoured or refused, never substituted.** If a
caller named a provider and it is unavailable, the answer is a failure
naming it. Silently sending the request elsewhere would answer a
different question and report success.

**Failover requires equivalence, not availability.** A healthy provider
that cannot carry the reference track is not a fallback for a
reference-conditioned request. Where equivalence fails, so does the
request — with the missing capability named.

**Failover is off by default.** Another provider existing is not a
reason to use it. The mode has to be chosen.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from luber_provider_resilience.capabilities import (
    ProviderProfile,
    RequestNeeds,
    check_equivalence,
)
from luber_provider_resilience.circuit import (
    CircuitIdentity,
    CircuitPolicy,
    CircuitRecord,
    CircuitState,
    allows,
    claim_probe,
    promote_to_half_open,
    ready_for_probe,
)
from luber_provider_resilience.store import CircuitStore, apply_with_retry, utcnow
from luber_provider_resilience.versions import (
    FAILOVER_POLICY_VERSION,
    ROUTING_POLICY_VERSION,
    version_block,
)


class FailoverMode(StrEnum):
    """How willing this deployment is to move a request.

    The default is `DISABLED`, and that is not timidity. With one
    production provider there is nowhere to fail over to, and a mode
    that silently became active the day a second provider was
    configured would change behaviour without anybody deciding to.
    """

    #: Never move a request. One provider is chosen; if it cannot serve
    #: the request, the request fails.
    DISABLED = "DISABLED"

    #: Move only to a provider that can represent the request unchanged,
    #: and only when the first choice is unavailable rather than merely
    #: slow.
    SAFE_EQUIVALENT_ONLY = "SAFE_EQUIVALENT_ONLY"

    #: An operator has pinned a provider. Used for draining one provider
    #: deliberately; still subject to equivalence, because "the operator
    #: said so" is not a reason to deliver a song without its lyrics.
    OPERATOR_FORCED = "OPERATOR_FORCED"


class RoutingOutcome(StrEnum):
    """What the router decided."""

    #: A provider was selected and normal traffic may proceed.
    SELECTED = "SELECTED"
    #: Selected as a bounded recovery probe against a HALF_OPEN circuit.
    SELECTED_AS_PROBE = "SELECTED_AS_PROBE"
    #: Every eligible provider's circuit is open.
    PROVIDER_UNAVAILABLE_CIRCUIT_OPEN = "PROVIDER_UNAVAILABLE_CIRCUIT_OPEN"
    #: The named provider is not configured on this deployment.
    PROVIDER_NOT_CONFIGURED = "PROVIDER_NOT_CONFIGURED"
    #: Providers are available but none can carry this request.
    NO_EQUIVALENT_PROVIDER = "NO_EQUIVALENT_PROVIDER"
    #: The caller named a provider that cannot serve them, and failover
    #: away from an explicit choice was not permitted.
    EXPLICIT_PROVIDER_UNAVAILABLE = "EXPLICIT_PROVIDER_UNAVAILABLE"
    #: The failover budget for this generation is spent.
    FAILOVER_BUDGET_EXHAUSTED = "FAILOVER_BUDGET_EXHAUSTED"
    #: HALF_OPEN, but the probe slots are taken. This request waits for
    #: another one to learn the answer rather than joining a stampede.
    PROBE_SLOTS_TAKEN = "PROBE_SLOTS_TAKEN"


@dataclass(frozen=True)
class RoutingPolicy:
    """How the router chooses, and how far it may go."""

    failover: str = FailoverMode.DISABLED.value

    #: Distinct providers one generation may touch. Two: the first
    #: choice and one alternative. This is the failover budget, and it
    #: is separate from — and multiplied by nothing against — Phase 29's
    #: attempt budget, because failover redirects attempts rather than
    #: adding them.
    maximum_providers_per_generation: int = 2

    #: Whether a caller's explicit provider choice may be overridden.
    #: False: substituting for an explicit request answers a different
    #: question. A product that wants otherwise sets it deliberately.
    allow_failover_from_explicit: bool = False

    #: Preference order when several providers can serve a request.
    #: Empty means "configuration order", which is the deployment's
    #: stated preference rather than one this module invents.
    preference: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "failover": self.failover,
            "maximum_providers_per_generation": self.maximum_providers_per_generation,
            "allow_failover_from_explicit": self.allow_failover_from_explicit,
            "preference": list(self.preference),
            "routing_policy_version": ROUTING_POLICY_VERSION,
            "failover_policy_version": FAILOVER_POLICY_VERSION,
        }


@dataclass(frozen=True)
class ConsideredProvider:
    """One provider the router looked at, and what it concluded.

    Every candidate is recorded, including the rejected ones. A trace
    that showed only the winner would leave an operator unable to answer
    "why not the other one", which is the question they actually have.
    """

    provider: str
    circuit_state: str
    eligible: bool
    reason: str
    equivalence: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "circuit_state": self.circuit_state,
            "eligible": self.eligible,
            "reason": self.reason,
            "equivalence": self.equivalence,
        }


@dataclass(frozen=True)
class RoutingDecision:
    """What the router decided, and everything behind it."""

    outcome: str
    at: datetime
    needs: RequestNeeds
    selected: str | None = None
    selected_revision: str | None = None
    circuit_state: str | None = None
    #: Set when this selection is a bounded recovery probe. The token
    #: must be returned to the circuit when the attempt finishes.
    probe_token: str | None = None
    fallback_used: bool = False
    requested_provider: str | None = None
    reason: str = ""
    considered: tuple[ConsideredProvider, ...] = ()
    providers_attempted: tuple[str, ...] = ()
    policy: dict[str, Any] = field(default_factory=dict)

    @property
    def permitted(self) -> bool:
        return self.outcome in {
            RoutingOutcome.SELECTED.value,
            RoutingOutcome.SELECTED_AS_PROBE.value,
        }

    def explain(self) -> str:
        """One line, for a log or an operator screen."""
        if self.permitted:
            probe = " as a recovery probe" if self.probe_token else ""
            fallback = " (fallback)" if self.fallback_used else ""
            return f"{self.selected} selected{probe}{fallback}: {self.reason}"
        return f"no provider selected — {self.outcome}: {self.reason}"

    def to_dict(self) -> dict[str, Any]:
        return {
            **version_block(),
            "outcome": self.outcome,
            "at": self.at.isoformat(),
            "selected_provider": self.selected,
            "selected_revision": self.selected_revision,
            "circuit_state": self.circuit_state,
            "is_probe": self.probe_token is not None,
            "fallback_used": self.fallback_used,
            "requested_provider": self.requested_provider,
            "reason": self.reason,
            "explanation": self.explain(),
            "needs": self.needs.to_dict(),
            "considered": [item.to_dict() for item in self.considered],
            "providers_attempted": list(self.providers_attempted),
            "policy": self.policy,
        }


class ProviderRouter:
    """Chooses a provider for a request. Calls none of them."""

    def __init__(
        self,
        profiles: Sequence[ProviderProfile],
        *,
        store: CircuitStore,
        policy: RoutingPolicy | None = None,
        circuit_policy: CircuitPolicy | None = None,
        clock: Any = utcnow,
    ) -> None:
        self.profiles = list(profiles)
        self.store = store
        self.policy = policy or RoutingPolicy()
        self.circuit_policy = circuit_policy or CircuitPolicy()
        self._clock = clock

    # ── selection ────────────────────────────────────────────────────

    async def select(
        self,
        needs: RequestNeeds,
        *,
        attempted: Sequence[str] = (),
        probe_token: str | None = None,
    ) -> RoutingDecision:
        """Choose a provider, or explain why none may be used.

        ``attempted`` is the providers this generation has already tried.
        Passing it is what makes the second call after a failure pick
        somebody else, and what makes the failover budget mean anything.
        """
        now = self._clock()
        already = tuple(attempted)
        considered: list[ConsideredProvider] = []

        ordered = self._ordered_profiles(needs)
        if not ordered:
            return self._refuse(
                RoutingOutcome.PROVIDER_NOT_CONFIGURED.value,
                needs=needs,
                now=now,
                reason=(
                    f"no provider named {needs.requested_provider!r} is configured"
                    if needs.requested_provider
                    else "no providers are configured on this deployment"
                ),
                considered=considered,
                attempted=already,
            )

        budget_spent = len(set(already)) >= self.policy.maximum_providers_per_generation

        # Providers that could have served this request and were passed
        # over — because a circuit was open, or a probe slot was taken.
        #
        # This, not "has anything been attempted", is what makes a
        # selection a failover. Routing to the second provider because
        # the first one's circuit is open is a failover on the very
        # first attempt, and defining it the other way let a deployment
        # with failover DISABLED still move the request.
        skipped_equivalent: list[str] = []

        for profile in ordered:
            equivalence = check_equivalence(profile, needs)
            record = await self.store.load(CircuitIdentity(profile.name, needs.task_type))

            if not equivalence.equivalent:
                # Not a fallback candidate at all: it could never have
                # served this request, so passing it over is not a
                # decision to move anything.
                considered.append(
                    ConsideredProvider(
                        provider=profile.name,
                        circuit_state=record.state,
                        eligible=False,
                        reason=equivalence.explain(),
                        equivalence=equivalence.to_dict(),
                    )
                )
                continue

            # Re-selecting a provider this generation already used is a
            # retry on the same provider — Phase 29's business — not a
            # failover. Only reaching a *different* one is.
            is_fallback = bool(skipped_equivalent) or (
                bool(already) and profile.name not in already
            )
            if is_fallback and not self._failover_permitted(needs):
                considered.append(
                    ConsideredProvider(
                        provider=profile.name,
                        circuit_state=record.state,
                        eligible=False,
                        reason=self._failover_refusal(needs),
                    )
                )
                continue

            if is_fallback and budget_spent:
                considered.append(
                    ConsideredProvider(
                        provider=profile.name,
                        circuit_state=record.state,
                        eligible=False,
                        reason=(
                            f"failover budget spent: {len(set(already))} of "
                            f"{self.policy.maximum_providers_per_generation} providers "
                            "already attempted"
                        ),
                    )
                )
                continue

            decision = await self._consider_circuit(
                profile,
                record,
                needs=needs,
                now=now,
                is_fallback=is_fallback,
                attempted=already,
                considered=considered,
                probe_token=probe_token,
            )
            if decision is not None:
                return decision

            # It could have served this request and did not. Anything
            # chosen after this point is a failover away from it.
            skipped_equivalent.append(profile.name)

        return self._refuse_after_consideration(
            needs=needs, now=now, considered=considered, attempted=already
        )

    async def _consider_circuit(
        self,
        profile: ProviderProfile,
        record: CircuitRecord,
        *,
        needs: RequestNeeds,
        now: datetime,
        is_fallback: bool,
        attempted: tuple[str, ...],
        considered: list[ConsideredProvider],
        probe_token: str | None,
    ) -> RoutingDecision | None:
        """Whether this provider's circuit lets the request through."""
        identity = CircuitIdentity(profile.name, needs.task_type)

        if allows(record, now):
            considered.append(
                ConsideredProvider(
                    provider=profile.name,
                    circuit_state=record.state,
                    eligible=True,
                    reason="circuit closed",
                )
            )
            return RoutingDecision(
                outcome=RoutingOutcome.SELECTED.value,
                at=now,
                needs=needs,
                selected=profile.name,
                selected_revision=profile.revision,
                circuit_state=record.state,
                fallback_used=is_fallback,
                requested_provider=needs.requested_provider,
                reason=(
                    "circuit closed and the provider can represent this request"
                    if not is_fallback
                    else "first choice unavailable; this provider is equivalent and healthy"
                ),
                considered=tuple(considered),
                providers_attempted=attempted,
                policy=self.policy.to_dict(),
            )

        # OPEN whose cooldown has expired becomes HALF_OPEN here — a
        # written transition rather than a state that quietly becomes
        # true, so two workers cannot disagree about when it happened.
        if ready_for_probe(record, now):
            record, _ = await apply_with_retry(
                self.store,
                identity,
                lambda current: promote_to_half_open(current, now=now),
                # Losing means another worker promoted it; re-reading
                # gives the same answer this was trying to write.
                required=False,
            )

        if record.state == CircuitState.HALF_OPEN.value and probe_token is not None:
            claimed = await self._claim(identity, probe_token, now)
            if claimed:
                considered.append(
                    ConsideredProvider(
                        provider=profile.name,
                        circuit_state=CircuitState.HALF_OPEN.value,
                        eligible=True,
                        reason="recovery probe slot claimed",
                    )
                )
                return RoutingDecision(
                    outcome=RoutingOutcome.SELECTED_AS_PROBE.value,
                    at=now,
                    needs=needs,
                    selected=profile.name,
                    selected_revision=profile.revision,
                    circuit_state=CircuitState.HALF_OPEN.value,
                    probe_token=probe_token,
                    fallback_used=is_fallback,
                    requested_provider=needs.requested_provider,
                    reason="cooldown expired; this request is a bounded recovery probe",
                    considered=tuple(considered),
                    providers_attempted=attempted,
                    policy=self.policy.to_dict(),
                )
            considered.append(
                ConsideredProvider(
                    provider=profile.name,
                    circuit_state=CircuitState.HALF_OPEN.value,
                    eligible=False,
                    reason="probe slots already taken",
                )
            )
            return None

        considered.append(
            ConsideredProvider(
                provider=profile.name,
                circuit_state=record.state,
                eligible=False,
                reason=record.open_reason or f"circuit {record.state}",
            )
        )
        return None

    async def _claim(self, identity: CircuitIdentity, token: str, now: datetime) -> bool:
        """Take a probe slot atomically, or find it gone."""
        claimed = False

        def mutate(current: CircuitRecord) -> tuple[CircuitRecord, None]:
            nonlocal claimed
            updated, ok = claim_probe(current, token=token, policy=self.circuit_policy, now=now)
            claimed = ok
            return updated, None

        try:
            await apply_with_retry(self.store, identity, mutate, required=False)
        except Exception:
            # Losing the race for a probe slot is the mechanism working:
            # somebody else is finding out whether the provider is back.
            return False
        return claimed

    # ── ordering and permission ──────────────────────────────────────

    def _ordered_profiles(self, needs: RequestNeeds) -> list[ProviderProfile]:
        """Candidates, most preferred first.

        An explicit request narrows the list to one. That is what makes
        "explicit provider unavailable" a distinct outcome rather than a
        silent substitution — there is nothing else in the list to fall
        through to.
        """
        if needs.requested_provider is not None:
            named = [item for item in self.profiles if item.name == needs.requested_provider]
            if named and not self.policy.allow_failover_from_explicit:
                return named
            if not named:
                return []
            others = [item for item in self.profiles if item.name != needs.requested_provider]
            return named + self._by_preference(others)
        return self._by_preference(self.profiles)

    def _by_preference(self, profiles: list[ProviderProfile]) -> list[ProviderProfile]:
        if not self.policy.preference:
            return list(profiles)
        order = {name: index for index, name in enumerate(self.policy.preference)}
        return sorted(profiles, key=lambda item: (order.get(item.name, len(order)), item.name))

    def _failover_permitted(self, needs: RequestNeeds) -> bool:
        if self.policy.failover == FailoverMode.DISABLED.value:
            return False
        if needs.requested_provider is not None:
            return self.policy.allow_failover_from_explicit
        return True

    def _failover_refusal(self, needs: RequestNeeds) -> str:
        if self.policy.failover == FailoverMode.DISABLED.value:
            return "failover is disabled on this deployment"
        return (
            "the caller named a provider explicitly, and moving an explicit request to "
            "another provider is not permitted"
        )

    # ── refusals ─────────────────────────────────────────────────────

    def _refuse(
        self,
        outcome: str,
        *,
        needs: RequestNeeds,
        now: datetime,
        reason: str,
        considered: list[ConsideredProvider],
        attempted: tuple[str, ...],
    ) -> RoutingDecision:
        return RoutingDecision(
            outcome=outcome,
            at=now,
            needs=needs,
            requested_provider=needs.requested_provider,
            reason=reason,
            considered=tuple(considered),
            providers_attempted=attempted,
            policy=self.policy.to_dict(),
        )

    def _refuse_after_consideration(
        self,
        *,
        needs: RequestNeeds,
        now: datetime,
        considered: list[ConsideredProvider],
        attempted: tuple[str, ...],
    ) -> RoutingDecision:
        """Name the reason nothing was chosen, as precisely as possible.

        The order matters to an operator: "your key is wrong" and "the
        other provider cannot do covers" send them to different places,
        and a generic "no provider available" sends them nowhere.
        """
        states = {item.provider: item.circuit_state for item in considered}
        open_circuits = [
            name
            for name, state in states.items()
            if state in {CircuitState.OPEN.value, CircuitState.HALF_OPEN.value}
        ]
        probe_blocked = any("probe slots" in item.reason for item in considered)
        equivalence_blocked = [item for item in considered if item.equivalence is not None]
        budget_blocked = any("failover budget spent" in item.reason for item in considered)

        if needs.requested_provider is not None and not self.policy.allow_failover_from_explicit:
            return self._refuse(
                RoutingOutcome.EXPLICIT_PROVIDER_UNAVAILABLE.value,
                needs=needs,
                now=now,
                reason=(
                    f"{needs.requested_provider} was requested explicitly and cannot serve "
                    "this request; substituting another provider would answer a different "
                    "question"
                ),
                considered=considered,
                attempted=attempted,
            )

        if budget_blocked:
            # Checked before the open circuits, and deliberately. Both
            # facts are true, but the operator's question is "why did it
            # not fail over", and the budget is the answer. Reporting
            # the open circuit alone is true and sends them to look at a
            # provider that was never the obstacle.
            blocked = ", ".join(
                sorted(
                    item.provider for item in considered if "failover budget spent" in item.reason
                )
            )
            detail = (
                f"; {', '.join(sorted(open_circuits))} also has an open circuit"
                if open_circuits
                else ""
            )
            return self._refuse(
                RoutingOutcome.FAILOVER_BUDGET_EXHAUSTED.value,
                needs=needs,
                now=now,
                reason=(
                    f"this generation has already tried "
                    f"{self.policy.maximum_providers_per_generation} providers, so "
                    f"{blocked or 'the remaining provider(s)'} cannot be tried{detail}"
                ),
                considered=considered,
                attempted=attempted,
            )

        if probe_blocked:
            return self._refuse(
                RoutingOutcome.PROBE_SLOTS_TAKEN.value,
                needs=needs,
                now=now,
                reason=(
                    "the circuit is testing recovery and its probe slots are taken; "
                    "this request is not joining the stampede"
                ),
                considered=considered,
                attempted=attempted,
            )

        if open_circuits:
            return self._refuse(
                RoutingOutcome.PROVIDER_UNAVAILABLE_CIRCUIT_OPEN.value,
                needs=needs,
                now=now,
                reason=(
                    "every provider that could serve this request has an open circuit: "
                    + ", ".join(sorted(open_circuits))
                ),
                considered=considered,
                attempted=attempted,
            )

        if equivalence_blocked:
            missing = sorted(
                {
                    capability
                    for item in equivalence_blocked
                    for capability in (item.equivalence or {}).get("missing", [])
                }
            )
            return self._refuse(
                RoutingOutcome.NO_EQUIVALENT_PROVIDER.value,
                needs=needs,
                now=now,
                reason=(
                    "no configured provider can represent this request unchanged; "
                    f"missing: {', '.join(missing) or 'unknown capability'}"
                ),
                considered=considered,
                attempted=attempted,
            )

        return self._refuse(
            RoutingOutcome.PROVIDER_UNAVAILABLE_CIRCUIT_OPEN.value,
            needs=needs,
            now=now,
            reason="no provider is currently able to serve this request",
            considered=considered,
            attempted=attempted,
        )


__all__ = [
    "ConsideredProvider",
    "FailoverMode",
    "ProviderRouter",
    "RoutingDecision",
    "RoutingOutcome",
    "RoutingPolicy",
]
