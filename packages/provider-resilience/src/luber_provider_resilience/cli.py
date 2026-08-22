"""The operator's way in: see the circuits, and decide about them.

Nine verbs. Six read, three record a human decision. Nothing here calls
a provider, starts a generation or changes a policy — the strongest
thing available is pinning a circuit open, which stops traffic and is
recorded with who did it and why.

The two mutating verbs both require a reason, and that is not
bureaucracy. A circuit somebody opened by hand is one the policy will
not close, so the next person to look needs to know whether it is still
true — and "opened by alex" without a reason is a question rather than
an answer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from luber_provider_resilience.capabilities import Capability, ProviderProfile
from luber_provider_resilience.circuit import CircuitIdentity, CircuitPolicy, CircuitState
from luber_provider_resilience.durable import DurableCircuitStore
from luber_provider_resilience.manager import ResilienceManager
from luber_provider_resilience.router import FailoverMode, RoutingPolicy
from luber_provider_resilience.versions import version_block

#: What a profile looks like when the CLI is only reporting.
#:
#: The CLI does not build providers — constructing an ACE-Step client to
#: print a circuit's state would open an HTTP connection to answer a
#: question about a database row. Capabilities are taken as complete so
#: readiness can be computed; the circuits themselves are the real
#: answer and they are read from storage.
_ALL_CAPABILITIES = frozenset({item.value for item in Capability})


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str))


async def _repository(url: str) -> Any:
    from luber_database import ResilienceRepository, create_async_engine_from_url

    engine = create_async_engine_from_url(url)
    return engine, ResilienceRepository(engine)


async def _with_manager(args: argparse.Namespace, work: Any) -> Any:
    engine, repository = await _repository(args.database_url)
    try:
        store = DurableCircuitStore(repository)
        names = _provider_names(args, await repository.all_circuits())
        profiles = [ProviderProfile(name=name, capabilities=_ALL_CAPABILITIES) for name in names]
        manager = ResilienceManager(
            profiles,
            store=store,
            routing_policy=RoutingPolicy(failover=args.failover),
            circuit_policy=CircuitPolicy(),
        )
        return await work(manager, repository)
    finally:
        await engine.dispose()


def _provider_names(args: argparse.Namespace, circuits: list[dict[str, Any]]) -> list[str]:
    """Which providers to report on.

    Taken from the circuits that exist, plus anything named on the
    command line. A provider that has never failed has no circuit row,
    so `--provider` is how an operator asks about one that has been
    healthy all week.
    """
    names = {row["provider"] for row in circuits}
    named = getattr(args, "provider", None)
    if named:
        names.add(named)
    return sorted(names) or ["(none)"]


# ── commands ─────────────────────────────────────────────────────────


def cmd_status(args: argparse.Namespace) -> int:
    async def work(manager: ResilienceManager, _repo: Any) -> Any:
        return await manager.status()

    _print(asyncio.run(_with_manager(args, work)))
    return 0


def cmd_circuits(args: argparse.Namespace) -> int:
    async def work(_manager: ResilienceManager, repo: Any) -> Any:
        rows = await repo.all_circuits()
        return {
            **version_block(),
            "circuits": [
                {
                    key: value
                    for key, value in row.items()
                    # The rolling window is evidence, not a display
                    # field: a busy provider's is hundreds of entries
                    # and nobody reads it in a list.
                    if key != "window"
                }
                for row in rows
            ],
            "count": len(rows),
        }

    result = asyncio.run(_with_manager(args, work))
    _print(result)
    # Non-zero when anything is not closed, so a scheduled check can be
    # wired to this rather than to a human reading it.
    return (
        1 if any(item["state"] != CircuitState.CLOSED.value for item in result["circuits"]) else 0
    )


def cmd_show(args: argparse.Namespace) -> int:
    identity = CircuitIdentity(args.provider, args.task)

    async def work(_manager: ResilienceManager, repo: Any) -> Any:
        row = await repo.load(identity.key())
        history = await repo.transitions(circuit_key=identity.key(), limit=args.history)
        if row is None:
            return {
                **version_block(),
                "identity": identity.to_dict(),
                "state": CircuitState.CLOSED.value,
                "note": (
                    "no circuit record exists: this provider and task have never "
                    "recorded a counted failure"
                ),
                "transitions": history,
            }
        return {**version_block(), **row, "transitions": history}

    _print(asyncio.run(_with_manager(args, work)))
    return 0


def cmd_readiness(args: argparse.Namespace) -> int:
    async def work(manager: ResilienceManager, _repo: Any) -> Any:
        return await manager.readiness()

    report = asyncio.run(_with_manager(args, work))
    if args.json:
        _print(report.to_dict())
    else:
        print(report.render())
    # Non-zero when nothing can be generated. Process health is a
    # different question and a different endpoint.
    return 0 if report.generation_available else 1


def cmd_open(args: argparse.Namespace) -> int:
    identity = CircuitIdentity(args.provider, args.task)

    async def work(manager: ResilienceManager, _repo: Any) -> Any:
        record = await manager.open(identity, operator=args.operator, reason=args.reason)
        return record.to_dict()

    _print(asyncio.run(_with_manager(args, work)))
    return 0


def cmd_close(args: argparse.Namespace) -> int:
    identity = CircuitIdentity(args.provider, args.task)

    async def work(manager: ResilienceManager, _repo: Any) -> Any:
        record = await manager.close(identity, operator=args.operator, reason=args.reason)
        return record.to_dict()

    _print(asyncio.run(_with_manager(args, work)))
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    identity = CircuitIdentity(args.provider, args.task)

    async def work(manager: ResilienceManager, _repo: Any) -> Any:
        record = await manager.reset(identity, operator=args.operator)
        return record.to_dict()

    _print(asyncio.run(_with_manager(args, work)))
    return 0


def cmd_transitions(args: argparse.Namespace) -> int:
    async def work(_manager: ResilienceManager, repo: Any) -> Any:
        rows = await repo.transitions(limit=args.limit)
        return {**version_block(), "transitions": rows, "count": len(rows)}

    _print(asyncio.run(_with_manager(args, work)))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Check the stored circuits for states that should be impossible."""

    async def work(_manager: ResilienceManager, repo: Any) -> Any:
        rows = await repo.all_circuits()
        issues: dict[str, list[str]] = {}

        def note(kind: str, detail: str) -> None:
            issues.setdefault(kind, []).append(detail)

        seen: set[str] = set()
        for row in rows:
            key = row["circuit_key"]
            if key in seen:
                note("DUPLICATE_CIRCUIT", key)
            seen.add(key)
            if row["state"] not in {item.value for item in CircuitState}:
                note("UNKNOWN_STATE", f"{key}: {row['state']}")
            if row["state"] == CircuitState.OPEN.value and row["control"] == "AUTOMATIC":
                if row.get("open_until") is None:
                    # An automatically opened circuit with no expiry
                    # would never be probed: an outage that outlives its
                    # cause and needs a human to notice.
                    note("OPEN_WITHOUT_COOLDOWN", key)
            if row["state"] != CircuitState.HALF_OPEN.value and row.get("probes"):
                note("PROBES_OUTSIDE_HALF_OPEN", key)
            if row.get("revision", 0) < 0:
                note("NEGATIVE_REVISION", key)
            if row.get("consecutive_failures", 0) < 0:
                note("NEGATIVE_COUNTER", key)
        return {
            **version_block(),
            "circuits": len(rows),
            "ok": not issues,
            "issues": {kind: sorted(set(items)) for kind, items in sorted(issues.items())},
        }

    report = asyncio.run(_with_manager(args, work))
    _print(report)
    return 0 if report["ok"] else 1


# ── parsing ──────────────────────────────────────────────────────────


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--database-url", default=None)
    parser.add_argument(
        "--failover",
        default=FailoverMode.DISABLED.value,
        choices=sorted(item.value for item in FailoverMode),
        help="only affects what readiness reports; it changes no stored policy",
    )


def _add_identity(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("provider")
    parser.add_argument(
        "--task",
        default="TEXT_TO_MUSIC",
        help="circuits are per provider and task; a broken cover endpoint "
        "must not take text-to-music offline with it",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="provider-resilience",
        description=(
            "Provider circuits, routing and readiness. Reports what the breaker has "
            "decided and records what an operator decides. Calls no provider."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="everything, in one call")
    _add_common(status)
    status.add_argument("--provider", default=None)
    status.set_defaults(handler=cmd_status)

    circuits = sub.add_parser("circuits", help="every circuit and its state")
    _add_common(circuits)
    circuits.add_argument("--provider", default=None)
    circuits.set_defaults(handler=cmd_circuits)

    show = sub.add_parser("show", help="one circuit, with its history")
    _add_common(show)
    _add_identity(show)
    show.add_argument("--history", type=int, default=20)
    show.set_defaults(handler=cmd_show)

    readiness = sub.add_parser("readiness", help="what the service can serve right now")
    _add_common(readiness)
    readiness.add_argument("--provider", default=None)
    readiness.add_argument("--json", action="store_true")
    readiness.set_defaults(handler=cmd_readiness)

    open_cmd = sub.add_parser("open", help="pin a circuit open; stops traffic")
    _add_common(open_cmd)
    _add_identity(open_cmd)
    open_cmd.add_argument("--operator", required=True)
    open_cmd.add_argument("--reason", required=True)
    open_cmd.set_defaults(handler=cmd_open)

    close_cmd = sub.add_parser("close", help="pin a circuit closed; resumes traffic")
    _add_common(close_cmd)
    _add_identity(close_cmd)
    close_cmd.add_argument("--operator", required=True)
    close_cmd.add_argument("--reason", required=True)
    close_cmd.set_defaults(handler=cmd_close)

    reset = sub.add_parser("reset", help="hand a circuit back to the policy")
    _add_common(reset)
    _add_identity(reset)
    reset.add_argument("--operator", required=True)
    reset.set_defaults(handler=cmd_reset)

    transitions = sub.add_parser("transitions", help="recent state changes")
    _add_common(transitions)
    transitions.add_argument("--provider", default=None)
    transitions.add_argument("--limit", type=int, default=50)
    transitions.set_defaults(handler=cmd_transitions)

    verify = sub.add_parser("verify", help="check stored circuits for impossible states")
    _add_common(verify)
    verify.add_argument("--provider", default=None)
    verify.set_defaults(handler=cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "database_url", None) is None:
        from luber_shared.settings import BaseServiceSettings

        args.database_url = BaseServiceSettings().database_url
    handler = args.handler
    return int(handler(args))


__all__ = ["build_parser", "main"]
