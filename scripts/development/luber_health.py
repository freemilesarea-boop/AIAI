#!/usr/bin/env python
"""Read-only operational health check for a running LUBER stack.

Answers the questions an operator actually asks — is anything down, is
anything stuck, does the database agree with what is on disk — without
changing a single row or object. Every query here is a SELECT and every
storage call is a stat; there is no flag that makes it delete, repair or
retry, because the moment a health check can mutate, running it becomes a
decision rather than a reflex.

    .venv/bin/python scripts/development/luber_health.py
    .venv/bin/python scripts/development/luber_health.py --json

Exit status is 0 when every check passes, 1 when any check reports a
problem, so it can gate a deploy or a cron. Nothing it prints contains a
credential: the database URL is reduced to host and database name, and
storage keys are printed relative to the storage root.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from luber_database import AuthRepository, create_async_engine_from_url, create_session_factory
from luber_shared import BaseServiceSettings

#: A generation older than this in a non-terminal state is reported as
#: stuck. Derived from measured behaviour rather than chosen: across the
#: generations in this deployment the slowest complete run took 110.5s
#: and the mean was 54.7s, so 15 minutes is roughly eight times the
#: worst observed case. It is deliberately far above the real
#: distribution — the cost of a false "stuck" is an operator chasing a
#: healthy job, and this check only reports, so a late flag costs
#: nothing while a wrong one costs trust.
STUCK_AFTER = timedelta(minutes=15)

#: Queue wait is bounded by inference, not by the queue, so a job still
#: QUEUED after this long means nothing is consuming — a different
#: failure from a slow generation, and worth separating.
QUEUED_STUCK_AFTER = timedelta(minutes=5)

TERMINAL = ("COMPLETED", "FAILED", "CANCELLED")


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


def _redact(url: str) -> str:
    """Host and database only — never the password."""
    tail = url.rsplit("@", 1)[-1]
    return tail or "(unparsed)"


def _http(url: str, timeout: float = 5.0) -> tuple[bool, str]:
    try:
        with urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 400, f"HTTP {response.status}"
    except URLError as exc:
        return False, f"unreachable ({exc.reason})"
    except Exception as exc:
        return False, f"unreachable ({type(exc).__name__})"


def check_api(base_url: str) -> Check:
    ok, detail = _http(f"{base_url}/health")
    return Check("api", ok, detail, {"url": base_url})


def check_web(base_url: str) -> Check:
    ok, detail = _http(base_url)
    return Check("web", ok, detail, {"url": base_url})


def check_provider(base_url: str) -> Check:
    # ACE-Step serves no route at "/", so its docs page is the cheapest
    # honest liveness probe; a 404 at the root would say nothing.
    ok, detail = _http(f"{base_url}/docs")
    return Check("ace_step", ok, detail, {"url": base_url})


def check_worker() -> Check:
    """Presence and uniqueness. Two workers is a finding, not a detail."""
    try:
        found = subprocess.run(
            ["pgrep", "-f", "arq .*luber_generation_worker"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return Check("worker", False, f"could not inspect processes ({type(exc).__name__})")
    pids = [pid for pid in found.stdout.split() if pid]
    if len(pids) == 1:
        return Check("worker", True, f"1 running (pid {pids[0]})", {"pids": pids})
    if not pids:
        return Check("worker", False, "no generation worker running", {"pids": []})
    return Check(
        "worker",
        False,
        f"{len(pids)} workers running — jobs may be processed by either",
        {"pids": pids},
    )


async def check_redis(redis_url: str, queue_name: str) -> Check:
    try:
        from redis.asyncio import Redis
    except ImportError:  # pragma: no cover - redis ships with arq
        return Check("redis", False, "redis client not installed")
    client = Redis.from_url(redis_url)
    try:
        await client.ping()
        depth = await client.zcard(queue_name)
        retries = len(await client.keys("arq:retry:*"))
        return Check(
            "redis",
            True,
            f"up · queue depth {depth} · {retries} awaiting retry",
            {"queue_depth": depth, "retrying": retries, "queue": queue_name},
        )
    except Exception as exc:
        return Check("redis", False, f"unreachable ({type(exc).__name__})")
    finally:
        await client.aclose()


async def check_database(engine: AsyncEngine) -> Check:
    async with engine.connect() as conn:
        rows = (
            await conn.execute(text("select status, count(*) from generations group by 1"))
        ).all()
    counts: dict[str, int] = {str(status): int(count) for status, count in rows}
    total = sum(counts.values())
    return Check("database", True, f"up · {total} generations", {"status_counts": counts})


async def check_stuck(engine: AsyncEngine, now: datetime) -> Check:
    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    text(
                        """
                    select id, status, created_at, started_at
                    from generations
                    where status not in ('COMPLETED','FAILED','CANCELLED')
                    order by created_at
                    """
                    )
                )
            )
            .mappings()
            .all()
        )

    stuck: list[dict[str, Any]] = []
    for row in rows:
        started = row["started_at"] or row["created_at"]
        limit = QUEUED_STUCK_AFTER if row["status"] == "QUEUED" else STUCK_AFTER
        age = now - started
        if age > limit:
            stuck.append(
                {
                    "id": str(row["id"]),
                    "status": row["status"],
                    "age_minutes": round(age.total_seconds() / 60, 1),
                }
            )
    in_flight = len(rows) - len(stuck)
    detail = f"{in_flight} in flight · {len(stuck)} stuck"
    return Check("stuck_generations", not stuck, detail, {"stuck": stuck, "in_flight": in_flight})


async def check_assets(engine: AsyncEngine, storage_root: Path) -> Check:
    """Does the database agree with what is actually on disk?"""
    async with engine.connect() as conn:
        assets = (
            (
                await conn.execute(
                    text("select generation_id, asset_type, storage_key from audio_assets")
                )
            )
            .mappings()
            .all()
        )
        completed = {
            str(row[0])
            for row in (
                await conn.execute(text("select id from generations where status = 'COMPLETED'"))
            ).all()
        }

    missing_object = [
        a["storage_key"] for a in assets if not (storage_root / a["storage_key"]).exists()
    ]
    roles = Counter((str(a["generation_id"]), a["asset_type"]) for a in assets)
    duplicate_roles = [f"{gid}:{role}" for (gid, role), n in roles.items() if n > 1]

    by_generation: dict[str, set[str]] = {}
    for asset in assets:
        by_generation.setdefault(str(asset["generation_id"]), set()).add(asset["asset_type"])
    # A COMPLETED generation the product cannot play or hand over is the
    # failure that matters most here: the UI says it is ready.
    undeliverable = sorted(
        gid
        for gid in completed
        if not ({"MASTER", "FINISHED_MASTER"} & by_generation.get(gid, set()))
    )

    known_keys = {a["storage_key"] for a in assets}
    audio_root = storage_root / "audio"
    orphan_objects: list[str] = []
    if audio_root.is_dir():
        for path in audio_root.rglob("*"):
            if path.is_file() and str(path.relative_to(storage_root)) not in known_keys:
                orphan_objects.append(str(path.relative_to(storage_root)))

    problems = missing_object or duplicate_roles or undeliverable
    detail = (
        f"{len(assets)} assets · {len(missing_object)} missing objects · "
        f"{len(duplicate_roles)} duplicate roles · {len(undeliverable)} undeliverable · "
        f"{len(orphan_objects)} orphan objects"
    )
    return Check(
        "asset_consistency",
        not problems,
        detail,
        {
            "missing_object": missing_object[:20],
            "duplicate_roles": duplicate_roles[:20],
            "completed_without_delivery_asset": undeliverable[:20],
            # Orphans are reported but do not fail the check: an
            # in-flight generation writes its object before the row.
            "orphan_objects": orphan_objects[:20],
        },
    )


async def check_lineage(engine: AsyncEngine) -> Check:
    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    text("select id, parent_generation_id, edit_kind from generations")
                )
            )
            .mappings()
            .all()
        )

    ids = {str(row["id"]) for row in rows}
    parent_of = {str(row["id"]): row["parent_generation_id"] for row in rows}
    self_parent, missing_parent, root_with_edit, cycles = [], [], [], []

    for row in rows:
        gid, parent, kind = str(row["id"]), row["parent_generation_id"], row["edit_kind"]
        if parent is not None and str(parent) == gid:
            self_parent.append(gid)
        elif parent is not None and str(parent) not in ids:
            missing_parent.append(gid)
        if parent is None and kind is not None:
            root_with_edit.append(gid)

    for start in ids:
        seen, current, depth = {start}, parent_of.get(start), 0
        while current is not None and depth < 64:
            key = str(current)
            if key in seen:
                cycles.append(start)
                break
            seen.add(key)
            current = parent_of.get(key)
            depth += 1

    problems = self_parent or missing_parent or root_with_edit or cycles
    detail = (
        f"{len(rows)} generations · {len(self_parent)} self-parent · "
        f"{len(missing_parent)} missing parent · {len(root_with_edit)} root with edit_kind · "
        f"{len(cycles)} in a cycle"
    )
    return Check(
        "lineage_consistency",
        not problems,
        detail,
        {
            "self_parent": self_parent[:20],
            "missing_parent": missing_parent[:20],
            "root_with_edit_kind": root_with_edit[:20],
            "cycles": sorted(set(cycles))[:20],
        },
    )


async def check_references(
    engine: AsyncEngine, storage_root: Path, now: datetime, grace_hours: int
) -> Check:
    async with engine.connect() as conn:
        refs = (
            (await conn.execute(text("select id, storage_key, created_at from reference_audio")))
            .mappings()
            .all()
        )
        used = {
            str(row[0])
            for row in (
                await conn.execute(
                    text(
                        "select distinct reference_audio_id from generations "
                        "where reference_audio_id is not null"
                    )
                )
            ).all()
        }

    missing_object, unused_old, unused_fresh, used_but_missing = [], [], [], []
    for ref in refs:
        rid = str(ref["id"])
        exists = (storage_root / ref["storage_key"]).exists()
        age = now - ref["created_at"]
        if rid in used:
            if not exists:
                used_but_missing.append(rid)
        elif not exists:
            missing_object.append(rid)
        elif age > timedelta(hours=grace_hours):
            unused_old.append(rid)
        else:
            unused_fresh.append(rid)

    detail = (
        f"{len(refs)} references · {len(used)} in use · {len(unused_old)} unused past grace · "
        f"{len(unused_fresh)} unused within grace · {len(used_but_missing)} used but missing"
    )
    # Only a reference a generation depends on is a real problem; an
    # unused one past its grace period is cleanup's job, not an alarm.
    return Check(
        "reference_consistency",
        not used_but_missing,
        detail,
        {
            "used_but_object_missing": used_but_missing[:20],
            "unused_past_grace": unused_old[:20],
            "unused_within_grace": len(unused_fresh),
            "row_missing_object": missing_object[:20],
        },
    )


async def run_checks(
    settings: BaseServiceSettings, api: str, web: str, storage_root: Path
) -> list[Check]:
    now = datetime.now(UTC)
    engine = create_async_engine_from_url(settings.database_url)
    checks = [
        check_api(api),
        check_web(web),
        check_provider(settings.ace_step_base_url),
        check_worker(),
        await check_redis(settings.redis_url, "luber:generation"),
    ]
    try:
        checks.append(await check_database(engine))
        checks.append(await check_stuck(engine, now))
        checks.append(await check_assets(engine, storage_root))
        checks.append(await check_lineage(engine))
        checks.append(
            await check_references(
                engine, storage_root, now, settings.reference_abandonment_grace_hours
            )
        )
    except Exception as exc:
        checks.append(Check("database", False, f"query failed ({type(exc).__name__}: {exc})"))
    finally:
        await engine.dispose()
    return checks


async def prune_sessions(settings: BaseServiceSettings) -> int:
    """Delete expired sessions. The only mutation in this file.

    Safe by construction: an expired session already fails to
    authenticate, so this reclaims rows rather than revoking access, and
    the query cannot reach a user or any product data.
    """
    engine = create_async_engine_from_url(settings.database_url)
    try:
        factory = create_session_factory(engine)
        async with factory() as session:
            return await AuthRepository(session).delete_expired_sessions(now=datetime.now(UTC))
    finally:
        await engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--prune-sessions",
        action="store_true",
        help="delete expired sessions and exit (the one mutating action here)",
    )
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--web", default="http://127.0.0.1:3000")
    args = parser.parse_args()

    settings = BaseServiceSettings()
    if args.prune_sessions:
        removed = asyncio.run(prune_sessions(settings))
        print(f"removed {removed} expired session(s)")
        return 0
    # Resolved here rather than inside the coroutine: filesystem calls in
    # an async function are a lint error, and this one only needs doing once.
    storage_root = Path(settings.audio_storage_dir).resolve()
    checks = asyncio.run(
        run_checks(settings, args.api.rstrip("/"), args.web.rstrip("/"), storage_root)
    )

    if args.json:
        print(
            json.dumps(
                {
                    "healthy": all(c.ok for c in checks),
                    "database": _redact(settings.database_url),
                    "checks": [
                        {"name": c.name, "ok": c.ok, "detail": c.detail, **c.data} for c in checks
                    ],
                },
                indent=2,
                default=str,
            )
        )
    else:
        print(f"LUBER health · db {_redact(settings.database_url)}")
        for check in checks:
            print(f"  [{'OK ' if check.ok else 'BAD'}] {check.name:<22} {check.detail}")
            if not check.ok:
                for key, value in check.data.items():
                    if value:
                        print(f"         {key}: {value}")
    return 0 if all(c.ok for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
