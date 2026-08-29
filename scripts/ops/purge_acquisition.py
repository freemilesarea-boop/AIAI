#!/usr/bin/env python3
"""Delete raw acquisition data past its retention period.

BOORDA keeps raw first-party acquisition records for twelve months. This
is what performs that — the privacy policy states the period, and a
period nothing enforces is a claim rather than a policy.

Defaults to a dry run. Deleting personal data is not something a script
should do because somebody typed its name; `--apply` is the deliberate
second step, and the count printed by the dry run is what makes it an
informed one.

Touches only `acquisition_visitors` and `acquisition_sessions`. Billing,
subscription and support records are retained under 전자상거래법 for
years and are never in scope here.

    uv run python scripts/ops/purge_acquisition.py
    uv run python scripts/ops/purge_acquisition.py --apply
    uv run python scripts/ops/purge_acquisition.py --days 365 --apply
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from luber_database import create_async_engine_from_url, create_session_factory  # noqa: E402
from luber_database.acquisition_retention import (  # noqa: E402
    RETENTION_DAYS,
    purge_acquisition,
)


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set.")
    return url


async def _run(days: int, apply: bool) -> int:
    engine = create_async_engine_from_url(_database_url())
    factory = create_session_factory(engine)
    try:
        async with factory() as session:
            report = await purge_acquisition(session, days=days, dry_run=not apply)

        label = "deleted" if apply else "would delete"
        print(f"retention          : {days} days")
        print(f"cutoff (UTC)       : {report.cutoff.isoformat()}")
        print(f"visitors {label:14}: {report.visitors_deleted}")
        print(f"sessions {label:14}: {report.sessions_deleted}")
        if not apply:
            print("\ndry run — nothing was deleted. Re-run with --apply to delete.")
        return 0
    finally:
        await engine.dispose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--days",
        type=int,
        default=RETENTION_DAYS,
        help=f"retention period in days (default: {RETENTION_DAYS})",
    )
    parser.add_argument(
        "--apply", action="store_true", help="actually delete; without it this is a dry run"
    )
    args = parser.parse_args(argv)
    if args.days < 1:
        parser.error("--days must be positive")
    return asyncio.run(_run(args.days, args.apply))


if __name__ == "__main__":
    raise SystemExit(main())
