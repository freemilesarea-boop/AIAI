#!/usr/bin/env python3
"""Find the payments that webhooks cannot tell us about.

Webhook delivery is not sufficient for a payment system, and the reason
is simple: a notification that is never delivered leaves no trace
anywhere. Nothing in the database says "PayApp tried to tell us
something and failed". The only way to notice is to have written down
what we expected and then to come back and check.

That is all this does. It compares our own records against our own
expectations and writes anomalies for the gaps:

  * an ACTIVE subscription whose period ended, with auto-renew on, and
    no renewal payment recorded → MISSING_EXPECTED_RENEWAL
  * a checkout registered with PayApp that nobody paid for → closed, and
    ABANDONED_CHECKOUT recorded
  * a subscription left in PAST_DUE past its period → UNRESOLVED_PAST_DUE

**What it deliberately cannot do.** PayApp's current published API
(https://docs.payapp.kr/dev_center01.html) documents commands to
register, cancel, pause and resume a recurring payment — `rebillRegist`,
`rebillCancel`, `rebillStop`, `rebillStart` — but no command that
authoritatively answers "what is the status of this rebill_no" or "list
the payments taken against it". So this job cannot confirm from the
provider whether a charge happened; it can only notice that we have no
record of one and say so.

That limitation is recorded here rather than worked around, because the
alternative — inventing a status-query endpoint and parsing whatever
comes back — would produce a reconciliation system that appears
authoritative and is not. If PayApp publishes such an API, this is the
one place that needs to change.

Nothing here ever grants entitlement, activates a subscription, or
invents a payment. It writes anomalies for a person to look at.

    uv run python scripts/ops/billing_reconcile.py
    uv run python scripts/ops/billing_reconcile.py --dry-run
    uv run python scripts/ops/billing_reconcile.py --list-anomalies
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
from luber_database.billing_reconciliation import reconcile  # noqa: E402
from luber_database.billing_repository import find_anomalies  # noqa: E402


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set.")
    return url


async def _run(dry_run: bool, list_only: bool) -> int:
    engine = create_async_engine_from_url(_database_url())
    factory = create_session_factory(engine)
    try:
        async with factory() as session:
            if list_only:
                rows = await find_anomalies(session)
                if not rows:
                    print("no unresolved billing anomalies")
                    return 0
                print(f"{len(rows)} unresolved billing anomaly/anomalies:\n")
                for row in rows:
                    print(f"  {row.detected_at.isoformat()}  {row.kind}")
                    print(f"    user={row.user_id}  rebill={row.provider_subscription_id}")
                    for key, value in row.detail.items():
                        print(f"    {key}: {value}")
                    print()
                return 1

            report = await reconcile(session, dry_run=dry_run)
            label = "would flag" if dry_run else "flagged"
            print(f"missing renewals      {label}: {len(report.missing_renewals)}")
            print(f"abandoned checkouts   {label}: {len(report.abandoned_checkouts)}")
            print(f"unresolved past due   {label}: {len(report.unresolved_past_due)}")
            # Non-zero when anything needs attention, so a scheduler can
            # alert on the exit code without parsing this output.
            return 1 if report.anomaly_count else 0
    finally:
        await engine.dispose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="report without writing anomalies")
    parser.add_argument(
        "--list-anomalies",
        action="store_true",
        help="print unresolved anomalies instead of reconciling",
    )
    args = parser.parse_args(argv)
    return asyncio.run(_run(args.dry_run, args.list_anomalies))


if __name__ == "__main__":
    raise SystemExit(main())
