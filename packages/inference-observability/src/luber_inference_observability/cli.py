"""The operator's way in, without a browser.

Nine verbs over the same engine the dashboard uses, so an answer given
here and an answer given there are the same answer. It reads generations
and writes observations and incidents; it starts no generation, changes
no policy and disables nothing.

Every command that could be destructive is not. `ingest` and `backfill`
write a projection keyed on the generation, so running either twice
changes no count. `verify` reads. `incident acknowledge` and
`incident dismiss` record what an operator decided and never delete the
history of what happened.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from luber_inference_observability.dimensions import Segment
from luber_inference_observability.incidents import IncidentStatus
from luber_inference_observability.queries import (
    compare_revisions,
    compare_windows,
    run_detection,
    summary,
    top_segments,
)
from luber_inference_observability.regressions import regressions
from luber_inference_observability.reports import (
    health_report,
    render_markdown,
    revision_report,
)
from luber_inference_observability.service import (
    ingest as ingest_generations,
)
from luber_inference_observability.service import (
    load_ledger,
    load_store,
    load_store_spanning,
    save_ledger,
)
from luber_inference_observability.storage import verify as verify_store
from luber_inference_observability.versions import version_block
from luber_inference_observability.windows import DURATIONS, TimeWindow

#: Where a deployment's database lives, if the caller does not say.
DEFAULT_WINDOW = "24h"


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str))


def _window(args: argparse.Namespace) -> TimeWindow:
    """The window a command runs over: named, or an explicit interval."""
    if getattr(args, "start", None) and getattr(args, "end", None):
        return TimeWindow.of(datetime.fromisoformat(args.start), datetime.fromisoformat(args.end))
    end = datetime.fromisoformat(args.end) if getattr(args, "end", None) else datetime.now(UTC)
    return TimeWindow.ending_at(end, args.window)


def _segment(args: argparse.Namespace) -> Segment | None:
    filters = {
        "provider": getattr(args, "provider", None),
        "provider_revision": getattr(args, "revision", None),
        "task_type": getattr(args, "task", None),
        "duration_bucket": getattr(args, "duration_bucket", None),
    }
    segment = Segment.of(**filters)
    return segment if segment.filters else None


async def _session(url: str) -> Any:
    """One session factory, imported late.

    The database packages are imported inside the command rather than at
    module scope so that `--help`, and every unit test of the pure
    engine, work without SQLAlchemy being installed or a database being
    reachable.
    """
    from luber_database import create_async_engine_from_url, create_session_factory

    engine = create_async_engine_from_url(url)
    return engine, create_session_factory(engine)


async def _with_repository(url: str, work: Any) -> Any:
    from luber_database import ObservabilityRepository

    engine, factory = await _session(url)
    try:
        async with factory() as session:
            return await work(ObservabilityRepository(session))
    finally:
        await engine.dispose()


# ── commands ─────────────────────────────────────────────────────────


def cmd_ingest(args: argparse.Namespace) -> int:
    async def work(repository: Any) -> Any:
        return await ingest_generations(
            repository,
            limit=args.limit,
            luber_revision=args.luber_revision,
            full=False,
        )

    result = asyncio.run(_with_repository(args.database_url, work))
    _print({**version_block(), "mode": "incremental", **result.to_dict()})
    return 0


def cmd_backfill(args: argparse.Namespace) -> int:
    """Project every finished generation, oldest first, in batches.

    Each round carries a watermark forward. Without it, every round
    would start from the beginning and re-read the same first batch
    until the round limit ran out — the table would look fully scanned
    while the newest generations, the ones an operator is about to ask
    about, were never ingested at all.

    The watermark advances by ``created_at``, and the source query is
    inclusive of it, so the boundary row is re-read once per round.
    That is harmless: the projection is keyed on the generation, so
    re-reading a row rewrites it rather than duplicating it.
    """

    async def work(repository: Any) -> Any:
        total = None
        rounds = 0
        since = None
        while True:
            result = await ingest_generations(
                repository,
                since=since,
                limit=args.limit,
                luber_revision=None,
                # Only the first round starts from the beginning. Later
                # rounds carry an explicit watermark, and `full` would
                # discard it.
                full=since is None,
            )
            rounds += 1
            if total is None:
                total = result
            else:
                total.scanned += result.scanned
                total.written += result.written
                total.failed += result.failed
                total.without_qc_trace += result.without_qc_trace
                total.errors.extend(result.errors)
                total.watermark = result.watermark or total.watermark

            # A short batch means the end. A watermark that did not move
            # means every row in this batch shares one timestamp, and
            # another round would read the same rows forever.
            if result.scanned < args.limit or rounds >= args.max_rounds:
                break
            if result.watermark is None or result.watermark == since:
                break
            since = result.watermark
        return total

    result = asyncio.run(_with_repository(args.database_url, work))
    _print({**version_block(), "mode": "backfill", **result.to_dict()})
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    window = _window(args)
    segment = _segment(args)

    async def work(repository: Any) -> Any:
        store = await load_store(repository, window=window, segment=segment)
        return summary(store, window=window, segment=segment)

    _print(asyncio.run(_with_repository(args.database_url, work)))
    return 0


def cmd_regressions(args: argparse.Namespace) -> int:
    window = _window(args)
    span = timedelta(days=args.baseline_days)
    gap = timedelta(hours=args.baseline_gap_hours)

    async def work(repository: Any) -> Any:
        store = await load_store_spanning(
            repository, current=window, baseline_span=span, baseline_gap=gap
        )
        from luber_inference_observability.queries import evaluate_segments

        findings = evaluate_segments(store, current=window, by=(args.group_by,), span=span, gap=gap)
        crossed = regressions(findings)
        return {
            **version_block(),
            "window": window.to_dict(),
            "grouped_by": args.group_by,
            "evaluated": len(findings),
            "regressions": [item.to_dict() for item in crossed],
            "statuses": sorted({item.status for item in findings}),
        }

    result = asyncio.run(_with_repository(args.database_url, work))
    _print(result)
    # Non-zero when something crossed, so a scheduled run can be wired
    # into a check rather than only read.
    return 1 if result["regressions"] else 0


def cmd_detect(args: argparse.Namespace) -> int:
    window = _window(args)
    span = timedelta(days=args.baseline_days)
    gap = timedelta(hours=args.baseline_gap_hours)

    async def work(repository: Any) -> Any:
        store = await load_store_spanning(
            repository, current=window, baseline_span=span, baseline_gap=gap
        )
        ledger = await load_ledger(repository)
        outcome = run_detection(
            store,
            current=window,
            ledger=ledger,
            by=(args.group_by,),
            span=span,
            gap=gap,
        )
        await save_ledger(repository, ledger)
        return outcome

    result = asyncio.run(_with_repository(args.database_url, work))
    _print(result)
    return 1 if result["regressions"] else 0


def cmd_incidents(args: argparse.Namespace) -> int:
    async def work(repository: Any) -> Any:
        statuses = (
            None if args.all else [IncidentStatus.OPEN.value, IncidentStatus.ACKNOWLEDGED.value]
        )
        rows = await repository.list_incidents(
            statuses=statuses, limit=args.limit, offset=args.offset
        )
        total = await repository.count_incidents(statuses=statuses)
        return {
            **version_block(),
            "total": total,
            "limit": args.limit,
            "offset": args.offset,
            "incidents": rows,
        }

    _print(asyncio.run(_with_repository(args.database_url, work)))
    return 0


def cmd_incident_show(args: argparse.Namespace) -> int:
    async def work(repository: Any) -> Any:
        return await repository.get_incident(args.incident_id)

    row = asyncio.run(_with_repository(args.database_url, work))
    if row is None:
        print(f"no incident {args.incident_id}", file=sys.stderr)
        return 2
    _print(row)
    return 0


def _act_on_incident(args: argparse.Namespace, action: str) -> int:
    async def work(repository: Any) -> Any:
        ledger = await load_ledger(repository)
        if ledger.get(args.incident_id) is None:
            return None
        now = datetime.now(UTC)
        if action == "acknowledge":
            ledger.acknowledge(args.incident_id, by=args.operator, at=now)
        else:
            ledger.dismiss(args.incident_id, by=args.operator, reason=args.reason, at=now)
        await save_ledger(repository, ledger)
        incident = ledger.get(args.incident_id)
        return incident.to_dict() if incident else None

    result = asyncio.run(_with_repository(args.database_url, work))
    if result is None:
        print(f"no incident {args.incident_id}", file=sys.stderr)
        return 2
    _print(result)
    return 0


def cmd_incident_acknowledge(args: argparse.Namespace) -> int:
    return _act_on_incident(args, "acknowledge")


def cmd_incident_dismiss(args: argparse.Namespace) -> int:
    return _act_on_incident(args, "dismiss")


def cmd_report(args: argparse.Namespace) -> int:
    window = _window(args)
    span = timedelta(days=args.baseline_days)
    gap = timedelta(hours=args.baseline_gap_hours)

    async def work(repository: Any) -> Any:
        store = await load_store_spanning(
            repository, current=window, baseline_span=span, baseline_gap=gap
        )
        ledger = await load_ledger(repository)
        return health_report(store, window=window, incidents=ledger.all())

    report = asyncio.run(_with_repository(args.database_url, work))
    destination = Path(args.output).expanduser() if args.output else None
    if destination is None:
        _print(report)
        return 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    json_path = destination.with_suffix(".json")
    md_path = destination.with_suffix(".md")
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True, default=str),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


def cmd_providers(args: argparse.Namespace) -> int:
    window = _window(args)

    async def work(repository: Any) -> Any:
        store = await load_store(repository, window=window)
        comparison = compare_revisions(
            store,
            window=window,
            left_revision=args.left,
            right_revision=args.right,
            minimum_samples=args.minimum_samples,
        )
        return comparison

    comparison = asyncio.run(_with_repository(args.database_url, work))
    if args.markdown:
        print(revision_report(comparison))
    else:
        _print(comparison)
    return 0


def cmd_deployment(args: argparse.Namespace) -> int:
    """Before and after a moment. Correlation, stated as such."""
    marker = datetime.fromisoformat(args.at)
    span = timedelta(hours=args.hours)
    before = TimeWindow.of(marker - span, marker)
    after = TimeWindow.of(marker, marker + span)

    async def work(repository: Any) -> Any:
        rows = await repository.select_observations(start=before.start, end=after.end)
        from luber_inference_observability.storage import (
            InMemoryObservationStore,
            from_mapping,
        )

        store = InMemoryObservationStore(from_mapping(row) for row in rows)
        return compare_windows(store, before=before, after=after)

    _print(asyncio.run(_with_repository(args.database_url, work)))
    return 0


def cmd_segments(args: argparse.Namespace) -> int:
    window = _window(args)

    async def work(repository: Any) -> Any:
        store = await load_store(repository, window=window)
        return top_segments(
            store,
            window=window,
            by=tuple(args.group_by.split(",")),
            metric=args.metric,
            minimum_samples=args.minimum_samples,
            limit=args.limit,
        )

    _print(asyncio.run(_with_repository(args.database_url, work)))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    window = _window(args)

    async def work(repository: Any) -> Any:
        store = await load_store(repository, window=window)
        ledger = await load_ledger(repository)
        return verify_store(list(store), incidents=ledger.all())

    report = asyncio.run(_with_repository(args.database_url, work))
    _print(report)
    return 0 if report["ok"] else 1


# ── argument parsing ─────────────────────────────────────────────────


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--database-url",
        default=None,
        help="SQLAlchemy async URL; defaults to the deployment's DATABASE_URL",
    )


def _add_window(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--window",
        default=DEFAULT_WINDOW,
        choices=sorted(DURATIONS),
        help="named window ending now",
    )
    parser.add_argument("--start", default=None, help="ISO start of an explicit interval")
    parser.add_argument("--end", default=None, help="ISO end of an explicit interval")


def _add_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", default=None)
    parser.add_argument("--revision", default=None, help="provider revision, name@version")
    parser.add_argument("--task", default=None)
    parser.add_argument("--duration-bucket", default=None)


def _add_baseline(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--baseline-days", type=int, default=7)
    parser.add_argument(
        "--baseline-gap-hours",
        type=int,
        default=1,
        help="how far before the window the baseline stops, so a live "
        "regression is not learned as normal",
    )
    parser.add_argument("--group-by", default="provider_revision")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inference-observability",
        description=(
            "Trends, regressions and incidents over Phase 29 inference traces. "
            "Detects and explains; changes nothing."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="project new finished generations")
    _add_common(ingest)
    ingest.add_argument("--limit", type=int, default=500)
    ingest.add_argument("--luber-revision", default=None)
    ingest.set_defaults(handler=cmd_ingest)

    backfill = sub.add_parser("backfill", help="project every finished generation")
    _add_common(backfill)
    backfill.add_argument("--limit", type=int, default=500)
    backfill.add_argument("--max-rounds", type=int, default=1000)
    backfill.set_defaults(handler=cmd_backfill)

    summary_cmd = sub.add_parser("summary", help="health for one window")
    _add_common(summary_cmd)
    _add_window(summary_cmd)
    _add_filters(summary_cmd)
    summary_cmd.set_defaults(handler=cmd_summary)

    regress = sub.add_parser("regressions", help="what crossed a threshold")
    _add_common(regress)
    _add_window(regress)
    _add_baseline(regress)
    regress.set_defaults(handler=cmd_regressions)

    detect_cmd = sub.add_parser("detect", help="evaluate and update incidents")
    _add_common(detect_cmd)
    _add_window(detect_cmd)
    _add_baseline(detect_cmd)
    detect_cmd.set_defaults(handler=cmd_detect)

    incidents = sub.add_parser("incidents", help="list incidents")
    _add_common(incidents)
    incidents.add_argument("--all", action="store_true", help="include closed")
    incidents.add_argument("--limit", type=int, default=50)
    incidents.add_argument("--offset", type=int, default=0)
    incidents.set_defaults(handler=cmd_incidents)

    incident = sub.add_parser("incident", help="one incident")
    incident_sub = incident.add_subparsers(dest="incident_command", required=True)

    show = incident_sub.add_parser("show")
    _add_common(show)
    show.add_argument("incident_id")
    show.set_defaults(handler=cmd_incident_show)

    ack = incident_sub.add_parser("acknowledge")
    _add_common(ack)
    ack.add_argument("incident_id")
    ack.add_argument("--operator", required=True)
    ack.set_defaults(handler=cmd_incident_acknowledge)

    dismiss = incident_sub.add_parser("dismiss")
    _add_common(dismiss)
    dismiss.add_argument("incident_id")
    dismiss.add_argument("--operator", required=True)
    dismiss.add_argument("--reason", required=True)
    dismiss.set_defaults(handler=cmd_incident_dismiss)

    report = sub.add_parser("report", help="write a health report")
    _add_common(report)
    _add_window(report)
    _add_baseline(report)
    report.add_argument(
        "--output", default=None, help="path stem; .json and .md are written beside it"
    )
    report.set_defaults(handler=cmd_report)

    providers = sub.add_parser("providers", help="compare two provider revisions")
    _add_common(providers)
    _add_window(providers)
    providers.add_argument("--left", required=True)
    providers.add_argument("--right", required=True)
    providers.add_argument("--minimum-samples", type=int, default=30)
    providers.add_argument("--markdown", action="store_true")
    providers.set_defaults(handler=cmd_providers)

    deployment = sub.add_parser("deployment", help="before and after a moment")
    _add_common(deployment)
    deployment.add_argument("--at", required=True, help="ISO timestamp of the change")
    deployment.add_argument("--hours", type=int, default=24)
    deployment.set_defaults(handler=cmd_deployment)

    segments = sub.add_parser("segments", help="worst segments with enough samples")
    _add_common(segments)
    _add_window(segments)
    segments.add_argument("--group-by", default="provider,duration_bucket")
    segments.add_argument("--metric", default="generation_failure_rate")
    segments.add_argument("--minimum-samples", type=int, default=30)
    segments.add_argument("--limit", type=int, default=10)
    segments.set_defaults(handler=cmd_segments)

    verify = sub.add_parser("verify", help="check the store's integrity and privacy")
    _add_common(verify)
    _add_window(verify)
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
