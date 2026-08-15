"""Reclaiming reference uploads nobody used.

A reference is stored the moment it is uploaded, before any generation
exists to justify it. Most of them earn their keep; some never do — the
user removed it, replaced it, closed the tab, or abandoned the flow. Those
would otherwise sit in storage forever.

The whole design is built around one asymmetry: **deleting a reference
that is still in use is unrecoverable, and leaving one behind costs
disk.** Every trade-off here is resolved in favour of the second.

Three rules follow from that.

*The database decides, never the browser.* "Remove" in the Create form
means "do not use this for my next generation". It is not evidence that
the upload is abandoned — the user may be mid-flow, and a second tab may
be about to submit with it. Nothing the frontend does deletes anything.

*Deletion is conditional, not checked-then-done.* The candidate scan and
the delete are separated by time, and a generation can attach in that
gap. The delete therefore re-tests "is anything referencing this?" inside
the statement itself, so a reference that became used in the meantime
matches zero rows instead of being destroyed. PostgreSQL refuses a second
time via ON DELETE RESTRICT.

*The row goes before the object.* If the object were deleted first and
the row delete then failed because the reference had just been attached,
a live generation would be left pointing at audio that no longer exists.
Reversed, a successful row delete is proof nothing referenced it, and a
failed object delete leaves an orphan that is reported rather than
hidden. A leaked object is a disk cost; a missing one is a broken song.

Provenance is never traded away: if *any* generation row cites a
reference — queued, running, completed or failed — it is not abandoned.
A failed generation is exactly when someone wants to know what the input
was.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from luber_audio_utils import AudioStorage, AudioStorageError, storage_from_settings
from luber_database import (
    GenerationRepository,
    create_async_engine_from_url,
    create_session_factory,
)
from luber_shared import BaseServiceSettings, configure_logging

logger = logging.getLogger(__name__)


@dataclass
class CleanupReport:
    """What one invocation actually did.

    ``deleted`` counts rows removed from the database. It is deliberately
    separate from ``storage_failures``: a row can be gone while its object
    remains, and reporting that as a clean success would hide a real leak
    from whoever has to reconcile storage later.
    """

    scanned: int = 0
    eligible: int = 0
    deleted: int = 0
    skipped_referenced: int = 0
    storage_failures: int = 0
    database_failures: int = 0
    dry_run: bool = False
    #: Ids of references a run would delete. Populated in dry-run so the
    #: decision can be reviewed before anything is destroyed.
    candidates: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "scanned": self.scanned,
            "eligible": self.eligible,
            "deleted": self.deleted,
            "skipped_referenced": self.skipped_referenced,
            "storage_failures": self.storage_failures,
            "database_failures": self.database_failures,
            "dry_run": self.dry_run,
        }


class CleanupConfig(BaseServiceSettings):
    """Grace period and batch size, read from the one shared home."""


async def cleanup_abandoned_references(
    repository: GenerationRepository,
    storage: AudioStorage,
    *,
    grace_hours: int,
    batch_size: int,
    dry_run: bool = False,
    now: datetime | None = None,
) -> CleanupReport:
    """Delete unused reference uploads older than the grace period.

    ``now`` is injectable so the cutoff boundary can be tested exactly
    rather than by waiting.
    """
    moment = now or datetime.now(UTC)
    cutoff = moment - timedelta(hours=grace_hours)
    report = CleanupReport(dry_run=dry_run)

    candidates = await repository.find_abandoned_references(cutoff=cutoff, limit=batch_size)
    report.scanned = len(candidates)
    report.eligible = len(candidates)

    for reference in candidates:
        reference_id = reference.id
        storage_key = reference.storage_key

        if dry_run:
            report.candidates.append(str(reference_id))
            continue

        # Row first. Zero rows means something attached since the scan,
        # which is the race this exists to survive.
        try:
            removed = await repository.delete_reference_audio_if_unused(reference_id)
        except Exception:
            report.database_failures += 1
            logger.warning(
                "could not delete reference row",
                extra={"reference_id": str(reference_id)},
                exc_info=True,
            )
            continue

        if not removed:
            report.skipped_referenced += 1
            logger.info(
                "reference became referenced during cleanup; keeping it",
                extra={"reference_id": str(reference_id)},
            )
            continue

        report.deleted += 1

        # Object second. Idempotent, so a retry of a partially-completed
        # run is safe, and a missing object is not an error.
        try:
            await storage.delete(storage_key)
        except AudioStorageError:
            report.storage_failures += 1
            # Reported, never swallowed: the row is gone, so no later run
            # will rediscover this object on its own.
            logger.error(
                "reference row deleted but its object could not be removed",
                extra={"reference_id": str(reference_id)},
                exc_info=True,
            )

    logger.info("reference cleanup finished", extra=report.as_dict())
    return report


async def _run(dry_run: bool, grace_hours: int | None, batch_size: int | None) -> CleanupReport:
    config = CleanupConfig()
    configure_logging(service="luber-reference-cleanup", level=config.log_level)
    engine = create_async_engine_from_url(config.database_url)
    session_factory = create_session_factory(engine)
    storage = storage_from_settings(config)
    try:
        async with session_factory() as session:
            return await cleanup_abandoned_references(
                GenerationRepository(session),
                storage,
                grace_hours=grace_hours
                if grace_hours is not None
                else config.reference_abandonment_grace_hours,
                batch_size=batch_size
                if batch_size is not None
                else config.reference_cleanup_batch_size,
                dry_run=dry_run,
            )
    finally:
        await engine.dispose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="luber-reference-cleanup",
        description="Remove reference uploads that no generation ever used.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be deleted without touching storage or the database.",
    )
    parser.add_argument(
        "--grace-hours",
        type=int,
        default=None,
        help="Override the configured abandonment grace period.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override the per-run ceiling."
    )
    args = parser.parse_args(argv)

    report = asyncio.run(_run(args.dry_run, args.grace_hours, args.batch_size))
    for key, value in report.as_dict().items():
        print(f"{key}: {value}")
    if report.dry_run and report.candidates:
        print("candidates:")
        for reference_id in report.candidates:
            print(f"  {reference_id}")
    # A storage failure leaves a real orphan, so it must not look like a
    # clean run to whatever scheduled this.
    return 1 if (report.storage_failures or report.database_failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
