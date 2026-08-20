"""Writing the dataset out, and freezing it.

Four files, because four different questions get asked of a build and
one file cannot answer them all:

* ``dataset_manifest.jsonl`` — one canonical track per line, the thing
  everything downstream reads;
* ``dataset_rejections.jsonl`` — what was excluded and why, so a build
  that lost six hundred tracks can account for them;
* ``dataset_duplicates.jsonl`` — what was folded into what;
* ``dataset_review_queue.jsonl`` — what a human still has to decide.

JSONL rather than one large JSON array: a forty-thousand-track manifest
is streamable line by line, and a truncated write costs the last record
instead of the file.

The freeze turns a build into something a training run can cite. Its
digest is taken over the *canonical* form of each record, so two builds
of unchanged audio produce the same lock even though their timestamps
differ — which is the only way "is this the dataset we trained on" can
be answered later.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from luber_dataset.factory.config import FACTORY_VERSION, SCHEMA_VERSION, FactoryConfig
from luber_dataset.factory.schemas import (
    RejectionRecord,
    ReviewItem,
    TrackRecord,
    canonical_json,
    manifest_digest,
)

MANIFEST_NAME = "dataset_manifest.jsonl"
SUMMARY_NAME = "dataset_summary.json"
REJECTIONS_NAME = "dataset_rejections.jsonl"
DUPLICATES_NAME = "dataset_duplicates.jsonl"
REVIEW_NAME = "dataset_review_queue.jsonl"
LOCK_NAME = "dataset_lock.json"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def write_manifest(output_root: Path, records: list[TrackRecord]) -> Path:
    """Sorted by track id, so the file itself is deterministic."""
    rows = [record.to_dict() for record in sorted(records, key=lambda r: r.track_id)]
    return _write_jsonl(output_root / MANIFEST_NAME, rows)


def write_rejections(output_root: Path, rejections: list[RejectionRecord]) -> Path:
    rows = [item.to_dict() for item in sorted(rejections, key=lambda r: r.track_id)]
    return _write_jsonl(output_root / REJECTIONS_NAME, rows)


def write_duplicates(output_root: Path, duplicates: list[dict[str, Any]]) -> Path:
    rows = sorted(duplicates, key=lambda d: str(d.get("track_id")))
    return _write_jsonl(output_root / DUPLICATES_NAME, rows)


def write_review_queue(output_root: Path, items: list[ReviewItem]) -> Path:
    rows = [item.to_dict() for item in sorted(items, key=lambda i: (i.track_id, i.reason))]
    return _write_jsonl(output_root / REVIEW_NAME, rows)


def write_summary(output_root: Path, summary: dict[str, Any]) -> Path:
    path = output_root / SUMMARY_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def canonical_manifest_digest(records: list[TrackRecord]) -> str:
    """Content digest ignoring timestamps and other run-varying fields."""
    return manifest_digest(records)


@dataclass
class DatasetLock:
    """What a training run cites to prove which dataset it used."""

    dataset_id: str
    created_at: str
    schema_version: str
    factory_version: str
    configuration_hash: str
    manifest_sha256: str
    track_count: int
    total_duration_seconds: float
    split_counts: dict[str, int]
    source_identity_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
            "factory_version": self.factory_version,
            "configuration_hash": self.configuration_hash,
            "manifest_sha256": self.manifest_sha256,
            "track_count": self.track_count,
            "total_duration_seconds": self.total_duration_seconds,
            "split_counts": dict(sorted(self.split_counts.items())),
            "source_identity_digest": self.source_identity_digest,
        }


def source_identity_digest(records: list[TrackRecord]) -> str:
    """Digest over the source audio itself, independent of the manifest.

    Two separate questions: "did the manifest change" and "did the audio
    change". A threshold tweak moves the first and must not move the
    second, and a silently altered source file must move the second even
    if the manifest happens to round to the same numbers.
    """
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda r: r.track_id):
        digest.update(str(record.source.get("sha256", "")).encode("utf-8"))
    return digest.hexdigest()


def freeze(
    output_root: Path,
    records: list[TrackRecord],
    config: FactoryConfig,
    *,
    dataset_id: str,
) -> DatasetLock:
    """Write ``dataset_lock.json`` for an approved build.

    Deliberately a separate step from building. A dataset is frozen when
    a person decides it is ready, not automatically because a run
    finished — otherwise the lock records whatever the last run happened
    to produce and proves nothing.
    """
    total = sum(float((r.analysis or {}).get("duration_seconds") or 0.0) for r in records)
    splits: dict[str, int] = {}
    for record in records:
        splits[record.split] = splits.get(record.split, 0) + 1

    lock = DatasetLock(
        dataset_id=dataset_id,
        created_at=datetime.now(UTC).isoformat(),
        schema_version=SCHEMA_VERSION,
        factory_version=FACTORY_VERSION,
        configuration_hash=config.configuration_hash(),
        manifest_sha256=canonical_manifest_digest(records),
        track_count=len(records),
        total_duration_seconds=round(total, 3),
        split_counts=splits,
        source_identity_digest=source_identity_digest(records),
    )
    path = output_root / LOCK_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(lock.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return lock


def read_manifest(path: Path) -> list[TrackRecord]:
    """Load a manifest back, skipping nothing silently."""
    records: list[TrackRecord] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} line {number} is not valid JSON: {exc}") from exc
        records.append(
            TrackRecord(
                track_id=payload["track_id"],
                source=payload.get("source", {}),
                audio=payload.get("audio", {}),
                analysis=payload.get("analysis", {}),
                music=payload.get("music", {}),
                vocals=payload.get("vocals", {}),
                text=payload.get("text", {}),
                quality=payload.get("quality", {}),
                provenance=payload.get("provenance", {}),
                dedup=payload.get("dedup", {}),
                eligibility=payload.get("eligibility", {}),
                metadata=payload.get("metadata", {}),
                split=payload.get("split", "EXCLUDED"),
                schema_version=payload.get("schema_version", SCHEMA_VERSION),
            )
        )
    return records


def verify_lock(lock_path: Path, records: list[TrackRecord]) -> list[str]:
    """Differences between a lock and the current records."""
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    if payload.get("manifest_sha256") != canonical_manifest_digest(records):
        problems.append("manifest content differs from the frozen digest")
    if payload.get("source_identity_digest") != source_identity_digest(records):
        problems.append("source audio differs from the frozen digest")
    if payload.get("track_count") != len(records):
        problems.append(f"track count changed: {payload.get('track_count')} -> {len(records)}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            f"schema version changed: {payload.get('schema_version')} -> {SCHEMA_VERSION}"
        )
    return problems


__all__ = [
    "DUPLICATES_NAME",
    "LOCK_NAME",
    "MANIFEST_NAME",
    "REJECTIONS_NAME",
    "REVIEW_NAME",
    "SUMMARY_NAME",
    "DatasetLock",
    "canonical_json",
    "canonical_manifest_digest",
    "freeze",
    "read_manifest",
    "source_identity_digest",
    "verify_lock",
    "write_duplicates",
    "write_manifest",
    "write_rejections",
    "write_review_queue",
    "write_summary",
]
