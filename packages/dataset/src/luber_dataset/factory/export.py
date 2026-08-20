"""Turning a manifest into what a trainer reads.

The exporter consumes canonical records and never touches the source
tree. It does not rescan, it does not re-analyse, and it does not decide
anything the manifest has not already decided — it filters and projects.
That matters because a second implementation of "is this track eligible"
is a second answer to the question, and the two would drift.

The default exclusions are the point of the module:

* quality tier ``REJECT``
* ``commercial_training_allowed`` FALSE
* ``commercial_training_allowed`` UNKNOWN
* duplicates awaiting review

Every one of them can only be relaxed by passing an explicit flag, and
the two that concern rights are refused outright for hard-blocked audio
regardless of any flag. An override that could clear a hard block would
make the block decorative.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from luber_dataset.factory.schemas import TrackRecord
from luber_dataset.factory.splitting import Split

TRAIN_NAME = "train.jsonl"
VALIDATION_NAME = "validation.jsonl"
TEST_NAME = "test.jsonl"


@dataclass
class ExportPolicy:
    """What may be relaxed, and what may not.

    ``allow_rights_unknown`` exists because an operator working entirely
    on their own material may legitimately decide the whole library is
    theirs. It is a decision they make once, visibly, and it still
    cannot admit audio that was hard-blocked.
    """

    allow_rights_unknown: bool = False
    allow_review_required: bool = False
    allow_quality_reject: bool = False
    min_tier: str = "B"

    def to_dict(self) -> dict[str, Any]:
        return {
            "allow_rights_unknown": self.allow_rights_unknown,
            "allow_review_required": self.allow_review_required,
            "allow_quality_reject": self.allow_quality_reject,
            "min_tier": self.min_tier,
        }


@dataclass
class ExportResult:
    counts: dict[str, int] = field(default_factory=dict)
    excluded: dict[str, int] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": dict(sorted(self.counts.items())),
            "excluded": dict(sorted(self.excluded.items())),
            "paths": dict(sorted(self.paths.items())),
        }


def _blocked_reasons(record: TrackRecord, policy: ExportPolicy) -> list[str]:
    """Why this record may not be exported. Empty means it may."""
    reasons: list[str] = []
    provenance = record.provenance or {}
    quality = record.quality or {}
    dedup = record.dedup or {}
    eligibility = record.eligibility or {}

    # Never overridable, by anything.
    if provenance.get("hard_blocks"):
        reasons.append("RIGHTS_HARD_BLOCK")

    permission = provenance.get("commercial_training_allowed", "UNKNOWN")
    if permission == "FALSE":
        reasons.append("RIGHTS_DENIED")
    elif permission != "TRUE" and not policy.allow_rights_unknown:
        reasons.append("RIGHTS_UNKNOWN")

    tier = quality.get("quality_tier", "REJECT")
    if tier == "REJECT" and not policy.allow_quality_reject:
        reasons.append("QUALITY_REJECTED")

    from luber_dataset.factory.quality import meets_tier

    if not meets_tier(tier, policy.min_tier):
        reasons.append("QUALITY_BELOW_MINIMUM_TIER")

    decision = dedup.get("dedup_decision", "KEEP")
    if decision == "MERGED":
        reasons.append("DUPLICATE_OF_ANOTHER_TRACK")
    elif decision == "REVIEW_REQUIRED" and not policy.allow_review_required:
        reasons.append("NEAR_DUPLICATE_REVIEW_REQUIRED")

    if not eligibility.get("analysis_eligible", False):
        reasons.append("DECODE_FAILED")
    return reasons


def training_row(record: TrackRecord) -> dict[str, Any]:
    """The projection a trainer consumes.

    Deliberately narrow. A trainer needs the audio path, the conditioning
    text and the measured attributes; it has no use for the rejection
    reasoning, and carrying the whole record would invite a training
    script to start making eligibility decisions of its own.
    """
    analysis = record.analysis or {}
    music = record.music or {}
    return {
        "track_id": record.track_id,
        "audio_path": (record.source or {}).get("source_path"),
        "sha256": (record.source or {}).get("sha256"),
        "duration_seconds": analysis.get("duration_seconds"),
        "sample_rate": analysis.get("sample_rate"),
        "channels": analysis.get("channels"),
        "bpm": music.get("bpm"),
        "key": music.get("key"),
        "mode": music.get("mode"),
        "language": ((record.metadata or {}).get("language") or {}).get("language"),
        "vocal_class": (record.vocals or {}).get("vocal_class"),
        "lyrics": (record.text or {}).get("lyrics"),
        "quality_tier": (record.quality or {}).get("quality_tier"),
        "split": record.split,
    }


def export(
    records: list[TrackRecord],
    output_root: Path,
    policy: ExportPolicy | None = None,
) -> ExportResult:
    """Write train/validation/test manifests from canonical records."""
    policy = policy or ExportPolicy()
    result = ExportResult()
    buckets: dict[str, list[dict[str, Any]]] = {
        Split.TRAIN.value: [],
        Split.VALIDATION.value: [],
        Split.TEST.value: [],
    }

    for record in sorted(records, key=lambda r: r.track_id):
        reasons = _blocked_reasons(record, policy)
        if reasons:
            for reason in reasons:
                result.excluded[reason] = result.excluded.get(reason, 0) + 1
            continue
        if record.split not in buckets:
            result.excluded["NOT_ASSIGNED_TO_A_SPLIT"] = (
                result.excluded.get("NOT_ASSIGNED_TO_A_SPLIT", 0) + 1
            )
            continue
        buckets[record.split].append(training_row(record))

    output_root.mkdir(parents=True, exist_ok=True)
    for split, name in (
        (Split.TRAIN.value, TRAIN_NAME),
        (Split.VALIDATION.value, VALIDATION_NAME),
        (Split.TEST.value, TEST_NAME),
    ):
        path = output_root / name
        with path.open("w", encoding="utf-8") as handle:
            for row in buckets[split]:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        result.counts[split] = len(buckets[split])
        result.paths[split] = str(path)
    return result
