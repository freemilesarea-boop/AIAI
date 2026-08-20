"""Writing a curation run out, including the parts nobody wants to read.

Six artifacts, because six different questions get asked of a curation
run and one file cannot answer them all: the curated manifest, a machine
summary, a human report, an acquisition wishlist, a prioritised review
queue, and the lock that freezes the decision.

The human report is the one that matters most and is the easiest to
write badly. It is deliberately organised around questions rather than
sections — what dominates this, what is missing, what cannot be assessed
— because a report organised around the code that produced it gets
skimmed, and the "cannot be assessed" part is exactly the part a reader
needs and would never think to ask for.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from luber_dataset.factory.intelligence.curation import CurationResult
from luber_dataset.factory.intelligence.profile import DatasetProfile
from luber_dataset.factory.intelligence.schemas import (
    CURATION_ENGINE_VERSION,
    CURATION_SCHEMA_VERSION,
    Severity,
)

CURATED_MANIFEST_NAME = "curated_manifest.jsonl"
SUMMARY_NAME = "curation_summary.json"
REPORT_NAME = "curation_report.md"
WISHLIST_NAME = "dataset_wishlist.json"
REVIEW_NAME = "prioritized_review_queue.jsonl"
WEIGHTS_NAME = "training_sampling_weights.jsonl"
LOCK_NAME = "curation_lock.json"
DIFF_JSON_NAME = "curation_diff.json"
DIFF_MD_NAME = "curation_diff.md"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_curated_manifest(output: Path, result: CurationResult) -> str:
    rows = sorted(result.curated_records, key=lambda r: str(r.get("track_id")))
    return _write_jsonl(output / CURATED_MANIFEST_NAME, rows)


def write_sampling_weights(output: Path, result: CurationResult) -> str:
    return _write_jsonl(output / WEIGHTS_NAME, result.sampling_plan.to_rows())


def _distribution_snapshot(profile: DatasetProfile | None) -> dict[str, Any]:
    """The few distributions a before/after comparison actually reads."""
    if profile is None:
        return {}
    keep = (
        "quality_tier",
        "language",
        "vocal_class",
        "tempo_bucket",
        "duration_bucket",
        "genre",
        "artist",
        "source_type",
    )
    return {
        "track_count": profile.track_count,
        "total_hours": round(profile.total_hours, 4),
        "distributions": {
            name: profile.categorical[name].to_dict()
            for name in keep
            if name in profile.categorical
        },
        "concentration": {
            name: metrics.to_dict() for name, metrics in sorted(profile.concentration.items())
        },
        "family_pressure": profile.family_pressure.to_dict(),
        "synthetic_share_by_count": round(profile.synthetic_share_by_count, 6),
    }


def build_summary(result: CurationResult) -> dict[str, Any]:
    corpus, eligible, selected = (
        result.corpus_profile,
        result.eligible_profile,
        result.selected_profile,
    )
    decisions = result.selection.decisions.values()
    by_action: dict[str, int] = {}
    for decision in decisions:
        by_action[decision.action] = by_action.get(decision.action, 0) + 1

    return {
        "curation_schema_version": CURATION_SCHEMA_VERSION,
        "curation_engine_version": CURATION_ENGINE_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_manifest_sha256": result.source_manifest_sha256,
        "source_dataset_lock_sha256": result.source_dataset_lock_sha256,
        "factory_schema_version": result.factory_schema_version,
        "factory_version": result.factory_version,
        "target_profile": result.target_profile.name,
        "target_profile_digest": result.target_profile.digest(),
        "config_digest": result.config.digest(),
        "tracks_input": corpus.track_count if corpus else 0,
        "hours_input": round(corpus.total_hours, 4) if corpus else 0.0,
        "training_eligible_input": eligible.track_count if eligible else 0,
        "training_eligible_hours": round(eligible.total_hours, 4) if eligible else 0.0,
        "tracks_selected": selected.track_count if selected else 0,
        "hours_selected": round(selected.total_hours, 4) if selected else 0.0,
        "actions": dict(sorted(by_action.items())),
        "excluded_by_reason": dict(sorted(result.selection.excluded_by_reason.items())),
        "findings": [finding.to_dict() for finding in result.findings],
        "sampling": result.sampling_plan.to_dict(),
        "before": _distribution_snapshot(eligible),
        "after": _distribution_snapshot(selected),
        "completeness": (
            {name: score.to_dict() for name, score in sorted(eligible.completeness.items())}
            if eligible
            else {}
        ),
    }


def write_summary(output: Path, result: CurationResult) -> str:
    return _write_json(output / SUMMARY_NAME, build_summary(result))


def build_wishlist(result: CurationResult) -> list[dict[str, Any]]:
    """What more material would improve coverage.

    Derived strictly from findings that name a target range. Nothing
    here is invented: an entry exists only where a profile declared a
    minimum the dataset does not meet, and the hours are computed from
    the stated deficit rather than guessed.
    """
    entries: list[dict[str, Any]] = []
    eligible = result.eligible_profile
    known_hours = eligible.total_hours if eligible else 0.0

    for finding in result.findings:
        if not finding.code.startswith("NEED_MORE_"):
            continue
        bounds = finding.target_range
        if bounds is None or finding.current_share is None:
            continue
        minimum = bounds[0]
        current = finding.current_share
        # Hours needed to reach the floor, holding everything else
        # constant: solving (current_hours + x) / (total + x) = minimum.
        current_hours = finding.affected_hours
        needed = None
        if 0.0 < minimum < 1.0 and known_hours > 0:
            needed = (minimum * known_hours - current_hours) / (1.0 - minimum)
            needed = round(max(0.0, needed), 2)
        entries.append(
            {
                "dimension": finding.dimension,
                "target": finding.code.replace(
                    f"NEED_MORE_{finding.dimension.upper()}_", ""
                ).lower(),
                "need": "more",
                "current_share": round(current, 6),
                "minimum_share": minimum,
                "estimated_hours_needed": needed,
                "estimation_basis": (
                    "hours required to reach the declared minimum with the rest of the "
                    "dataset held constant"
                    if needed is not None
                    else "not derivable from the declared range"
                ),
                "priority": "high" if finding.severity == Severity.CRITICAL.value else "medium",
            }
        )

    # Concentration findings ask for breadth rather than a category.
    for finding in result.findings:
        if finding.code not in ("LOW_EFFECTIVE_ARTIST_COUNT", "ONE_ARTIST_DOMINATES"):
            continue
        entries.append(
            {
                "dimension": "artist",
                "target": "distinct artists",
                "need": "more",
                "current_share": finding.current_share,
                "minimum_share": None,
                "estimated_hours_needed": None,
                "estimation_basis": (
                    "not derivable: breadth is a count of sources, not a share of hours"
                ),
                "priority": "high",
            }
        )
    return sorted(entries, key=lambda e: (e["priority"] != "high", e["dimension"], e["target"]))


def write_wishlist(output: Path, result: CurationResult) -> str:
    return _write_json(
        output / WISHLIST_NAME,
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "target_profile": result.target_profile.name,
            "source_manifest_sha256": result.source_manifest_sha256,
            "entries": build_wishlist(result),
        },
    )


#: How much a review decision is worth unblocking, highest first. Rights
#: lead because they are the only category that unlocks training hours
#: outright — a resolved rights question can move a track from excluded
#: to selected, while a resolved genre question only improves analysis.
REVIEW_PRIORITY: dict[str, int] = {
    "RIGHTS_UNKNOWN": 0,
    "NEAR_DUPLICATE": 1,
    "METADATA_CONFLICT": 2,
    "VOCAL_CLASS_UNCERTAIN": 3,
    "QUALITY_BORDERLINE": 4,
    "LANGUAGE_UNCERTAIN": 5,
}


def build_review_queue(
    result: CurationResult, existing: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Order Phase 23's review items by what they would unblock.

    Phase 23 decides *what* needs review; this decides what to look at
    first. Human decisions are never overridden — an item's reason and
    recommended action pass through untouched.
    """
    hours_by_track = {
        str(record.get("track_id")): float(
            (record.get("analysis") or {}).get("duration_seconds") or 0.0
        )
        / 3600.0
        for record in result.curated_records
    }

    prioritized: list[dict[str, Any]] = []
    for item in existing:
        reason = str(item.get("reason", ""))
        track_id = str(item.get("track_id", ""))
        hours = hours_by_track.get(track_id, 0.0)
        prioritized.append(
            {
                **item,
                "priority_rank": REVIEW_PRIORITY.get(reason, 9),
                "unblocks_hours": round(hours, 4),
                "why_this_order": _why(reason),
            }
        )
    # Rank first, then hours descending: within rights review, the track
    # that unlocks the most material is worth the operator's attention
    # before one that unlocks thirty seconds.
    prioritized.sort(
        key=lambda i: (i["priority_rank"], -i["unblocks_hours"], str(i.get("track_id")))
    )
    return prioritized


def _why(reason: str) -> str:
    return {
        "RIGHTS_UNKNOWN": "resolving rights is the only review that can add training hours",
        "NEAR_DUPLICATE": "a large family distorts training weight until it is resolved",
        "METADATA_CONFLICT": "conflicting metadata blocks target-profile analysis",
        "VOCAL_CLASS_UNCERTAIN": "improves analysis; does not change eligibility",
        "QUALITY_BORDERLINE": "affects tier, which affects selection order",
        "LANGUAGE_UNCERTAIN": "improves analysis; does not change eligibility",
    }.get(reason, "no specific unblocking effect recorded")


def write_review_queue(output: Path, result: CurationResult, existing: list[dict[str, Any]]) -> str:
    return _write_jsonl(output / REVIEW_NAME, build_review_queue(result, existing))


@dataclass
class CurationLock:
    """Freezes one curation decision so a training run can cite it."""

    curation_id: str
    created_at: str
    engine_version: str
    schema_version: str
    source_manifest_sha256: str
    source_dataset_lock_sha256: str | None
    target_profile_sha256: str
    config_sha256: str
    curated_manifest_sha256: str
    sampling_weights_sha256: str | None
    selected_track_count: int
    selected_hours: float
    distribution_summary_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "curation_id": self.curation_id,
            "created_at": self.created_at,
            "engine_version": self.engine_version,
            "schema_version": self.schema_version,
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_dataset_lock_sha256": self.source_dataset_lock_sha256,
            "target_profile_sha256": self.target_profile_sha256,
            "config_sha256": self.config_sha256,
            "curated_manifest_sha256": self.curated_manifest_sha256,
            "sampling_weights_sha256": self.sampling_weights_sha256,
            "selected_track_count": self.selected_track_count,
            "selected_hours": self.selected_hours,
            "distribution_summary_digest": self.distribution_summary_digest,
        }


def distribution_digest(result: CurationResult) -> str:
    """Digest over the after-distributions, timestamps excluded."""
    payload = json.dumps(
        _distribution_snapshot(result.selected_profile),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def freeze(
    output: Path,
    result: CurationResult,
    *,
    curation_id: str,
    weights_digest: str | None = None,
) -> CurationLock:
    """Write ``curation_lock.json``.

    The curated-manifest digest is the *canonical* one — computed over
    record content, not over the file — so a lock survives a rewrite
    that changes nothing but formatting, and cannot survive one that
    changes a decision.
    """
    selected = result.selected_profile
    lock = CurationLock(
        curation_id=curation_id,
        created_at=datetime.now(UTC).isoformat(),
        engine_version=CURATION_ENGINE_VERSION,
        schema_version=CURATION_SCHEMA_VERSION,
        source_manifest_sha256=result.source_manifest_sha256,
        source_dataset_lock_sha256=result.source_dataset_lock_sha256,
        target_profile_sha256=result.target_profile.digest(),
        config_sha256=result.config.digest(),
        curated_manifest_sha256=result.canonical_digest(),
        sampling_weights_sha256=weights_digest,
        selected_track_count=selected.track_count if selected else 0,
        selected_hours=round(selected.total_hours, 4) if selected else 0.0,
        distribution_summary_digest=distribution_digest(result),
    )
    _write_json(output / LOCK_NAME, lock.to_dict())
    return lock


def verify(lock_path: Path, result: CurationResult) -> list[str]:
    """Differences between a lock and a freshly computed curation."""
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    problems: list[str] = []

    checks = (
        ("source_manifest_sha256", result.source_manifest_sha256, "source manifest"),
        ("target_profile_sha256", result.target_profile.digest(), "target profile"),
        ("config_sha256", result.config.digest(), "curation config"),
        ("curated_manifest_sha256", result.canonical_digest(), "curated manifest"),
        ("distribution_summary_digest", distribution_digest(result), "distribution summary"),
    )
    for key, actual, label in checks:
        if payload.get(key) != actual:
            problems.append(f"{label} digest differs from the lock")

    selected = result.selected_profile
    if payload.get("selected_track_count") != (selected.track_count if selected else 0):
        problems.append(
            f"selected track count changed: {payload.get('selected_track_count')} -> "
            f"{selected.track_count if selected else 0}"
        )
    if payload.get("schema_version") != CURATION_SCHEMA_VERSION:
        problems.append(
            f"curation schema version changed: {payload.get('schema_version')} -> "
            f"{CURATION_SCHEMA_VERSION}"
        )
    return problems
