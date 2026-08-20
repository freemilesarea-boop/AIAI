"""The canonical manifest record, versioned.

One record per canonical track, sectioned so a consumer can read the
part it needs without understanding the rest. The sections are stable;
fields may be added within them, and a change to their meaning bumps
``SCHEMA_VERSION``.

The record carries a *canonical* form deliberately distinct from its
serialised form. Canonical excludes anything that varies between two
runs over identical inputs — timestamps, absolute machine paths, wall
times — so two runs can be compared by hash. Without that separation
"is this dataset the same as last week's" is unanswerable, because every
record would differ in its `created_at` alone.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from luber_dataset.factory.config import SCHEMA_VERSION

#: Keys excluded from the canonical hash: real information, but not
#: information about the dataset's *content*.
NON_CANONICAL_KEYS: frozenset[str] = frozenset(
    {"created_at", "generated_at", "source_mtime", "analysis_seconds", "factory_host"}
)


@dataclass
class TrackRecord:
    """One canonical track, as it appears in ``dataset_manifest.jsonl``."""

    track_id: str
    source: dict[str, Any] = field(default_factory=dict)
    audio: dict[str, Any] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)
    music: dict[str, Any] = field(default_factory=dict)
    vocals: dict[str, Any] = field(default_factory=dict)
    text: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    dedup: dict[str, Any] = field(default_factory=dict)
    eligibility: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    split: str = "EXCLUDED"
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "track_id": self.track_id,
            "source": self.source,
            "audio": self.audio,
            "analysis": self.analysis,
            "music": self.music,
            "vocals": self.vocals,
            "text": self.text,
            "quality": self.quality,
            "provenance": self.provenance,
            "dedup": self.dedup,
            "eligibility": self.eligibility,
            "metadata": self.metadata,
            "split": self.split,
        }

    def canonical_dict(self) -> dict[str, Any]:
        """The record with run-varying fields removed."""
        stripped = strip_non_canonical(self.to_dict())
        assert isinstance(stripped, dict)
        return stripped


def strip_non_canonical(payload: Any) -> Any:
    """Recursively drop keys that legitimately differ between runs."""
    if isinstance(payload, dict):
        return {
            key: strip_non_canonical(value)
            for key, value in sorted(payload.items())
            if key not in NON_CANONICAL_KEYS
        }
    if isinstance(payload, list):
        return [strip_non_canonical(item) for item in payload]
    return payload


def canonical_json(payload: Any) -> str:
    """Sorted, compact, unicode-preserving. The hashing form."""
    return json.dumps(
        strip_non_canonical(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def manifest_digest(records: list[TrackRecord]) -> str:
    """Content hash over every canonical record.

    Records are sorted by track id first, so a change in processing
    order — which parallelism makes inevitable — does not change the
    digest. Only the content does.
    """
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda r: r.track_id):
        digest.update(canonical_json(record.to_dict()).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass
class ReviewItem:
    """One thing a human needs to decide."""

    track_id: str
    reason: str
    detail: str
    source_path: str
    recommended_action: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "reason": self.reason,
            "detail": self.detail,
            "source_path": self.source_path,
            "recommended_action": self.recommended_action,
            "metrics": self.metrics,
        }


# ── review reasons ───────────────────────────────────────────────────
REVIEW_NEAR_DUPLICATE = "NEAR_DUPLICATE"
REVIEW_RIGHTS_UNKNOWN = "RIGHTS_UNKNOWN"
REVIEW_VOCAL_CLASS_UNCERTAIN = "VOCAL_CLASS_UNCERTAIN"
REVIEW_LANGUAGE_UNCERTAIN = "LANGUAGE_UNCERTAIN"
REVIEW_QUALITY_BORDERLINE = "QUALITY_BORDERLINE"
REVIEW_METADATA_CONFLICT = "METADATA_CONFLICT"


@dataclass
class RejectionRecord:
    """A track that will not be trained on, and why."""

    track_id: str
    source_path: str
    reasons: list[str] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)
    quality_tier: str = "REJECT"
    decode_status: str = "VALID"
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "source_path": self.source_path,
            "reasons": sorted(set(self.reasons)),
            "quality_flags": sorted(set(self.quality_flags)),
            "quality_tier": self.quality_tier,
            "decode_status": self.decode_status,
            "detail": self.detail,
        }
