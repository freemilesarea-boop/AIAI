"""Training-set manifest assembly.

Builds `LUBER_TRAINSET_V1`-style manifests from candidate tracks,
applying the rights gate and the quality gate. Excluded tracks are
recorded with their reason rather than silently dropped, so a manifest
doubles as an audit trail of what was rejected and why.

A manifest carries a content hash over its accepted tracks, which is
what a training run cites for reproducibility (see Step 15).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from luber_dataset.rights import RightsError, validate_rights
from luber_dataset.schema import (
    ACCEPTABLE_GRADES,
    DISCOURAGED_STYLES,
    QualityGrade,
    TrainingTrack,
)


@dataclass
class Exclusion:
    track_id: str
    reason: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"track_id": self.track_id, "reason": self.reason, "detail": self.detail}


@dataclass
class DatasetManifest:
    dataset_id: str
    created_at: str
    tracks: list[TrainingTrack] = field(default_factory=list)
    exclusions: list[Exclusion] = field(default_factory=list)
    notes: str = ""

    @property
    def track_count(self) -> int:
        return len(self.tracks)

    @property
    def total_duration_seconds(self) -> float:
        return sum(t.duration_seconds for t in self.tracks)

    def content_hash(self) -> str:
        """Stable hash over accepted track identity and audio digests."""
        digest = hashlib.sha256()
        for track in sorted(self.tracks, key=lambda t: t.track_id):
            digest.update(track.track_id.encode("utf-8"))
            digest.update(track.audio_sha256.encode("utf-8"))
        return digest.hexdigest()

    def style_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for track in self.tracks:
            key = str(track.vocal.vocal_style) if track.vocal else "instrumental"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def language_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for track in self.tracks:
            counts[track.language] = counts.get(track.language, 0) + 1
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "created_at": self.created_at,
            "track_count": self.track_count,
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "content_hash": self.content_hash(),
            "style_distribution": self.style_distribution(),
            "language_distribution": self.language_distribution(),
            "tracks": [t.to_dict() for t in self.tracks],
            "exclusions": [e.to_dict() for e in self.exclusions],
            "notes": self.notes,
        }

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return path


def build_manifest(
    dataset_id: str,
    candidates: list[TrainingTrack],
    *,
    notes: str = "",
    allow_discouraged_styles: bool = False,
) -> DatasetManifest:
    """Apply both gates and assemble a manifest.

    Rights are checked first: a track with unconfirmed rights is
    excluded before its audio quality is even considered, because no
    amount of quality makes unlicensed material usable.
    """
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        created_at=datetime.now(UTC).isoformat(),
        notes=notes,
    )

    for track in candidates:
        try:
            validate_rights(
                track.rights,
                has_lyrics=track.lyrics_available,
                has_vocals=track.has_vocals,
            )
        except RightsError as exc:
            manifest.exclusions.append(Exclusion(track.track_id, "RIGHTS", str(exc)))
            continue

        if track.quality_grade not in ACCEPTABLE_GRADES:
            manifest.exclusions.append(
                Exclusion(track.track_id, "QUALITY_GRADE", f"grade is {track.quality_grade}")
            )
            continue

        if track.audio_quality_flags:
            manifest.exclusions.append(
                Exclusion(
                    track.track_id,
                    "AUDIO_QUALITY",
                    ", ".join(track.audio_quality_flags),
                )
            )
            continue

        if track.lyrics_qa_flags:
            manifest.exclusions.append(
                Exclusion(track.track_id, "LYRICS_QA", ", ".join(track.lyrics_qa_flags))
            )
            continue

        if (
            not allow_discouraged_styles
            and track.vocal
            and track.vocal.vocal_style in DISCOURAGED_STYLES
        ):
            manifest.exclusions.append(
                Exclusion(
                    track.track_id,
                    "DISCOURAGED_STYLE",
                    f"{track.vocal.vocal_style} is the bias this training set exists to correct",
                )
            )
            continue

        manifest.tracks.append(track)

    return manifest


@dataclass
class TrainingRunManifest:
    """Everything needed to reproduce one training run (Step 15)."""

    run_id: str
    base_model: str
    ace_step_commit: str
    dataset_id: str
    dataset_content_hash: str
    lora_rank: int
    lora_alpha: int
    learning_rate: float
    optimizer: str
    batch_size: int
    gradient_accumulation_steps: int
    max_epochs: int
    save_every_n_epochs: int
    val_split: float
    seed: int
    gpu: str
    cuda_version: str
    started_at: str = ""
    finished_at: str = ""
    training_seconds: float | None = None
    gpu_hours: float | None = None
    estimated_cost_usd: float | None = None
    checkpoint_hashes: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return path


def validate_run_manifest(manifest: TrainingRunManifest) -> None:
    """Refuse a run manifest that could not reproduce the run."""
    required = {
        "run_id": manifest.run_id,
        "base_model": manifest.base_model,
        "ace_step_commit": manifest.ace_step_commit,
        "dataset_id": manifest.dataset_id,
        "dataset_content_hash": manifest.dataset_content_hash,
        "gpu": manifest.gpu,
    }
    missing = sorted(name for name, value in required.items() if not str(value).strip())
    if missing:
        raise ValueError(f"training run manifest is missing: {', '.join(missing)}")
    if manifest.lora_rank <= 0 or manifest.lora_alpha <= 0:
        raise ValueError("lora rank and alpha must be positive")
    if manifest.learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    if not 0.0 <= manifest.val_split < 1.0:
        raise ValueError("val_split must be in [0, 1)")
    # Step 16: without validation there is no way to detect overtraining.
    if manifest.val_split == 0.0:
        raise ValueError(
            "val_split is 0: overtraining cannot be detected without a validation split"
        )


def summarize_quality_grade(audio_flags: list[str], lyrics_flags: list[str]) -> QualityGrade:
    """Derive a grade from measured flags rather than opinion."""
    if audio_flags or lyrics_flags:
        return QualityGrade.REJECTED
    return QualityGrade.GOOD
