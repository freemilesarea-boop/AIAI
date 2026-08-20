"""Synthetic Phase 23 manifests with distributions chosen on purpose.

Curation is arithmetic over metadata, so its tests need manifests rather
than audio — and building them directly is not a shortcut, it is the
only way to test a claim like "detects when one artist holds 90%". Real
audio cannot be made to have a stated concentration; a fixture can.

The records here match the *audited* Phase 23 schema, including the
detail that artist, album, genre and mood live inside
``metadata.sidecar`` rather than at the top level. A fixture that
invented those as first-class fields would let every accessor test pass
against a shape the factory never produces.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

FACTORY_SCHEMA_VERSION = "luber-dataset-factory/1"


def record(
    track_id: str,
    *,
    duration: float = 200.0,
    quality_tier: str = "A",
    quality_flags: list[str] | None = None,
    quality_score: float = 1.0,
    language: str | None = None,
    vocal_class: str = "UNCERTAIN",
    vocal_gender: str | None = None,
    bpm: float | None = None,
    bpm_confidence: float = 0.9,
    key: str | None = None,
    mode: str | None = None,
    key_confidence: float = 0.5,
    artist: str | None = None,
    album: str | None = None,
    genre: str | None = None,
    mood: str | None = None,
    embedded: dict[str, str] | None = None,
    rights_status: str = "USER_OWNED",
    permission: str = "TRUE",
    source_type: str = "USER_ORIGINAL",
    source_reference: str = "library",
    hard_blocks: list[str] | None = None,
    training_eligible: bool = True,
    eligibility_reasons: list[str] | None = None,
    split: str = "TRAIN",
    duplicate_group: str | None = None,
    dedup_decision: str = "KEEP",
    sample_rate: int = 44_100,
    channels: int = 2,
    lyrics: str | None = None,
    technical: dict[str, float] | None = None,
) -> dict[str, Any]:
    """One manifest record, shaped exactly as Phase 23 writes them."""
    sidecar: dict[str, Any] = {}
    for name, value in (
        ("artist", artist),
        ("album", album),
        ("genre", genre),
        ("mood", mood),
        ("language", language),
    ):
        if value is not None:
            sidecar[name] = value

    analysis: dict[str, Any] = {
        "duration_seconds": duration,
        "sample_rate": sample_rate,
        "channels": channels,
        "integrated_lufs": -14.0,
        "true_peak_dbtp": -1.0,
        "crest_factor_db": 12.0,
        "clipping_sample_ratio": 0.0,
        "silence_ratio": 0.02,
        "stereo_width": 0.18,
        "phase_correlation": 0.8,
        "dynamic_range_proxy_db": 8.0,
        "high_frequency_cutoff_hz": 20_000.0,
        "spectral_centroid_hz": 2_500.0,
        "unavailable": {},
        "analysis_error": None,
    }
    analysis.update(technical or {})

    return {
        "schema_version": FACTORY_SCHEMA_VERSION,
        "track_id": track_id,
        "source": {
            "source_path": f"/library/{track_id}.wav",
            "source_filename": f"{track_id}.wav",
            "source_extension": ".wav",
            "source_size_bytes": 1_000_000,
            "source_mtime": 1_700_000_000.0,
            "sha256": track_id.ljust(64, "0"),
        },
        "audio": {
            "status": "VALID",
            "decode_error": None,
            "duration_seconds": duration,
            "sample_rate": sample_rate,
            "channels": channels,
            "bit_depth": 16,
            "codec": "pcm_s16le",
            "container": "wav",
        },
        "analysis": analysis,
        "music": {
            "bpm": bpm,
            "bpm_confidence": bpm_confidence if bpm is not None else None,
            "key": key,
            "key_confidence": key_confidence if key is not None else None,
            "mode": mode,
            "estimated_downbeat_seconds": None,
            "estimated_structure": None,
            "structure_status": "UNAVAILABLE",
            "unavailable": {},
        },
        "vocals": {
            "vocal_class": vocal_class,
            "vocal_confidence": 1.0 if vocal_class != "UNCERTAIN" else None,
            "vocal_source": "USER" if vocal_class != "UNCERTAIN" else "NONE",
            "vocal_gender": vocal_gender,
            "vocal_gender_source": "USER" if vocal_gender else "NONE",
            "centre_dominance_db": None,
            "reason": "",
        },
        "text": {
            "lyrics": lyrics,
            "lyrics_source": "USER" if lyrics else "NONE",
            "lyrics_confidence": 1.0 if lyrics else None,
            "transcript": None,
            "transcript_source": "NONE",
            "transcript_confidence": None,
            "notes": [],
        },
        "quality": {
            "quality_flags": quality_flags or [],
            "quality_score": quality_score,
            "quality_tier": quality_tier,
            "reasons": [],
        },
        "provenance": {
            "source_type": source_type,
            "source_reference": source_reference,
            "rights_status": rights_status,
            "license": None,
            "commercial_training_allowed": permission,
            "provenance_notes": "",
            "field_sources": {},
            "hard_blocks": hard_blocks or [],
            "training_permitted": (
                permission == "TRUE"
                and rights_status in ("VERIFIED", "USER_OWNED", "LICENSED", "PUBLIC_DOMAIN")
                and not (hard_blocks or [])
            ),
        },
        "dedup": {
            "canonical_track_id": track_id,
            "duplicate_of": None,
            "duplicate_group_id": duplicate_group,
            "duplicate_type": "EXACT_FILE" if duplicate_group else "NONE",
            "similarity_score": 1.0 if duplicate_group else None,
            "dedup_decision": dedup_decision,
            "all_source_paths": [f"/library/{track_id}.wav"],
            "fingerprint": "ab" * 100,
        },
        "eligibility": {
            "analysis_eligible": True,
            "training_eligible": training_eligible,
            "validation_eligible": training_eligible,
            "test_eligible": training_eligible,
            "eligibility_reasons": eligibility_reasons or [],
        },
        "metadata": {
            "language": {
                "language": language or "unknown",
                "language_confidence": 1.0 if language else None,
                "language_source": "USER" if language else "NONE",
                "reason": "",
            },
            "embedded_tags": embedded or {},
            "sidecar": sidecar or None,
            "sidecar_error": None,
            "worker_error": None,
        },
        "split": split,
    }


def write_manifest(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records),
        encoding="utf-8",
    )
    return path


# ── deliberately biased corpora ──────────────────────────────────────


def dominated_by_one_artist(total: int = 20, dominant: int = 12) -> list[dict[str, Any]]:
    """Most of the corpus from a single artist."""
    records = []
    for index in range(dominant):
        records.append(
            record(
                f"trk_dom{index:03d}",
                artist="Dominant Artist",
                album="Dominant Record",
                language="ko",
                vocal_class="VOCAL",
                bpm=100.0,
            )
        )
    for index in range(total - dominant):
        records.append(
            record(
                f"trk_oth{index:03d}",
                artist=f"Other {index}",
                album=f"Album {index}",
                language="en",
                vocal_class="VOCAL",
                bpm=130.0,
            )
        )
    return records


def one_big_duplicate_family(family_size: int = 20, others: int = 10) -> list[dict[str, Any]]:
    records = [
        record(
            f"trk_fam{index:03d}",
            duplicate_group="dup_family",
            artist=f"Artist {index}",
            language="ko",
        )
        for index in range(family_size)
    ]
    records.extend(
        record(f"trk_uni{index:03d}", artist=f"Unique {index}", language="en")
        for index in range(others)
    )
    return records


def mixed_rights(total: int = 12) -> list[dict[str, Any]]:
    """Every rights state that must be treated differently."""
    return [
        record("trk_ok1", permission="TRUE", rights_status="USER_OWNED"),
        record("trk_ok2", permission="TRUE", rights_status="LICENSED"),
        record(
            "trk_unknown",
            permission="UNKNOWN",
            rights_status="UNKNOWN",
            training_eligible=False,
            eligibility_reasons=["RIGHTS_UNKNOWN"],
        ),
        record(
            "trk_denied",
            permission="FALSE",
            rights_status="RESTRICTED",
            training_eligible=False,
            eligibility_reasons=["RIGHTS_DENIED"],
        ),
        record(
            "trk_restricted",
            permission="TRUE",
            rights_status="RESTRICTED",
            training_eligible=False,
            eligibility_reasons=["RIGHTS_DENIED"],
        ),
        record(
            "trk_selfmodel",
            permission="TRUE",
            rights_status="USER_OWNED",
            source_type="SELF_MODEL_OUTPUT",
            hard_blocks=["SELF_MODEL_OUTPUT"],
            training_eligible=False,
            eligibility_reasons=["RIGHTS_HARD_BLOCK"],
        ),
    ][:total]


def sparse_genre(total: int = 20, labelled: int = 2) -> list[dict[str, Any]]:
    """Almost no genre metadata: the missingness trap."""
    records = [
        record(f"trk_g{index:03d}", genre="pop", artist=f"A{index}") for index in range(labelled)
    ]
    records.extend(
        record(f"trk_u{index:03d}", artist=f"B{index}") for index in range(total - labelled)
    )
    return records


@pytest.fixture
def biased_manifest(tmp_path: Path) -> Path:
    """The Step 47 corpus: 20 tracks, 12 from one source, mixed quality."""
    records: list[dict[str, Any]] = []
    for index in range(12):
        records.append(
            record(
                f"trk_dom{index:03d}",
                artist="Dominant Artist",
                album="Dominant Record",
                source_reference="dominant_folder",
                language="ko",
                vocal_class="VOCAL",
                bpm=100.0,
                quality_tier="B" if index % 3 else "A",
                duration=180.0,
            )
        )
    for index in range(4):
        records.append(
            record(
                f"trk_min{index:03d}",
                artist=f"Minority {index}",
                album=f"Minority Record {index}",
                source_reference="minority_folder",
                language="en",
                vocal_class="INSTRUMENTAL",
                bpm=145.0,
                quality_tier="B",
                duration=210.0,
            )
        )
    records.append(
        record(
            "trk_unknown_rights",
            artist="Unknown Rights",
            language="ko",
            permission="UNKNOWN",
            rights_status="UNKNOWN",
            training_eligible=False,
            eligibility_reasons=["RIGHTS_UNKNOWN"],
        )
    )
    records.append(
        record(
            "trk_denied_rights",
            artist="Denied",
            language="ko",
            permission="FALSE",
            rights_status="RESTRICTED",
            training_eligible=False,
            eligibility_reasons=["RIGHTS_DENIED"],
        )
    )
    records.append(record("trk_eval_only", artist="Benchmark", language="ko", quality_tier="A"))
    records.append(
        record(
            "trk_review",
            artist="Review Me",
            language="ko",
            dedup_decision="REVIEW_REQUIRED",
            duplicate_group="dup_review",
        )
    )
    return write_manifest(tmp_path / "build" / "dataset_manifest.jsonl", records)
