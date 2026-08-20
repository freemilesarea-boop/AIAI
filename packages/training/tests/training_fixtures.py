"""Synthetic locks, manifests and entities for orchestration tests.

Everything is built rather than borrowed. Curation and dataset locks are
produced by the *real* Phase 23/24 code so the gates are tested against
the contracts that actually exist — a hand-written lock would let a gate
pass against a shape the factory never emits.

No audio, no model weights, no network.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from luber_training.config import TrainingConfig, preset
from luber_training.entities import (
    ModelBaseline,
    TrainingDatasetRef,
    TrainingStrategySupport,
    TrainingWorker,
    WorkerCapabilities,
    WorkerClass,
)
from luber_training.ids import EntityKind, new_id
from luber_training.orchestrator import Orchestrator
from luber_training.registry import Registry

FACTORY_SCHEMA_VERSION = "luber-dataset-factory/1"
CURATION_SCHEMA_VERSION = "luber-dataset-curation/1"
ACE_STEP_COMMIT = "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0"


def manifest_record(
    track_id: str,
    *,
    permission: str = "TRUE",
    rights_status: str = "USER_OWNED",
    source_type: str = "USER_ORIGINAL",
    hard_blocks: list[str] | None = None,
    training_eligible: bool = True,
    split: str = "TRAIN",
    duration: float = 200.0,
    language: str = "ko",
    vocal_class: str = "VOCAL",
    lyrics: str | None = "[Verse]\nline",
    genre: str = "ballad",
    sha256: str | None = None,
) -> dict[str, Any]:
    """A Phase 23 manifest record, shaped as the factory writes them."""
    digest = sha256 or hashlib.sha256(track_id.encode()).hexdigest()
    return {
        "schema_version": FACTORY_SCHEMA_VERSION,
        "track_id": track_id,
        "source": {
            "source_path": f"/library/{track_id}.wav",
            "source_filename": f"{track_id}.wav",
            "source_extension": ".wav",
            "source_size_bytes": 1_000_000,
            "source_mtime": 1_700_000_000.0,
            "sha256": digest,
        },
        "audio": {
            "status": "VALID",
            "decode_error": None,
            "duration_seconds": duration,
            "sample_rate": 44_100,
            "channels": 2,
            "bit_depth": 16,
            "codec": "pcm_s16le",
            "container": "wav",
        },
        "analysis": {"duration_seconds": duration, "sample_rate": 44_100, "channels": 2},
        "music": {
            "bpm": 120.0,
            "bpm_confidence": 0.9,
            "key": "C",
            "key_confidence": 0.4,
            "mode": "major",
            "estimated_structure": None,
            "structure_status": "UNAVAILABLE",
        },
        "vocals": {"vocal_class": vocal_class, "vocal_source": "USER"},
        "text": {"lyrics": lyrics, "lyrics_source": "USER" if lyrics else "NONE"},
        "quality": {"quality_flags": [], "quality_score": 1.0, "quality_tier": "A"},
        "provenance": {
            "source_type": source_type,
            "source_reference": "library",
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
            "duplicate_group_id": None,
            "duplicate_type": "NONE",
            "dedup_decision": "KEEP",
            "all_source_paths": [f"/library/{track_id}.wav"],
        },
        "eligibility": {
            "analysis_eligible": True,
            "training_eligible": training_eligible,
            "validation_eligible": training_eligible,
            "test_eligible": training_eligible,
            "eligibility_reasons": [],
        },
        "metadata": {
            "language": {
                "language": language,
                "language_confidence": 1.0,
                "language_source": "USER",
            },
            "embedded_tags": {},
            "sidecar": {"artist": "Operator", "album": "Record", "genre": genre},
            "sidecar_error": None,
        },
        "split": split,
    }


def curated_record(record: dict[str, Any], *, action: str = "KEEP") -> dict[str, Any]:
    return {
        "curation_schema_version": CURATION_SCHEMA_VERSION,
        "curation_engine_version": "luber-dataset-curation/1.0.0",
        "profile_version": "NEUTRAL",
        **record,
        "curation_action": action,
        "curation_reasons": [],
        "curation_score": 0.8,
        "score_components": {},
        "sampling_weight": 1.0 if action in ("KEEP", "KEEP_PRIORITY") else None,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def build_locked_dataset(
    root: Path, records: list[dict[str, Any]], *, actions: dict[str, str] | None = None
) -> tuple[Path, Path]:
    """Write a dataset build and a curation build, both properly locked.

    Uses the real Phase 23/24 lock writers so the digests are the ones
    the gates recompute, rather than values invented here.
    """
    from luber_dataset.factory import manifest as manifest_io

    dataset_dir = root / "dataset-build"
    curation_dir = root / "curated-build"
    _write_jsonl(dataset_dir / "dataset_manifest.jsonl", records)

    loaded = manifest_io.read_manifest(dataset_dir / "dataset_manifest.jsonl")
    total = sum(float((r.analysis or {}).get("duration_seconds") or 0.0) for r in loaded)
    lock = {
        "dataset_id": "ds-test-001",
        "created_at": "2026-08-20T00:00:00+00:00",
        "schema_version": FACTORY_SCHEMA_VERSION,
        "factory_version": "luber-dataset-factory/1.0.0",
        "configuration_hash": "0" * 64,
        "manifest_sha256": manifest_io.canonical_manifest_digest(loaded),
        "track_count": len(loaded),
        "total_duration_seconds": round(total, 3),
        "split_counts": {"TRAIN": len(loaded)},
        "source_identity_digest": manifest_io.source_identity_digest(loaded),
    }
    (dataset_dir / "dataset_lock.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    curated = [
        curated_record(record, action=(actions or {}).get(record["track_id"], "KEEP"))
        for record in records
    ]
    _write_jsonl(curation_dir / "curated_manifest.jsonl", curated)

    digest = hashlib.sha256()
    for row in sorted(curated, key=lambda r: str(r.get("track_id"))):
        digest.update(
            json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")

    dataset_lock_digest = hashlib.sha256(
        (dataset_dir / "dataset_lock.json").read_bytes()
    ).hexdigest()
    curation_lock = {
        "curation_id": "cur-test-001",
        "created_at": "2026-08-20T00:00:00+00:00",
        "engine_version": "luber-dataset-curation/1.0.0",
        "schema_version": CURATION_SCHEMA_VERSION,
        "source_manifest_sha256": lock["manifest_sha256"],
        "source_dataset_lock_sha256": dataset_lock_digest,
        "target_profile_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "curated_manifest_sha256": digest.hexdigest(),
        "sampling_weights_sha256": None,
        "selected_track_count": sum(
            1 for row in curated if row["curation_action"] in ("KEEP", "KEEP_PRIORITY")
        ),
        "selected_hours": round(total / 3600.0, 4),
        "distribution_summary_digest": "3" * 64,
    }
    (curation_dir / "curation_lock.json").write_text(
        json.dumps(curation_lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return dataset_dir, curation_dir


@pytest.fixture
def registry(tmp_path: Path) -> Registry:
    return Registry(tmp_path / "registry")


@pytest.fixture
def orchestrator(registry: Registry, tmp_path: Path) -> Orchestrator:
    return Orchestrator(
        registry,
        artifacts_root=tmp_path / "runs",
        repository_root=Path(__file__).resolve().parents[3],
    )


@pytest.fixture
def baseline(orchestrator: Orchestrator) -> ModelBaseline:
    return orchestrator.register_baseline(
        ModelBaseline(
            model_id=new_id(EntityKind.MODEL),
            provider="ACE-Step",
            model_family="acestep",
            model_name="acestep-v15-turbo",
            model_version="1.5",
            upstream_commit=ACE_STEP_COMMIT,
            architecture="DiT + VAE",
            training_strategy_support=[
                TrainingStrategySupport.LORA.value,
                TrainingStrategySupport.LOKR.value,
            ],
        )
    )


@pytest.fixture
def gpu_worker(orchestrator: Orchestrator) -> TrainingWorker:
    """A worker that has reported real CUDA capability."""
    return orchestrator.register_worker(
        TrainingWorker(
            worker_id=new_id(EntityKind.WORKER),
            name="rented-gpu",
            backend_type="dry-run",
            host_identity="gpu-host-1",
            worker_class=WorkerClass.GPU_TRAINING_READY.value,
            capabilities=WorkerCapabilities(
                gpu_vendor="NVIDIA",
                gpu_model="Test GPU",
                gpu_count=1,
                vram_total_mb=24_000,
                cuda_available=True,
                cuda_version="12.1",
                bf16_supported=True,
                free_disk_mb=500_000,
                reported_by="test probe",
                reported_at="2026-08-20T00:00:00+00:00",
            ),
        )
    )


@pytest.fixture
def mac_worker(orchestrator: Orchestrator) -> TrainingWorker:
    """The local Mac: development only, no CUDA ever reported."""
    return orchestrator.register_worker(
        TrainingWorker(
            worker_id=new_id(EntityKind.WORKER),
            name="local-mac",
            backend_type="dry-run",
            host_identity="mac-1",
            worker_class=WorkerClass.DEVELOPMENT_ONLY.value,
            capabilities=WorkerCapabilities(cpu_count=12, reported_by="test probe"),
        )
    )


@pytest.fixture
def dataset_ref() -> TrainingDatasetRef:
    return TrainingDatasetRef(
        dataset_id="ds-test-001",
        dataset_lock_sha256="a" * 64,
        curation_id="cur-test-001",
        curation_lock_sha256="b" * 64,
        curated_manifest_sha256="c" * 64,
        manifest_artifact_ref="curation://cur-test-001/curated_manifest",
        selected_track_count=4,
        selected_hours=0.22,
    )


@pytest.fixture
def smoke_config() -> TrainingConfig:
    return preset("SMOKE")
