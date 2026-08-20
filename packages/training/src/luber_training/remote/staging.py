"""Assembling what goes to the worker — and refusing to, when it must not.

This is the last point at which forbidden data can be stopped. After
staging, bytes leave the operator's machine for a rented box owned by
somebody else, and there is no recalling them. So the gates run *here*,
again, immediately before the manifest is built:

**Rights are re-checked, not inherited.** Phase 25's `rights_gate`
already ran when the run was validated. It runs again because time
passed: a curated manifest can be regenerated, a lock can be replaced,
and a run validated on Monday can be dispatched on Friday. Re-reading
the file that is about to be transferred is the only check that
describes what will actually be sent.

**Leakage is re-checked the same way.** P20 material reaching a training
host would silently destroy the benchmark's meaning, and the damage
would not be visible in any metric — the model would simply score well
on songs it had memorised.

Both are hard failures that abort before the first file is opened for
transfer. There is no override flag in this module, and adding one would
defeat every gate upstream of it.

The staging directory is derived entirely from immutable inputs, so the
same run staged twice produces the same tree and the same digest. It
contains only what the run needs: the approved, selected, training-split
audio, and nothing else from the library.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from luber_training.gates import GateReport, leakage_gate, rights_gate, selected_records
from luber_training.plan import TrainingPlan
from luber_training.remote.manifest import (
    ArtifactEntry,
    ArtifactRole,
    RemoteArtifactManifest,
    sha256_file,
)
from luber_training.remote.paths import validate_relative
from luber_training.remote.protocol import now
from luber_training.remote.transport import verified_copy

STAGING_SCHEMA_VERSION = "luber-remote-staging/1"

#: Where each role lands under the run root on the worker. Fixed, so the
#: worker can find things without being told twice.
PLAN_PATH = "plan.json"
ENVIRONMENT_PATH = "environment_lock.json"
TRAINER_DATASET_PATH = "trainer/dataset.json"
DATASET_MANIFEST_PATH = "metadata/dataset_manifest.jsonl"
CURATED_MANIFEST_PATH = "metadata/curated_manifest.jsonl"
CURATION_LOCK_PATH = "metadata/curation_lock.json"
DATASET_LOCK_PATH = "metadata/dataset_lock.json"
SAMPLING_WEIGHTS_PATH = "metadata/sampling_weights.json"
#: Audio goes here. The trainer's dataset.json indexes by filename, so
#: the layout is flat and the names come from the manifest.
DATASET_AUDIO_PREFIX = "dataset"


class StagingError(RuntimeError):
    """Raised when a run may not be staged for transfer."""


class RightsViolation(StagingError):
    """Raised when unauthorised material would be transferred."""


class LeakageViolation(StagingError):
    """Raised when evaluation material would be transferred."""


@dataclass
class StagingInputs:
    """The immutable files a staging build reads.

    Paths rather than loaded state, so a build always reflects what is
    on disk at the moment of dispatch rather than what was true when the
    run was created.
    """

    dataset_build_dir: Path
    curation_build_dir: Path
    audio_root: Path
    evaluation_only_ids: frozenset[str] = frozenset()
    evaluation_only_digests: frozenset[str] = frozenset()

    @property
    def curated_manifest(self) -> Path:
        return Path(self.curation_build_dir) / "curated_manifest.jsonl"

    @property
    def curation_lock(self) -> Path:
        return Path(self.curation_build_dir) / "curation_lock.json"

    @property
    def dataset_manifest(self) -> Path:
        return Path(self.dataset_build_dir) / "dataset_manifest.jsonl"

    @property
    def dataset_lock(self) -> Path:
        return Path(self.dataset_build_dir) / "dataset_lock.json"

    @property
    def sampling_weights(self) -> Path:
        return Path(self.curation_build_dir) / "sampling_weights.json"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        raise StagingError(f"{path} does not exist")
    records: list[dict[str, Any]] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StagingError(f"{path}:{number} is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise StagingError(f"{path}:{number} is not a record")
        records.append(payload)
    return records


@dataclass
class StagingResult:
    """A staged run: the tree, the manifest, and what was checked."""

    run_id: str
    staging_dir: Path
    manifest: RemoteArtifactManifest
    gate_report: GateReport
    selected_tracks: int
    staged_bytes: int
    staging_manifest_sha256: str
    schema_version: str = STAGING_SCHEMA_VERSION
    built_at: str = field(default_factory=now)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "staging_dir": str(self.staging_dir),
            "staging_manifest_sha256": self.staging_manifest_sha256,
            "manifest_sha256": self.manifest.digest(),
            "training_plan_sha256": self.manifest.training_plan_sha256,
            "selected_tracks": self.selected_tracks,
            "entry_count": len(self.manifest.entries),
            "staged_bytes": self.staged_bytes,
            "transfer_bytes": self.manifest.transfer_bytes(),
            "gates": self.gate_report.to_dict(),
            "warnings": self.warnings,
            "built_at": self.built_at,
        }


def revalidate_before_transfer(
    curated_records: list[dict[str, Any]],
    *,
    evaluation_only_ids: frozenset[str] = frozenset(),
    evaluation_only_digests: frozenset[str] = frozenset(),
) -> GateReport:
    """Rights and leakage, checked against what is about to be sent.

    Returns a report; raising is the caller's job so that the failure
    can be recorded before it propagates. Both gates are hard — there is
    no argument to this function that makes either advisory.
    """
    report = GateReport()
    report.results.append(rights_gate(curated_records))
    report.results.append(
        leakage_gate(
            curated_records,
            evaluation_only_ids=evaluation_only_ids,
            evaluation_only_digests=evaluation_only_digests,
        )
    )
    return report


def _audio_source(record: dict[str, Any], audio_root: Path) -> tuple[Path, str]:
    """Where a track's audio is, and what it will be called remotely.

    The remote name comes from the recorded source filename because the
    trainer's dataset.json indexes by filename — the two have to agree
    or the loader finds nothing. The path is validated like any other
    untrusted string.
    """
    source = record.get("source") or {}
    filename = source.get("source_filename")
    if not isinstance(filename, str) or not filename.strip():
        raise StagingError(
            f"track {record.get('track_id')!r} has no source filename; it cannot be "
            "staged, because the trainer indexes samples by filename"
        )

    relative = source.get("relative_path")
    candidate = (
        Path(audio_root) / str(relative)
        if isinstance(relative, str) and relative.strip()
        else Path(audio_root) / filename
    )
    target = validate_relative(f"{DATASET_AUDIO_PREFIX}/{Path(filename).name}")
    return candidate, target


def build_staging(
    *,
    plan: TrainingPlan,
    inputs: StagingInputs,
    staging_root: Path,
    environment_lock: dict[str, Any] | None = None,
    trainer_dataset: dict[str, Any] | None = None,
    copy_audio: bool = True,
) -> StagingResult:
    """Assemble a run's transfer set, after re-checking it may be sent.

    Order is the point. Gates first, on the records that are about to
    become the manifest; the staging directory is not created until they
    pass, so a blocked dispatch leaves nothing behind that a later
    transfer could pick up by accident.
    """
    curated = _read_jsonl(inputs.curated_manifest)
    selected = selected_records(curated)
    if not selected:
        raise StagingError("curation selected no tracks; there is nothing to train on")

    # ── the gates, before anything is copied ──
    report = revalidate_before_transfer(
        curated,
        evaluation_only_ids=inputs.evaluation_only_ids,
        evaluation_only_digests=inputs.evaluation_only_digests,
    )
    rights = next(result for result in report.results if result.name == "rights")
    if not rights.passed:
        raise RightsViolation(
            f"staging refused: {rights.detail}. {rights.offending_count} track(s) are not "
            "authorised for training and no remote host will receive them"
        )
    leakage = next(result for result in report.results if result.name == "evaluation_leakage")
    if not leakage.passed:
        raise LeakageViolation(
            f"staging refused: {leakage.detail}. Benchmark or held-out material in a "
            "training transfer would destroy the benchmark's meaning permanently"
        )

    staging_dir = Path(staging_root) / validate_relative(plan.run_id)
    staging_dir.mkdir(parents=True, exist_ok=True)

    manifest = RemoteArtifactManifest(run_id=plan.run_id, training_plan_sha256=plan.digest())
    warnings: list[str] = []
    staged_bytes = 0

    def stage_json(payload: dict[str, Any], target: str, role: str) -> None:
        nonlocal staged_bytes
        local = staging_dir / target
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        entry = ArtifactEntry.from_file(local, role=role, target_path=target)
        manifest.add(entry)
        staged_bytes += entry.size_bytes

    def stage_copy(source: Path, target: str, role: str, *, required: bool = True) -> None:
        nonlocal staged_bytes
        source = Path(source)
        if not source.is_file():
            if required:
                raise StagingError(f"{source} is required for this run and does not exist")
            warnings.append(f"{target}: optional artifact {source} is absent")
            return
        local = staging_dir / target
        digest = verified_copy(source, local)
        entry = ArtifactEntry(
            artifact_id=digest[:16],
            role=role,
            target_path=target,
            sha256=digest,
            size_bytes=local.stat().st_size,
            required=required,
            source_reference=str(source),
        )
        manifest.add(entry)
        staged_bytes += entry.size_bytes

    stage_json(plan.to_dict(), PLAN_PATH, ArtifactRole.PLAN.value)
    if environment_lock is not None:
        stage_json(environment_lock, ENVIRONMENT_PATH, ArtifactRole.ENVIRONMENT_LOCK.value)
    if trainer_dataset is not None:
        stage_json(trainer_dataset, TRAINER_DATASET_PATH, ArtifactRole.TRAINER_DATASET.value)

    stage_copy(inputs.dataset_lock, DATASET_LOCK_PATH, ArtifactRole.METADATA.value)
    stage_copy(inputs.curation_lock, CURATION_LOCK_PATH, ArtifactRole.METADATA.value)
    stage_copy(inputs.curated_manifest, CURATED_MANIFEST_PATH, ArtifactRole.CURATED_MANIFEST.value)
    stage_copy(
        inputs.dataset_manifest,
        DATASET_MANIFEST_PATH,
        ArtifactRole.DATASET_MANIFEST.value,
        required=False,
    )
    stage_copy(
        inputs.sampling_weights,
        SAMPLING_WEIGHTS_PATH,
        ArtifactRole.SAMPLING_WEIGHTS.value,
        required=False,
    )

    # ── audio: only the selected, authorised, training-split tracks ──
    for record in selected:
        source, target = _audio_source(record, inputs.audio_root)
        track_id = str(record.get("track_id", ""))
        if not source.is_file():
            raise StagingError(
                f"track {track_id}: audio is missing at {source}. A run cannot be staged "
                "from a manifest whose files are not where it says they are"
            )

        if copy_audio:
            local = staging_dir / target
            digest = verified_copy(source, local)
            size = local.stat().st_size
        else:
            # Manifest-only mode, for planning a transfer whose bytes
            # already sit where they are. Nothing is duplicated.
            digest, size = sha256_file(source)

        declared = str((record.get("source") or {}).get("sha256", ""))
        if declared and declared != digest:
            raise StagingError(
                f"track {track_id}: the file at {source} hashes to {digest[:12]} but the "
                f"curated manifest records {declared[:12]}. The audio has changed since "
                "curation, so this is not the dataset that was approved"
            )

        manifest.add(
            ArtifactEntry(
                artifact_id=digest[:16],
                role=ArtifactRole.DATASET_AUDIO.value,
                target_path=target,
                sha256=digest,
                size_bytes=size,
                required=True,
                source_reference=str(source if not copy_audio else staging_dir / target),
                track_id=track_id or None,
            )
        )
        staged_bytes += size if copy_audio else 0

    manifest.write(staging_dir / "artifact_manifest.json")

    result = StagingResult(
        run_id=plan.run_id,
        staging_dir=staging_dir,
        manifest=manifest,
        gate_report=report,
        selected_tracks=len(selected),
        staged_bytes=staged_bytes,
        staging_manifest_sha256=staging_digest(manifest, plan),
        warnings=warnings,
    )
    (staging_dir / "staging.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def staging_digest(manifest: RemoteArtifactManifest, plan: TrainingPlan) -> str:
    """One identity for "this run, staged from these inputs".

    Over the manifest digest and the plan digest rather than over the
    directory: two staging builds of the same run on two machines write
    files at different absolute paths and at different times, and both
    must hash the same or the identity is useless.
    """
    return hashlib.sha256(
        json.dumps(
            {
                "schema_version": STAGING_SCHEMA_VERSION,
                "run_id": manifest.run_id,
                "manifest_sha256": manifest.digest(),
                "training_plan_sha256": plan.digest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def verify_staging(staging_dir: Path, *, plan: TrainingPlan | None = None) -> list[str]:
    """Recheck a staged tree against its own manifest.

    Recomputes every digest from the files on disk. Run before dispatch,
    so that a staging directory edited or damaged after it was built is
    caught on the operator's machine rather than after an upload.
    """
    staging_dir = Path(staging_dir)
    manifest_path = staging_dir / "artifact_manifest.json"
    if not manifest_path.is_file():
        return [f"{manifest_path} is missing; this directory was never staged"]

    problems: list[str] = []
    try:
        manifest = RemoteArtifactManifest.read(manifest_path)
    except Exception as exc:
        return [f"the manifest cannot be read: {exc}"]

    if plan is not None and manifest.training_plan_sha256 != plan.digest():
        problems.append(
            f"the staged manifest cites plan {manifest.training_plan_sha256[:12]} but the "
            f"run's plan is {plan.digest()[:12]}"
        )

    for entry in manifest.entries:
        local = staging_dir / entry.target_path
        if not local.is_file():
            if entry.required:
                problems.append(f"{entry.target_path}: staged file is missing")
            continue
        digest, size = sha256_file(local)
        if size != entry.size_bytes:
            problems.append(
                f"{entry.target_path}: size is {size}, manifest says {entry.size_bytes}"
            )
        elif digest != entry.sha256:
            problems.append(
                f"{entry.target_path}: digest is {digest[:12]}, manifest says {entry.sha256[:12]}"
            )
    return problems


__all__ = [
    "CURATED_MANIFEST_PATH",
    "DATASET_AUDIO_PREFIX",
    "ENVIRONMENT_PATH",
    "PLAN_PATH",
    "STAGING_SCHEMA_VERSION",
    "TRAINER_DATASET_PATH",
    "LeakageViolation",
    "RightsViolation",
    "StagingError",
    "StagingInputs",
    "StagingResult",
    "build_staging",
    "revalidate_before_transfer",
    "staging_digest",
    "verify_staging",
]
