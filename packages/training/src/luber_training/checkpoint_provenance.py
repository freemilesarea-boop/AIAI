"""What produced a checkpoint, written beside the checkpoint.

A trainer writes an adapter and an optimizer state. It knows nothing
about experiments, splits, rights or code versions, so a checkpoint
found on disk later is a directory of tensors whose origin is a guess.
Phase 35B's pilot exposed exactly that: its checkpoints were sound —
they reopened, they resumed — and the integrity check still reported
them as incomplete, because nothing beside them said what they were.

This module is the answer. One record, one schema, written next to
every checkpoint a LUBER experiment produces, and verified rather than
trusted: :func:`verify_checkpoint_provenance` refuses a record that is
missing a required field, because a provenance file with holes in it is
the failure mode that matters. A checkpoint whose dataset digest is
blank cannot be tied to the data that made it, and that is the whole
reason the file exists.

Nothing here asserts anything about the *quality* of a checkpoint. It
records identity: which experiment, which data, which splits, which
configuration, which code.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CHECKPOINT_PROVENANCE_SCHEMA_VERSION = "luber-checkpoint-provenance/1"

#: The file this module writes. Distinct from the canary's own record so
#: a canary checkpoint keeps its existing name and still verifies.
CHECKPOINT_PROVENANCE_NAME = "luber_checkpoint_provenance.json"

#: Older, canary-specific record. Read for compatibility, never written.
LEGACY_PROVENANCE_NAMES: tuple[str, ...] = ("luber_canary_provenance.json",)

#: Fields that must carry a value. A record missing any of them cannot
#: tie its checkpoint to an experiment, to data, or to a code version,
#: which is the only thing this file is for.
REQUIRED_FIELDS: tuple[str, ...] = (
    "experiment_id",
    "run_id",
    "checkpoint_path",
    "created_at",
    "base_model_id",
    "dataset_id",
    "dataset_lock_sha256",
    "curation_id",
    "curation_lock_sha256",
    "train_split_digest",
    "validation_split_digest",
    "evaluation_split_digest",
    "config_digest",
    "precision",
    "device",
    "optimizer",
    "code_commit",
)

#: Numeric fields that must be present and non-negative.
REQUIRED_NUMERIC_FIELDS: tuple[str, ...] = ("epoch", "step", "lora_rank", "seed")


class ProvenanceError(RuntimeError):
    """Raised when a provenance record cannot be written or read."""


@dataclass
class CheckpointProvenance:
    """Everything needed to say what a checkpoint is.

    ``base_model_digest`` is optional and honestly so: it is a digest
    over the base weight *files*, and a worker that streamed its weights
    from elsewhere may not be able to compute one. Absent means nobody
    measured it, never "unchanged".
    """

    experiment_id: str
    run_id: str
    checkpoint_path: str
    epoch: int
    step: int
    base_model_id: str
    dataset_id: str
    dataset_lock_sha256: str
    curation_id: str
    curation_lock_sha256: str
    train_split_digest: str
    validation_split_digest: str
    evaluation_split_digest: str
    config_digest: str
    lora_rank: int
    precision: str
    device: str
    optimizer: str
    learning_rate: float
    seed: int
    code_commit: str

    base_model_upstream_commit: str = ""
    base_model_digest: str | None = None
    ace_step_commit: str = ""
    plan_id: str = ""
    plan_digest: str = ""
    dataset_kind: str = ""
    segment: str = ""
    #: What this checkpoint may be used for. Never widened by code.
    artifact_class: tuple[str, ...] = ("EXPERIMENTAL", "NON_PRODUCTION", "NEVER_AUTO_PROMOTE")
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    schema_version: str = CHECKPOINT_PROVENANCE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifact_class"] = list(self.artifact_class)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CheckpointProvenance:
        known = set(cls.__dataclass_fields__)
        data = {key: value for key, value in payload.items() if key in known}
        if "artifact_class" in data and data["artifact_class"] is not None:
            data["artifact_class"] = tuple(data["artifact_class"])
        return cls(**data)


@dataclass(frozen=True)
class ProvenanceVerdict:
    """Whether a checkpoint's provenance can be relied on."""

    present: bool
    complete: bool
    path: str | None = None
    schema_version: str = ""
    missing_fields: tuple[str, ...] = ()
    mismatches: tuple[str, ...] = ()
    legacy: bool = False
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.present and self.complete and not self.mismatches

    def to_dict(self) -> dict[str, Any]:
        return {
            "present": self.present,
            "complete": self.complete,
            "ok": self.ok,
            "path": self.path,
            "schema_version": self.schema_version,
            "missing_fields": list(self.missing_fields),
            "mismatches": list(self.mismatches),
            "legacy": self.legacy,
            "detail": self.detail,
        }


def provenance_path(checkpoint_dir: Path) -> Path:
    """Where this module's record lives for *checkpoint_dir*."""
    return Path(checkpoint_dir) / CHECKPOINT_PROVENANCE_NAME


def _existing_record(checkpoint_dir: Path) -> Path | None:
    current = provenance_path(checkpoint_dir)
    if current.is_file():
        return current
    for name in LEGACY_PROVENANCE_NAMES:
        legacy = Path(checkpoint_dir) / name
        if legacy.is_file():
            return legacy
    return None


def write_checkpoint_provenance(checkpoint_dir: Path, provenance: CheckpointProvenance) -> Path:
    """Write the record beside the checkpoint, refusing an empty one.

    Validated before it is written rather than after. A provenance file
    that exists and says nothing is worse than none at all: the first
    reports a problem, the second looks like a pass.
    """
    verdict = _completeness(provenance.to_dict())
    if verdict.missing_fields:
        raise ProvenanceError(
            "refusing to write a provenance record missing "
            f"{', '.join(verdict.missing_fields)}; a record with holes in it reads as "
            "evidence and is not"
        )
    target = provenance_path(checkpoint_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(provenance.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def read_checkpoint_provenance(checkpoint_dir: Path) -> dict[str, Any] | None:
    """The record beside a checkpoint, or ``None`` if there is none."""
    path = _existing_record(Path(checkpoint_dir))
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _completeness(payload: dict[str, Any]) -> ProvenanceVerdict:
    missing = [name for name in REQUIRED_FIELDS if not str(payload.get(name) or "").strip()]
    for name in REQUIRED_NUMERIC_FIELDS:
        value = payload.get(name)
        if value is None or not isinstance(value, int) or isinstance(value, bool) or value < 0:
            missing.append(name)
    return ProvenanceVerdict(
        present=True,
        complete=not missing,
        missing_fields=tuple(missing),
    )


def verify_checkpoint_provenance(
    checkpoint_dir: Path,
    *,
    expected: dict[str, Any] | None = None,
) -> ProvenanceVerdict:
    """Whether this checkpoint's provenance is present, whole and right.

    ``expected`` names field values the caller already knows — a plan
    digest, a split digest — and each one that disagrees is reported as
    a mismatch rather than an absence, because a checkpoint from the
    wrong dataset is a different failure from a checkpoint from no
    recorded dataset.

    A legacy canary record is reported as present but never complete:
    it predates these fields and cannot answer them, and saying so is
    more useful than pretending either way.
    """
    directory = Path(checkpoint_dir)
    path = _existing_record(directory)
    if path is None:
        return ProvenanceVerdict(
            present=False,
            complete=False,
            detail=f"no provenance record beside {directory.name}",
        )

    payload = read_checkpoint_provenance(directory)
    if payload is None:
        return ProvenanceVerdict(
            present=True,
            complete=False,
            path=str(path),
            detail="the provenance record is unreadable",
        )

    legacy = path.name != CHECKPOINT_PROVENANCE_NAME
    if legacy:
        return ProvenanceVerdict(
            present=True,
            complete=False,
            path=str(path),
            schema_version=str(payload.get("schema_version", "")),
            legacy=True,
            missing_fields=tuple(REQUIRED_FIELDS),
            detail=(
                f"{path.name} predates the checkpoint provenance schema and cannot answer "
                "the fields an experiment needs"
            ),
        )

    verdict = _completeness(payload)
    mismatches: list[str] = []
    for name, wanted in (expected or {}).items():
        found = payload.get(name)
        if wanted is not None and found != wanted:
            mismatches.append(f"{name}: recorded {found!r}, expected {wanted!r}")

    detail = "provenance is present and complete"
    if verdict.missing_fields:
        detail = f"provenance is missing {', '.join(verdict.missing_fields)}"
    elif mismatches:
        detail = "provenance disagrees with what was expected"

    return ProvenanceVerdict(
        present=True,
        complete=verdict.complete,
        path=str(path),
        schema_version=str(payload.get("schema_version", "")),
        missing_fields=verdict.missing_fields,
        mismatches=tuple(mismatches),
        detail=detail,
    )


__all__ = [
    "CHECKPOINT_PROVENANCE_NAME",
    "CHECKPOINT_PROVENANCE_SCHEMA_VERSION",
    "LEGACY_PROVENANCE_NAMES",
    "REQUIRED_FIELDS",
    "REQUIRED_NUMERIC_FIELDS",
    "CheckpointProvenance",
    "ProvenanceError",
    "ProvenanceVerdict",
    "provenance_path",
    "read_checkpoint_provenance",
    "verify_checkpoint_provenance",
    "write_checkpoint_provenance",
]
