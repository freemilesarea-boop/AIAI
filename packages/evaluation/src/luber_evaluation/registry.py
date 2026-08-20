"""Evaluation storage, built on the Phase 25 registry rather than beside it.

Model and checkpoint identity comes from Phase 25 and is never
duplicated here. Two registries claiming to know what a checkpoint is
would eventually disagree, and the one people trusted would be whichever
they happened to read.

So this adds a namespace — evaluations, decisions, reviews — and reuses
the same atomic writes, the same reentrant lock and the same append-only
audit log. Partial qualification files must never appear final: a
half-written verdict that reads as a verdict is worse than a missing
one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from luber_evaluation.schemas import (
    EVALUATION_SCHEMA_VERSION,
    EvaluationRunStatus,
    now,
)
from luber_training.registry import Registry as TrainingRegistry
from luber_training.registry import RegistryError

#: Namespaces this package owns. Models, checkpoints, experiments and
#: runs stay with Phase 25.
EVALUATION_COLLECTIONS: tuple[str, ...] = (
    "evaluations",
    "qualifications",
    "promotion_reviews",
)

# ── audit events ─────────────────────────────────────────────────────
EVALUATION_CREATED = "EVALUATION_CREATED"
EVALUATION_STARTED = "EVALUATION_STARTED"
EVALUATION_COMPLETED = "EVALUATION_COMPLETED"
EVALUATION_FAILED = "EVALUATION_FAILED"
EVALUATION_CANCELLED = "EVALUATION_CANCELLED"
QUALIFICATION_DECIDED = "QUALIFICATION_DECIDED"
HUMAN_REVIEW_REQUESTED = "HUMAN_REVIEW_REQUESTED"
HUMAN_REVIEW_RECORDED = "HUMAN_REVIEW_RECORDED"
PROMOTION_REVIEW_RECORDED = "PROMOTION_REVIEW_RECORDED"


class EvaluationRegistryError(RegistryError):
    """Raised when evaluation storage cannot honour a request."""


@dataclass
class EvaluationArtifacts:
    """Where one evaluation's files live.

    Fixed layout so a verification pass knows where to look and an
    operator can find a report without being told.
    """

    root: Path
    evaluation_id: str

    @property
    def directory(self) -> Path:
        return self.root / self.evaluation_id

    @property
    def evaluation_json(self) -> Path:
        return self.directory / "evaluation.json"

    @property
    def suite_json(self) -> Path:
        return self.directory / "suite.json"

    @property
    def policy_json(self) -> Path:
        return self.directory / "policy.json"

    @property
    def metrics_jsonl(self) -> Path:
        return self.directory / "metrics.jsonl"

    @property
    def comparisons_json(self) -> Path:
        return self.directory / "comparisons.json"

    @property
    def qualification_json(self) -> Path:
        return self.directory / "qualification.json"

    @property
    def report_md(self) -> Path:
        return self.directory / "report.md"

    @property
    def samples_jsonl(self) -> Path:
        return self.directory / "samples.jsonl"

    @property
    def baseline_dir(self) -> Path:
        return self.directory / "baseline"

    @property
    def candidate_dir(self) -> Path:
        return self.directory / "candidate"

    @property
    def human_review_dir(self) -> Path:
        return self.directory / "human_review"

    def ensure(self) -> None:
        for path in (
            self.directory,
            self.baseline_dir,
            self.candidate_dir,
            self.human_review_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


class EvaluationRegistry:
    """Evaluation records, sharing the training registry's durability."""

    def __init__(
        self, training_registry: TrainingRegistry, artifacts_root: Path | None = None
    ) -> None:
        self.training = training_registry
        self.root = training_registry.root
        for name in EVALUATION_COLLECTIONS:
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.artifacts_root = Path(artifacts_root or self.root / "evaluations")
        self.artifacts_root.mkdir(parents=True, exist_ok=True)

    # ── generic access, mirroring the training registry ──────────────
    def _path(self, collection: str, entity_id: str) -> Path:
        if collection not in EVALUATION_COLLECTIONS:
            raise EvaluationRegistryError(f"unknown evaluation collection {collection!r}")
        if "/" in entity_id or ".." in entity_id or not entity_id:
            raise EvaluationRegistryError(f"unsafe entity id {entity_id!r}")
        return self.root / collection / f"{entity_id}.json"

    def write(
        self, collection: str, entity_id: str, payload: dict[str, Any], *, overwrite: bool = False
    ) -> None:
        from luber_training.registry import _atomic_write

        with self.training.lock():
            path = self._path(collection, entity_id)
            if path.exists() and not overwrite:
                raise EvaluationRegistryError(f"{collection[:-1]} {entity_id} already exists")
            _atomic_write(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))

    def read(self, collection: str, entity_id: str) -> dict[str, Any]:
        path = self._path(collection, entity_id)
        if not path.is_file():
            raise EvaluationRegistryError(f"{collection[:-1]} {entity_id} is not registered")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise EvaluationRegistryError(f"{path.name} is not a registry record")
        return payload

    def exists(self, collection: str, entity_id: str) -> bool:
        return self._path(collection, entity_id).is_file()

    def list_ids(self, collection: str) -> list[str]:
        directory = self.root / collection
        if not directory.is_dir():
            return []
        return sorted(path.stem for path in directory.glob("*.json"))

    def list_all(self, collection: str) -> list[dict[str, Any]]:
        return [self.read(collection, entity_id) for entity_id in self.list_ids(collection)]

    def find(self, collection: str, **criteria: Any) -> list[dict[str, Any]]:
        return [
            record
            for record in self.list_all(collection)
            if all(record.get(key) == value for key, value in criteria.items())
        ]

    def artifacts(self, evaluation_id: str) -> EvaluationArtifacts:
        return EvaluationArtifacts(root=self.artifacts_root, evaluation_id=evaluation_id)

    def audit(self, event: str, entity_id: str, **metadata: Any) -> None:
        """Append to the shared training audit log.

        Shared on purpose: an operator asking "what happened to this
        checkpoint" should get training and evaluation events in one
        ordered history rather than having to interleave two files.
        """
        self.training.append_audit(event, entity_id, "evaluation", **metadata)

    def audit_events(self, entity_id: str | None = None) -> list[dict[str, Any]]:
        return self.training.audit_events(entity_id)

    # ── evaluation runs ──────────────────────────────────────────────
    def save_evaluation(self, payload: dict[str, Any], *, overwrite: bool = True) -> None:
        """Record a run, and mirror it into its artifact directory.

        Both copies are written here rather than by each caller,
        because the artifact directory is what a verification pass and
        an archived bundle read. A directory whose ``evaluation.json``
        lagged the registry would describe a run that never happened in
        exactly the situation someone went looking for it.
        """
        from luber_training.registry import _atomic_write

        evaluation_id = str(payload["evaluation_id"])
        self.write("evaluations", evaluation_id, payload, overwrite=overwrite)
        artifacts = self.artifacts(evaluation_id)
        artifacts.ensure()
        _atomic_write(
            artifacts.evaluation_json,
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        )

    def load_evaluation(self, evaluation_id: str) -> dict[str, Any]:
        return self.read("evaluations", evaluation_id)

    def save_qualification(self, payload: dict[str, Any]) -> None:
        """A verdict is written once, atomically, and never edited.

        Re-deciding produces a new evaluation rather than overwriting a
        recorded decision: a qualification that could be revised in
        place would make the audit log describe a history that no
        longer exists.
        """
        evaluation_id = str(payload["evaluation_id"])
        self.write("qualifications", evaluation_id, payload, overwrite=False)
        self.audit(
            QUALIFICATION_DECIDED,
            evaluation_id,
            outcome=payload.get("outcome"),
            policy_id=payload.get("policy_id"),
        )

    def save_promotion_review(self, payload: dict[str, Any]) -> None:
        review_id = str(payload["review_id"])
        self.write("promotion_reviews", review_id, payload, overwrite=False)
        self.audit(
            PROMOTION_REVIEW_RECORDED,
            review_id,
            decision=payload.get("decision"),
            candidate_id=payload.get("candidate_id"),
        )


def transition(current: str, target: str) -> str:
    """Move an evaluation run's status, honouring the state machine."""
    from luber_evaluation.schemas import ALLOWED_EVALUATION_TRANSITIONS

    if target == current:
        return current
    if target not in ALLOWED_EVALUATION_TRANSITIONS.get(current, frozenset()):
        raise EvaluationRegistryError(f"an evaluation cannot move {current} -> {target}")
    return target


def stamp_for(status: str) -> str | None:
    return {
        EvaluationRunStatus.RUNNING.value: "started_at",
        EvaluationRunStatus.COMPLETED.value: "completed_at",
        EvaluationRunStatus.CANCELLED.value: "cancelled_at",
        EvaluationRunStatus.FAILED.value: "failed_at",
    }.get(status)


__all__ = [
    "EVALUATION_CANCELLED",
    "EVALUATION_COLLECTIONS",
    "EVALUATION_COMPLETED",
    "EVALUATION_CREATED",
    "EVALUATION_FAILED",
    "EVALUATION_SCHEMA_VERSION",
    "EVALUATION_STARTED",
    "HUMAN_REVIEW_RECORDED",
    "HUMAN_REVIEW_REQUESTED",
    "PROMOTION_REVIEW_RECORDED",
    "QUALIFICATION_DECIDED",
    "EvaluationArtifacts",
    "EvaluationRegistry",
    "EvaluationRegistryError",
    "now",
    "stamp_for",
    "transition",
]
