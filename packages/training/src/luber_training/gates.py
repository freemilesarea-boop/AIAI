"""The gates a run must clear before it may be queued.

These are the reason this package exists. Everything else — registries,
plans, backends — is bookkeeping around the question "is it legitimate
to train on this?", and that question has to be answered before a GPU
is rented, not after a model exists.

Five gates, applied in order, all of them hard:

1. **Dataset lock** — the manifest is the one the lock froze
2. **Curation lock** — the curation is the one the lock froze, *and* it
   derives from that same dataset
3. **Rights** — every training entry is permitted
4. **Evaluation leakage** — no benchmark or held-out material
5. **Self-generated data** — model output is not fed back by default

There is no `--ignore-rights`, no `--force`, and no override parameter
anywhere in this module. Phase 23 has an `include_rights_unknown`
setting for *analysis*, and Step 10 asks that production training not be
able to reach it accidentally: :func:`rights_gate` therefore re-checks
every record's own provenance rather than trusting the eligibility flag
a permissive curation may have set.

Verification works on **stable identity** — track ids and digests. Never
filenames. A file renamed between the lock and the run is the same
audio; a different file with the same name is not.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from luber_training.entities import FailureCode

#: Rights states that may never be trained on.
FORBIDDEN_PERMISSIONS: frozenset[str] = frozenset({"FALSE", "UNKNOWN"})
FORBIDDEN_RIGHTS_STATUSES: frozenset[str] = frozenset({"RESTRICTED"})
#: Provenance describing machine-made audio.
SELF_GENERATED_SOURCE_TYPES: frozenset[str] = frozenset({"SELF_MODEL_OUTPUT"})
SYNTHETIC_SOURCE_TYPES: frozenset[str] = frozenset({"AI_GENERATED", "SELF_MODEL_OUTPUT"})

#: Curation actions whose tracks belong in a training selection.
SELECTED_ACTIONS: frozenset[str] = frozenset({"KEEP", "KEEP_PRIORITY"})


@dataclass
class GateResult:
    """The outcome of one gate, with everything needed to act on it."""

    name: str
    passed: bool
    detail: str = ""
    failure_code: str | None = None
    #: Track ids that caused the failure, capped for readability. The
    #: full count is always reported.
    offending_ids: list[str] = field(default_factory=list)
    offending_count: int = 0
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "failure_code": self.failure_code,
            "offending_ids": self.offending_ids[:20],
            "offending_count": self.offending_count,
            "evidence": self.evidence,
        }


@dataclass
class GateReport:
    results: list[GateResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    @property
    def first_failure(self) -> GateResult | None:
        for result in self.results:
            if not result.passed:
                return result
        return None

    def failure_code(self) -> str | None:
        failure = self.first_failure
        return failure.failure_code if failure else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failure_code": self.failure_code(),
            "gates": [result.to_dict() for result in self.results],
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} line {number} is not valid JSON: {exc}") from exc
        records.append(record)
    return records


# ── 1. dataset lock ──────────────────────────────────────────────────


def dataset_lock_gate(dataset_lock_path: Path, manifest_path: Path) -> GateResult:
    """The manifest on disk is the one the lock froze.

    Phase 23's lock records a *canonical* manifest digest — over record
    content, ignoring timestamps — so this recomputes it the same way
    rather than hashing the file, which would fail on a harmless
    reformat and pass on nothing useful.
    """
    name = "dataset_lock"
    if not dataset_lock_path.is_file():
        return GateResult(
            name=name,
            passed=False,
            detail=f"no dataset lock at {dataset_lock_path.name}",
            failure_code=FailureCode.DATASET_LOCK_INVALID.value,
        )
    if not manifest_path.is_file():
        return GateResult(
            name=name,
            passed=False,
            detail=f"no dataset manifest at {manifest_path.name}",
            failure_code=FailureCode.DATASET_LOCK_INVALID.value,
        )

    lock = _load_json(dataset_lock_path)
    from luber_dataset.factory import manifest as manifest_io

    records = manifest_io.read_manifest(manifest_path)
    actual_manifest = manifest_io.canonical_manifest_digest(records)
    actual_sources = manifest_io.source_identity_digest(records)

    problems: list[str] = []
    if lock.get("manifest_sha256") != actual_manifest:
        problems.append("manifest content differs from the frozen digest")
    if lock.get("source_identity_digest") != actual_sources:
        problems.append("source audio identity differs from the frozen digest")
    if lock.get("track_count") != len(records):
        problems.append(f"track count {lock.get('track_count')} -> {len(records)}")

    if problems:
        return GateResult(
            name=name,
            passed=False,
            detail="; ".join(problems),
            failure_code=FailureCode.DATASET_LOCK_INVALID.value,
            evidence={"dataset_id": lock.get("dataset_id")},
        )
    return GateResult(
        name=name,
        passed=True,
        detail=f"dataset {lock.get('dataset_id')} matches its lock ({len(records)} tracks)",
        evidence={"dataset_id": lock.get("dataset_id"), "track_count": len(records)},
    )


# ── 2. curation lock ─────────────────────────────────────────────────


def curation_lock_gate(
    curation_lock_path: Path, curated_manifest_path: Path, dataset_lock_path: Path
) -> GateResult:
    """The curation matches its lock, and derives from *this* dataset.

    The linkage check is the one that catches the real mistake: a
    curation computed against last week's manifest, still internally
    consistent, silently paired with today's dataset. Phase 24's lock
    records the digest of the dataset lock it read, so the two can be
    tied together instead of merely each being valid.
    """
    name = "curation_lock"
    required = (
        (curation_lock_path, "curation lock"),
        (curated_manifest_path, "curated manifest"),
    )
    for path, label in required:
        if not path.is_file():
            return GateResult(
                name=name,
                passed=False,
                detail=f"no {label} at {path.name}",
                failure_code=FailureCode.CURATION_LOCK_INVALID.value,
            )

    lock = _load_json(curation_lock_path)
    records = _load_jsonl(curated_manifest_path)

    digest = hashlib.sha256()
    for record in sorted(records, key=lambda r: str(r.get("track_id"))):
        digest.update(
            json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")

    problems: list[str] = []
    if lock.get("curated_manifest_sha256") != digest.hexdigest():
        problems.append("curated manifest content differs from the frozen digest")

    recorded_dataset_lock = lock.get("source_dataset_lock_sha256")
    if recorded_dataset_lock is None:
        problems.append(
            "the curation lock records no dataset lock digest, so it cannot be tied to this dataset"
        )
    elif dataset_lock_path.is_file():
        actual = _sha256_file(dataset_lock_path)
        if recorded_dataset_lock != actual:
            problems.append(
                "the curation was computed against a different dataset lock than the one supplied"
            )

    if problems:
        return GateResult(
            name=name,
            passed=False,
            detail="; ".join(problems),
            failure_code=FailureCode.CURATION_LOCK_INVALID.value,
            evidence={"curation_id": lock.get("curation_id")},
        )
    return GateResult(
        name=name,
        passed=True,
        detail=(
            f"curation {lock.get('curation_id')} matches its lock and derives from this dataset"
        ),
        evidence={
            "curation_id": lock.get("curation_id"),
            "selected_track_count": lock.get("selected_track_count"),
        },
    )


def selected_records(curated_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only the tracks curation actually selected for training."""
    return [
        record
        for record in curated_records
        if str(record.get("curation_action", "")) in SELECTED_ACTIONS
    ]


# ── 3. rights ────────────────────────────────────────────────────────


def rights_gate(curated_records: list[dict[str, Any]]) -> GateResult:
    """Every selected track is permitted, checked from its own record.

    Deliberately re-derived rather than trusting `training_eligible`.
    Phase 23 has an `include_rights_unknown` build option and Phase 24
    an export policy, both legitimate for analysis and inventory. Either
    could mark an unknown-rights track eligible, and Step 10 requires
    that production training cannot reach that path by accident. So the
    provenance block itself is the authority here.
    """
    name = "rights"
    selected = selected_records(curated_records)
    offending: list[str] = []
    reasons: dict[str, int] = {}

    for record in selected:
        track_id = str(record.get("track_id", "<unknown>"))
        provenance = record.get("provenance")
        provenance = provenance if isinstance(provenance, dict) else {}

        failures: list[str] = []
        if provenance.get("hard_blocks"):
            failures.append("HARD_BLOCK")
        if str(provenance.get("commercial_training_allowed", "UNKNOWN")) in FORBIDDEN_PERMISSIONS:
            failures.append("PERMISSION_NOT_TRUE")
        if str(provenance.get("rights_status", "UNKNOWN")) in FORBIDDEN_RIGHTS_STATUSES:
            failures.append("RESTRICTED")
        if not provenance.get("training_permitted", False):
            failures.append("NOT_PERMITTED")
        eligibility = record.get("eligibility")
        eligibility = eligibility if isinstance(eligibility, dict) else {}
        if not eligibility.get("training_eligible", False):
            failures.append("NOT_ELIGIBLE")

        if failures:
            offending.append(track_id)
            for reason in failures:
                reasons[reason] = reasons.get(reason, 0) + 1

    if offending:
        return GateResult(
            name=name,
            passed=False,
            detail=(
                f"{len(offending)} of {len(selected)} selected tracks are not permitted "
                f"for training"
            ),
            failure_code=FailureCode.RIGHTS_GATE_FAILED.value,
            offending_ids=sorted(offending),
            offending_count=len(offending),
            evidence={"reasons": dict(sorted(reasons.items()))},
        )
    return GateResult(
        name=name,
        passed=True,
        detail=f"all {len(selected)} selected tracks carry explicit training permission",
        evidence={"selected": len(selected)},
    )


# ── 4. evaluation leakage ────────────────────────────────────────────


def leakage_gate(
    curated_records: list[dict[str, Any]],
    *,
    evaluation_only_ids: frozenset[str] = frozenset(),
    evaluation_only_digests: frozenset[str] = frozenset(),
) -> GateResult:
    """No benchmark or held-out material may enter training.

    Checked on **track id and content digest**, never on filename. A
    benchmark track copied under a different name is the same audio and
    the same leak, and a filename check would miss it — which is exactly
    how a benchmark quietly stops measuring generalisation.

    Split membership is checked too: a track Phase 23 placed in
    VALIDATION or TEST must not appear in a training selection whatever
    curation decided.
    """
    name = "evaluation_leakage"
    selected = selected_records(curated_records)
    offending: list[str] = []
    reasons: dict[str, int] = {}

    for record in selected:
        track_id = str(record.get("track_id", "<unknown>"))
        source = record.get("source")
        source = source if isinstance(source, dict) else {}
        digest = str(source.get("sha256", ""))

        failures: list[str] = []
        if track_id in evaluation_only_ids:
            failures.append("EVALUATION_ONLY_ID")
        if digest and digest in evaluation_only_digests:
            failures.append("EVALUATION_ONLY_DIGEST")
        if str(record.get("split", "")) not in ("TRAIN", ""):
            failures.append(f"SPLIT_{record.get('split')}")

        if failures:
            offending.append(track_id)
            for reason in failures:
                reasons[reason] = reasons.get(reason, 0) + 1

    if offending:
        return GateResult(
            name=name,
            passed=False,
            detail=(
                f"{len(offending)} selected track(s) belong to evaluation or a non-training split"
            ),
            failure_code=FailureCode.EVALUATION_LEAKAGE.value,
            offending_ids=sorted(offending),
            offending_count=len(offending),
            evidence={"reasons": dict(sorted(reasons.items()))},
        )
    return GateResult(
        name=name,
        passed=True,
        detail=(
            f"no evaluation-only or held-out material among {len(selected)} selected "
            f"tracks (checked by id and digest)"
        ),
        evidence={
            "protected_ids": len(evaluation_only_ids),
            "protected_digests": len(evaluation_only_digests),
        },
    )


# ── 4a. split contamination ──────────────────────────────────────────


def split_leakage_gate(splits_payload: dict[str, Any]) -> GateResult:
    """No track may hold more than one role in an experiment.

    The gate that has to hold before Phase 36's experiment starts. A
    track appearing in both training and evaluation makes every later
    measurement a statement about memorisation, and the mistake is
    silent: the numbers simply come out better than they should.

    The check is delegated to :func:`luber_dataset.leakage_report`,
    which compares every pair of splits by audio digest *and* by track
    id, and it runs against the split manifest **as written to disk**
    rather than against the builder's in-memory result. A builder that
    is correct and a file that is correct are two different claims.
    """
    name = "split_leakage"
    from luber_dataset.splits import (
        EVALUATION,
        TRAIN,
        VALIDATION,
        ExperimentSplits,
        Split,
        SplitMember,
        leakage_report,
    )

    def _split(key: str) -> Split:
        raw = splits_payload.get(key) or {}
        members = tuple(
            SplitMember(
                track_id=str(item.get("track_id", "")),
                audio_sha256=str(item.get("audio_sha256", "")),
                source_group=str(item.get("source_group", "")),
                duration_seconds=float(item.get("duration_seconds", 0.0)),
            )
            for item in raw.get("tracks") or []
        )
        return Split(name=raw.get("name") or key.upper(), members=members)

    try:
        splits = ExperimentSplits(
            dataset_id=str(splits_payload.get("dataset_id", "")),
            library_content_hash=str(splits_payload.get("library_content_hash", "")),
            seed=int(splits_payload.get("seed", 0)),
            train=_split("train"),
            validation=_split("validation"),
            evaluation=_split("evaluation"),
        )
    except (TypeError, ValueError) as exc:
        return GateResult(
            name=name,
            passed=False,
            detail=f"the split manifest could not be read: {exc}",
            failure_code=FailureCode.EVALUATION_LEAKAGE.value,
        )

    empty = [split.name for split in splits.splits if not split.members]
    if empty:
        return GateResult(
            name=name,
            passed=False,
            detail=f"{', '.join(sorted(empty))} is empty; an empty split proves nothing",
            failure_code=FailureCode.EVALUATION_LEAKAGE.value,
        )

    report = leakage_report(splits)
    if not report.passed:
        offending = sorted({item for f in report.findings for item in f.identities})
        return GateResult(
            name=name,
            passed=False,
            detail=report.detail,
            failure_code=FailureCode.EVALUATION_LEAKAGE.value,
            offending_ids=offending[:20],
            offending_count=len(offending),
            evidence={"findings": [f.to_dict() for f in report.findings]},
        )

    return GateResult(
        name=name,
        passed=True,
        detail=report.detail,
        evidence={
            "splits_digest": splits.digest(),
            TRAIN: splits.train.digest(),
            VALIDATION: splits.validation.digest(),
            EVALUATION: splits.evaluation.digest(),
        },
    )


# ── 5. self-generated data ───────────────────────────────────────────


def self_generated_gate(
    curated_records: list[dict[str, Any]], *, allow_self_generated: bool = False
) -> GateResult:
    """Model output must not silently feed back into the model.

    Off by default. Training a model on its own generations teaches it
    its own artifacts, and the Phase 5 human verdict rated that output
    2/10 — so the default is not caution for its own sake.

    A record whose provenance cannot distinguish origin **blocks**
    rather than being assumed human. Guessing here is exactly the
    failure mode: unknown provenance is how self-generated audio gets
    in unnoticed.
    """
    name = "self_generated"
    selected = selected_records(curated_records)
    self_generated: list[str] = []
    indeterminate: list[str] = []

    for record in selected:
        track_id = str(record.get("track_id", "<unknown>"))
        provenance = record.get("provenance")
        provenance = provenance if isinstance(provenance, dict) else {}
        source_type = provenance.get("source_type")

        if source_type is None or not str(source_type).strip():
            indeterminate.append(track_id)
        elif str(source_type) in SELF_GENERATED_SOURCE_TYPES:
            self_generated.append(track_id)

    if indeterminate:
        return GateResult(
            name=name,
            passed=False,
            detail=(
                f"{len(indeterminate)} selected track(s) have no source_type, so "
                f"self-generated audio cannot be ruled out"
            ),
            failure_code=FailureCode.SELF_GENERATED_BLOCKED.value,
            offending_ids=sorted(indeterminate),
            offending_count=len(indeterminate),
            evidence={"reason": "INDETERMINATE_PROVENANCE"},
        )

    if self_generated and not allow_self_generated:
        return GateResult(
            name=name,
            passed=False,
            detail=(
                f"{len(self_generated)} selected track(s) are this project's own model "
                f"output and ALLOW_SELF_GENERATED is false"
            ),
            failure_code=FailureCode.SELF_GENERATED_BLOCKED.value,
            offending_ids=sorted(self_generated),
            offending_count=len(self_generated),
            evidence={"allow_self_generated": allow_self_generated},
        )

    synthetic = sum(
        1
        for record in selected
        if str((record.get("provenance") or {}).get("source_type", "")) in SYNTHETIC_SOURCE_TYPES
    )
    return GateResult(
        name=name,
        passed=True,
        detail=(
            "no self-model output in the selection"
            if not self_generated
            else f"{len(self_generated)} self-generated track(s) admitted by explicit policy"
        ),
        evidence={
            "self_generated": len(self_generated),
            "synthetic_total": synthetic,
            "allow_self_generated": allow_self_generated,
        },
    )


# ── the whole battery ────────────────────────────────────────────────


@dataclass(frozen=True)
class GateInputs:
    """Everything the gates read. Paths, not loaded state, so a gate
    run always reflects what is on disk right now."""

    dataset_lock_path: Path
    dataset_manifest_path: Path
    curation_lock_path: Path
    curated_manifest_path: Path
    evaluation_only_ids: frozenset[str] = frozenset()
    evaluation_only_digests: frozenset[str] = frozenset()
    allow_self_generated: bool = False


def run_all(inputs: GateInputs) -> GateReport:
    """Every gate, in order, short-circuiting only where it must.

    The lock gates run first because the later ones read the curated
    manifest, and reading an unverified manifest to decide about rights
    would be answering the right question from the wrong file.
    """
    report = GateReport()

    dataset = dataset_lock_gate(inputs.dataset_lock_path, inputs.dataset_manifest_path)
    report.results.append(dataset)

    curation = curation_lock_gate(
        inputs.curation_lock_path, inputs.curated_manifest_path, inputs.dataset_lock_path
    )
    report.results.append(curation)

    if not (dataset.passed and curation.passed):
        # The remaining gates would read a manifest nobody has verified.
        # Reporting them as "not run" is honest; running them would give
        # answers computed from a file that may not be the right one.
        for name, code in (
            ("rights", FailureCode.RIGHTS_GATE_FAILED),
            ("evaluation_leakage", FailureCode.EVALUATION_LEAKAGE),
            ("self_generated", FailureCode.SELF_GENERATED_BLOCKED),
        ):
            report.results.append(
                GateResult(
                    name=name,
                    passed=False,
                    detail="not evaluated: the dataset or curation lock did not verify",
                    failure_code=code.value,
                    evidence={"skipped": True},
                )
            )
        return report

    curated = _load_jsonl(inputs.curated_manifest_path)
    report.results.append(rights_gate(curated))
    report.results.append(
        leakage_gate(
            curated,
            evaluation_only_ids=inputs.evaluation_only_ids,
            evaluation_only_digests=inputs.evaluation_only_digests,
        )
    )
    report.results.append(
        self_generated_gate(curated, allow_self_generated=inputs.allow_self_generated)
    )
    return report
