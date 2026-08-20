"""The orchestrator: manifest in, curation plan out.

Reads a Phase 23 manifest, profiles it, evaluates it against a target,
selects, weights, and writes derived artifacts. It never rescans audio,
never re-decodes, and never writes to ``dataset_manifest.jsonl``. The
input is treated as immutable and its digest is recorded so any output
can be traced back to exactly the manifest it came from.

The profile is computed three times over different populations, because
they answer different questions and conflating them hides the answer:

* **corpus** — everything in the manifest, which explains what was
  excluded and why;
* **eligible** — what Phase 23 would allow into training, which is what
  a curation target should be measured against;
* **selected** — what survived curation, which is what will actually be
  trained on and is therefore the population the sampling weights and
  the after-distributions describe.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from luber_dataset.factory.intelligence import findings as findings_module
from luber_dataset.factory.intelligence import profile as profile_module
from luber_dataset.factory.intelligence import sampling, selection, targets
from luber_dataset.factory.intelligence.schemas import (
    CURATION_ENGINE_VERSION,
    CURATION_SCHEMA_VERSION,
    Finding,
    TrackView,
)
from luber_dataset.factory.intelligence.scoring import DEFAULT_WEIGHTS, Scorer


@dataclass(frozen=True)
class CurationConfig:
    """Everything that changes the answer, in one hashable place."""

    seed: int = 42
    #: Confidence a measured tempo or key needs before it counts as
    #: known. Phase 23 reports a tempo for material with no pulse.
    min_music_confidence: float = 0.55
    scoring_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    max_sampling_weight: float = sampling.DEFAULT_MAX_WEIGHT
    min_sampling_weight: float = sampling.DEFAULT_MIN_WEIGHT
    #: Track ids that must never enter training. Phase 23 has no such
    #: field, so this is the mechanism — configuration, not inference.
    evaluation_only: tuple[str, ...] = ()
    head_cumulative: float = 0.5
    mid_cumulative: float = 0.9

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evaluation_only"] = sorted(self.evaluation_only)
        return payload

    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class CurationResult:
    """Everything one curation run produced."""

    source_manifest_sha256: str = ""
    source_dataset_lock_sha256: str | None = None
    factory_schema_version: str = ""
    factory_version: str | None = None

    corpus_profile: profile_module.DatasetProfile | None = None
    eligible_profile: profile_module.DatasetProfile | None = None
    selected_profile: profile_module.DatasetProfile | None = None

    findings: list[Finding] = field(default_factory=list)
    selection: selection.SelectionResult = field(default_factory=selection.SelectionResult)
    sampling_plan: sampling.SamplingPlan = field(default_factory=sampling.SamplingPlan)
    curated_records: list[dict[str, Any]] = field(default_factory=list)
    target_profile: targets.TargetProfile = field(default_factory=targets.neutral)
    config: CurationConfig = field(default_factory=CurationConfig)

    @property
    def selected_hours(self) -> float:
        return self.selected_profile.total_hours if self.selected_profile else 0.0

    def canonical_digest(self) -> str:
        """Digest over the curated records, timestamps excluded.

        The whole point of curating deterministically is being able to
        say "this is the same plan as yesterday", and a digest that
        moved with the clock could never say it.
        """
        digest = hashlib.sha256()
        for record in sorted(self.curated_records, key=lambda r: str(r.get("track_id"))):
            digest.update(
                json.dumps(
                    record, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ).encode("utf-8")
            )
            digest.update(b"\n")
        return digest.hexdigest()


def read_manifest(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Load a manifest and its digest, reading the file exactly once.

    The digest is over the bytes on disk rather than over the parsed
    records, so it identifies the artifact the operator actually has.
    """
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    records: list[dict[str, Any]] = []
    for number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} line {number} is not valid JSON: {exc}") from exc
        if not isinstance(record, dict) or "track_id" not in record:
            raise ValueError(f"{path.name} line {number} is not a manifest record")
        records.append(record)
    return records, digest


def curate(
    manifest_path: Path,
    *,
    target: targets.TargetProfile | None = None,
    config: CurationConfig | None = None,
    dataset_lock_path: Path | None = None,
) -> CurationResult:
    """Profile, evaluate, select and weight. Writes nothing."""
    target = target or targets.neutral()
    targets.validate(target)
    config = config or CurationConfig()

    records, digest = read_manifest(manifest_path)
    result = CurationResult(source_manifest_sha256=digest, target_profile=target, config=config)

    if dataset_lock_path is not None and dataset_lock_path.is_file():
        lock_bytes = dataset_lock_path.read_bytes()
        result.source_dataset_lock_sha256 = hashlib.sha256(lock_bytes).hexdigest()
        try:
            lock = json.loads(lock_bytes)
            result.factory_version = str(lock.get("factory_version") or "") or None
        except json.JSONDecodeError:
            result.factory_version = None

    versions = {str(r.get("schema_version", "")) for r in records}
    result.factory_schema_version = ", ".join(sorted(v for v in versions if v))

    views = [TrackView(record, min_confidence=config.min_music_confidence) for record in records]

    def profiled(population_tracks: list[TrackView], name: str) -> profile_module.DatasetProfile:
        return profile_module.build(
            population_tracks,
            population=name,
            duplicate_family_cap=target.selection.max_records_per_duplicate_family,
            head_cumulative=config.head_cumulative,
            mid_cumulative=config.mid_cumulative,
        )

    result.corpus_profile = profiled(views, "corpus")
    eligible = [view for view in views if view.training_eligible]
    result.eligible_profile = profiled(eligible, "training_eligible")

    # Findings are evaluated against the eligible population: a target
    # describes the dataset that would be trained on, and measuring it
    # against a corpus half of which is barred would report gaps that
    # curation could never close.
    result.findings = findings_module.evaluate(result.eligible_profile, target)

    scorer = Scorer(result.eligible_profile, target, weights=config.scoring_weights)
    result.selection = selection.select(
        views,
        scorer,
        target,
        evaluation_only=frozenset(config.evaluation_only),
    )

    selected_views = [view for view in views if view.track_id in set(result.selection.selected_ids)]
    result.selected_profile = profiled(selected_views, "selected")
    result.sampling_plan = sampling.build(
        selected_views,
        result.selected_profile,
        target,
        max_weight=config.max_sampling_weight,
        min_weight=config.min_sampling_weight,
    )

    for view in sorted(views, key=lambda v: v.track_id):
        decision = result.selection.decisions[view.track_id]
        if decision.selected:
            decision.sampling_weight = result.sampling_plan.weights.get(view.track_id, 1.0)
        result.curated_records.append(_curated_record(view, decision, result))
    return result


def _curated_record(
    view: TrackView, decision: selection.Decision, result: CurationResult
) -> dict[str, Any]:
    """A curated record: the original identity plus what was decided.

    The Phase 23 record is carried whole rather than summarised. A
    curated manifest that dropped fields would force consumers back to
    the original to answer basic questions, and the two would drift.
    """
    return {
        "curation_schema_version": CURATION_SCHEMA_VERSION,
        "curation_engine_version": CURATION_ENGINE_VERSION,
        "profile_version": result.target_profile.name,
        "profile_digest": result.target_profile.digest(),
        **view.raw,
        **decision.to_dict(),
    }
