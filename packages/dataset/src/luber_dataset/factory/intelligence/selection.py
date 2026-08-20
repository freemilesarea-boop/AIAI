"""Deciding what enters training, in an order anyone can re-derive.

The pipeline is fixed and each stage can only ever remove:

1. **Rights and policy.** A hard gate, applied first, that curation may
   not reach past. Nothing later can readmit what this excludes.
2. **Evaluation protection.** Anything designated evaluation-only is
   withheld regardless of how good it is.
3. **Split respect.** Only tracks Phase 23 placed in TRAIN are candidates.
   Validation and test are left exactly as frozen.
4. **Duplicate family caps.** Versions of the same content beyond the cap.
5. **Concentration caps.** Artist, album and source ceilings.

Within each cap the *lowest-scoring* tracks are dropped, so the surviving
selection is the best of an overrepresented region rather than an
arbitrary slice of it. Ties break on track id, which makes the result
deterministic without a random number generator being involved at all —
a seed is recorded for reproducibility of anything that does use one,
but selection does not need it.

Every exclusion carries its reason. A track that vanishes without one is
indistinguishable from a bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from luber_dataset.factory.intelligence.schemas import (
    SELECTED_ACTIONS,
    CurationAction,
    TrackView,
)
from luber_dataset.factory.intelligence.scoring import ScoreComponents, Scorer
from luber_dataset.factory.intelligence.targets import TargetProfile

# ── exclusion reasons ────────────────────────────────────────────────
RIGHTS_NOT_PERMITTED = "RIGHTS_NOT_PERMITTED"
RIGHTS_HARD_BLOCK = "RIGHTS_HARD_BLOCK"
NOT_TRAINING_ELIGIBLE = "NOT_TRAINING_ELIGIBLE"
EVALUATION_ONLY = "EVALUATION_ONLY"
NOT_IN_TRAIN_SPLIT = "NOT_IN_TRAIN_SPLIT"
DUPLICATE_FAMILY_CAP = "DUPLICATE_FAMILY_CAP"
ARTIST_CAP = "ARTIST_CAP"
ARTIST_HOURS_CAP = "ARTIST_HOURS_CAP"
ALBUM_CAP = "ALBUM_CAP"
NEEDS_REVIEW = "NEEDS_REVIEW"

#: Rights states that may never be trained on, whatever else is true.
FORBIDDEN_PERMISSIONS: frozenset[str] = frozenset({"FALSE", "UNKNOWN"})
FORBIDDEN_RIGHTS_STATUSES: frozenset[str] = frozenset({"RESTRICTED"})


@dataclass
class Decision:
    """What was decided for one track, and why."""

    track_id: str
    action: str
    reasons: list[str] = field(default_factory=list)
    score: float = 0.0
    components: ScoreComponents = field(default_factory=ScoreComponents)
    sampling_weight: float | None = None

    @property
    def selected(self) -> bool:
        return self.action in SELECTED_ACTIONS

    def to_dict(self) -> dict[str, Any]:
        return {
            "curation_action": self.action,
            "curation_reasons": sorted(set(self.reasons)),
            "curation_score": round(self.score, 6),
            "score_components": self.components.to_dict(),
            "sampling_weight": self.sampling_weight,
        }


def rights_barred(track: TrackView) -> list[str]:
    """Reasons this track may never be trained on.

    Read straight from Phase 23's provenance verdict rather than
    recomputed. A second implementation of the rights rule is a second
    answer to the question, and the two would eventually disagree.
    """
    reasons: list[str] = []
    if track.hard_blocks:
        reasons.append(RIGHTS_HARD_BLOCK)
    if track.commercial_training_allowed in FORBIDDEN_PERMISSIONS:
        reasons.append(RIGHTS_NOT_PERMITTED)
    if track.rights_status in FORBIDDEN_RIGHTS_STATUSES:
        reasons.append(RIGHTS_NOT_PERMITTED)
    if not track.training_permitted:
        reasons.append(RIGHTS_NOT_PERMITTED)
    if not track.training_eligible:
        reasons.append(NOT_TRAINING_ELIGIBLE)
    return reasons


@dataclass
class SelectionResult:
    decisions: dict[str, Decision] = field(default_factory=dict)
    selected_ids: list[str] = field(default_factory=list)
    excluded_by_reason: dict[str, int] = field(default_factory=dict)

    def to_summary(self) -> dict[str, Any]:
        return {
            "selected": len(self.selected_ids),
            "total": len(self.decisions),
            "excluded_by_reason": dict(sorted(self.excluded_by_reason.items())),
        }


def select(
    tracks: list[TrackView],
    scorer: Scorer,
    target: TargetProfile,
    *,
    evaluation_only: frozenset[str] = frozenset(),
    curate_holdout: bool = False,
) -> SelectionResult:
    """Run the full selection pipeline.

    ``evaluation_only`` names track ids that must never enter training —
    benchmark material above all. Phase 23 has no such field, so it is
    supplied by configuration rather than read from the manifest, and it
    is applied before anything else can weigh in.
    """
    result = SelectionResult()
    ordered = sorted(tracks, key=lambda t: t.track_id)

    # ── 1-3: gates nothing later may reopen ──────────────────────────
    candidates: list[TrackView] = []
    for track in ordered:
        score, components = scorer.score(track)
        decision = Decision(
            track_id=track.track_id,
            action=CurationAction.KEEP.value,
            score=score,
            components=components,
        )
        result.decisions[track.track_id] = decision

        barred = rights_barred(track)
        if barred:
            decision.action = CurationAction.EXCLUDE_POLICY.value
            decision.reasons.extend(barred)
            _count(result, barred)
            continue
        if track.track_id in evaluation_only:
            decision.action = CurationAction.HOLDOUT.value
            decision.reasons.append(EVALUATION_ONLY)
            _count(result, [EVALUATION_ONLY])
            continue
        if track.split != "TRAIN":
            decision.action = CurationAction.HOLDOUT.value
            decision.reasons.append(NOT_IN_TRAIN_SPLIT)
            _count(result, [NOT_IN_TRAIN_SPLIT])
            continue
        if track.dedup_decision == "REVIEW_REQUIRED":
            decision.action = CurationAction.REVIEW.value
            decision.reasons.append(NEEDS_REVIEW)
            _count(result, [NEEDS_REVIEW])
            continue
        candidates.append(track)

    # ── 4: duplicate family caps ─────────────────────────────────────
    limits = target.selection
    survivors = _apply_cap(
        candidates,
        result,
        key=lambda t: t.duplicate_family,
        cap=limits.max_records_per_duplicate_family,
        reason=DUPLICATE_FAMILY_CAP,
        action=CurationAction.EXCLUDE_DUPLICATE_PRESSURE.value,
    )

    # ── 5: concentration caps ────────────────────────────────────────
    if limits.max_tracks_per_artist is not None:
        survivors = _apply_cap(
            survivors,
            result,
            key=lambda t: str(t.artist().value) if t.artist().known else None,
            cap=limits.max_tracks_per_artist,
            reason=ARTIST_CAP,
            action=CurationAction.DOWNSAMPLE.value,
        )
    if limits.max_tracks_per_album is not None:
        survivors = _apply_cap(
            survivors,
            result,
            key=lambda t: str(t.album().value) if t.album().known else None,
            cap=limits.max_tracks_per_album,
            reason=ALBUM_CAP,
            action=CurationAction.DOWNSAMPLE.value,
        )
    if limits.max_hours_per_artist is not None:
        survivors = _apply_hours_cap(survivors, result, limits.max_hours_per_artist)

    # Priority marks the top of what survived, so a sampler that has to
    # choose knows which tracks the plan actually wants.
    ranked = sorted(survivors, key=lambda t: (-result.decisions[t.track_id].score, t.track_id))
    priority_cut = max(1, len(ranked) // 4) if ranked else 0
    for index, track in enumerate(ranked):
        decision = result.decisions[track.track_id]
        decision.action = (
            CurationAction.KEEP_PRIORITY.value
            if index < priority_cut
            else CurationAction.KEEP.value
        )
    result.selected_ids = sorted(track.track_id for track in survivors)
    return result


def _count(result: SelectionResult, reasons: list[str]) -> None:
    for reason in reasons:
        result.excluded_by_reason[reason] = result.excluded_by_reason.get(reason, 0) + 1


def _apply_cap(
    tracks: list[TrackView],
    result: SelectionResult,
    *,
    key: Any,
    cap: int,
    reason: str,
    action: str,
) -> list[TrackView]:
    """Keep the best ``cap`` tracks per group; drop the rest with a reason.

    A track whose group key is unknown is never capped. Grouping
    unknowns together would treat "we do not know the artist" as an
    artist and cap unrelated tracks against each other.
    """
    grouped: dict[str, list[TrackView]] = {}
    survivors: list[TrackView] = []
    for track in tracks:
        group = key(track)
        if group is None:
            survivors.append(track)
            continue
        grouped.setdefault(str(group), []).append(track)

    for group in sorted(grouped):
        members = sorted(
            grouped[group],
            key=lambda t: (-result.decisions[t.track_id].score, t.track_id),
        )
        survivors.extend(members[:cap])
        for dropped in members[cap:]:
            decision = result.decisions[dropped.track_id]
            decision.action = action
            decision.reasons.append(reason)
            _count(result, [reason])
    return survivors


def _apply_hours_cap(
    tracks: list[TrackView], result: SelectionResult, max_hours: float
) -> list[TrackView]:
    """Cap total hours per artist, best tracks first."""
    grouped: dict[str, list[TrackView]] = {}
    survivors: list[TrackView] = []
    for track in tracks:
        observation = track.artist()
        if not observation.known:
            survivors.append(track)
            continue
        grouped.setdefault(str(observation.value), []).append(track)

    for artist in sorted(grouped):
        members = sorted(
            grouped[artist], key=lambda t: (-result.decisions[t.track_id].score, t.track_id)
        )
        accumulated = 0.0
        for track in members:
            if accumulated + track.hours <= max_hours:
                accumulated += track.hours
                survivors.append(track)
            else:
                decision = result.decisions[track.track_id]
                decision.action = CurationAction.DOWNSAMPLE.value
                decision.reasons.append(ARTIST_HOURS_CAP)
                _count(result, [ARTIST_HOURS_CAP])
    return survivors
