"""Sampling weights: rebalance by how often, never by how many copies.

The naive way to correct an imbalance is to duplicate the rare tracks
until the counts look right. It is also the reliable way to teach a
model to memorise them: the same forty seconds arriving twelve times an
epoch is not more data, it is the same data twelve times, and the model
learns it verbatim while the loss curve looks fine.

So nothing is ever copied. A rare track appears once in the curated
manifest and carries a *weight*, and the trainer's sampler decides how
often to draw it. That keeps the dataset honest about its own size —
selected hours stay the hours that actually exist.

Two rules keep the weights from recreating the problem they solve:

*Weights are capped.* An uncapped inverse-frequency weight on a category
holding one track out of ten thousand asks for that track ten thousand
times as often, which is memorisation with extra arithmetic.

*Validation and test get none.* They exist to be a stable, representative
measurement. Reweighting them changes what the number means between runs
and makes two evaluations incomparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from luber_dataset.factory.intelligence.distributions import duration_bucket
from luber_dataset.factory.intelligence.profile import DatasetProfile
from luber_dataset.factory.intelligence.schemas import TrackView
from luber_dataset.factory.intelligence.targets import TargetProfile

#: The hard ceiling. Four is enough to lift a genuinely thin region
#: without any single track dominating a batch; measured against the
#: alternative, an uncapped inverse-frequency scheme on a 1-in-5000
#: category asks for 5000x and is simply memorisation.
DEFAULT_MAX_WEIGHT = 4.0
DEFAULT_MIN_WEIGHT = 0.25

#: Dimensions weights may respond to. Reliable ones only — weighting on
#: a mostly-unknown dimension would hand the largest boost to whichever
#: tracks happen to be unlabelled.
WEIGHTED_DIMENSIONS: tuple[str, ...] = ("language", "vocal_class", "duration_bucket")


@dataclass
class SamplingPlan:
    """Per-track weights for the training split only."""

    weights: dict[str, float] = field(default_factory=dict)
    max_weight: float = DEFAULT_MAX_WEIGHT
    min_weight: float = DEFAULT_MIN_WEIGHT
    #: Why each weighted track got what it got.
    rationale: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def bounded(self) -> bool:
        """Whether every weight sits inside the declared range."""
        return all(
            self.min_weight - 1e-9 <= weight <= self.max_weight + 1e-9
            for weight in self.weights.values()
        )

    def to_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "track_id": track_id,
                "sampling_weight": round(self.weights[track_id], 6),
                "rationale": {
                    k: round(v, 6) for k, v in sorted(self.rationale.get(track_id, {}).items())
                },
            }
            for track_id in sorted(self.weights)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_weight": self.max_weight,
            "min_weight": self.min_weight,
            "weighted_tracks": len(self.weights),
            "bounded": self.bounded,
        }


def build(
    selected: list[TrackView],
    profile: DatasetProfile,
    target: TargetProfile,
    *,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    min_weight: float = DEFAULT_MIN_WEIGHT,
) -> SamplingPlan:
    """Weights that lift under-target regions, bounded on both sides.

    The profile passed in should be the profile *of the selection*, not
    of the whole corpus: weights correct what is actually going into
    training, and the corpus includes everything selection just removed.

    A track with no target-relevant, known dimension gets exactly 1.0.
    Neutrality is the correct answer for "nothing about this track says
    it should be seen more or less often".
    """
    if max_weight < 1.0 or min_weight > 1.0:
        raise ValueError("the weight range must contain 1.0, or nothing is neutral")

    plan = SamplingPlan(max_weight=max_weight, min_weight=min_weight)
    for track in sorted(selected, key=lambda t: t.track_id):
        factors: dict[str, float] = {}

        for dimension in WEIGHTED_DIMENSIONS:
            distribution = profile.categorical.get(dimension)
            if distribution is None or distribution.known_count == 0:
                continue
            label = _label(track, dimension)
            if label is None:
                continue
            bounds = target.range_for(dimension, label)
            if bounds is None or bounds.minimum is None:
                continue
            share = distribution.share(label)
            if share <= 0 or share >= bounds.minimum:
                continue
            # Exactly the factor that would bring this category to its
            # declared minimum if applied to every member of it, before
            # the cap. Stated this way the weight is explainable: "this
            # region is at 12% and was asked for 30%".
            factors[dimension] = bounds.minimum / share

        weight = 1.0
        for factor in factors.values():
            weight *= factor
        weight = max(min_weight, min(max_weight, weight))
        plan.weights[track.track_id] = weight
        if factors:
            plan.rationale[track.track_id] = factors
    return plan


def _label(track: TrackView, dimension: str) -> str | None:
    if dimension == "language":
        observation = track.language()
    elif dimension == "vocal_class":
        observation = track.vocal_class()
    elif dimension == "duration_bucket":
        return duration_bucket(track.duration_seconds) if track.duration_seconds > 0 else None
    else:
        return None
    return str(observation.value) if observation.known else None
