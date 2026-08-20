"""Ranking eligible tracks, transparently, and never ranking rights.

The score answers exactly one question: **among tracks that are already
allowed into training, which should be preferred when something has to
choose?** It is a tie-breaker for selection, not a judgement of worth,
and it has no authority to admit anything.

That boundary is the important part. Rights are a hard gate applied
*before* scoring, and they are deliberately absent from the components
below. If provenance were a weighted term, a track with unknown rights
could out-score a cleared one on quality and coverage and be selected —
turning "we are not sure we may use this" into an arithmetic
comparison. There is no weight at which that is acceptable, so the
weight does not exist.

Every component is explicit, bounded and stored per track. No opaque
model, no learned weighting: a selection nobody can explain is a
selection nobody can correct, and this decides what a model is trained
on.

The components trade off deliberately. A rare Tier B track *can* beat a
redundant Tier A one, because coverage contribution is worth more than
the gap between adjacent tiers. That is the whole reason to have a
coverage term — a dataset of uniformly excellent duplicates is worse
than a varied one of merely good tracks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from luber_dataset.factory.intelligence.profile import DatasetProfile
from luber_dataset.factory.intelligence.schemas import TrackView
from luber_dataset.factory.intelligence.targets import TargetProfile

#: Weights, summing to 1.0 so a score is always in [0, 1]. Coverage is
#: weighted above quality on purpose: the tier gap between A and B is
#: 0.2 here, while moving a track from a saturated region to an empty
#: one is worth up to 0.30 — which is exactly the "rare B beats
#: redundant A" behaviour the design calls for.
DEFAULT_WEIGHTS: dict[str, float] = {
    "quality": 0.30,
    "coverage_contribution": 0.30,
    "metadata_completeness": 0.15,
    "source_diversity": 0.15,
    "duplicate_pressure": 0.10,
}

#: Tier scores. Ordered, and closer together than the coverage term's
#: range so tier alone cannot dominate selection.
TIER_SCORES: dict[str, float] = {"A": 1.0, "B": 0.8, "C": 0.5, "REJECT": 0.0}

#: Dimensions whose rarity counts toward coverage contribution. Only
#: reliable ones: a track cannot earn a rarity bonus for being unknown,
#: or "no genre" would become the rarest and most valuable category in
#: any poorly-labelled corpus.
COVERAGE_DIMENSIONS: tuple[str, ...] = (
    "language",
    "vocal_class",
    "tempo_bucket",
    "duration_bucket",
    "genre",
)


@dataclass
class ScoreComponents:
    """The arithmetic behind one track's rank, kept for audit."""

    quality: float = 0.0
    coverage_contribution: float = 0.0
    metadata_completeness: float = 0.0
    source_diversity: float = 0.0
    duplicate_pressure: float = 0.0
    #: Per-dimension detail behind ``coverage_contribution``.
    coverage_detail: dict[str, float] = field(default_factory=dict)

    def total(self, weights: dict[str, float]) -> float:
        return (
            self.quality * weights["quality"]
            + self.coverage_contribution * weights["coverage_contribution"]
            + self.metadata_completeness * weights["metadata_completeness"]
            + self.source_diversity * weights["source_diversity"]
            + self.duplicate_pressure * weights["duplicate_pressure"]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality": round(self.quality, 6),
            "coverage_contribution": round(self.coverage_contribution, 6),
            "metadata_completeness": round(self.metadata_completeness, 6),
            "source_diversity": round(self.source_diversity, 6),
            "duplicate_pressure": round(self.duplicate_pressure, 6),
            "coverage_detail": {k: round(v, 6) for k, v in sorted(self.coverage_detail.items())},
        }


class Scorer:
    """Scores tracks against a profile that is computed once."""

    def __init__(
        self,
        profile: DatasetProfile,
        target: TargetProfile,
        *,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.profile = profile
        self.target = target
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"scoring weights must sum to 1.0, got {total}")

        family_sizes: dict[str, int] = {}
        for name, members in self._families().items():
            family_sizes[name] = len(members)
        self._family_sizes = family_sizes

    def _families(self) -> dict[str, list[str]]:
        distribution = self.profile.categorical.get("duplicate_family")
        if distribution is None:
            return {}
        return {bucket.label: [""] * bucket.count for bucket in distribution.buckets}

    # ── components ───────────────────────────────────────────────────
    def _quality(self, track: TrackView) -> float:
        return TIER_SCORES.get(track.quality_tier, 0.0)

    def _coverage(self, track: TrackView) -> tuple[float, dict[str, float]]:
        """How much this track helps a thin region.

        A category's contribution is ``1 - share``: the rarer the region
        the track occupies, the more it is worth. When a target names a
        minimum the dataset is below, the contribution is raised toward
        1.0, so a profile can pull selection toward what it asked for
        without a separate mechanism.

        Averaged over the dimensions this track is actually known on,
        not over all of them. Counting an unknown as 0.0 sounds
        conservative and is not: it divides the one real signal by five
        and erases the term on any sparsely-labelled corpus, which is
        the normal case. Measured on nine redundant Tier A tracks and
        one rare Tier B, dilution made the rare track *lose* 0.504 to
        0.516 — the opposite of what the coverage weight exists for.

        An unknown still earns nothing. It is skipped rather than
        rewarded, so being unlabelled never becomes the most valuable
        property a track can have.
        """
        detail: dict[str, float] = {}
        accessors = {
            "language": track.language,
            "vocal_class": track.vocal_class,
            "genre": track.genre,
        }
        for dimension in COVERAGE_DIMENSIONS:
            distribution = self.profile.categorical.get(dimension)
            if distribution is None or distribution.known_count == 0:
                # The dataset knows nothing about this dimension, so it
                # can say nothing about anyone's coverage. Skipped, not
                # scored zero — see the note on the unknown-label branch.
                continue

            if dimension in accessors:
                observation = accessors[dimension]()
                label = str(observation.value) if observation.known else None
            elif dimension == "tempo_bucket":
                from luber_dataset.factory.intelligence.distributions import tempo_bucket

                observation = track.bpm()
                label = (
                    tempo_bucket(float(observation.value))
                    if observation.known and isinstance(observation.value, (int, float))
                    else None
                )
            else:
                from luber_dataset.factory.intelligence.distributions import duration_bucket

                label = (
                    duration_bucket(track.duration_seconds) if track.duration_seconds > 0 else None
                )

            if label is None:
                # Not counted at all, rather than counted as zero.
                # Averaging over every dimension including the unknown
                # ones divides the single real signal by five, and on a
                # sparsely-labelled corpus — the normal case — that
                # erases the coverage term entirely. Measured on a
                # fixture of nine redundant Tier A tracks and one rare
                # Tier B: diluted, the rare track lost 0.504 to 0.516;
                # averaged over known dimensions only, it wins.
                continue

            share = distribution.share(label)
            contribution = max(0.0, 1.0 - share)
            bounds = self.target.range_for(dimension, label)
            if bounds is not None and bounds.minimum is not None and share < bounds.minimum:
                # Below a declared floor: this region is not merely rare,
                # it is short of what was asked for.
                contribution = max(contribution, 0.9)
            detail[dimension] = contribution

        # Averaged over the dimensions this track is actually known on.
        # A track known on nothing scores 0.0, which is correct: it
        # demonstrates no coverage contribution, and rewarding it would
        # make being unlabelled the most valuable property available.
        return (sum(detail.values()) / len(detail) if detail else 0.0), detail

    def _metadata(self, track: TrackView) -> float:
        """Share of the useful metadata fields this track actually has.

        A well-described track is more useful for conditioning and for
        every future analysis, so it wins ties — but at 0.15 weight it
        cannot outrank a real quality or coverage difference.
        """
        checks = (
            track.artist().known,
            track.album().known,
            track.genre().known,
            track.language().known,
            track.vocal_class().known,
            track.bpm().known,
            bool((track.raw.get("text") or {}).get("lyrics")),
        )
        return sum(1 for present in checks if present) / len(checks)

    def _source_diversity(self, track: TrackView) -> float:
        """Lower when the track comes from an already-dominant source."""
        scores: list[float] = []
        for dimension, accessor in (
            ("artist", track.artist),
            ("source_reference", lambda: None),
        ):
            distribution = self.profile.categorical.get(dimension)
            if distribution is None or distribution.known_count == 0:
                continue
            if dimension == "artist":
                observation = accessor()
                label = str(observation.value) if observation and observation.known else None
            else:
                label = track.source_reference or None
            if label is None:
                # Unknown source: neutral, not rewarded. It might come
                # from the dominant source for all anyone knows.
                scores.append(0.5)
                continue
            scores.append(max(0.0, 1.0 - distribution.share(label)))
        return sum(scores) / len(scores) if scores else 0.5

    def _duplicate_pressure(self, track: TrackView) -> float:
        """1.0 for a unique track, falling as its family grows."""
        size = self._family_sizes.get(track.duplicate_family, 1)
        return 1.0 / float(max(1, size))

    # ── public ───────────────────────────────────────────────────────
    def components(self, track: TrackView) -> ScoreComponents:
        coverage, detail = self._coverage(track)
        return ScoreComponents(
            quality=self._quality(track),
            coverage_contribution=coverage,
            metadata_completeness=self._metadata(track),
            source_diversity=self._source_diversity(track),
            duplicate_pressure=self._duplicate_pressure(track),
            coverage_detail=detail,
        )

    def score(self, track: TrackView) -> tuple[float, ScoreComponents]:
        components = self.components(track)
        return components.total(self.weights), components
