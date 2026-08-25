"""Sorting measured audio into quality tiers, on four named axes.

Phase 38's question is which authorised material has live high end, a
steady pulse and a busy arrangement — and the answer has to be a
*decision*, separate from the measurement it rests on, so the two are in
different modules and the decision carries its thresholds with it.

Four axes, because those are the four things a listener is later asked
about:

**HIGH_END** — high-frequency energy share, spectral centroid, and the
level of the high band. A dull master loses this first.

**RHYTHM** — beat stability, tempo consistency, and the drum/bass
alignment proxy. Whether there is a pulse and whether it holds.

**ARRANGEMENT** — layer density, onset density, active-band occupancy.
How much is sounding at once.

**VOCAL** — deliberately **not scored here.** Nothing in this repository
can tell a sung note from a lead synth, and a number invented for it
would be the most misleading value in the file. The axis exists in the
vocabulary because the listening evaluation is organised around it; the
tiering leaves it unmeasured and says so.

Scores are percentile ranks *within the library being classified*, not
absolute quality. A tier-A track is in the top part of this library. It
is not "good", and the enum says `TIER_A`, not `GOOD`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from luber_dataset.audio_features import AudioFeatures


class QualityAxis(StrEnum):
    """The four axes the listening evaluation is organised around."""

    HIGH_END = "HIGH_END"
    RHYTHM = "RHYTHM"
    ARRANGEMENT = "ARRANGEMENT"
    #: Present in the vocabulary, never scored. See the module docstring.
    VOCAL = "VOCAL"


class QualityTier(StrEnum):
    """Where a track sits in *this* library. Not a verdict on the music."""

    TIER_A = "TIER_A"
    TIER_B = "TIER_B"
    TIER_C = "TIER_C"
    #: Not enough measurement to place it.
    UNRANKED = "UNRANKED"


#: Internal key for the texture half of HIGH_END. Not a member of
#: :class:`QualityAxis`, because the listening evaluation asks about four
#: axes and this is one half of one of them, not a fifth.
_HIGH_END_TEXTURE = "HIGH_END_TEXTURE"

#: Which measured features feed each scored axis, and which way is up.
#: Every one of these is a choice, recorded here rather than buried in
#: an expression.
AXIS_FEATURES: dict[str, tuple[tuple[str, bool], ...]] = {
    QualityAxis.HIGH_END.value: (
        ("high_frequency_energy_ratio", True),
        ("spectral_centroid_hz", True),
        ("high_band_rms_db", True),
    ),
    #: Texture, scored separately from the energy above and combined with
    #: it at equal weight. Phase 38 tiered on energy alone, selected the
    #: brightest material in the library, and still produced a high band
    #: the operator called metallic — because every feature above is a
    #: *level* measure and the complaint was about *texture*. Flatness up,
    #: narrow resonances down.
    _HIGH_END_TEXTURE: (
        ("high_band_flatness", True),
        ("high_band_resonance_ratio", False),
    ),
    QualityAxis.RHYTHM.value: (
        ("beat_stability", True),
        ("tempo_consistency", True),
        ("drum_bass_alignment", True),
    ),
    QualityAxis.ARRANGEMENT.value: (
        ("layer_density", True),
        ("onset_density_per_second", True),
        ("active_band_fraction", True),
    ),
}

#: Percentile a combined score must reach for each tier. Chosen, not
#: measured, and reported alongside every classification that uses them.
TIER_A_PERCENTILE = 0.70
TIER_B_PERCENTILE = 0.40


@dataclass(frozen=True)
class AxisScores:
    """One item's position on each scored axis, in [0, 1]."""

    #: Level in the high band: energy share, centroid, high-band RMS.
    high_end_energy: float
    #: Texture in the high band: flatness up, narrow resonances down.
    high_end_texture: float
    rhythm: float
    arrangement: float
    #: Never a number. Nothing here can measure it.
    vocal: None = None

    @property
    def high_end(self) -> float:
        """Energy and texture at equal weight.

        Equal because there is no evidence for any other split. What
        there *is* evidence for is that energy alone was not enough:
        Phase 38 maximised it and the operator still reported a metallic
        high band.
        """
        return (self.high_end_energy + self.high_end_texture) / 2.0

    @property
    def combined(self) -> float:
        """Equal weight across the three measurable axes.

        Equal because there is no evidence for any other weighting, and
        inventing one would be a preference dressed as a finding.
        """
        return (self.high_end + self.rhythm + self.arrangement) / 3.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "HIGH_END": round(self.high_end, 4),
            "HIGH_END_ENERGY": round(self.high_end_energy, 4),
            "HIGH_END_TEXTURE": round(self.high_end_texture, 4),
            "RHYTHM": round(self.rhythm, 4),
            "ARRANGEMENT": round(self.arrangement, 4),
            "VOCAL": None,
            "combined": round(self.combined, 4),
            "vocal_note": (
                "not scored: nothing in this repository distinguishes a sung note from a "
                "lead synth, and a number invented for it would mislead"
            ),
        }


@dataclass(frozen=True)
class TierAssignment:
    """One item's tier, and everything the decision rested on."""

    item_id: str
    tier: str
    scores: AxisScores
    source_group: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "source_group": self.source_group,
            "tier": self.tier,
            "scores": self.scores.to_dict(),
            "detail": self.detail,
        }


def _percentile_ranks(values: Sequence[float]) -> list[float]:
    """Each value's rank in [0, 1]. Ties share the average rank.

    A rank rather than a normalised value, because these features have
    wildly different scales and long tails — one very bright track would
    otherwise flatten every other high-end score toward zero.
    """
    count = len(values)
    if count == 0:
        return []
    if count == 1:
        return [0.5]
    order = sorted(range(count), key=lambda index: values[index])
    ranks = [0.0] * count
    position = 0
    while position < count:
        end = position
        while end + 1 < count and values[order[end + 1]] == values[order[position]]:
            end += 1
        shared = (position + end) / 2.0 / (count - 1)
        for index in range(position, end + 1):
            ranks[order[index]] = shared
        position = end + 1
    return ranks


def score_population(
    items: Sequence[tuple[str, AudioFeatures]],
) -> dict[str, AxisScores]:
    """Score every item against the rest of the population.

    Ranks are computed once across the whole population so a caller
    cannot accidentally rank a track against a different reference set
    than its neighbours.
    """
    if not items:
        return {}
    feature_ranks: dict[str, list[float]] = {}
    for axis_features in AXIS_FEATURES.values():
        for name, _ in axis_features:
            if name in feature_ranks:
                continue
            values = [float(getattr(features, name)) for _, features in items]
            feature_ranks[name] = _percentile_ranks(values)

    scores: dict[str, AxisScores] = {}
    for index, (item_id, _) in enumerate(items):
        axis_values: dict[str, float] = {}
        for axis, axis_features in AXIS_FEATURES.items():
            parts = []
            for name, higher_is_better in axis_features:
                rank = feature_ranks[name][index]
                parts.append(rank if higher_is_better else 1.0 - rank)
            axis_values[axis] = sum(parts) / len(parts)
        scores[item_id] = AxisScores(
            high_end_energy=axis_values[QualityAxis.HIGH_END.value],
            high_end_texture=axis_values[_HIGH_END_TEXTURE],
            rhythm=axis_values[QualityAxis.RHYTHM.value],
            arrangement=axis_values[QualityAxis.ARRANGEMENT.value],
        )
    return scores


def classify_population(
    items: Sequence[tuple[str, AudioFeatures]],
    *,
    groups: dict[str, str] | None = None,
    tier_a_percentile: float = TIER_A_PERCENTILE,
    tier_b_percentile: float = TIER_B_PERCENTILE,
) -> list[TierAssignment]:
    """Assign a tier to every item, by combined rank within the population."""
    scores = score_population(items)
    if not scores:
        return []
    combined = [(item_id, scores[item_id].combined) for item_id, _ in items]
    ranks = dict(
        zip(
            [item_id for item_id, _ in combined],
            _percentile_ranks([value for _, value in combined]),
            strict=True,
        )
    )

    assignments: list[TierAssignment] = []
    for item_id, _ in items:
        rank = ranks[item_id]
        if rank >= tier_a_percentile:
            tier = QualityTier.TIER_A.value
        elif rank >= tier_b_percentile:
            tier = QualityTier.TIER_B.value
        else:
            tier = QualityTier.TIER_C.value
        assignments.append(
            TierAssignment(
                item_id=item_id,
                tier=tier,
                scores=scores[item_id],
                source_group=(groups or {}).get(item_id, ""),
                detail=(
                    f"combined rank {rank:.3f} within a population of {len(items)}; "
                    f"tier A at {tier_a_percentile:.2f}, tier B at {tier_b_percentile:.2f}. "
                    "A rank inside this library, not a judgement of the music"
                ),
            )
        )
    return assignments


def tier_summary(assignments: Iterable[TierAssignment]) -> dict[str, Any]:
    """Counts by tier and by group, for a report."""
    by_tier: dict[str, int] = {}
    by_group: dict[str, dict[str, int]] = {}
    for item in assignments:
        by_tier[item.tier] = by_tier.get(item.tier, 0) + 1
        group = by_group.setdefault(item.source_group or "", {})
        group[item.tier] = group.get(item.tier, 0) + 1
    return {"by_tier": dict(sorted(by_tier.items())), "by_group": by_group}


__all__ = [
    "AXIS_FEATURES",
    "TIER_A_PERCENTILE",
    "TIER_B_PERCENTILE",
    "AxisScores",
    "QualityAxis",
    "QualityTier",
    "TierAssignment",
    "classify_population",
    "score_population",
    "tier_summary",
]
