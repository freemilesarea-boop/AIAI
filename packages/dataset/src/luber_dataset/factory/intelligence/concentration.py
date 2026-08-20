"""How concentrated a distribution is, measured several ways on purpose.

No single number tells you whether a dataset is dominated. Top-1 share
misses a dataset split evenly between three artists. HHI misses a long
tail of singletons behind a moderate head. Entropy is insensitive to
*which* category dominates. So all of them are reported, and a finding
is raised on the one that is actually diagnostic for the question being
asked.

The measure that carries the most weight here is **effective category
count** — the reciprocal of HHI. It answers "how many categories does
this dataset behave as though it has", which is the number that matters
for training: a corpus labelled with forty artists but behaving like
three will teach the model three artists.

Everything is computed over the *known* subset, and the denominator
travels with the result. Concentration measured over 10% coverage is a
statement about that 10%.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from luber_dataset.factory.intelligence.distributions import CategoricalDistribution


@dataclass
class ConcentrationMetrics:
    """Several views of the same question."""

    dimension: str
    known_count: int = 0
    category_count: int = 0
    top1_label: str | None = None
    top1_share: float = 0.0
    top5_share: float = 0.0
    top10_share: float = 0.0
    #: Herfindahl-Hirschman Index: sum of squared shares. 1.0 is one
    #: category holding everything; 1/n is perfectly even.
    hhi: float = 0.0
    #: 1/HHI — how many categories the distribution behaves as though it
    #: has, which is rarely how many it is labelled with.
    effective_categories: float = 0.0
    entropy: float = 0.0
    #: Entropy over log(n), so datasets with different category counts
    #: are comparable. 1.0 is perfectly even.
    normalized_entropy: float = 0.0
    #: Categories holding exactly one track.
    singleton_categories: int = 0
    coverage: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "known_count": self.known_count,
            "category_count": self.category_count,
            "top1_label": self.top1_label,
            "top1_share": round(self.top1_share, 6),
            "top5_share": round(self.top5_share, 6),
            "top10_share": round(self.top10_share, 6),
            "hhi": round(self.hhi, 6),
            "effective_categories": round(self.effective_categories, 4),
            "entropy": round(self.entropy, 6),
            "normalized_entropy": round(self.normalized_entropy, 6),
            "singleton_categories": self.singleton_categories,
            "coverage": round(self.coverage, 6),
        }


def measure(
    distribution: CategoricalDistribution, *, by_duration: bool = False
) -> ConcentrationMetrics:
    """Concentration of *known* values in a distribution.

    ``by_duration`` switches from counting tracks to weighing hours,
    which is the honest view when track lengths vary — thirty seconds
    and six minutes are one track each and not remotely equal exposure.
    """
    metrics = ConcentrationMetrics(
        dimension=distribution.dimension,
        known_count=distribution.known_count,
        category_count=distribution.category_count,
        coverage=distribution.coverage,
    )
    if not distribution.buckets:
        return metrics

    weighted = [
        (
            bucket.label,
            bucket.share_by_duration if by_duration else bucket.share_by_count,
        )
        for bucket in distribution.buckets
    ]
    total = sum(share for _, share in weighted)
    if total <= 0:
        return metrics
    # Renormalise: rounding in the bucket shares should not leak into
    # an index that is meant to sum to one.
    weighted = [(label, share / total) for label, share in weighted]

    # Ranked by *this* weighting. Buckets arrive ordered by count, so
    # taking the label from position zero while taking the share from a
    # duration ordering would name one category and quote another's
    # number — and any finding built on the pair would be wrong.
    ranked_pairs = sorted(weighted, key=lambda item: (-item[1], item[0]))
    shares = [share for _, share in weighted]
    ordered = [share for _, share in ranked_pairs]

    metrics.top1_label = ranked_pairs[0][0]
    metrics.top1_share = ordered[0]
    metrics.top5_share = sum(ordered[:5])
    metrics.top10_share = sum(ordered[:10])
    metrics.hhi = sum(share * share for share in shares)
    metrics.effective_categories = 1.0 / metrics.hhi if metrics.hhi > 0 else 0.0
    metrics.entropy = -sum(share * math.log(share) for share in shares if share > 0)
    if len(shares) > 1:
        metrics.normalized_entropy = metrics.entropy / math.log(len(shares))
    else:
        # One category is perfectly concentrated, and log(1) is zero.
        # Reporting 0.0 rather than dividing is the honest answer.
        metrics.normalized_entropy = 0.0
    metrics.singleton_categories = sum(1 for bucket in distribution.buckets if bucket.count == 1)
    return metrics


@dataclass
class LongTail:
    """Head, mid and tail of a categorical distribution.

    Split by cumulative share rather than by rank: "the categories
    covering the first 50%" describes the dataset, while "the top ten"
    describes an arbitrary choice of ten.
    """

    dimension: str
    head_categories: list[str] = field(default_factory=list)
    mid_categories: list[str] = field(default_factory=list)
    tail_categories: list[str] = field(default_factory=list)
    head_share: float = 0.0
    mid_share: float = 0.0
    tail_share: float = 0.0
    singletons: list[str] = field(default_factory=list)
    #: Categories with fewer tracks than the rarity floor.
    rare_categories: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "head_categories": self.head_categories,
            "mid_categories": self.mid_categories,
            "tail_categories": self.tail_categories,
            "head_share": round(self.head_share, 6),
            "mid_share": round(self.mid_share, 6),
            "tail_share": round(self.tail_share, 6),
            "singletons": self.singletons,
            "rare_categories": self.rare_categories,
        }


def long_tail(
    distribution: CategoricalDistribution,
    *,
    head_cumulative: float = 0.5,
    mid_cumulative: float = 0.9,
    rare_below: int = 3,
) -> LongTail:
    """Partition categories into head, mid-tail and long tail."""
    result = LongTail(dimension=distribution.dimension)
    cumulative = 0.0
    for bucket in distribution.buckets:
        share = bucket.share_by_count
        if cumulative < head_cumulative:
            result.head_categories.append(bucket.label)
            result.head_share += share
        elif cumulative < mid_cumulative:
            result.mid_categories.append(bucket.label)
            result.mid_share += share
        else:
            result.tail_categories.append(bucket.label)
            result.tail_share += share
        cumulative += share
        if bucket.count == 1:
            result.singletons.append(bucket.label)
        if bucket.count < rare_below:
            result.rare_categories.append(bucket.label)
    return result


@dataclass
class FamilyPressure:
    """How much of the dataset is versions of the same thing.

    A corpus can be perfectly deduplicated and still be four encodes of
    the same three hundred songs. Deduplication removes *identical*
    records; family pressure measures what the survivors still share.
    """

    unique_families: int = 0
    total_tracks: int = 0
    largest_family: int = 0
    largest_family_id: str | None = None
    families_over_cap: int = 0
    tracks_over_cap: int = 0
    #: family size -> how many families are that size.
    size_distribution: dict[int, int] = field(default_factory=dict)
    #: Share of tracks belonging to a family with more than one member.
    multi_member_share: float = 0.0
    #: Effective family count, as for any other concentration measure.
    effective_families: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "unique_families": self.unique_families,
            "total_tracks": self.total_tracks,
            "largest_family": self.largest_family,
            "largest_family_id": self.largest_family_id,
            "families_over_cap": self.families_over_cap,
            "tracks_over_cap": self.tracks_over_cap,
            "size_distribution": {str(k): v for k, v in sorted(self.size_distribution.items())},
            "multi_member_share": round(self.multi_member_share, 6),
            "effective_families": round(self.effective_families, 4),
        }


def family_pressure(families: dict[str, list[str]], *, cap: int) -> FamilyPressure:
    """Measure duplicate-family concentration.

    ``families`` maps a family id to the track ids in it. Solo tracks
    count as families of one — excluding them would make a dataset of
    entirely unique tracks look like it had no families at all, and the
    arithmetic would flatter every corpus that had been deduplicated.
    """
    result = FamilyPressure(unique_families=len(families))
    if not families:
        return result

    sizes = {name: len(members) for name, members in families.items()}
    result.total_tracks = sum(sizes.values())
    largest = max(sizes.items(), key=lambda item: (item[1], item[0]))
    result.largest_family_id, result.largest_family = largest

    for size in sizes.values():
        result.size_distribution[size] = result.size_distribution.get(size, 0) + 1

    result.families_over_cap = sum(1 for size in sizes.values() if size > cap)
    result.tracks_over_cap = sum(max(0, size - cap) for size in sizes.values())
    result.multi_member_share = (
        sum(size for size in sizes.values() if size > 1) / result.total_tracks
        if result.total_tracks
        else 0.0
    )
    hhi = sum((size / result.total_tracks) ** 2 for size in sizes.values())
    result.effective_families = 1.0 / hhi if hhi > 0 else 0.0
    return result
