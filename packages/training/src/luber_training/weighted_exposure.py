"""Making per-sample weights actually change what the trainer sees.

Phase 37 and Phase 38 both computed per-window sampling weights, wrote
them down as evidence, and then trained without them. The installed
``PreprocessedDataModule`` builds its loader with ``shuffle=True`` and no
sampler, and its dataset yields no weight field, so nothing downstream
could have used them. Every weighted-exposure claim in those phases was
therefore a claim about a file, not about training.

This module closes that gap and nothing else. It turns a weight per
sample into an explicit visiting order, which a sampler can hand to a
DataLoader in place of shuffling.

Two properties matter more than cleverness here:

**Deterministic.** The same weights and the same seed give the same
order, every time, on any machine. That is why this allocates exposure
arithmetically — largest-remainder, the method used to apportion seats —
rather than drawing from ``WeightedRandomSampler``. A random draw gives
the right *expectation* and a different epoch every run, which is not
something a controlled experiment can rest on.

**No silent fallback.** A sample with no weight raises. A weight that is
negative, infinite or NaN raises. The failure mode this replaces was
weights that were quietly ignored, and swapping it for weights that are
quietly defaulted would repeat the mistake in a new place.

Nothing here judges audio or decides what a weight should be. It takes
the weights it is given and makes them real.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Every sample is visited at least this many times per epoch. Without a
#: floor, a low-weighted sample in a large set rounds to zero visits and
#: silently leaves training — which is a dataset change disguised as a
#: weighting, and not one anybody authorised.
MINIMUM_OCCURRENCES = 1


class WeightedExposureError(ValueError):
    """Raised when weights cannot be turned into an honest exposure plan."""


@dataclass(frozen=True)
class ExposurePlan:
    """The order samples will be visited in, and how it was arrived at.

    ``order`` is the epoch: one entry per visit, already shuffled. A name
    appearing three times is visited three times.
    """

    order: tuple[str, ...]
    repeats: dict[str, int]
    weights: dict[str, float]
    seed: int
    minimum_applied: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.order)

    @property
    def exposure(self) -> dict[str, float]:
        """Share of the epoch each sample receives."""
        total = len(self.order) or 1
        return {name: count / total for name, count in self.repeats.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "luber-exposure-plan/1",
            "epoch_length": len(self.order),
            "sample_count": len(self.repeats),
            "seed": self.seed,
            "minimum_occurrences": MINIMUM_OCCURRENCES,
            "minimum_applied": list(self.minimum_applied),
            "repeats": dict(sorted(self.repeats.items())),
            "weights": {name: round(value, 6) for name, value in sorted(self.weights.items())},
            "exposure": {name: round(value, 6) for name, value in sorted(self.exposure.items())},
            "note": (
                "Visits per sample for one epoch, allocated by largest remainder from the "
                "given weights and then ordered with a seeded shuffle. Deterministic: the "
                "same weights and seed reproduce this exactly."
            ),
        }


def _validate(names: Sequence[str], weights: Mapping[str, float]) -> dict[str, float]:
    if not names:
        raise WeightedExposureError("no samples to plan exposure for")
    duplicates = len(names) - len(set(names))
    if duplicates:
        raise WeightedExposureError(f"{duplicates} duplicate sample name(s); names must be unique")

    missing = [name for name in names if name not in weights]
    if missing:
        raise WeightedExposureError(
            f"{len(missing)} sample(s) have no weight, e.g. {missing[:3]}. "
            "Refusing to default them: an unweighted sample is a question for the caller"
        )

    checked: dict[str, float] = {}
    for name in names:
        value = float(weights[name])
        if not math.isfinite(value) or value <= 0.0:
            raise WeightedExposureError(
                f"{name}: weight {weights[name]!r} is not a positive finite number"
            )
        checked[name] = value
    return checked


def _allocate(names: Sequence[str], weights: Mapping[str, float], length: int) -> dict[str, int]:
    """Largest-remainder apportionment of ``length`` visits.

    Floors first, then hands the leftover visits to the largest fractional
    parts. Ties break on the name so two runs never disagree.
    """
    total_weight = sum(weights[name] for name in names)
    quotas = {name: weights[name] / total_weight * length for name in names}
    counts = {name: max(MINIMUM_OCCURRENCES, int(quotas[name])) for name in names}

    # The floor plus the minimum rarely lands on `length` exactly, so the
    # difference is settled deterministically rather than left to drift.
    remainder = length - sum(counts.values())
    if remainder > 0:
        ranked = sorted(names, key=lambda name: (-(quotas[name] - int(quotas[name])), name))
        for index in range(remainder):
            counts[ranked[index % len(ranked)]] += 1
    elif remainder < 0:
        # Only take back from samples that are above the floor, so the
        # floor stays a floor.
        ranked = sorted(names, key=lambda name: (-counts[name], name))
        taken = 0
        while taken < -remainder:
            progressed = False
            for name in ranked:
                if taken >= -remainder:
                    break
                if counts[name] > MINIMUM_OCCURRENCES:
                    counts[name] -= 1
                    taken += 1
                    progressed = True
            if not progressed:
                raise WeightedExposureError(
                    f"cannot fit {len(names)} sample(s) into an epoch of {length} "
                    f"with a floor of {MINIMUM_OCCURRENCES} visit(s) each"
                )
    return counts


def build_exposure_plan(
    names: Sequence[str],
    weights: Mapping[str, float],
    *,
    seed: int,
    length: int | None = None,
) -> ExposurePlan:
    """Turn weights into a deterministic visiting order.

    ``length`` defaults to one visit per sample, so a weighted epoch costs
    the same as an unweighted one and the weighting changes *which*
    samples fill it rather than how long it is.
    """
    checked = _validate(names, weights)
    epoch = len(names) if length is None else int(length)
    if epoch < len(names) * MINIMUM_OCCURRENCES:
        raise WeightedExposureError(
            f"epoch of {epoch} cannot give {len(names)} sample(s) "
            f"{MINIMUM_OCCURRENCES} visit(s) each"
        )

    counts = _allocate(list(names), checked, epoch)

    total_weight = sum(checked.values())
    floored = tuple(
        name
        for name in sorted(names)
        if counts[name] == MINIMUM_OCCURRENCES
        and checked[name] / total_weight * epoch < MINIMUM_OCCURRENCES
    )

    order: list[str] = []
    for name in sorted(names):
        order.extend([name] * counts[name])
    random.Random(seed).shuffle(order)

    return ExposurePlan(
        order=tuple(order),
        repeats=dict(sorted(counts.items())),
        weights=dict(sorted(checked.items())),
        seed=seed,
        minimum_applied=floored,
    )


class DeterministicWeightedSampler:
    """A torch-compatible sampler that yields dataset indices in plan order.

    Deliberately not a subclass of ``torch.utils.data.Sampler``: this
    package must import without torch, and a DataLoader only requires
    ``__iter__`` and ``__len__``.
    """

    def __init__(self, plan: ExposurePlan, index_of: Mapping[str, int]) -> None:
        missing = [name for name in plan.repeats if name not in index_of]
        if missing:
            raise WeightedExposureError(
                f"{len(missing)} planned sample(s) are absent from the dataset, "
                f"e.g. {missing[:3]}. Refusing to skip them silently"
            )
        self.plan = plan
        self._indices = tuple(index_of[name] for name in plan.order)

    def __iter__(self) -> Iterator[int]:
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


def load_window_weights(path: Path) -> dict[str, float]:
    """Read ``sampling_weights`` out of a ``windows_{split}.json`` document.

    The weights the dataset build already writes. Reading them here is
    the whole point: until now nothing did.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    weights = document.get("sampling_weights")
    if not isinstance(weights, dict) or not weights:
        raise WeightedExposureError(
            f"{path} carries no sampling_weights; there is nothing to enforce"
        )
    return {str(key): float(value) for key, value in weights.items()}


__all__ = [
    "MINIMUM_OCCURRENCES",
    "DeterministicWeightedSampler",
    "ExposurePlan",
    "WeightedExposureError",
    "build_exposure_plan",
    "load_window_weights",
]
