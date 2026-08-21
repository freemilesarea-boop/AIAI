"""What "normal" is compared against, and the two ways to say it.

A baseline is a reference value for one metric in one segment, with the
window it was measured over and the number of samples behind it. The
sample count is not decoration: comparing today against a baseline built
from eleven requests is comparing against noise, and a baseline that did
not carry its count would let that happen silently.

Two kinds, and they answer different questions.

**Rolling** — the previous seven days, ending some distance before the
current window. It answers "is this different from recently?" and it is
what runs by default. The lag matters: a rolling baseline that ran right
up to the current window would absorb the first hours of a regression
into "normal" and then report that nothing had changed.

**Frozen** — a fixed interval somebody chose and named: a known-good
week, the revision before a rollout, a release candidate. It answers "is
this different from *that*?", it never moves, and nothing recomputes it.
It exists because after a bad deploy the rolling baseline is
contaminated by definition, and the only useful comparison is against a
period an operator can point at.

A frozen baseline is immutable. Recomputing one on new data would make
every historical comparison against it unreproducible, which is the one
thing a reference point may not do.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta
from enum import StrEnum
from typing import Any

from luber_inference_observability.aggregation import Aggregate, Distribution, Rate
from luber_inference_observability.dimensions import Segment
from luber_inference_observability.versions import AGGREGATION_VERSION
from luber_inference_observability.windows import TimeWindow


class BaselineKind(StrEnum):
    ROLLING = "ROLLING"
    FROZEN = "FROZEN"


class BaselineStatus(StrEnum):
    """Whether this baseline may be compared against."""

    READY = "READY"
    #: Not enough history yet. A new provider revision starts here, and
    #: staying silent is the correct behaviour rather than a gap.
    BASELINE_BUILDING = "BASELINE_BUILDING"
    #: Nothing at all in the reference window.
    NO_DATA = "NO_DATA"


#: How long a rolling baseline looks back.
DEFAULT_BASELINE_SPAN = timedelta(days=7)

#: How far before the current window a rolling baseline stops.
#:
#: One hour, so a regression that started in the current window has not
#: already taught the baseline that it is normal. Longer would make the
#: baseline stale after a legitimate change; shorter lets contamination
#: in. It is configurable per policy for the cases where neither default
#: is right.
DEFAULT_BASELINE_GAP = timedelta(hours=1)

#: Below this a baseline is BASELINE_BUILDING rather than READY.
#: A rate estimated from fewer samples than this has a confidence
#: interval wider than most of the regressions worth finding.
DEFAULT_MINIMUM_BASELINE_SAMPLES = 50


@dataclass(frozen=True)
class Baseline:
    """One metric's reference value, with everything needed to argue with it."""

    metric: str
    segment: Segment
    window: TimeWindow
    kind: str = BaselineKind.ROLLING.value
    #: For a rate: the proportion. For a latency metric: the quantile
    #: named by `quantile_fraction`.
    value: float | None = None
    numerator: int | None = None
    denominator: int | None = None
    sample_count: int = 0
    quantile_fraction: float | None = None
    #: Set only for a frozen baseline somebody named.
    label: str | None = None
    aggregation_version: str = AGGREGATION_VERSION
    minimum_samples: int = DEFAULT_MINIMUM_BASELINE_SAMPLES

    @property
    def status(self) -> str:
        if self.sample_count == 0 or self.value is None:
            return BaselineStatus.NO_DATA.value
        if self.sample_count < self.minimum_samples:
            return BaselineStatus.BASELINE_BUILDING.value
        return BaselineStatus.READY.value

    @property
    def ready(self) -> bool:
        return self.status == BaselineStatus.READY.value

    def frozen(self, label: str) -> Baseline:
        """Take a permanent copy under a name. The original is untouched."""
        return replace(self, kind=BaselineKind.FROZEN.value, label=label)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "segment": self.segment.to_dict(),
            "window": self.window.to_dict(),
            "kind": self.kind,
            "label": self.label,
            "value": None if self.value is None else round(self.value, 6),
            "numerator": self.numerator,
            "denominator": self.denominator,
            "sample_count": self.sample_count,
            "quantile_fraction": self.quantile_fraction,
            "status": self.status,
            "minimum_samples": self.minimum_samples,
            "aggregation_version": self.aggregation_version,
        }


def from_rate(
    rate: Rate,
    *,
    segment: Segment,
    window: TimeWindow,
    kind: str = BaselineKind.ROLLING.value,
    minimum_samples: int = DEFAULT_MINIMUM_BASELINE_SAMPLES,
) -> Baseline:
    """A baseline for a proportion, keeping the counts behind it."""
    return Baseline(
        metric=rate.name,
        segment=segment,
        window=window,
        kind=kind,
        value=rate.value,
        numerator=rate.numerator,
        denominator=rate.denominator,
        sample_count=rate.denominator,
        minimum_samples=minimum_samples,
    )


def from_distribution(
    distribution: Distribution,
    *,
    segment: Segment,
    window: TimeWindow,
    fraction: float = 0.95,
    kind: str = BaselineKind.ROLLING.value,
    minimum_samples: int = DEFAULT_MINIMUM_BASELINE_SAMPLES,
) -> Baseline:
    """A baseline for a latency quantile.

    The quantile is named in the baseline, so a comparison cannot
    accidentally hold P95 against P50 — which would look like a
    catastrophic regression on a perfectly healthy system.
    """
    return Baseline(
        metric=distribution.name,
        segment=segment,
        window=window,
        kind=kind,
        value=distribution.at(fraction),
        sample_count=distribution.count,
        quantile_fraction=fraction,
        minimum_samples=minimum_samples,
    )


def rolling_window(
    current: TimeWindow,
    *,
    span: timedelta = DEFAULT_BASELINE_SPAN,
    gap: timedelta = DEFAULT_BASELINE_GAP,
) -> TimeWindow:
    """The reference interval for a rolling comparison against *current*."""
    return current.preceding(span, gap=gap)


def baselines_from(
    aggregate: Aggregate,
    *,
    metrics: tuple[str, ...],
    latency_fraction: float = 0.95,
    kind: str = BaselineKind.ROLLING.value,
    minimum_samples: int = DEFAULT_MINIMUM_BASELINE_SAMPLES,
) -> dict[str, Baseline]:
    """Extract baselines for several metrics from one aggregated window."""
    out: dict[str, Baseline] = {}
    for metric in metrics:
        if metric in aggregate.rates:
            out[metric] = from_rate(
                aggregate.rates[metric],
                segment=aggregate.segment,
                window=aggregate.window,
                kind=kind,
                minimum_samples=minimum_samples,
            )
        elif metric in aggregate.distributions:
            out[metric] = from_distribution(
                aggregate.distributions[metric],
                segment=aggregate.segment,
                window=aggregate.window,
                fraction=latency_fraction,
                kind=kind,
                minimum_samples=minimum_samples,
            )
        elif metric in aggregate.averages:
            average = aggregate.averages[metric]
            out[metric] = Baseline(
                metric=metric,
                segment=aggregate.segment,
                window=aggregate.window,
                kind=kind,
                value=average.value,
                sample_count=average.count,
                minimum_samples=minimum_samples,
            )
    return out


__all__ = [
    "DEFAULT_BASELINE_GAP",
    "DEFAULT_BASELINE_SPAN",
    "DEFAULT_MINIMUM_BASELINE_SAMPLES",
    "Baseline",
    "BaselineKind",
    "BaselineStatus",
    "baselines_from",
    "from_distribution",
    "from_rate",
    "rolling_window",
]
