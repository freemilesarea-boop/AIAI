"""The dataset profile: everything measurable, measured once.

Built in a single pass and handed to every later stage, so findings,
scoring and reporting all argue from the same numbers rather than each
recomputing them slightly differently.

Two decisions worth stating.

*The profile is computed over a stated population.* Curation cares about
what would actually be trained on, so the default population is the
training-eligible subset — but the full-corpus profile is computed too,
because "the eligible half looks balanced" and "the corpus looks
balanced" are different claims and confusing them hides why tracks were
excluded.

*Nothing here decides anything.* The profile has no thresholds and no
opinions; it reports. Findings live in :mod:`findings`, and keeping the
two apart is what lets a target profile change without recomputing a
single distribution.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from luber_dataset.factory.intelligence import concentration as conc
from luber_dataset.factory.intelligence import distributions as dist
from luber_dataset.factory.intelligence.schemas import (
    COMPLETENESS_DIMENSIONS,
    Observation,
    Source,
    TrackView,
)

#: Technical metrics summarised for outlier detection. Measurement is
#: Phase 22/23's job; this only asks how the corpus is spread across it.
TECHNICAL_METRICS: tuple[str, ...] = (
    "integrated_lufs",
    "true_peak_dbtp",
    "crest_factor_db",
    "clipping_sample_ratio",
    "silence_ratio",
    "stereo_width",
    "phase_correlation",
    "dynamic_range_proxy_db",
    "high_frequency_cutoff_hz",
    "spectral_centroid_hz",
)

#: Provenance source types that describe machine-made audio. Used only
#: for the synthetic share; never inferred from the audio itself.
SYNTHETIC_SOURCE_TYPES: frozenset[str] = frozenset({"AI_GENERATED", "SELF_MODEL_OUTPUT"})


@dataclass
class DatasetProfile:
    """Every distribution the manifest supports, over one population."""

    population: str
    track_count: int = 0
    total_hours: float = 0.0

    categorical: dict[str, dist.CategoricalDistribution] = field(default_factory=dict)
    numeric: dict[str, dist.NumericSummary] = field(default_factory=dict)
    concentration: dict[str, conc.ConcentrationMetrics] = field(default_factory=dict)
    concentration_by_duration: dict[str, conc.ConcentrationMetrics] = field(default_factory=dict)
    long_tail: dict[str, conc.LongTail] = field(default_factory=dict)
    completeness: dict[str, dist.CompletenessScore] = field(default_factory=dict)
    family_pressure: conc.FamilyPressure = field(default_factory=conc.FamilyPressure)
    synthetic_share_by_count: float = 0.0
    synthetic_share_by_duration: float = 0.0

    def share(self, dimension: str, category: str) -> float:
        distribution = self.categorical.get(dimension)
        return distribution.share(category) if distribution else 0.0

    def coverage(self, dimension: str) -> float:
        distribution = self.categorical.get(dimension)
        return distribution.coverage if distribution else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "population": self.population,
            "track_count": self.track_count,
            "total_hours": round(self.total_hours, 4),
            "completeness": {
                name: score.to_dict() for name, score in sorted(self.completeness.items())
            },
            "categorical": {
                name: value.to_dict() for name, value in sorted(self.categorical.items())
            },
            "numeric": {name: value.to_dict() for name, value in sorted(self.numeric.items())},
            "concentration": {
                name: value.to_dict() for name, value in sorted(self.concentration.items())
            },
            "concentration_by_duration": {
                name: value.to_dict()
                for name, value in sorted(self.concentration_by_duration.items())
            },
            "long_tail": {name: value.to_dict() for name, value in sorted(self.long_tail.items())},
            "family_pressure": self.family_pressure.to_dict(),
            "synthetic_share_by_count": round(self.synthetic_share_by_count, 6),
            "synthetic_share_by_duration": round(self.synthetic_share_by_duration, 6),
        }


def _always(value: Any, source: str = Source.MEASURED.value) -> Observation:
    """Wrap a field that is always present, so it joins the same machinery."""
    return Observation(value=value, source=source)


def build(
    tracks: list[TrackView],
    *,
    population: str,
    duplicate_family_cap: int = 1,
    head_cumulative: float = 0.5,
    mid_cumulative: float = 0.9,
) -> DatasetProfile:
    """Profile one population of tracks."""
    profile = DatasetProfile(population=population, track_count=len(tracks))
    profile.total_hours = sum(track.hours for track in tracks)

    # ── categorical ──────────────────────────────────────────────────
    definitions: dict[str, Callable[[TrackView], Observation]] = {
        "quality_tier": lambda t: _always(t.quality_tier),
        "split": lambda t: _always(t.split),
        "rights_status": lambda t: _always(t.rights_status),
        "commercial_training_allowed": lambda t: _always(t.commercial_training_allowed),
        "source_type": lambda t: _always(t.source_type),
        "source_reference": lambda t: (
            _always(t.source_reference) if t.source_reference else Observation(value=None)
        ),
        "language": lambda t: t.language(),
        "vocal_class": lambda t: t.vocal_class(),
        "vocal_gender": lambda t: t.vocal_gender(),
        "artist": lambda t: t.artist(),
        "album": lambda t: t.album(),
        "genre": lambda t: t.genre(),
        "subgenre": lambda t: t.subgenre(),
        "mood": lambda t: t.mood(),
        "mode": lambda t: t.mode(),
        "key": lambda t: t.key(),
        "sample_rate": lambda t: t.sample_rate(),
        "channels": lambda t: t.channels(),
        "duplicate_family": lambda t: _always(t.duplicate_family),
    }
    for name, accessor in definitions.items():
        profile.categorical[name] = dist.categorical(tracks, name, accessor)

    # Derived buckets. Duration is always known; tempo is gated, so a
    # low-confidence estimate lands in `unknown` rather than a bucket.
    profile.categorical["duration_bucket"] = dist.categorical(
        tracks,
        "duration_bucket",
        lambda t: (
            _always(dist.duration_bucket(t.duration_seconds))
            if t.duration_seconds > 0
            else Observation(value=None)
        ),
    )
    profile.categorical["tempo_bucket"] = dist.categorical(
        tracks,
        "tempo_bucket",
        lambda t: _bucketed_tempo(t),
    )

    # ── numeric ──────────────────────────────────────────────────────
    profile.numeric["duration_seconds"] = dist.numeric(
        tracks,
        "duration_seconds",
        lambda t: _always(t.duration_seconds) if t.duration_seconds > 0 else Observation(None),
    )
    profile.numeric["bpm"] = dist.numeric(tracks, "bpm", lambda t: t.bpm())
    profile.numeric["quality_score"] = dist.numeric(
        tracks, "quality_score", lambda t: _always(t.quality_score)
    )
    for metric in TECHNICAL_METRICS:
        profile.numeric[metric] = dist.numeric(tracks, metric, _technical_accessor(metric))

    # ── concentration ────────────────────────────────────────────────
    for name in (
        "artist",
        "album",
        "source_reference",
        "source_type",
        "duplicate_family",
        "genre",
        "language",
    ):
        distribution = profile.categorical[name]
        profile.concentration[name] = conc.measure(distribution)
        profile.concentration_by_duration[name] = conc.measure(distribution, by_duration=True)
        profile.long_tail[name] = conc.long_tail(
            distribution, head_cumulative=head_cumulative, mid_cumulative=mid_cumulative
        )

    # ── completeness ─────────────────────────────────────────────────
    for name in COMPLETENESS_DIMENSIONS:
        categorical_view = profile.categorical.get(name)
        if categorical_view is None:
            summary = profile.numeric.get(name)
            if summary is None:
                continue
            profile.completeness[name] = dist.CompletenessScore(
                dimension=name,
                known=summary.known_count,
                unknown=summary.unknown_count,
                low_confidence=summary.low_confidence_count,
            )
            continue
        profile.completeness[name] = dist.CompletenessScore(
            dimension=name,
            known=categorical_view.known_count,
            unknown=categorical_view.unknown_count,
            low_confidence=categorical_view.low_confidence_count,
        )

    # ── duplicate families ───────────────────────────────────────────
    families: dict[str, list[str]] = {}
    for track in tracks:
        families.setdefault(track.duplicate_family, []).append(track.track_id)
    profile.family_pressure = conc.family_pressure(families, cap=duplicate_family_cap)

    # ── synthetic share, from provenance only ────────────────────────
    synthetic = [t for t in tracks if t.source_type in SYNTHETIC_SOURCE_TYPES]
    if tracks:
        profile.synthetic_share_by_count = len(synthetic) / len(tracks)
    if profile.total_hours > 0:
        profile.synthetic_share_by_duration = sum(t.hours for t in synthetic) / profile.total_hours
    return profile


def _technical_accessor(name: str) -> Callable[[TrackView], Observation]:
    """Bind a metric name without a late-binding closure over the loop."""

    def accessor(track: TrackView) -> Observation:
        return track.technical(name)

    return accessor


def _bucketed_tempo(track: TrackView) -> Observation:
    """A tempo bucket, or an honest unknown.

    A low-confidence tempo keeps its LOW_CONFIDENCE source so the
    distribution counts it separately — visible in the profile, absent
    from every share.
    """
    observation = track.bpm()
    if observation.value is None:
        return Observation(value=None)
    if not observation.known:
        return Observation(
            value=None, source=Source.LOW_CONFIDENCE.value, confidence=observation.confidence
        )
    return Observation(
        value=dist.tempo_bucket(float(observation.value)),
        source=observation.source,
        confidence=observation.confidence,
    )
