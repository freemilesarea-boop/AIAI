"""Turning a profile into findings, and refusing to invent goals.

Two rules govern everything here.

**A gap requires a target.** "Only 12% Korean" is a measurement. It
becomes a gap when a profile says Korean should be at least 30%, and not
before. An engine that generated gaps from a neutral profile would be
asserting objectives nobody set, and every one of them would look like
analysis.

**A finding requires coverage.** A target on a dimension that is 5%
known cannot be evaluated — the share is computed from almost nothing,
and reporting it as a gap gives a confident answer built on twenty
tracks. Below the profile's coverage floor the dimension yields a single
`NOT_ASSESSABLE` finding saying exactly why, which is more useful than a
number that would mislead.

Concentration findings are the exception to the first rule, and
deliberately so: one artist owning a quarter of the corpus is a problem
for any training objective, so the neutral profile raises those and
nothing else.
"""

from __future__ import annotations

from luber_dataset.factory.intelligence import targets as target_module
from luber_dataset.factory.intelligence.distributions import TEMPO_BUCKETS
from luber_dataset.factory.intelligence.profile import DatasetProfile
from luber_dataset.factory.intelligence.schemas import Finding, Severity

# ── codes ────────────────────────────────────────────────────────────
ONE_ARTIST_DOMINATES = "ONE_ARTIST_DOMINATES"
ONE_SOURCE_DOMINATES = "ONE_SOURCE_DOMINATES"
ONE_GENRE_DOMINATES = "ONE_GENRE_DOMINATES"
ONE_LANGUAGE_DOMINATES = "ONE_LANGUAGE_DOMINATES"
DUPLICATE_FAMILY_DOMINATES = "DUPLICATE_FAMILY_DOMINATES"
LOW_EFFECTIVE_ARTIST_COUNT = "LOW_EFFECTIVE_ARTIST_COUNT"
EXCESSIVE_SYNTHETIC_SHARE = "EXCESSIVE_SYNTHETIC_SHARE"

TOO_MANY_FROM_ARTIST = "TOO_MANY_FROM_ARTIST"
TOO_MANY_FROM_SOURCE = "TOO_MANY_FROM_SOURCE"
GENRE_OVERREPRESENTED = "GENRE_OVERREPRESENTED"
LANGUAGE_OVERREPRESENTED = "LANGUAGE_OVERREPRESENTED"
DURATION_BUCKET_OVERREPRESENTED = "DURATION_BUCKET_OVERREPRESENTED"
DUPLICATE_FAMILY_OVERREPRESENTED = "DUPLICATE_FAMILY_OVERREPRESENTED"
TEMPO_NARROWNESS = "TEMPO_NARROWNESS"
TEMPO_GAPS = "TEMPO_GAPS"
SHORT_TRACK_DOMINANCE = "SHORT_TRACK_DOMINANCE"
LONG_TRACK_DOMINANCE = "LONG_TRACK_DOMINANCE"
VOCAL_DOMINANCE = "VOCAL_DOMINANCE"
INSTRUMENTAL_DOMINANCE = "INSTRUMENTAL_DOMINANCE"
VOCAL_CLASS_UNKNOWN_HIGH = "VOCAL_CLASS_UNKNOWN_HIGH"
METADATA_COVERAGE_LOW = "METADATA_COVERAGE_LOW"
NOT_ASSESSABLE = "NOT_ASSESSABLE"
KEY_CONCENTRATION = "KEY_CONCENTRATION"

#: Every tempo bucket a track could land in, including the open-ended
#: top one, so "which buckets are empty" has something to subtract from.
ALL_TEMPO_BUCKET_LABELS: tuple[str, ...] = (
    *(label for _, label in TEMPO_BUCKETS),
    ">180",
)


#: Gap codes are generated from the profile's own dimension names, so a
#: custom profile constraining `genre` produces NEED_MORE_GENRE_<x>
#: without any code here having to anticipate it.
def gap_code(dimension: str, category: str) -> str:
    return f"NEED_MORE_{dimension.upper()}_{str(category).upper()}"


def over_code(dimension: str) -> str:
    return {
        "language": LANGUAGE_OVERREPRESENTED,
        "genre": GENRE_OVERREPRESENTED,
        "duration_bucket": DURATION_BUCKET_OVERREPRESENTED,
    }.get(dimension, f"{dimension.upper()}_OVERREPRESENTED")


def _hours_for(profile: DatasetProfile, dimension: str, category: str) -> tuple[int, float]:
    distribution = profile.categorical.get(dimension)
    if distribution is None:
        return 0, 0.0
    for bucket in distribution.buckets:
        if bucket.label == category:
            return bucket.count, bucket.hours
    return 0, 0.0


def evaluate(profile: DatasetProfile, target: target_module.TargetProfile) -> list[Finding]:
    """Every finding the profile and the target together support."""
    findings: list[Finding] = []
    findings.extend(_concentration_findings(profile, target))
    findings.extend(_target_findings(profile, target))
    findings.extend(_informational_findings(profile))
    # Sorted so two runs over the same data produce the same list, and
    # so the worst news is at the top where it will be read.
    order = {Severity.CRITICAL.value: 0, Severity.WARNING.value: 1, Severity.INFO.value: 2}
    findings.sort(key=lambda f: (order.get(f.severity, 3), f.code, f.dimension))
    return findings


def _concentration_findings(
    profile: DatasetProfile, target: target_module.TargetProfile
) -> list[Finding]:
    """Domination, which is a problem for any objective."""
    findings: list[Finding] = []
    limits = target.concentration

    checks = (
        ("artist", limits.max_artist_share, ONE_ARTIST_DOMINATES, TOO_MANY_FROM_ARTIST),
        ("album", limits.max_album_share, "ONE_ALBUM_DOMINATES", "TOO_MANY_FROM_ALBUM"),
        (
            "source_reference",
            limits.max_source_reference_share,
            ONE_SOURCE_DOMINATES,
            TOO_MANY_FROM_SOURCE,
        ),
        ("source_type", limits.max_source_type_share, ONE_SOURCE_DOMINATES, TOO_MANY_FROM_SOURCE),
        ("genre", None, ONE_GENRE_DOMINATES, GENRE_OVERREPRESENTED),
        ("language", None, ONE_LANGUAGE_DOMINATES, LANGUAGE_OVERREPRESENTED),
    )

    for dimension, ceiling, dominate_code, _over in checks:
        if ceiling is None:
            continue
        metrics = profile.concentration.get(dimension)
        distribution = profile.categorical.get(dimension)
        if metrics is None or distribution is None or metrics.known_count == 0:
            continue
        if distribution.coverage < target.min_coverage_to_evaluate:
            continue
        if metrics.top1_share > ceiling and metrics.top1_label is not None:
            count, hours = _hours_for(profile, dimension, metrics.top1_label)
            findings.append(
                Finding(
                    code=dominate_code,
                    severity=(
                        Severity.CRITICAL.value
                        if metrics.top1_share > ceiling * 1.5
                        else Severity.WARNING.value
                    ),
                    dimension=dimension,
                    detail=(
                        f"{metrics.top1_label!r} holds {metrics.top1_share:.1%} of known "
                        f"{dimension} values, above the {ceiling:.0%} ceiling"
                    ),
                    current_share=metrics.top1_share,
                    threshold=ceiling,
                    affected_tracks=count,
                    affected_hours=hours,
                    recommended_action=(
                        f"downsample {metrics.top1_label!r} or acquire other {dimension} values"
                    ),
                    known_denominator=metrics.known_count,
                    evidence={
                        "top1_share_by_duration": profile.concentration_by_duration[
                            dimension
                        ].top1_share,
                        "effective_categories": metrics.effective_categories,
                        "coverage": distribution.coverage,
                    },
                )
            )

    artists = profile.concentration.get("artist")
    artist_distribution = profile.categorical.get("artist")
    if (
        artists is not None
        and artist_distribution is not None
        and artists.known_count > 0
        and artist_distribution.coverage >= target.min_coverage_to_evaluate
        and artists.effective_categories < limits.min_effective_artists
    ):
        findings.append(
            Finding(
                code=LOW_EFFECTIVE_ARTIST_COUNT,
                severity=Severity.CRITICAL.value,
                dimension="artist",
                detail=(
                    f"the corpus names {artists.category_count} artists but behaves as "
                    f"though it has {artists.effective_categories:.1f}"
                ),
                current_share=None,
                threshold=limits.min_effective_artists,
                affected_tracks=artists.known_count,
                recommended_action="acquire material from more distinct artists",
                known_denominator=artists.known_count,
                evidence={"hhi": artists.hhi, "normalized_entropy": artists.normalized_entropy},
            )
        )

    pressure = profile.family_pressure
    if pressure.total_tracks > 0 and pressure.largest_family > 1:
        share = pressure.largest_family / pressure.total_tracks
        if share > limits.max_duplicate_family_share:
            findings.append(
                Finding(
                    code=DUPLICATE_FAMILY_DOMINATES,
                    severity=Severity.WARNING.value,
                    dimension="duplicate_family",
                    detail=(
                        f"family {pressure.largest_family_id!r} holds "
                        f"{pressure.largest_family} of {pressure.total_tracks} tracks "
                        f"({share:.1%})"
                    ),
                    current_share=share,
                    threshold=limits.max_duplicate_family_share,
                    affected_tracks=pressure.largest_family,
                    recommended_action="cap records per duplicate family",
                    known_denominator=pressure.total_tracks,
                    evidence=pressure.to_dict(),
                )
            )

    if profile.synthetic_share_by_count > limits.max_synthetic_share:
        findings.append(
            Finding(
                code=EXCESSIVE_SYNTHETIC_SHARE,
                severity=Severity.WARNING.value,
                dimension="source_type",
                detail=(
                    f"{profile.synthetic_share_by_count:.1%} of tracks are declared "
                    f"synthetic, above the {limits.max_synthetic_share:.0%} ceiling"
                ),
                current_share=profile.synthetic_share_by_count,
                threshold=limits.max_synthetic_share,
                recommended_action="acquire human-produced material",
                known_denominator=profile.track_count,
                evidence={"by_duration": profile.synthetic_share_by_duration},
            )
        )
    return findings


def _target_findings(profile: DatasetProfile, target: target_module.TargetProfile) -> list[Finding]:
    """Gaps and overrepresentation, strictly against declared ranges."""
    findings: list[Finding] = []

    for dimension, categories in sorted(target.shares.items()):
        distribution = profile.categorical.get(dimension)
        if distribution is None:
            continue
        if distribution.known_count == 0:
            findings.append(
                Finding(
                    code=NOT_ASSESSABLE,
                    severity=Severity.WARNING.value,
                    dimension=dimension,
                    detail=(
                        f"the profile constrains {dimension} and no track has a known value for it"
                    ),
                    recommended_action=f"supply {dimension} metadata before targeting it",
                    known_denominator=0,
                )
            )
            continue
        if distribution.coverage < target.min_coverage_to_evaluate:
            findings.append(
                Finding(
                    code=NOT_ASSESSABLE,
                    severity=Severity.WARNING.value,
                    dimension=dimension,
                    detail=(
                        f"{dimension} is known for only {distribution.coverage:.1%} of "
                        f"tracks, below the {target.min_coverage_to_evaluate:.0%} floor "
                        f"this profile requires; its targets were not evaluated"
                    ),
                    current_share=distribution.coverage,
                    threshold=target.min_coverage_to_evaluate,
                    affected_tracks=distribution.unknown_count,
                    affected_hours=distribution.unknown_hours,
                    recommended_action=(
                        f"raise {dimension} coverage, or lower min_coverage_to_evaluate "
                        f"and accept a share computed from {distribution.known_count} tracks"
                    ),
                    known_denominator=distribution.known_count,
                )
            )
            continue

        for category, bounds in sorted(categories.items()):
            share = distribution.share(category)
            count, hours = _hours_for(profile, dimension, category)

            if bounds.minimum is not None and share < bounds.minimum:
                deficit = bounds.minimum - share
                findings.append(
                    Finding(
                        code=gap_code(dimension, category),
                        severity=(
                            Severity.CRITICAL.value
                            if share < bounds.minimum * 0.5
                            else Severity.WARNING.value
                        ),
                        dimension=dimension,
                        detail=(
                            f"{dimension}={category} is {share:.1%} of known values, below "
                            f"the {bounds.minimum:.0%} minimum"
                        ),
                        current_share=share,
                        target_range=(bounds.minimum, bounds.maximum or 1.0),
                        affected_tracks=count,
                        affected_hours=hours,
                        recommended_action=f"acquire more {dimension}={category} material",
                        known_denominator=distribution.known_count,
                        evidence={
                            "deficit_share": round(deficit, 6),
                            "share_by_duration": distribution.share_by_duration(category),
                        },
                    )
                )

            if bounds.maximum is not None and share > bounds.maximum:
                findings.append(
                    Finding(
                        code=over_code(dimension),
                        severity=(
                            Severity.CRITICAL.value
                            if share > min(1.0, bounds.maximum * 1.5)
                            else Severity.WARNING.value
                        ),
                        dimension=dimension,
                        detail=(
                            f"{dimension}={category} is {share:.1%} of known values, above "
                            f"the {bounds.maximum:.0%} maximum"
                        ),
                        current_share=share,
                        target_range=(bounds.minimum or 0.0, bounds.maximum),
                        affected_tracks=count,
                        affected_hours=hours,
                        recommended_action=(
                            f"downsample {dimension}={category}, or acquire other categories"
                        ),
                        known_denominator=distribution.known_count,
                        evidence={
                            "excess_share": round(share - bounds.maximum, 6),
                            "share_by_duration": distribution.share_by_duration(category),
                        },
                    )
                )
    return findings


def _informational_findings(profile: DatasetProfile) -> list[Finding]:
    """Things worth knowing that no target has to justify.

    Reported at INFO unless the dataset is unusable for a reason that
    holds regardless of intent — a corpus that is 90% unknown vocal
    class cannot support a vocal target whoever writes one.
    """
    findings: list[Finding] = []

    for name, score in sorted(profile.completeness.items()):
        if score.total == 0 or score.completeness >= 0.5:
            continue
        findings.append(
            Finding(
                code=METADATA_COVERAGE_LOW,
                severity=Severity.INFO.value,
                dimension=name,
                detail=(
                    f"{name} is unknown for {score.missing_percentage:.0f}% of tracks "
                    f"({score.unknown} of {score.total})"
                    + (
                        f"; a further {score.low_confidence} were measured below the "
                        "confidence gate"
                        if score.low_confidence
                        else ""
                    )
                ),
                current_share=score.completeness,
                affected_tracks=score.unknown,
                recommended_action=(
                    f"no {name}-based analysis can be trusted until coverage improves"
                ),
                known_denominator=score.known,
            )
        )

    vocals = profile.categorical.get("vocal_class")
    if vocals is not None and vocals.total_count > 0:
        unknown_share = vocals.unknown_count / vocals.total_count
        if unknown_share > 0.5:
            findings.append(
                Finding(
                    code=VOCAL_CLASS_UNKNOWN_HIGH,
                    severity=Severity.WARNING.value,
                    dimension="vocal_class",
                    detail=(
                        f"vocal class is UNCERTAIN for {unknown_share:.0%} of tracks; "
                        "no validated detector exists, so this needs operator metadata"
                    ),
                    current_share=unknown_share,
                    affected_tracks=vocals.unknown_count,
                    recommended_action="declare vocal_type in sidecars",
                    known_denominator=vocals.known_count,
                )
            )
        elif vocals.known_count:
            dominance = (
                ("VOCAL", VOCAL_DOMINANCE),
                ("INSTRUMENTAL", INSTRUMENTAL_DOMINANCE),
            )
            for label, code in dominance:
                share = vocals.share(label)
                if share > 0.95:
                    findings.append(
                        Finding(
                            code=code,
                            severity=Severity.INFO.value,
                            dimension="vocal_class",
                            detail=f"{share:.0%} of classified tracks are {label}",
                            current_share=share,
                            known_denominator=vocals.known_count,
                            recommended_action="acquire the other class if the product needs it",
                        )
                    )

    tempo = profile.categorical.get("tempo_bucket")
    if tempo is not None and tempo.known_count >= 10:
        metrics_source = tempo.buckets
        occupied = len(metrics_source)
        if occupied <= 2:
            findings.append(
                Finding(
                    code=TEMPO_NARROWNESS,
                    severity=Severity.WARNING.value,
                    dimension="tempo_bucket",
                    detail=(
                        f"confidently-measured tempi occupy only {occupied} of 7 buckets: "
                        + ", ".join(b.label for b in metrics_source)
                    ),
                    affected_tracks=tempo.known_count,
                    recommended_action="acquire material at other tempi",
                    known_denominator=tempo.known_count,
                )
            )
        occupied_labels = {bucket.label for bucket in metrics_source}
        missing = [label for label in ALL_TEMPO_BUCKET_LABELS if label not in occupied_labels]
        if missing and occupied > 2:
            findings.append(
                Finding(
                    code=TEMPO_GAPS,
                    severity=Severity.INFO.value,
                    dimension="tempo_bucket",
                    detail="no confidently-measured tracks in: " + ", ".join(missing),
                    affected_tracks=0,
                    recommended_action="acquire material at these tempi if the product needs it",
                    known_denominator=tempo.known_count,
                    evidence={"missing_buckets": missing},
                )
            )

    durations = profile.categorical.get("duration_bucket")
    if durations is not None and durations.known_count >= 10:
        short = sum(b.share_by_count for b in durations.buckets if b.label in ("<30s", "30-60s"))
        long = sum(b.share_by_count for b in durations.buckets if b.label in ("240-360s", ">360s"))
        if short > 0.6:
            findings.append(
                Finding(
                    code=SHORT_TRACK_DOMINANCE,
                    severity=Severity.INFO.value,
                    dimension="duration_bucket",
                    detail=f"{short:.0%} of tracks are under a minute",
                    current_share=short,
                    known_denominator=durations.known_count,
                    recommended_action="confirm the training pipeline expects short material",
                )
            )
        if long > 0.6:
            findings.append(
                Finding(
                    code=LONG_TRACK_DOMINANCE,
                    severity=Severity.INFO.value,
                    dimension="duration_bucket",
                    detail=f"{long:.0%} of tracks are over four minutes",
                    current_share=long,
                    known_denominator=durations.known_count,
                    recommended_action=(
                        "long material usually needs segmenting; the factory does not "
                        "chunk audio and never mutates it"
                    ),
                )
            )

    key_distribution = profile.categorical.get("key")
    if key_distribution is not None and key_distribution.known_count >= 20:
        top = key_distribution.buckets[0] if key_distribution.buckets else None
        if top is not None and top.share_by_count > 0.4:
            findings.append(
                Finding(
                    code=KEY_CONCENTRATION,
                    severity=Severity.INFO.value,
                    dimension="key",
                    detail=(
                        f"{top.share_by_count:.0%} of confidently-detected keys are "
                        f"{top.label}; informational only, since natural corpora are "
                        "not uniform across keys"
                    ),
                    current_share=top.share_by_count,
                    known_denominator=key_distribution.known_count,
                    recommended_action="none unless a profile constrains key",
                )
            )
    return findings
