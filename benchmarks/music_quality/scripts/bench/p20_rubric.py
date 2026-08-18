"""Phase 20 rubric and taxonomy, as data the tool can enforce.

Kept separate from the v1 vocabulary in ``scoring.py`` rather than
replacing it. The v1 rubric is frozen against everything already scored
with it; editing it in place would silently redefine those results. Two
vocabularies coexisting is the honest cost of having scored anything at
all.

Every name here is transcribed from ``listening/RUBRIC_P20.md`` and
``listening/TAXONOMY.md``. Those documents are the specification; this
module must not drift from them, and a test asserts it does not.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Dimensions every case is scored on, whatever it is.
UNIVERSAL_DIMENSIONS: tuple[str, ...] = (
    # COMPOSITION
    "melody_quality",
    "harmonic_coherence",
    "phrasing",
    "hook_strength",
    "commercial_plausibility",
    # ARRANGEMENT
    "section_structure",
    "transitions",
    "energy_progression",
    "repetition_control",
    # INSTRUMENT QUALITY
    "timbral_realism",
    "instrument_definition",
    "transient_quality",
    "separation",
    "production_resolution",
    # MIX / SONICS
    "frequency_balance",
    "low_mid_clarity",
    "presence",
    "high_frequency_detail",
    "harshness",
    "sibilance",
    "stereo_image",
    "depth",
    "ambience",
    "dynamics",
    # OVERALL
    "listenability",
    "commercial_readiness",
    "overall_preference",
)

#: Asked only where there is a voice to judge.
VOCAL_DIMENSIONS: tuple[str, ...] = (
    "vocal_naturalness",
    "vocal_timbre",
    "pitch_stability",
    "vocal_phrasing",
    "emotional_appropriateness",
    # VOCAL STYLE — the trot measurement lives here
    "trot_absence",
    "vibrato_control",
    "ornament_appropriateness",
    "genre_appropriateness",
)

#: Asked only for Korean vocal cases, and scored against the expected
#: lyrics displayed beside the player.
KOREAN_DIMENSIONS: tuple[str, ...] = (
    "pronunciation",
    "lyric_completeness",
    "syllable_timing",
    "phrase_omission",
    "segmentation_naturalness",
)

#: Asked only where there is enough music for drift to show.
LONG_FORM_DIMENSIONS: tuple[str, ...] = ("long_form_coherence",)

#: Below this, "does it hold together over time" is not a real question.
LONG_FORM_MIN_SECONDS = 100

ALL_DIMENSIONS: tuple[str, ...] = (
    UNIVERSAL_DIMENSIONS + VOCAL_DIMENSIONS + KOREAN_DIMENSIONS + LONG_FORM_DIMENSIONS
)

#: Transcribed from TAXONOMY.md. Unknown tags are rejected: a typo that
#: silently becomes a new category would make tag frequencies useless.
ARTIFACT_TAGS: tuple[str, ...] = (
    "VOCAL_SYNTHETIC",
    "VOCAL_TROT_STYLE",
    "VOCAL_EXCESSIVE_VIBRATO",
    "VOCAL_PITCH_INSTABILITY",
    "KOREAN_PRONUNCIATION",
    "KOREAN_LYRIC_OMISSION",
    "KOREAN_SYLLABLE_TIMING",
    "LYRIC_PHRASE_SKIPPED",
    "INSTRUMENT_SYNTHETIC",
    "INSTRUMENT_BLUR",
    "TRANSIENT_WEAK",
    "ARRANGEMENT_COLLAPSE",
    "LONG_FORM_DRIFT",
    "REPETITION_EXCESS",
    "LOW_END_EXCESS",
    "LOW_MID_MUD",
    "MID_HOLLOW",
    "PRESENCE_EXCESS",
    "PRESENCE_DEFICIT",
    "HIGH_FREQUENCY_DEFICIT",
    "HIGH_FREQUENCY_EXCESS",
    "HARSHNESS",
    "SIBILANCE",
    "STEREO_NARROW",
    "STEREO_UNSTABLE",
    "DEPTH_FLAT",
    "REVERB_UNNATURAL",
    "MELODY_WEAK",
    "MELODY_TROT_LIKE",
    "GENRE_MISMATCH",
    "EARLY_FADE",
    "ABRUPT_END",
)

#: Groups, purely for laying the form out in a readable order.
DIMENSION_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Composition", UNIVERSAL_DIMENSIONS[0:5]),
    ("Arrangement", UNIVERSAL_DIMENSIONS[5:9] + LONG_FORM_DIMENSIONS),
    ("Instrument quality", UNIVERSAL_DIMENSIONS[9:14]),
    ("Vocal quality", VOCAL_DIMENSIONS[0:5]),
    ("Vocal style", VOCAL_DIMENSIONS[5:9]),
    ("Korean vocal", KOREAN_DIMENSIONS),
    ("Mix / sonics", UNIVERSAL_DIMENSIONS[14:24]),
    ("Overall", UNIVERSAL_DIMENSIONS[24:27]),
)


class P20ScoreError(ValueError):
    """A submitted score set the rubric does not accept."""


def expected_dimensions(
    *, instrumental: bool, korean: bool, duration_seconds: float
) -> tuple[str, ...]:
    """Exactly what this case should be scored on.

    Asking for vocal dimensions on an instrumental is how a rubric
    collects meaningless numbers: the listener either invents one or
    enters 5, and both distort the average more than an absent score
    would.
    """
    dimensions = list(UNIVERSAL_DIMENSIONS)
    if not instrumental:
        dimensions += list(VOCAL_DIMENSIONS)
        if korean:
            dimensions += list(KOREAN_DIMENSIONS)
    if duration_seconds >= LONG_FORM_MIN_SECONDS:
        dimensions += list(LONG_FORM_DIMENSIONS)
    return tuple(dimensions)


def validate_scores(
    scores: Mapping[str, Any],
    *,
    instrumental: bool,
    korean: bool,
    duration_seconds: float,
) -> dict[str, int]:
    """Accept a complete, in-range score set, or explain the refusal."""
    required = set(
        expected_dimensions(
            instrumental=instrumental, korean=korean, duration_seconds=duration_seconds
        )
    )

    unknown = sorted(set(scores) - set(ALL_DIMENSIONS))
    if unknown:
        raise P20ScoreError(f"unknown dimension(s): {', '.join(unknown)}")

    not_applicable = sorted(set(scores) - required)
    if not_applicable:
        raise P20ScoreError(
            f"dimension(s) that do not apply to this case: {', '.join(not_applicable)}"
        )

    missing = sorted(required - set(scores))
    if missing:
        raise P20ScoreError(f"missing score(s): {', '.join(missing)}")

    cleaned: dict[str, int] = {}
    for name, value in scores.items():
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise P20ScoreError(f"{name}: not a number") from exc
        if not 1 <= number <= 10:
            raise P20ScoreError(f"{name}: {number} is outside 1-10")
        cleaned[name] = number
    return cleaned


def validate_tags(tags: list[str]) -> list[str]:
    unknown = sorted(set(tags) - set(ARTIFACT_TAGS))
    if unknown:
        raise P20ScoreError(f"unknown artifact tag(s): {', '.join(unknown)}")
    # Deduplicated but order-stable, so a frequency count is honest.
    seen: list[str] = []
    for tag in tags:
        if tag not in seen:
            seen.append(tag)
    return seen
