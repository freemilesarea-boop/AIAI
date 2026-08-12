"""Human evaluation rubric, artifact taxonomy, and blind A/B pairing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

#: Rubric dimensions, scored 1-10. Definitions live in listening/RUBRIC.md.
RUBRIC_DIMENSIONS: tuple[str, ...] = (
    "overall_musical_quality",
    "composition_melody",
    "harmony",
    "rhythm_groove",
    "arrangement",
    "song_structure",
    "vocal_naturalness",
    "vocal_tone",
    "lyrics_pronunciation",
    "lyrics_alignment",
    "prompt_adherence",
    "genre_authenticity",
    "mix_balance",
    "artifact_free",
    "commercial_release_readiness",
)

#: Dimensions that do not apply to instrumental tracks.
VOCAL_ONLY_DIMENSIONS: frozenset[str] = frozenset(
    {
        "vocal_naturalness",
        "vocal_tone",
        "lyrics_pronunciation",
        "lyrics_alignment",
    }
)

ARTIFACT_TAGS: tuple[str, ...] = (
    "VOCAL_ROBOTIC",
    "VOCAL_WOBBLE",
    "BAD_PRONUNCIATION",
    "LYRIC_OMISSION",
    "LYRIC_REPETITION",
    "MELODY_REPETITIVE",
    "STRUCTURE_COLLAPSE",
    "BAD_TRANSITION",
    "RHYTHM_DRIFT",
    "HARMONY_WEIRD",
    "INSTRUMENT_ARTIFACT",
    "MIX_MUDDY",
    "MIX_HARSH",
    "HIGH_FREQ_ARTIFACT",
    "LOW_END_PROBLEM",
    "UNNATURAL_REVERB",
    "UNWANTED_NOISE",
    "GENERIC_COMPOSITION",
    "PROMPT_MISS",
    "GENRE_MISS",
    "OTHER",
)

#: Internal quality gate. These are targets, not achievements, and are
#: never lowered to make a run pass.
QUALITY_TARGETS: dict[str, float] = {
    "overall_musical_quality": 8.0,
    "commercial_release_readiness": 8.0,
    "vocal_naturalness": 8.0,
    "lyrics_pronunciation": 8.0,
    "prompt_adherence": 8.0,
    "song_structure": 8.0,
}
MAX_TECHNICAL_FAILURE_RATE = 0.02
MAX_ARTIFACT_RATE = 0.10


class ScoreValidationError(Exception):
    """Raised when a submitted score set is not usable."""


def validate_scores(scores: dict[str, int], *, instrumental: bool = False) -> dict[str, int]:
    """Check a submitted score set against the rubric.

    Vocal dimensions are omitted (not zero-filled) for instrumentals, so
    they never drag down an average they have no business affecting.
    """
    if not scores:
        raise ScoreValidationError("no scores submitted")

    expected = set(RUBRIC_DIMENSIONS)
    if instrumental:
        expected -= VOCAL_ONLY_DIMENSIONS

    unknown = sorted(set(scores) - set(RUBRIC_DIMENSIONS))
    if unknown:
        raise ScoreValidationError(f"unknown rubric dimensions: {', '.join(unknown)}")

    if instrumental:
        forbidden = sorted(set(scores) & VOCAL_ONLY_DIMENSIONS)
        if forbidden:
            raise ScoreValidationError(
                f"vocal dimensions scored on an instrumental track: {', '.join(forbidden)}"
            )

    missing = sorted(expected - set(scores))
    if missing:
        raise ScoreValidationError(f"missing rubric dimensions: {', '.join(missing)}")

    for dimension, value in scores.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise ScoreValidationError(f"{dimension}: score must be an integer")
        if not 1 <= value <= 10:
            raise ScoreValidationError(f"{dimension}: score {value} outside 1-10")
    return scores


def validate_artifact_tags(tags: list[str]) -> list[str]:
    unknown = sorted(set(tags) - set(ARTIFACT_TAGS))
    if unknown:
        raise ScoreValidationError(f"unknown artifact tags: {', '.join(unknown)}")
    return tags


@dataclass(frozen=True)
class BlindPair:
    """One blind A/B comparison. Which side is A is deterministic but
    not guessable from the configuration, so an evaluator cannot infer
    the system from position."""

    pair_id: str
    track_a: str
    track_b: str

    def reveal(self, choice: str) -> str:
        """Map an A/B/tie choice back to a benchmark id (or 'tie')."""
        if choice == "A":
            return self.track_a
        if choice == "B":
            return self.track_b
        if choice.lower() == "tie":
            return "tie"
        raise ScoreValidationError(f"invalid choice: {choice!r} (expected A, B, or tie)")


def make_blind_pair(left: str, right: str, *, salt: str = "") -> BlindPair:
    """Deterministically randomize which candidate is presented as A.

    Deterministic so a pairing can be reproduced and audited; hashed so
    ordering carries no information about the configurations.
    """
    if left == right:
        raise ScoreValidationError("cannot compare a track against itself")
    ordered = sorted((left, right))
    digest = hashlib.sha256(f"{salt}|{ordered[0]}|{ordered[1]}".encode()).hexdigest()
    swap = int(digest[:8], 16) % 2 == 1
    a, b = (ordered[1], ordered[0]) if swap else (ordered[0], ordered[1])
    return BlindPair(pair_id=digest[:16], track_a=a, track_b=b)


def meets_targets(averages: dict[str, float]) -> dict[str, bool]:
    """Compare measured averages against the internal quality gate."""
    return {
        dimension: averages.get(dimension, 0.0) >= target
        for dimension, target in QUALITY_TARGETS.items()
    }
