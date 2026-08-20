"""What a dataset is supposed to look like — declared, never assumed.

A gap only exists relative to a goal. Without a stated target, "40%
Korean" is a measurement and nothing else; it is neither good nor bad,
and an engine that called it a gap would be inventing an objective
nobody set.

So target profiles are explicit, versioned, hashable data. The default
one deliberately sets almost no targets: it detects *severe
concentration* — one artist owning the corpus, one duplicate family
dominating — and leaves everything else alone. Music datasets are not
supposed to be uniform, and a neutral profile that pushed toward
equality would be a strong opinion wearing a neutral name.

Named profiles exist for real product intentions. Each one is only
allowed to constrain dimensions the manifest can actually measure, and
:func:`validate` refuses a profile that constrains anything else —
because a target on a field that is 95% unknown produces confident
findings computed from almost nothing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

TARGET_PROFILE_SCHEMA_VERSION = "luber-target-profile/1"

#: Dimensions a profile may constrain. Deliberately closed: each of
#: these is either measured by Phase 23 or read from operator metadata,
#: and a target on anything else could not be evaluated.
CONSTRAINABLE_DIMENSIONS: frozenset[str] = frozenset(
    {
        "language",
        "vocal_class",
        "quality_tier",
        "tempo_bucket",
        "duration_bucket",
        "genre",
        "source_type",
        "mode",
    }
)


@dataclass(frozen=True)
class Range:
    """A share the dataset should sit inside.

    All three are optional because they answer different questions.
    ``minimum`` is a floor worth acquiring toward; ``maximum`` is a
    ceiling worth downsampling toward; ``target`` is where to aim when
    choosing between otherwise equal tracks.
    """

    minimum: float | None = None
    target: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        for name in ("minimum", "target", "maximum"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a share in [0, 1], got {value}")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError(f"minimum {self.minimum} exceeds maximum {self.maximum}")
        for bound, comparison in (("minimum", "below"), ("maximum", "above")):
            edge = getattr(self, bound)
            if self.target is None or edge is None:
                continue
            if (bound == "minimum" and self.target < edge) or (
                bound == "maximum" and self.target > edge
            ):
                raise ValueError(f"target {self.target} is {comparison} {bound} {edge}")

    def contains(self, share: float) -> bool:
        if self.minimum is not None and share < self.minimum:
            return False
        return not (self.maximum is not None and share > self.maximum)

    def to_dict(self) -> dict[str, Any]:
        return {"min": self.minimum, "target": self.target, "max": self.maximum}


@dataclass(frozen=True)
class ConcentrationLimits:
    """Ceilings on how concentrated the dataset may be.

    These are the neutral profile's entire content. Unlike a share
    target they encode no aesthetic preference — a dataset where one
    artist holds 80% of the hours is a problem whatever the intended
    style, because the model will learn that artist.
    """

    max_artist_share: float = 0.25
    max_album_share: float = 0.20
    max_source_reference_share: float = 0.40
    max_source_type_share: float = 0.90
    max_duplicate_family_share: float = 0.05
    #: Below this many effective artists, the corpus behaves as though
    #: it came from a handful of people however many it names.
    min_effective_artists: float = 4.0
    #: Share of the training selection that may be synthetic. Applied
    #: only to provenance-declared origin, never inferred from audio.
    max_synthetic_share: float = 0.50

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SelectionLimits:
    """Caps the downsampler enforces. Conservative by default."""

    max_tracks_per_artist: int | None = None
    max_hours_per_artist: float | None = None
    max_tracks_per_album: int | None = None
    #: The one cap that is on by default. A family is versions of the
    #: same content; four of them teach that content four times.
    max_records_per_duplicate_family: int = 1
    #: Whether a filename may stand in for a missing artist. Off: a
    #: filename is not an identity, and inferring one would apply artist
    #: caps to tracks that share nothing but a naming convention.
    infer_artist_from_filename: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TargetProfile:
    """A named, hashable statement of intent."""

    name: str
    description: str = ""
    schema_version: str = TARGET_PROFILE_SCHEMA_VERSION
    #: dimension -> {category: Range}
    shares: dict[str, dict[str, Range]] = field(default_factory=dict)
    concentration: ConcentrationLimits = field(default_factory=ConcentrationLimits)
    selection: SelectionLimits = field(default_factory=SelectionLimits)
    #: Minimum coverage a dimension needs before its targets are
    #: evaluated at all. A target on 5%-known genre data would generate
    #: confident findings from almost nothing.
    min_coverage_to_evaluate: float = 0.60

    def range_for(self, dimension: str, category: str) -> Range | None:
        return self.shares.get(dimension, {}).get(category)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "schema_version": self.schema_version,
            "min_coverage_to_evaluate": self.min_coverage_to_evaluate,
            "shares": {
                dimension: {
                    category: bounds.to_dict() for category, bounds in sorted(categories.items())
                }
                for dimension, categories in sorted(self.shares.items())
            },
            "concentration": self.concentration.to_dict(),
            "selection": self.selection.to_dict(),
        }

    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ProfileError(ValueError):
    """Raised when a profile could not be evaluated as written."""


def validate(profile: TargetProfile) -> None:
    """Refuse a profile that cannot mean what it says.

    Two failures, both of which would otherwise produce findings that
    look authoritative and are not: constraining a dimension nothing
    measures, and minimums across one dimension summing past 1.0.

    A third check was tried and removed — "if every declared category
    has a maximum they must sum to at least 1.0". It assumed the
    declared categories exhaust the dimension, and they routinely do
    not: KOREAN_POP constrains ``vocal_class`` for VOCAL alone and says
    nothing about INSTRUMENTAL, which is a coherent profile the rule
    rejected. A validation rule that fails legitimate input is worse
    than the absent check, because it pushes people toward writing
    profiles that satisfy the validator rather than their intent.
    """
    unknown = sorted(set(profile.shares) - CONSTRAINABLE_DIMENSIONS)
    if unknown:
        raise ProfileError(
            f"profile {profile.name!r} constrains dimension(s) the manifest does not "
            f"provide: {', '.join(unknown)}. "
            f"Constrainable: {', '.join(sorted(CONSTRAINABLE_DIMENSIONS))}"
        )

    for dimension, categories in profile.shares.items():
        minimums = sum(r.minimum for r in categories.values() if r.minimum is not None)
        if minimums > 1.0 + 1e-9:
            raise ProfileError(
                f"profile {profile.name!r}: minimums for {dimension} sum to "
                f"{minimums:.2f}, which no dataset can satisfy"
            )


def neutral() -> TargetProfile:
    """The default. Detects domination; imposes no aesthetic.

    It states no share targets at all. Everything it says is of the form
    "no single artist should own a quarter of this corpus", which is
    true regardless of what the corpus is for.
    """
    return TargetProfile(
        name="NEUTRAL",
        description=(
            "Detects severe concentration without imposing a stylistic target. "
            "Music datasets are not meant to be uniform, and a profile that "
            "pushed toward equality would be an opinion wearing a neutral name."
        ),
    )


def korean_pop() -> TargetProfile:
    """A Korean-leaning vocal corpus.

    Constrains only language, vocal class and quality — the three
    dimensions the manifest measures or the operator declares reliably.
    Genre is *not* constrained, because genre coverage in this project
    depends entirely on operator sidecars and is typically sparse; a
    target on it would fail :func:`validate`'s coverage gate anyway.
    """
    return TargetProfile(
        name="KOREAN_POP",
        description="Korean-leaning contemporary vocal material with an English minority.",
        shares={
            "language": {
                "ko": Range(minimum=0.30, target=0.45, maximum=0.60),
                "en": Range(minimum=0.15, target=0.30, maximum=0.50),
            },
            "vocal_class": {"VOCAL": Range(minimum=0.60, maximum=0.90)},
            "quality_tier": {"A": Range(minimum=0.40), "REJECT": Range(maximum=0.0)},
        },
    )


def instrumental() -> TargetProfile:
    return TargetProfile(
        name="INSTRUMENTAL",
        description="Instrumental-dominant corpus for non-vocal generation.",
        shares={
            "vocal_class": {
                "INSTRUMENTAL": Range(minimum=0.70, target=0.85),
                "VOCAL": Range(maximum=0.30),
            },
            "quality_tier": {"REJECT": Range(maximum=0.0)},
        },
    )


def global_pop() -> TargetProfile:
    return TargetProfile(
        name="GLOBAL_POP",
        description="Language-diverse contemporary vocal material.",
        shares={
            "language": {
                "en": Range(minimum=0.25, target=0.40, maximum=0.60),
                "ko": Range(minimum=0.10, target=0.20, maximum=0.40),
            },
            "vocal_class": {"VOCAL": Range(minimum=0.60)},
            "quality_tier": {"REJECT": Range(maximum=0.0)},
        },
    )


#: Profiles whose required dimensions the manifest can actually supply.
BUILT_IN: dict[str, Callable[[], TargetProfile]] = {
    "NEUTRAL": neutral,
    "KOREAN_POP": korean_pop,
    "GLOBAL_POP": global_pop,
    "INSTRUMENTAL": instrumental,
}

#: Profiles the product will eventually want and the data cannot yet
#: support, with what each one is waiting on. Listed rather than
#: half-implemented: a profile that silently evaluated against absent
#: annotations would report success it had not earned.
FUTURE_PROFILES: dict[str, str] = {
    "VOCAL_QUALITY": (
        "needs per-track vocal quality annotation — timbre, pitch stability, "
        "sibilance. Phase 23 measures none of these, and the Phase 20H "
        "listening rubric that does is a human process, not a manifest field."
    ),
    "KOREAN_LYRICS": (
        "needs verified Korean lyrics with section structure. Lyrics are "
        "operator-supplied and optional today, and no transcript exists at all."
    ),
    "MODERN_NON_TROT": (
        "needs a reliable trot/non-trot style label. Nothing in the manifest "
        "distinguishes them; the P20 benchmark scores it by ear."
    ),
}


def load(payload: dict[str, Any]) -> TargetProfile:
    """Build a profile from JSON, rejecting anything unrecognised.

    Unknown keys are an error rather than an ignored no-op: a misspelled
    dimension that silently does nothing is a target the operator
    believes they set and did not.
    """
    if not isinstance(payload, dict):
        raise ProfileError("a target profile must be a JSON object")

    permitted = {
        "name",
        "description",
        "schema_version",
        "shares",
        "concentration",
        "selection",
        "min_coverage_to_evaluate",
    }
    unknown = sorted(set(payload) - permitted)
    if unknown:
        raise ProfileError(
            f"unrecognised profile key(s): {', '.join(unknown)}. "
            f"Permitted: {', '.join(sorted(permitted))}"
        )

    version = str(payload.get("schema_version", TARGET_PROFILE_SCHEMA_VERSION))
    if version != TARGET_PROFILE_SCHEMA_VERSION:
        raise ProfileError(
            f"profile schema version {version!r} is not {TARGET_PROFILE_SCHEMA_VERSION!r}"
        )

    shares: dict[str, dict[str, Range]] = {}
    raw_shares = payload.get("shares") or {}
    if not isinstance(raw_shares, dict):
        raise ProfileError("`shares` must be an object")
    for dimension, categories in raw_shares.items():
        if not isinstance(categories, dict):
            raise ProfileError(f"`shares.{dimension}` must be an object")
        shares[str(dimension)] = {
            str(category): Range(
                minimum=_share(bounds, "min", dimension, category),
                target=_share(bounds, "target", dimension, category),
                maximum=_share(bounds, "max", dimension, category),
            )
            for category, bounds in categories.items()
        }

    profile = TargetProfile(
        name=str(payload.get("name", "CUSTOM")),
        description=str(payload.get("description", "")),
        shares=shares,
        concentration=ConcentrationLimits(**(payload.get("concentration") or {})),
        selection=SelectionLimits(**(payload.get("selection") or {})),
        min_coverage_to_evaluate=float(payload.get("min_coverage_to_evaluate", 0.60)),
    )
    validate(profile)
    return profile


def _share(bounds: Any, key: str, dimension: str, category: str) -> float | None:
    if not isinstance(bounds, dict):
        raise ProfileError(f"`shares.{dimension}.{category}` must be an object")
    value = bounds.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise ProfileError(f"`shares.{dimension}.{category}.{key}` must be a number")
    return float(value)


def by_name(name: str) -> TargetProfile:
    key = name.strip().upper()
    if key in BUILT_IN:
        profile: TargetProfile = BUILT_IN[key]()
        validate(profile)
        return profile
    if key in FUTURE_PROFILES:
        raise ProfileError(f"profile {key!r} is not available yet: {FUTURE_PROFILES[key]}")
    raise ProfileError(f"unknown profile {name!r}. Available: {', '.join(sorted(BUILT_IN))}")
