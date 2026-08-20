"""What a manifest record actually offers, and how to ask for it.

Every dimension Phase 24 analyses is read through :class:`TrackView`,
which returns an :class:`Observation` — a value together with whether it
is *known*. That is not ceremony. The failure this whole layer exists to
avoid is reporting "pop = 60%" when only a tenth of the dataset carries
a genre label, and the only reliable way to prevent it is to make
"unknown" impossible to drop on the floor.

The accessor also absorbs a fact discovered by auditing the real
manifest rather than assuming its shape: **artist, album, genre,
subgenre and mood are not first-class fields.** They exist only inside
``metadata.sidecar`` when an operator wrote one, or inside
``metadata.embedded_tags`` when the container carried them. Phase 23 was
right not to promote them — nobody verified an ID3 frame — so Phase 24
reads them from where they live and records which source they came from.

Two dimensions are gated on confidence rather than presence. Phase 23
reports a tempo for almost anything, including material with no pulse;
the fixture used to audit this schema came back at 50.17 BPM with 0.70
confidence. A number attached to a confidence is not yet a measurement,
so the gate is configurable and closed by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

#: Bumped when a curated record's *shape* changes.
CURATION_SCHEMA_VERSION = "luber-dataset-curation/1"
#: Bumped when a scoring or selection *algorithm* changes in a way that
#: would give a different answer for the same manifest.
CURATION_ENGINE_VERSION = "luber-dataset-curation/1.0.0"


class Source(StrEnum):
    """Where an observed value came from. Never collapsed away."""

    #: The operator wrote it in a sidecar.
    USER = "USER"
    #: A container tag. Real, and nobody verified it.
    EMBEDDED = "EMBEDDED"
    #: Measured by Phase 23.
    MEASURED = "MEASURED"
    #: Measured, but below the confidence this analysis requires.
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    #: Nobody said and nothing measured it.
    NONE = "NONE"


@dataclass(frozen=True)
class Observation:
    """A value that knows whether it is real."""

    value: Any
    source: str = Source.NONE.value
    confidence: float | None = None

    @property
    def known(self) -> bool:
        """Present, and trusted enough to count toward a denominator."""
        return self.value is not None and self.source not in (
            Source.NONE.value,
            Source.LOW_CONFIDENCE.value,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "source": self.source, "confidence": self.confidence}


UNKNOWN = Observation(value=None, source=Source.NONE.value)


class CurationAction(StrEnum):
    """What the plan proposes for one track.

    Deliberately no DELETE. The factory never removes source audio, and
    a vocabulary that could express it would eventually be used.
    """

    KEEP = "KEEP"
    #: Kept, and preferred when a sampler has to choose.
    KEEP_PRIORITY = "KEEP_PRIORITY"
    #: Kept in the dataset, dropped from this training selection because
    #: its region is overrepresented.
    DOWNSAMPLE = "DOWNSAMPLE"
    #: Deliberately withheld from training.
    HOLDOUT = "HOLDOUT"
    #: A human decides.
    REVIEW = "REVIEW"
    #: Barred by a policy that curation may not override — rights above
    #: all.
    EXCLUDE_POLICY = "EXCLUDE_POLICY"
    #: Dropped because its duplicate family is over its cap.
    EXCLUDE_DUPLICATE_PRESSURE = "EXCLUDE_DUPLICATE_PRESSURE"


#: Actions whose tracks appear in a training selection.
SELECTED_ACTIONS: frozenset[str] = frozenset(
    {CurationAction.KEEP.value, CurationAction.KEEP_PRIORITY.value}
)


class Severity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Finding:
    """One thing worth knowing about the dataset, with its evidence."""

    code: str
    severity: str
    dimension: str
    detail: str
    #: The measurement behind the claim.
    current_share: float | None = None
    target_range: tuple[float, float] | None = None
    threshold: float | None = None
    affected_tracks: int = 0
    affected_hours: float = 0.0
    recommended_action: str = ""
    #: Denominator the share was computed over, so a finding drawn from
    #: 10% of the dataset cannot be read as describing all of it.
    known_denominator: int | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "dimension": self.dimension,
            "detail": self.detail,
            "current_share": self.current_share,
            "target_range": list(self.target_range) if self.target_range else None,
            "threshold": self.threshold,
            "affected_tracks": self.affected_tracks,
            "affected_hours": round(self.affected_hours, 4),
            "recommended_action": self.recommended_action,
            "known_denominator": self.known_denominator,
            "evidence": self.evidence,
        }


class TrackView:
    """Read one manifest record without guessing at anything.

    A thin, total accessor: every dimension has a method, every method
    returns an :class:`Observation`, and nothing raises on a record that
    is missing a section.
    """

    def __init__(self, record: dict[str, Any], *, min_confidence: float = 0.0) -> None:
        self._record = record
        self._min_confidence = min_confidence

    # ── identity ─────────────────────────────────────────────────────
    @property
    def track_id(self) -> str:
        return str(self._record.get("track_id", ""))

    @property
    def raw(self) -> dict[str, Any]:
        return self._record

    def _section(self, name: str) -> dict[str, Any]:
        value = self._record.get(name)
        return value if isinstance(value, dict) else {}

    @property
    def sidecar(self) -> dict[str, Any]:
        value = self._section("metadata").get("sidecar")
        return value if isinstance(value, dict) else {}

    @property
    def embedded(self) -> dict[str, Any]:
        value = self._section("metadata").get("embedded_tags")
        return value if isinstance(value, dict) else {}

    # ── measured, always present ─────────────────────────────────────
    @property
    def duration_seconds(self) -> float:
        for section in ("analysis", "audio"):
            value = self._section(section).get("duration_seconds")
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
        return 0.0

    @property
    def hours(self) -> float:
        return self.duration_seconds / 3600.0

    @property
    def split(self) -> str:
        return str(self._record.get("split", "EXCLUDED"))

    @property
    def training_eligible(self) -> bool:
        return bool(self._section("eligibility").get("training_eligible", False))

    @property
    def eligibility_reasons(self) -> list[str]:
        reasons = self._section("eligibility").get("eligibility_reasons")
        return [str(r) for r in reasons] if isinstance(reasons, list) else []

    @property
    def quality_tier(self) -> str:
        return str(self._section("quality").get("quality_tier", "REJECT"))

    @property
    def quality_flags(self) -> list[str]:
        flags = self._section("quality").get("quality_flags")
        return [str(f) for f in flags] if isinstance(flags, list) else []

    @property
    def quality_score(self) -> float:
        value = self._section("quality").get("quality_score")
        return float(value) if isinstance(value, (int, float)) else 0.0

    # ── rights: never an Observation, because it is never optional ───
    @property
    def training_permitted(self) -> bool:
        return bool(self._section("provenance").get("training_permitted", False))

    @property
    def commercial_training_allowed(self) -> str:
        return str(self._section("provenance").get("commercial_training_allowed", "UNKNOWN"))

    @property
    def rights_status(self) -> str:
        return str(self._section("provenance").get("rights_status", "UNKNOWN"))

    @property
    def hard_blocks(self) -> list[str]:
        blocks = self._section("provenance").get("hard_blocks")
        return [str(b) for b in blocks] if isinstance(blocks, list) else []

    @property
    def source_type(self) -> str:
        return str(self._section("provenance").get("source_type", "UNKNOWN"))

    @property
    def source_reference(self) -> str:
        return str(self._section("provenance").get("source_reference", "") or "")

    # ── dedup ────────────────────────────────────────────────────────
    @property
    def duplicate_group_id(self) -> str | None:
        value = self._section("dedup").get("duplicate_group_id")
        return str(value) if value else None

    @property
    def dedup_decision(self) -> str:
        return str(self._section("dedup").get("dedup_decision", "KEEP"))

    @property
    def duplicate_family(self) -> str:
        """The family this track belongs to, falling back to itself.

        A track with no duplicate group is a family of one. Treating it
        as "no family" would exclude it from family-pressure arithmetic
        and quietly understate how concentrated the dataset is.
        """
        return self.duplicate_group_id or f"solo:{self.track_id}"

    # ── observations: value plus whether it is known ─────────────────
    def language(self) -> Observation:
        block = self._section("metadata").get("language")
        block = block if isinstance(block, dict) else {}
        value = block.get("language")
        source = str(block.get("language_source", Source.NONE.value))
        if not value or value == "unknown" or source == Source.NONE.value:
            return UNKNOWN
        confidence = block.get("language_confidence")
        return Observation(
            value=str(value),
            source=source,
            confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
        )

    def vocal_class(self) -> Observation:
        block = self._section("vocals")
        value = str(block.get("vocal_class", "UNCERTAIN"))
        if value == "UNCERTAIN":
            return UNKNOWN
        confidence = block.get("vocal_confidence")
        return Observation(
            value=value,
            source=str(block.get("vocal_source", Source.NONE.value)),
            confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
        )

    def vocal_gender(self) -> Observation:
        block = self._section("vocals")
        value = block.get("vocal_gender")
        if not value:
            return UNKNOWN
        return Observation(value=str(value), source=str(block.get("vocal_gender_source", "USER")))

    def bpm(self, *, min_confidence: float | None = None) -> Observation:
        """Tempo, gated on confidence.

        The gate is the point. Phase 23 will report a tempo for material
        with no pulse at all, and a low-confidence number recorded as
        fact would put ambient tracks into a tempo bucket and skew every
        distribution built on it.
        """
        floor = self._min_confidence if min_confidence is None else min_confidence
        block = self._section("music")
        value = block.get("bpm")
        if not isinstance(value, (int, float)):
            return UNKNOWN
        confidence = block.get("bpm_confidence")
        confidence = float(confidence) if isinstance(confidence, (int, float)) else 0.0
        if confidence < floor:
            return Observation(
                value=float(value), source=Source.LOW_CONFIDENCE.value, confidence=confidence
            )
        return Observation(value=float(value), source=Source.MEASURED.value, confidence=confidence)

    def key(self, *, min_confidence: float | None = None) -> Observation:
        floor = self._min_confidence if min_confidence is None else min_confidence
        block = self._section("music")
        value = block.get("key")
        if not value:
            return UNKNOWN
        confidence = block.get("key_confidence")
        confidence = float(confidence) if isinstance(confidence, (int, float)) else 0.0
        if confidence < floor:
            return Observation(
                value=str(value), source=Source.LOW_CONFIDENCE.value, confidence=confidence
            )
        return Observation(value=str(value), source=Source.MEASURED.value, confidence=confidence)

    def mode(self, *, min_confidence: float | None = None) -> Observation:
        floor = self._min_confidence if min_confidence is None else min_confidence
        block = self._section("music")
        value = block.get("mode")
        if not value:
            return UNKNOWN
        confidence = block.get("key_confidence")
        confidence = float(confidence) if isinstance(confidence, (int, float)) else 0.0
        if confidence < floor:
            return Observation(
                value=str(value), source=Source.LOW_CONFIDENCE.value, confidence=confidence
            )
        return Observation(value=str(value), source=Source.MEASURED.value, confidence=confidence)

    def _tagged(self, name: str) -> Observation:
        """A field that exists only in a sidecar or a container tag.

        Sidecar wins: an operator statement outranks whatever a tag
        happens to say, and the two disagreeing is itself worth seeing.
        """
        value = self.sidecar.get(name)
        if isinstance(value, str) and value.strip():
            return Observation(value=value.strip(), source=Source.USER.value, confidence=1.0)
        value = self.embedded.get(name)
        if isinstance(value, str) and value.strip():
            return Observation(value=value.strip(), source=Source.EMBEDDED.value, confidence=0.5)
        return UNKNOWN

    def artist(self) -> Observation:
        return self._tagged("artist")

    def album(self) -> Observation:
        return self._tagged("album")

    def genre(self) -> Observation:
        return self._tagged("genre")

    def subgenre(self) -> Observation:
        return self._tagged("subgenre")

    def mood(self) -> Observation:
        return self._tagged("mood")

    def title(self) -> Observation:
        return self._tagged("title")

    def metadata_conflict(self) -> bool:
        """Sidecar and container tag disagreeing about the same field."""
        for name in ("artist", "album", "genre", "title"):
            declared = self.sidecar.get(name)
            tagged = self.embedded.get(name)
            if (
                isinstance(declared, str)
                and isinstance(tagged, str)
                and declared.strip().casefold() != tagged.strip().casefold()
            ):
                return True
        return False

    # ── numeric technical metrics ────────────────────────────────────
    def technical(self, name: str) -> Observation:
        value = self._section("analysis").get(name)
        if not isinstance(value, (int, float)):
            return UNKNOWN
        return Observation(value=float(value), source=Source.MEASURED.value)

    def sample_rate(self) -> Observation:
        for section in ("analysis", "audio"):
            value = self._section(section).get("sample_rate")
            if isinstance(value, int) and value > 0:
                return Observation(value=value, source=Source.MEASURED.value)
        return UNKNOWN

    def channels(self) -> Observation:
        for section in ("analysis", "audio"):
            value = self._section(section).get("channels")
            if isinstance(value, int) and value > 0:
                return Observation(value=value, source=Source.MEASURED.value)
        return UNKNOWN


#: Dimensions the metadata completeness scorecard reports on. Chosen
#: because each one either gates an analysis or drives a curation
#: decision; a field nothing depends on does not need a score.
COMPLETENESS_DIMENSIONS: tuple[str, ...] = (
    "artist",
    "album",
    "genre",
    "subgenre",
    "mood",
    "language",
    "vocal_class",
    "bpm",
    "key",
    "mode",
)
