"""Song structure, advanced musical parameters, and pre-flight advisories.

This module is the single source of truth for everything the Phase 8
"advanced control" surface needs to agree on: which section tags exist,
which musical parameters the pinned ACE-Step build actually accepts,
and which advisories a request should carry.

Two hard rules shape the design.

**Advisories never mutate user input.** Every function here reads
lyrics and returns findings. Nothing rewrites, reflows, re-tags, or
"fixes" what the user typed. The original text is what gets stored and
what gets sent; an advisory is a note shown next to it, never an edit.

**Parameter ranges come from the pinned engine, not from taste.** The
constants below were read out of ACE-Step at commit
``6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`` (``acestep/constants.py``)
rather than chosen. Where LUBER is stricter than the engine that is
stated in the constant's comment.

The density and structure heuristics *are* judgement calls. They are
labelled as heuristics, they only ever produce warnings, and the
thresholds are named constants so they can be tuned against listening
evidence instead of argued about in prose.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import StrEnum

# ── Engine-verified parameter surface ─────────────────────────────────
# Source: acestep/constants.py @ 6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0

#: ``BPM_MIN`` / ``BPM_MAX`` upstream.
BPM_MIN = 30
BPM_MAX = 300

#: Upstream ``DURATION_MIN`` / ``DURATION_MAX`` are 10 and 600. LUBER
#: caps at 360 because nothing above that has been verified end to end
#: on the Apple Silicon path.
DURATION_MIN = 10
DURATION_MAX = 360

#: Upstream ``VALID_TIME_SIGNATURES = [2, 3, 4, 6]``. The REST field is
#: typed ``str``, and the value that reaches the DiT metadata block is
#: the bare numerator ("4"), not "4/4".
VALID_TIME_SIGNATURE_VALUES: tuple[str, ...] = ("2", "3", "4", "6")

#: Upstream builds ``VALID_KEYSCALES`` as notes x accidentals x modes.
#: Unicode accidentals are accepted upstream; LUBER offers the ASCII
#: forms only, so one key has exactly one representation in our data.
KEYSCALE_NOTES: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G")
KEYSCALE_ACCIDENTALS: tuple[str, ...] = ("", "#", "b")
KEYSCALE_MODES: tuple[str, ...] = ("major", "minor")
VALID_KEY_SCALES: tuple[str, ...] = tuple(
    f"{note}{accidental} {mode}"
    for note in KEYSCALE_NOTES
    for accidental in KEYSCALE_ACCIDENTALS
    for mode in KEYSCALE_MODES
)

#: Accepted upstream too, but deliberately not exposed as product
#: controls. Kept here so the reason is recorded next to the surface it
#: is absent from.
UNEXPOSED_ENGINE_PARAMETERS: dict[str, str] = {
    "guidance_scale": (
        "auto-corrected to 1.0 by the engine for turbo checkpoints; "
        "exposing a control that the engine overwrites would be a lie"
    ),
    "infer_method": (
        "ode/sde sampler choice; Phase 6 A/B showed no user-visible win "
        "and sde worsened end-level drift"
    ),
    "lm_negative_prompt": (
        "the only negative-prompt field upstream and it is LM-only; the "
        "DiT has no negative prompt and the LM is disabled on this host"
    ),
    "thinking": "requires the 1.7B LM, which is disabled on this host",
    "use_adg": "base-model only; not applicable to acestep-v15-turbo",
}

#: Long-form presets. 240s is the longest verified in Phase 8.
DURATION_PRESETS: tuple[int, ...] = (30, 60, 120, 180, 240)
LONG_FORM_THRESHOLD_SECONDS = 120


# ── Song structure ────────────────────────────────────────────────────


class SectionKind(StrEnum):
    """Section roles LUBER's editor knows how to reason about."""

    INTRO = "intro"
    VERSE = "verse"
    PRE_CHORUS = "pre-chorus"
    CHORUS = "chorus"
    POST_CHORUS = "post-chorus"
    BRIDGE = "bridge"
    BREAK = "break"
    INSTRUMENTAL = "instrumental"
    OUTRO = "outro"


#: Canonical tag text offered by the editor, in song order.
SECTION_TAG_PALETTE: tuple[str, ...] = (
    "[Intro]",
    "[Verse]",
    "[Verse 1]",
    "[Verse 2]",
    "[Pre-Chorus]",
    "[Chorus]",
    "[Post-Chorus]",
    "[Bridge]",
    "[Break]",
    "[Instrumental]",
    "[Outro]",
)

#: Sections that carry no sung words by definition.
NON_VOCAL_SECTIONS = frozenset({SectionKind.BREAK, SectionKind.INSTRUMENTAL})

_SECTION_LINE = re.compile(r"^\s*\[([^\]]{1,40})\]\s*$")
_SECTION_NAME = re.compile(r"^(?P<name>[a-z][a-z \-]*?)\s*(?P<index>\d+)?$")

#: Spelling variants accepted from the user, mapped to a canonical kind.
_SECTION_ALIASES: dict[str, SectionKind] = {
    "intro": SectionKind.INTRO,
    "introduction": SectionKind.INTRO,
    "verse": SectionKind.VERSE,
    "pre-chorus": SectionKind.PRE_CHORUS,
    "prechorus": SectionKind.PRE_CHORUS,
    "pre chorus": SectionKind.PRE_CHORUS,
    "chorus": SectionKind.CHORUS,
    "hook": SectionKind.CHORUS,
    "post-chorus": SectionKind.POST_CHORUS,
    "postchorus": SectionKind.POST_CHORUS,
    "post chorus": SectionKind.POST_CHORUS,
    "bridge": SectionKind.BRIDGE,
    "break": SectionKind.BREAK,
    "breakdown": SectionKind.BREAK,
    "instrumental": SectionKind.INSTRUMENTAL,
    "inst": SectionKind.INSTRUMENTAL,
    "solo": SectionKind.INSTRUMENTAL,
    "outro": SectionKind.OUTRO,
    "ending": SectionKind.OUTRO,
}


@dataclass(frozen=True)
class Section:
    """One tagged block of lyrics, exactly as the user wrote it."""

    #: Canonical role, or ``None`` when the tag is not one we recognise.
    kind: SectionKind | None
    #: The tag text as typed, without brackets ("Verse 2").
    label: str
    #: Trailing number in the tag, if any ("Verse 2" -> 2).
    index: int | None
    #: Line number (1-based) of the tag itself.
    line_number: int
    #: The lines under this tag, verbatim and unmodified.
    lines: tuple[str, ...]

    @property
    def is_recognised(self) -> bool:
        return self.kind is not None

    @property
    def has_content(self) -> bool:
        return any(line.strip() for line in self.lines)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@dataclass(frozen=True)
class ParsedStructure:
    """Result of reading section tags out of a lyric sheet."""

    sections: tuple[Section, ...]
    #: Lines that appear before the first tag. Legitimate — plenty of
    #: lyric sheets are untagged — so this is reported, not an error.
    preamble: tuple[str, ...] = ()

    @property
    def is_tagged(self) -> bool:
        return bool(self.sections)

    @property
    def kinds(self) -> tuple[SectionKind | None, ...]:
        return tuple(section.kind for section in self.sections)

    def count(self, kind: SectionKind) -> int:
        return sum(1 for section in self.sections if section.kind is kind)


def parse_structure(lyrics: str) -> ParsedStructure:
    """Split *lyrics* into tagged sections without altering the text.

    A tag is a line consisting only of ``[...]``. Bracketed text that
    shares a line with lyrics is left alone: it is far more likely to be
    an ad-lib or a stage direction than a section marker.
    """
    preamble: list[str] = []
    sections: list[Section] = []
    pending: list[str] | None = None
    current: tuple[SectionKind | None, str, int | None, int] | None = None

    def flush() -> None:
        if current is None or pending is None:
            return
        kind, label, index, line_number = current
        sections.append(
            Section(
                kind=kind,
                label=label,
                index=index,
                line_number=line_number,
                lines=tuple(pending),
            )
        )

    for line_number, line in enumerate(lyrics.splitlines(), start=1):
        match = _SECTION_LINE.match(line)
        if match is None:
            if current is None:
                preamble.append(line)
            else:
                assert pending is not None
                pending.append(line)
            continue
        flush()
        label = match.group(1).strip()
        kind, index = classify_section_label(label)
        current = (kind, label, index, line_number)
        pending = []

    flush()
    return ParsedStructure(sections=tuple(sections), preamble=tuple(preamble))


def classify_section_label(label: str) -> tuple[SectionKind | None, int | None]:
    """Map a tag's inner text to a canonical kind and ordinal."""
    normalized = re.sub(r"\s+", " ", label.strip().lower())
    match = _SECTION_NAME.match(normalized)
    if match is None:
        return _SECTION_ALIASES.get(normalized), None
    name = match.group("name").strip()
    raw_index = match.group("index")
    index = int(raw_index) if raw_index is not None else None
    return _SECTION_ALIASES.get(name), index


# ── Advisories ────────────────────────────────────────────────────────


class AdvisoryLevel(StrEnum):
    INFO = "info"
    WARNING = "warning"


@dataclass(frozen=True)
class Advisory:
    """A non-blocking finding about a generation request.

    Advisories are surfaced to the user and persisted with the request
    trace. None of them prevents a generation from running: the user is
    allowed to overrule every heuristic in this module.
    """

    code: str
    level: AdvisoryLevel
    message: str
    #: Machine-readable context for the diagnostics drawer and tests.
    detail: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "level": self.level.value,
            "message": self.message,
            "detail": dict(self.detail),
        }


# ── Structure advisories (Step 4) ─────────────────────────────────────

#: A section shorter than this many seconds of its share of the song is
#: unlikely to read as a section at all.
MIN_SECONDS_PER_SECTION = 15


def validate_structure(structure: ParsedStructure, *, instrumental: bool = False) -> list[Advisory]:
    """Warn about structural problems. Never blocks, never rewrites."""
    advisories: list[Advisory] = []
    if not structure.is_tagged:
        return advisories

    unknown = [s.label for s in structure.sections if not s.is_recognised]
    if unknown:
        advisories.append(
            Advisory(
                code="UNKNOWN_SECTION_TAG",
                level=AdvisoryLevel.WARNING,
                message=(
                    f"{len(unknown)} section tag(s) are not recognised: "
                    f"{', '.join(sorted(set(unknown))[:5])}. They are sent to the model "
                    f"exactly as written."
                ),
                detail={"labels": sorted(set(unknown))},
            )
        )

    if not instrumental:
        empty = [
            s.label
            for s in structure.sections
            if s.kind not in NON_VOCAL_SECTIONS and not s.has_content
        ]
        if empty:
            advisories.append(
                Advisory(
                    code="EMPTY_SECTION",
                    level=AdvisoryLevel.WARNING,
                    message=(
                        f"{len(empty)} section(s) have a tag but no lyrics: {', '.join(empty[:5])}."
                    ),
                    detail={"labels": empty},
                )
            )

        if structure.count(SectionKind.CHORUS) == 0:
            advisories.append(
                Advisory(
                    code="NO_CHORUS",
                    level=AdvisoryLevel.INFO,
                    message="No [Chorus] section. Songs without a chorus tend to feel shapeless.",
                    detail={},
                )
            )

    duplicates = _duplicate_numbering(structure)
    if duplicates:
        advisories.append(
            Advisory(
                code="DUPLICATE_SECTION_NUMBER",
                level=AdvisoryLevel.WARNING,
                message=(
                    f"Repeated section numbering: {', '.join(duplicates[:5])}. "
                    f"Numbered sections normally appear once each."
                ),
                detail={"labels": duplicates},
            )
        )

    kinds = [s.kind for s in structure.sections]
    if SectionKind.INTRO in kinds and kinds[0] is not SectionKind.INTRO:
        advisories.append(
            Advisory(
                code="INTRO_NOT_FIRST",
                level=AdvisoryLevel.INFO,
                message="[Intro] is not the first section.",
                detail={"position": kinds.index(SectionKind.INTRO) + 1},
            )
        )
    if SectionKind.OUTRO in kinds and kinds[-1] is not SectionKind.OUTRO:
        advisories.append(
            Advisory(
                code="OUTRO_NOT_LAST",
                level=AdvisoryLevel.INFO,
                message="[Outro] is not the last section.",
                detail={"position": kinds.index(SectionKind.OUTRO) + 1},
            )
        )

    if instrumental:
        vocal_sections = [
            s.label
            for s in structure.sections
            if s.kind not in NON_VOCAL_SECTIONS and s.has_content
        ]
        if vocal_sections:
            advisories.append(
                Advisory(
                    code="LYRICS_IN_INSTRUMENTAL",
                    level=AdvisoryLevel.WARNING,
                    message=(
                        "This is an instrumental track, so the lyrics in "
                        f"{', '.join(vocal_sections[:5])} will not be sung."
                    ),
                    detail={"labels": vocal_sections},
                )
            )

    return advisories


def _duplicate_numbering(structure: ParsedStructure) -> list[str]:
    seen: dict[tuple[SectionKind | None, int], int] = {}
    duplicates: list[str] = []
    for section in structure.sections:
        if section.index is None:
            continue
        key = (section.kind, section.index)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 2:
            duplicates.append(f"[{section.label}]")
    return duplicates


# ── Duration / density advisories (Step 6) ────────────────────────────

#: Share of a track that typically carries sung words; the rest is
#: intro, fills, instrumental passages and outro. Heuristic.
SINGABLE_FRACTION = 0.75
#: Syllables per singable second. Below the floor a lyric sheet leaves
#: the model improvising; above the ceiling it has to rush. Heuristic,
#: derived from the Phase 5 benchmark set, not from the model.
MIN_SYLLABLES_PER_SECOND = 1.2
MAX_SYLLABLES_PER_SECOND = 3.2

_HANGUL_SYLLABLES = re.compile(r"[가-힣]")
_LATIN_VOWEL_GROUPS = re.compile(r"[aeiouy]+", re.IGNORECASE)
_CJK = re.compile(r"[぀-ヿ一-鿿]")


def estimate_syllables(text: str) -> int:
    """Rough syllable count for mixed Korean/Latin lyric text.

    Hangul and CJK are counted per character because each block is one
    syllable. Latin words are counted by vowel group, which over-counts
    silent vowels and under-counts diphthongs — good enough to separate
    "far too many words" from "far too few", which is all it is used for.
    """
    stripped = _strip_non_lyrics(text)
    hangul = len(_HANGUL_SYLLABLES.findall(stripped))
    cjk = len(_CJK.findall(stripped))
    latin_only = _CJK.sub(" ", _HANGUL_SYLLABLES.sub(" ", stripped))
    latin = 0
    for word in latin_only.split():
        groups = len(_LATIN_VOWEL_GROUPS.findall(word))
        if groups == 0 and any(c.isalpha() for c in word):
            groups = 1
        latin += groups
    return hangul + cjk + latin


def _strip_non_lyrics(text: str) -> str:
    """Drop section tags so they are not counted as words to sing."""
    return "\n".join(line for line in text.splitlines() if _SECTION_LINE.match(line) is None)


def analyze_density(
    lyrics: str, duration_seconds: int, *, instrumental: bool = False
) -> list[Advisory]:
    """Compare lyric volume and section count against the target length."""
    if instrumental:
        return []
    advisories: list[Advisory] = []
    structure = parse_structure(lyrics)
    syllables = estimate_syllables(lyrics)
    singable_seconds = max(duration_seconds * SINGABLE_FRACTION, 1.0)
    density = syllables / singable_seconds

    if syllables == 0:
        return advisories

    if density > MAX_SYLLABLES_PER_SECOND:
        suggested = int(syllables / (MAX_SYLLABLES_PER_SECOND * SINGABLE_FRACTION))
        advisories.append(
            Advisory(
                code="LYRICS_DENSE_FOR_DURATION",
                level=AdvisoryLevel.WARNING,
                message=(
                    f"About {syllables} syllables in {duration_seconds}s is dense "
                    f"({density:.1f}/s). Consider roughly {suggested}s, or fewer words."
                ),
                detail={
                    "syllables": syllables,
                    "density_per_second": round(density, 2),
                    "suggested_duration_seconds": suggested,
                },
            )
        )
    elif density < MIN_SYLLABLES_PER_SECOND:
        suggested = max(
            DURATION_MIN, int(syllables / (MIN_SYLLABLES_PER_SECOND * SINGABLE_FRACTION))
        )
        advisories.append(
            Advisory(
                code="LYRICS_SPARSE_FOR_DURATION",
                level=AdvisoryLevel.INFO,
                message=(
                    f"About {syllables} syllables in {duration_seconds}s is sparse "
                    f"({density:.1f}/s); the model will fill the gaps itself. "
                    f"Consider roughly {suggested}s, or more lyrics."
                ),
                detail={
                    "syllables": syllables,
                    "density_per_second": round(density, 2),
                    "suggested_duration_seconds": suggested,
                },
            )
        )

    section_count = len(structure.sections)
    max_sections = max(1, duration_seconds // MIN_SECONDS_PER_SECTION)
    if section_count > max_sections:
        advisories.append(
            Advisory(
                code="MANY_SECTIONS_FOR_DURATION",
                level=AdvisoryLevel.WARNING,
                message=(
                    f"{section_count} sections in {duration_seconds}s leaves under "
                    f"{MIN_SECONDS_PER_SECTION}s each. Consider fewer sections or a "
                    f"longer track."
                ),
                detail={"sections": section_count, "max_recommended": max_sections},
            )
        )
    return advisories


# ── Korean lyric pre-flight (Step 7) ──────────────────────────────────

#: Sung Korean lines much longer than this rarely fit a musical phrase.
KOREAN_LINE_SYLLABLE_LIMIT = 24
#: Share of lyric characters that must be Hangul to call it Korean.
KOREAN_DOMINANCE_THRESHOLD = 0.30

_LATIN_WORD = re.compile(r"[A-Za-z]{2,}")


def korean_ratio(text: str) -> float:
    """Fraction of letter characters that are Hangul."""
    stripped = _strip_non_lyrics(text)
    letters = [c for c in stripped if c.isalpha()]
    if not letters:
        return 0.0
    hangul = sum(1 for c in letters if _HANGUL_SYLLABLES.match(c))
    return hangul / len(letters)


def audit_korean_lyrics(lyrics: str, language: str | None) -> list[Advisory]:
    """Pre-flight checks for Korean vocal requests.

    Reports only. The lyrics that reach the model are byte-for-byte the
    lyrics the user typed, whatever this function finds.
    """
    advisories: list[Advisory] = []
    text = _strip_non_lyrics(lyrics)
    if not text.strip():
        return advisories

    # macOS hands back NFD for Korean; compare on NFC so a decomposed
    # jamo sequence is not mistaken for a non-Hangul character.
    text = unicodedata.normalize("NFC", text)
    ratio = korean_ratio(text)
    declared_korean = (language or "").lower() == "ko"

    if ratio >= KOREAN_DOMINANCE_THRESHOLD and not declared_korean:
        advisories.append(
            Advisory(
                code="KOREAN_LYRICS_LANGUAGE_MISMATCH",
                level=AdvisoryLevel.WARNING,
                message=(
                    f"The lyrics are {ratio:.0%} Korean but the vocal language is "
                    f"'{language or 'unset'}'. Set it to Korean for correct pronunciation."
                ),
                detail={"korean_ratio": round(ratio, 3), "language": language},
            )
        )
    if declared_korean and ratio < KOREAN_DOMINANCE_THRESHOLD:
        advisories.append(
            Advisory(
                code="KOREAN_LANGUAGE_WITHOUT_KOREAN_LYRICS",
                level=AdvisoryLevel.WARNING,
                message=(
                    f"Vocal language is Korean but only {ratio:.0%} of the lyrics are "
                    f"Hangul. Latin text will be sung as-is."
                ),
                detail={"korean_ratio": round(ratio, 3)},
            )
        )

    if ratio < KOREAN_DOMINANCE_THRESHOLD and not declared_korean:
        return advisories

    long_lines = [
        number
        for number, line in enumerate(text.splitlines(), start=1)
        if estimate_syllables(line) > KOREAN_LINE_SYLLABLE_LIMIT
    ]
    if long_lines:
        advisories.append(
            Advisory(
                code="KOREAN_LINE_TOO_LONG",
                level=AdvisoryLevel.INFO,
                message=(
                    f"{len(long_lines)} line(s) exceed {KOREAN_LINE_SYLLABLE_LIMIT} "
                    f"syllables and may be rushed. Breaking them into shorter lines "
                    f"usually sings better."
                ),
                detail={"lines": long_lines[:20]},
            )
        )

    mixed = [
        number
        for number, line in enumerate(text.splitlines(), start=1)
        if _HANGUL_SYLLABLES.search(line) and len(_LATIN_WORD.findall(line)) >= 3
    ]
    if mixed:
        advisories.append(
            Advisory(
                code="KOREAN_MIXED_SCRIPT_LINE",
                level=AdvisoryLevel.INFO,
                message=(
                    f"{len(mixed)} line(s) mix Korean with several English words. "
                    f"Korean vocal models often mispronounce embedded English."
                ),
                detail={"lines": mixed[:20]},
            )
        )
    return advisories


# ── Combined pre-flight ───────────────────────────────────────────────


def preflight(
    *,
    lyrics: str,
    duration_seconds: int,
    language: str | None,
    instrumental: bool,
) -> list[Advisory]:
    """Every advisory for one request, in reporting order."""
    structure = parse_structure(lyrics)
    return [
        *validate_structure(structure, instrumental=instrumental),
        *analyze_density(lyrics, duration_seconds, instrumental=instrumental),
        *audit_korean_lyrics(lyrics, language),
    ]
