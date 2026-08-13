"""Song presets and structure templates for full-song writing.

These are **composition aids**, not controls. A template is a block of
section tags the user can choose to insert; a preset bundles a suggested
duration with a template and a short description of the shape it makes.

Three rules the whole module is built around.

**Nothing here is applied automatically.** Every value is a *suggestion*
returned to the caller. Insertion is always an explicit user action, and
:func:`apply_template` refuses to touch lyrics that already have content
unless the caller passes ``replace=True`` — which the UI only does after
the user confirms.

**Templates do not make the model obey them.** ACE-Step conditions on
lyric text; section tags are part of that text and nothing more. A
template raises the odds of a recognisable arrangement. It does not
enforce one, and the docs must not imply otherwise.

**Durations here are suggestions inside the validated range.** The
Phase 9 product ceiling is 240s (``PRODUCT_MAX_DURATION``), which is
below the engine's 600s and below LUBER's own 360s schema cap. That gap
is deliberate: only up to 240s has been validated end to end.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from luber_schemas.songcraft import (
    DURATION_MAX,
    ParsedStructure,
    SectionKind,
    parse_structure,
)

#: Longest duration the product offers. Below the engine's 600s and
#: below the schema's 360s: only up to here has been validated end to
#: end on the deployed stack (see PHASE9_LONG_FORM_ENGINE_AUDIT.md).
PRODUCT_MAX_DURATION = 240

#: Durations the UI offers, shortest first. Each is a real validated
#: point, not a round number: 30/60 came from Phase 3, 120/180/240 from
#: the Phase 9 long-form gates.
PRODUCT_DURATIONS: tuple[int, ...] = (30, 60, 120, 180, 240)


class StructureTemplateId(StrEnum):
    POP = "pop"
    BALLAD = "ballad"
    RNB = "rnb"
    BAND = "band"
    MINIMAL = "minimal"


@dataclass(frozen=True)
class StructureTemplate:
    """An ordered run of section tags the user may insert."""

    id: StructureTemplateId
    name: str
    description: str
    sections: tuple[str, ...]
    #: Duration this shape is written for. Shorter and it gets crowded.
    suggested_duration: int

    @property
    def text(self) -> str:
        """The template as lyric-sheet text: one tag per line, blank line between."""
        return "\n\n".join(self.sections) + "\n"

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id.value,
            "name": self.name,
            "description": self.description,
            "sections": list(self.sections),
            "suggested_duration": self.suggested_duration,
            "text": self.text,
        }


STRUCTURE_TEMPLATES: tuple[StructureTemplate, ...] = (
    StructureTemplate(
        id=StructureTemplateId.POP,
        name="Pop",
        description="Two verses into a repeated chorus, with a bridge before the last one.",
        sections=(
            "[Intro]",
            "[Verse 1]",
            "[Pre-Chorus]",
            "[Chorus]",
            "[Verse 2]",
            "[Pre-Chorus]",
            "[Chorus]",
            "[Bridge]",
            "[Final Chorus]",
            "[Outro]",
        ),
        suggested_duration=180,
    ),
    StructureTemplate(
        id=StructureTemplateId.BALLAD,
        name="Ballad",
        description="Verse-led and slower to arrive; the chorus lands after more story.",
        sections=(
            "[Intro]",
            "[Verse 1]",
            "[Verse 2]",
            "[Chorus]",
            "[Verse 3]",
            "[Chorus]",
            "[Bridge]",
            "[Final Chorus]",
            "[Outro]",
        ),
        suggested_duration=240,
    ),
    StructureTemplate(
        id=StructureTemplateId.RNB,
        name="R&B",
        description="Tighter frame with room for phrasing rather than more sections.",
        sections=(
            "[Intro]",
            "[Verse 1]",
            "[Pre-Chorus]",
            "[Chorus]",
            "[Verse 2]",
            "[Chorus]",
            "[Bridge]",
            "[Outro]",
        ),
        suggested_duration=180,
    ),
    StructureTemplate(
        id=StructureTemplateId.BAND,
        name="Band",
        description="Leaves an instrumental slot where a solo would sit.",
        sections=(
            "[Intro]",
            "[Verse 1]",
            "[Chorus]",
            "[Verse 2]",
            "[Chorus]",
            "[Instrumental]",
            "[Final Chorus]",
            "[Outro]",
        ),
        suggested_duration=180,
    ),
    StructureTemplate(
        id=StructureTemplateId.MINIMAL,
        name="Verse / Chorus",
        description="The smallest shape that still reads as a song.",
        sections=("[Verse]", "[Chorus]"),
        suggested_duration=60,
    ),
)

TEMPLATES_BY_ID: dict[str, StructureTemplate] = {t.id.value: t for t in STRUCTURE_TEMPLATES}


class PresetId(StrEnum):
    SHORT_DEMO = "short_demo"
    FULL_POP_SONG = "full_pop_song"
    BALLAD = "ballad"
    RNB = "rnb"
    BAND_SONG = "band_song"
    INSTRUMENTAL = "instrumental"


@dataclass(frozen=True)
class SongPreset:
    """A starting point: a duration, and usually a structure to go with it.

    A preset never carries a prompt or lyrics. It shapes the *frame* of
    the song and leaves the writing to the user.
    """

    id: PresetId
    name: str
    description: str
    duration: int
    template_id: StructureTemplateId | None
    instrumental: bool = False

    @property
    def template(self) -> StructureTemplate | None:
        return TEMPLATES_BY_ID.get(self.template_id.value) if self.template_id else None

    def to_dict(self) -> dict[str, object]:
        template = self.template
        return {
            "id": self.id.value,
            "name": self.name,
            "description": self.description,
            "duration": self.duration,
            "instrumental": self.instrumental,
            "template": template.to_dict() if template else None,
        }


SONG_PRESETS: tuple[SongPreset, ...] = (
    SongPreset(
        id=PresetId.SHORT_DEMO,
        name="Short Demo",
        description="One verse and a chorus, for trying an idea quickly.",
        duration=60,
        template_id=StructureTemplateId.MINIMAL,
    ),
    SongPreset(
        id=PresetId.FULL_POP_SONG,
        name="Full Pop Song",
        description="A complete pop arrangement with a bridge and a final chorus.",
        duration=180,
        template_id=StructureTemplateId.POP,
    ),
    SongPreset(
        id=PresetId.BALLAD,
        name="Ballad",
        description="Longer and verse-led, for a slower emotional build.",
        duration=240,
        template_id=StructureTemplateId.BALLAD,
    ),
    SongPreset(
        id=PresetId.RNB,
        name="R&B",
        description="A tighter frame that leaves space for vocal phrasing.",
        duration=180,
        template_id=StructureTemplateId.RNB,
    ),
    SongPreset(
        id=PresetId.BAND_SONG,
        name="Band Song",
        description="Verse/chorus with an instrumental section for a solo.",
        duration=180,
        template_id=StructureTemplateId.BAND,
    ),
    SongPreset(
        id=PresetId.INSTRUMENTAL,
        name="Instrumental",
        description="No vocals. Structure tags are left out entirely.",
        duration=120,
        template_id=None,
        instrumental=True,
    ),
)

PRESETS_BY_ID: dict[str, SongPreset] = {p.id.value: p for p in SONG_PRESETS}


def lyrics_have_content(lyrics: str) -> bool:
    """Whether the sheet holds anything the user would mind losing.

    Section tags alone do not count as content — replacing a bare
    skeleton with a different skeleton loses no writing. Any other
    non-blank line does count.
    """
    parsed = parse_structure(lyrics)
    if any(line.strip() for line in parsed.preamble):
        return True
    return any(section.has_content for section in parsed.sections)


def apply_template(lyrics: str, template: StructureTemplate, *, replace: bool = False) -> str:
    """Return the lyric sheet with *template* applied.

    Refuses to discard writing: if the sheet already has content and
    ``replace`` is False, the template is **appended** after it rather
    than overwriting it. The caller must ask for ``replace=True``
    explicitly, which the UI only does behind a confirmation.

    This function returns new text; it never mutates anything, and the
    caller decides whether to use it.
    """
    if not lyrics.strip():
        return template.text
    if replace:
        return template.text
    if not lyrics_have_content(lyrics):
        # Only a bare skeleton is present: swapping shapes loses nothing.
        return template.text
    separator = "" if lyrics.endswith("\n") else "\n"
    return f"{lyrics}{separator}\n{template.text}"


def describe_template_fit(template: StructureTemplate, duration_seconds: int) -> str | None:
    """A one-line note when a template and a duration disagree.

    Advisory text, not a rule — the user may well want a ten-section
    song in ninety seconds and is allowed to have one.
    """
    sections = len(template.sections)
    if sections == 0 or duration_seconds <= 0:
        return None
    per_section = duration_seconds / sections
    if per_section < 12:
        return (
            f"{sections} sections in {duration_seconds}s is about {per_section:.0f}s each — "
            f"{template.name} usually needs {template.suggested_duration}s to breathe."
        )
    if per_section > 45:
        return (
            f"{sections} sections across {duration_seconds}s is about {per_section:.0f}s each; "
            f"consider a longer template or a shorter track."
        )
    return None


def template_for_structure(structure: ParsedStructure) -> StructureTemplateId | None:
    """Best-effort guess at which template a lyric sheet already follows.

    Used only to label the editor; a wrong guess costs nothing. Returns
    ``None`` rather than forcing a match.
    """
    kinds = [s.kind for s in structure.sections if s.kind is not None]
    if not kinds:
        return None
    for template in STRUCTURE_TEMPLATES:
        expected = [
            kind
            for kind in (_TAG_TO_KIND.get(tag) for tag in template.sections)
            if kind is not None
        ]
        if expected == kinds:
            return template.id
    return None


_TAG_TO_KIND: dict[str, SectionKind | None] = {
    "[Intro]": SectionKind.INTRO,
    "[Verse]": SectionKind.VERSE,
    "[Verse 1]": SectionKind.VERSE,
    "[Verse 2]": SectionKind.VERSE,
    "[Verse 3]": SectionKind.VERSE,
    "[Pre-Chorus]": SectionKind.PRE_CHORUS,
    "[Chorus]": SectionKind.CHORUS,
    "[Final Chorus]": SectionKind.CHORUS,
    "[Post-Chorus]": SectionKind.POST_CHORUS,
    "[Bridge]": SectionKind.BRIDGE,
    "[Break]": SectionKind.BREAK,
    "[Instrumental]": SectionKind.INSTRUMENTAL,
    "[Outro]": SectionKind.OUTRO,
}


def validate_product_duration(duration_seconds: int) -> bool:
    """Whether the product offers this duration today."""
    return duration_seconds in PRODUCT_DURATIONS


assert PRODUCT_MAX_DURATION <= DURATION_MAX, "product ceiling must stay inside the schema cap"
assert PRODUCT_DURATIONS[-1] == PRODUCT_MAX_DURATION
