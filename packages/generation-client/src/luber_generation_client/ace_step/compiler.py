"""AceStepPromptCompiler — LUBER input → ACE-Step conditioning.

Upstream has **no** ``vocal_gender`` parameter (verified at the pinned
commit), so vocal gender compiles into descriptive prompt conditioning.
Instrumental uses the official upstream mechanism: lyrics ``[inst]``
(``acestep/api/server_utils.py::is_instrumental`` also treats empty
lyrics as instrumental).

The user's original prompt is preserved unmodified in the LUBER
database; only the compiled prompt is sent to the model. Both are
logged for reproducibility.

**Conditioning is added only when the prompt does not already carry
it.** Phase 5 found the earlier unconditional-append behaviour produced
``"…, no vocals, instrumental, no vocals"`` and stated "female vocal"
three times over. Repetition spends prompt budget restating conditions
instead of describing music, so each conditioning phrase is now checked
against what the user already wrote.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from luber_generation_client.provider import GenerationRequest
from luber_schemas import VocalGender

INSTRUMENTAL_LYRICS = "[inst]"

#: Conditioning phrases appended per vocal selection, in priority order.
#: Each is added only if the prompt does not already express it.
_VOCAL_CONDITIONING: dict[VocalGender, tuple[str, ...]] = {
    VocalGender.FEMALE: ("female lead vocal", "natural female singing voice"),
    VocalGender.MALE: ("male lead vocal", "natural male singing voice"),
}
_INSTRUMENTAL_CONDITIONING: tuple[str, ...] = ("instrumental", "no vocals")

#: Markers that mean a *concept group* is already established in the
#: prompt. Matched on word boundaries so "male" never matches inside
#: "female". When any marker for a group is present, the whole group is
#: skipped: once the user has said "breathy female lead vocal", adding
#: "natural female singing voice" restates the gender and can fight the
#: user's own descriptors.
_GROUP_MARKERS: dict[VocalGender | str, tuple[str, ...]] = {
    VocalGender.FEMALE: (
        "female vocal",
        "female vocals",
        "female lead",
        "female singer",
        "female singing",
        "female voice",
        "woman singing",
    ),
    VocalGender.MALE: (
        "male vocal",
        "male vocals",
        "male lead",
        "male singer",
        "male singing",
        "male voice",
        "man singing",
    ),
    "instrumental": (
        "instrumental",
        "no vocals",
        "no vocal",
        "without vocals",
        "vocal-free",
    ),
}


@dataclass(frozen=True)
class CompiledAceStepInput:
    original_prompt: str
    prompt: str
    lyrics: str
    vocal_language: str
    instrumental: bool
    #: Conditioning phrases actually appended, for A/B analysis.
    added_conditioning: tuple[str, ...] = ()
    #: Phrases skipped because the prompt already expressed them.
    skipped_conditioning: tuple[str, ...] = ()


def _contains_phrase(haystack: str, needle: str) -> bool:
    """Whole-word containment, so "male" never matches inside "female"."""
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None


def _group_already_expressed(prompt_lower: str, group: VocalGender | str) -> bool:
    """Whether the prompt already establishes this conditioning concept."""
    return any(_contains_phrase(prompt_lower, marker) for marker in _GROUP_MARKERS.get(group, ()))


class AceStepPromptCompiler:
    def compile(self, request: GenerationRequest) -> CompiledAceStepInput:
        instrumental = (
            request.instrumental
            or request.vocal_gender is VocalGender.INSTRUMENTAL
            or not request.lyrics.strip()
        )

        base_prompt = request.prompt.strip()
        if instrumental:
            group: VocalGender | str = "instrumental"
            candidates = _INSTRUMENTAL_CONDITIONING
        else:
            group = request.vocal_gender
            candidates = _VOCAL_CONDITIONING[request.vocal_gender]

        # All-or-nothing per concept: conditioning is useful when the
        # prompt lacks it entirely, and noise once the prompt has it.
        if _group_already_expressed(base_prompt.lower(), group):
            added: tuple[str, ...] = ()
            skipped: tuple[str, ...] = candidates
        else:
            added = candidates
            skipped = ()

        prompt = ", ".join([base_prompt, *added]) if added else base_prompt
        lyrics = INSTRUMENTAL_LYRICS if instrumental else request.lyrics

        return CompiledAceStepInput(
            original_prompt=request.prompt,
            prompt=prompt,
            lyrics=lyrics,
            vocal_language=request.language or "en",
            instrumental=instrumental,
            added_conditioning=added,
            skipped_conditioning=skipped,
        )
