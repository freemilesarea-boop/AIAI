"""AceStepPromptCompiler — LUBER input → ACE-Step conditioning.

Upstream has **no** ``vocal_gender`` parameter (verified at the pinned
commit), so vocal gender compiles into descriptive prompt conditioning.
Instrumental uses the official upstream mechanism: lyrics ``[inst]``
(``acestep/api/server_utils.py::is_instrumental`` also treats empty
lyrics as instrumental).

The user's original prompt is preserved unmodified in the LUBER
database; only the compiled prompt is sent to the model. Both are
logged for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass

from luber_generation_client.provider import GenerationRequest
from luber_schemas import VocalGender

INSTRUMENTAL_LYRICS = "[inst]"

_VOCAL_CONDITIONING = {
    VocalGender.FEMALE: "female lead vocal, natural female singing voice",
    VocalGender.MALE: "male lead vocal, natural male singing voice",
}


@dataclass(frozen=True)
class CompiledAceStepInput:
    original_prompt: str
    prompt: str
    lyrics: str
    vocal_language: str
    instrumental: bool


class AceStepPromptCompiler:
    def compile(self, request: GenerationRequest) -> CompiledAceStepInput:
        instrumental = (
            request.instrumental
            or request.vocal_gender is VocalGender.INSTRUMENTAL
            or not request.lyrics.strip()
        )

        if instrumental:
            prompt = f"{request.prompt.strip()}, instrumental, no vocals"
            lyrics = INSTRUMENTAL_LYRICS
        else:
            conditioning = _VOCAL_CONDITIONING[request.vocal_gender]
            prompt = f"{request.prompt.strip()}, {conditioning}"
            lyrics = request.lyrics  # section tags preserved verbatim

        return CompiledAceStepInput(
            original_prompt=request.prompt,
            prompt=prompt,
            lyrics=lyrics,
            vocal_language=request.language or "en",
            instrumental=instrumental,
        )
