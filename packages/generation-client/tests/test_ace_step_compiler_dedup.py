"""Prompt compiler de-duplication (Phase 6).

Phase 5 found the compiler appended conditioning unconditionally, so a
prompt ending "no vocals" became "…no vocals, instrumental, no vocals".
These tests pin both halves of the fix: redundant conditioning is
dropped, and genuinely absent conditioning is still added.
"""

import pytest

from luber_generation_client.ace_step.compiler import AceStepPromptCompiler
from luber_generation_client.provider import GenerationRequest
from luber_schemas import VocalGender


def _compile(prompt: str, *, vocal=VocalGender.FEMALE, lyrics="[Verse]\nline", language="ko"):
    return AceStepPromptCompiler().compile(
        GenerationRequest(
            title="t",
            prompt=prompt,
            lyrics=lyrics,
            vocal_gender=vocal,
            duration_seconds=60,
            language=language,
        )
    )


# ── the exact Phase 5 defects ─────────────────────────────────────────


def test_instrumental_no_longer_repeats_no_vocals():
    """The literal Phase 5 defect: '…no vocals, instrumental, no vocals'."""
    result = _compile(
        "Instrumental jazz trio at 120 BPM, brushed drums, no vocals",
        vocal=VocalGender.INSTRUMENTAL,
        lyrics="",
    )
    assert result.prompt.lower().count("no vocals") == 1
    assert result.prompt.lower().count("instrumental") == 1
    assert result.added_conditioning == ()


def test_female_vocal_not_restated_three_times():
    result = _compile("bright K-pop with female vocal")
    assert result.prompt == "bright K-pop with female vocal"
    assert result.added_conditioning == ()
    assert result.prompt.lower().count("female") == 1


def test_male_vocal_not_restated():
    result = _compile("smooth Korean R&B with male vocal", vocal=VocalGender.MALE)
    assert result.prompt == "smooth Korean R&B with male vocal"
    assert result.added_conditioning == ()


# ── useful conditioning is preserved ──────────────────────────────────


def test_conditioning_still_added_when_prompt_lacks_it():
    """De-duplication must not become 'never condition'."""
    result = _compile("dreamy synth pop at 100 BPM")
    assert result.added_conditioning == ("female lead vocal", "natural female singing voice")
    assert result.prompt.endswith("female lead vocal, natural female singing voice")


def test_instrumental_conditioning_added_when_absent():
    result = _compile("lo-fi chill beat to study to", vocal=VocalGender.INSTRUMENTAL, lyrics="")
    assert result.added_conditioning == ("instrumental", "no vocals")
    assert "instrumental" in result.prompt and "no vocals" in result.prompt


def test_male_conditioning_added_when_absent():
    result = _compile("gritty blues rock", vocal=VocalGender.MALE)
    assert result.added_conditioning == ("male lead vocal", "natural male singing voice")


# ── the gender-substring trap ─────────────────────────────────────────


def test_female_prompt_does_not_satisfy_male_conditioning():
    """'female' contains 'male' — word boundaries must prevent a match."""
    result = _compile("ballad with a female vocal", vocal=VocalGender.MALE)
    assert result.added_conditioning == ("male lead vocal", "natural male singing voice")


def test_male_marker_matches_only_as_a_whole_word():
    assert _compile("song with male vocals", vocal=VocalGender.MALE).added_conditioning == ()
    # "females" must not be read as establishing a male vocal.
    assert _compile("song with females", vocal=VocalGender.MALE).added_conditioning != ()


# ── marker coverage ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "prompt",
    [
        "pop with female vocals",
        "track with a female lead",
        "song with a female singer",
        "warm female singing throughout",
        "bright female voice",
        "a woman singing softly",
    ],
)
def test_female_markers_all_suppress_conditioning(prompt):
    assert _compile(prompt).added_conditioning == ()


@pytest.mark.parametrize(
    "prompt",
    [
        "instrumental post-rock",
        "beat with no vocals",
        "track with no vocal",
        "piece without vocals",
        "a vocal-free arrangement",
    ],
)
def test_instrumental_markers_all_suppress_conditioning(prompt):
    result = _compile(prompt, vocal=VocalGender.INSTRUMENTAL, lyrics="")
    assert result.added_conditioning == ()


def test_matching_is_case_insensitive():
    assert _compile("Bright K-pop with FEMALE VOCAL").added_conditioning == ()


# ── unchanged behaviour from earlier phases ───────────────────────────


def test_original_prompt_is_preserved_verbatim():
    result = _compile("  bright K-pop with female vocal  ")
    assert result.original_prompt == "  bright K-pop with female vocal  "
    assert result.prompt == "bright K-pop with female vocal"


def test_instrumental_lyrics_use_the_upstream_marker():
    result = _compile("jazz trio", vocal=VocalGender.INSTRUMENTAL, lyrics="")
    assert result.lyrics == "[inst]"
    assert result.instrumental is True


def test_empty_lyrics_force_instrumental_even_for_a_vocal_selection():
    result = _compile("a song", vocal=VocalGender.FEMALE, lyrics="   ")
    assert result.instrumental is True
    assert result.lyrics == "[inst]"


def test_korean_lyrics_pass_through_untouched():
    lyrics = "[Verse]\n오늘 밤 너를 생각해\n\n[Chorus]\n조금만 더"
    result = _compile("Korean ballad", lyrics=lyrics)
    assert result.lyrics == lyrics


def test_language_defaults_to_english_when_unset():
    result = AceStepPromptCompiler().compile(
        GenerationRequest(
            title="t",
            prompt="a song",
            lyrics="[Verse]\nline",
            vocal_gender=VocalGender.FEMALE,
            duration_seconds=60,
            language=None,
        )
    )
    assert result.vocal_language == "en"


def test_skipped_conditioning_is_reported_for_analysis():
    """A/B analysis needs to know what was suppressed and why."""
    result = _compile("bright K-pop with female vocal")
    assert result.skipped_conditioning == (
        "female lead vocal",
        "natural female singing voice",
    )
    assert result.added_conditioning == ()


def test_compiled_prompt_never_repeats_a_conditioning_phrase():
    """Property check across a spread of realistic prompts."""
    prompts = [
        ("bright K-pop with female vocal", VocalGender.FEMALE, "[Verse]\nx"),
        ("dreamy synth pop", VocalGender.FEMALE, "[Verse]\nx"),
        ("gritty rock with male vocal", VocalGender.MALE, "[Verse]\nx"),
        ("blues rock", VocalGender.MALE, "[Verse]\nx"),
        ("instrumental jazz, no vocals", VocalGender.INSTRUMENTAL, ""),
        ("lo-fi beat", VocalGender.INSTRUMENTAL, ""),
    ]
    for prompt, vocal, lyrics in prompts:
        compiled = _compile(prompt, vocal=vocal, lyrics=lyrics).prompt.lower()
        for phrase in ("no vocals", "instrumental", "female lead vocal", "male lead vocal"):
            assert compiled.count(phrase) <= 1, f"{phrase!r} repeated in {compiled!r}"
