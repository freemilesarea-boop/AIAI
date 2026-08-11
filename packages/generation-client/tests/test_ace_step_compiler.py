from luber_generation_client import GenerationRequest
from luber_generation_client.ace_step import AceStepPromptCompiler
from luber_generation_client.ace_step.compiler import INSTRUMENTAL_LYRICS
from luber_schemas import VocalGender


def _request(**overrides):
    defaults = dict(
        title="PHASE 2 REAL TEST",
        prompt="Dreamy Korean indie pop with warm electric piano",
        lyrics="[Verse]\n오늘 밤 너를 생각해\n[Chorus]\n내 곁에 조금만 더 있어줘",
        vocal_gender=VocalGender.FEMALE,
        duration_seconds=30,
        language="ko",
    )
    defaults.update(overrides)
    return GenerationRequest(**defaults)


def test_female_vocal_compiles_to_descriptive_conditioning():
    compiled = AceStepPromptCompiler().compile(_request())
    assert "female lead vocal, natural female singing voice" in compiled.prompt
    assert compiled.prompt.startswith("Dreamy Korean indie pop")
    # Original prompt preserved separately, unmodified.
    assert compiled.original_prompt == "Dreamy Korean indie pop with warm electric piano"
    assert compiled.instrumental is False


def test_male_vocal_conditioning():
    compiled = AceStepPromptCompiler().compile(_request(vocal_gender=VocalGender.MALE))
    assert "male lead vocal, natural male singing voice" in compiled.prompt


def test_lyrics_and_section_tags_preserved_verbatim():
    compiled = AceStepPromptCompiler().compile(_request())
    assert compiled.lyrics.startswith("[Verse]\n오늘 밤")
    assert "[Chorus]" in compiled.lyrics


def test_instrumental_uses_official_inst_lyrics_mechanism():
    compiled = AceStepPromptCompiler().compile(
        _request(vocal_gender=VocalGender.INSTRUMENTAL, instrumental=True)
    )
    assert compiled.lyrics == INSTRUMENTAL_LYRICS  # upstream: "[inst]"
    assert "instrumental, no vocals" in compiled.prompt
    assert compiled.instrumental is True


def test_empty_lyrics_becomes_instrumental():
    compiled = AceStepPromptCompiler().compile(_request(lyrics="   "))
    assert compiled.instrumental is True
    assert compiled.lyrics == INSTRUMENTAL_LYRICS


def test_language_passthrough_with_english_default():
    assert AceStepPromptCompiler().compile(_request()).vocal_language == "ko"
    assert AceStepPromptCompiler().compile(_request(language=None)).vocal_language == "en"
