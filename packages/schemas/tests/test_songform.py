"""Full-song presets, structure templates, and the lyric budget engine.

The rules these tests defend, in order of importance:

1. **A template never destroys writing.** Applying a template to a sheet
   that has lyrics in it appends; it does not overwrite. Overwriting
   requires an explicit ``replace=True`` from a caller that has already
   asked the user.
2. **Templates are conditioning aids, not controls.** Nothing here
   claims the model obeys them, and the tests only assert what the text
   contains.
3. **One owner per advisory.** Total lyric density is reported by
   ``analyze_density`` below the full-song threshold and by
   ``analyze_lyric_budget`` at or above it — never by both, so the
   editor never shows the same advice twice.
"""

from __future__ import annotations

import pytest

from luber_schemas import (
    FULL_SONG_THRESHOLD_SECONDS,
    PRODUCT_DURATIONS,
    PRODUCT_MAX_DURATION,
    SONG_PRESETS,
    STRUCTURE_TEMPLATES,
    TEMPLATES_BY_ID,
    SectionKind,
    analyze_density,
    analyze_lyric_budget,
    apply_template,
    compute_lyric_budget,
    count_hangul_syllables,
    describe_template_fit,
    lyrics_have_content,
    parse_structure,
    preflight,
    validate_product_duration,
)
from luber_schemas.songform import PRESETS_BY_ID, template_for_structure


def codes(advisories) -> set[str]:
    return {a.code for a in advisories}


# ── Product duration surface ──────────────────────────────────────────


def test_product_durations_are_the_validated_set():
    # 30/60 from Phase 3; 120/180/240 from the Phase 9 long-form gates.
    assert PRODUCT_DURATIONS == (30, 60, 120, 180, 240)


def test_product_ceiling_is_240_not_the_engine_maximum():
    # Upstream accepts 600s. We do not offer what we have not validated.
    assert PRODUCT_MAX_DURATION == 240
    assert 360 not in PRODUCT_DURATIONS
    assert 600 not in PRODUCT_DURATIONS


@pytest.mark.parametrize("duration", PRODUCT_DURATIONS)
def test_every_offered_duration_validates(duration):
    assert validate_product_duration(duration)


@pytest.mark.parametrize("duration", [0, 45, 300, 360, 600])
def test_unoffered_durations_do_not_validate(duration):
    assert not validate_product_duration(duration)


# ── Structure templates ───────────────────────────────────────────────


def test_every_template_uses_only_recognised_tags():
    # A template that warns about its own tags would be embarrassing —
    # and did happen on the first real 180s run with [Final Chorus].
    for template in STRUCTURE_TEMPLATES:
        parsed = parse_structure(template.text)
        unknown = [s.label for s in parsed.sections if not s.is_recognised]
        assert unknown == [], f"{template.id}: {unknown}"


def test_every_template_parses_to_its_own_section_count():
    for template in STRUCTURE_TEMPLATES:
        parsed = parse_structure(template.text)
        assert len(parsed.sections) == len(template.sections), template.id


def test_pop_template_matches_the_specified_shape():
    kinds = [s.kind for s in parse_structure(TEMPLATES_BY_ID["pop"].text).sections]
    assert kinds == [
        SectionKind.INTRO,
        SectionKind.VERSE,
        SectionKind.PRE_CHORUS,
        SectionKind.CHORUS,
        SectionKind.VERSE,
        SectionKind.PRE_CHORUS,
        SectionKind.CHORUS,
        SectionKind.BRIDGE,
        SectionKind.CHORUS,
        SectionKind.OUTRO,
    ]


def test_ballad_template_is_verse_led():
    kinds = [s.kind for s in parse_structure(TEMPLATES_BY_ID["ballad"].text).sections]
    # Three verses before the bridge is what makes it a ballad shape.
    assert kinds.count(SectionKind.VERSE) == 3
    assert kinds[0] is SectionKind.INTRO and kinds[-1] is SectionKind.OUTRO


def test_rnb_template_matches_the_specified_shape():
    kinds = [s.kind for s in parse_structure(TEMPLATES_BY_ID["rnb"].text).sections]
    assert kinds == [
        SectionKind.INTRO,
        SectionKind.VERSE,
        SectionKind.PRE_CHORUS,
        SectionKind.CHORUS,
        SectionKind.VERSE,
        SectionKind.CHORUS,
        SectionKind.BRIDGE,
        SectionKind.OUTRO,
    ]


def test_templates_never_warn_about_their_own_tags():
    for template in STRUCTURE_TEMPLATES:
        found = codes(
            preflight(
                lyrics=template.text,
                duration_seconds=template.suggested_duration,
                language=None,
                instrumental=False,
            )
        )
        assert "UNKNOWN_SECTION_TAG" not in found, template.id


def test_an_unfilled_template_asks_for_words_but_not_for_intro_or_outro():
    """A bare skeleton *should* say its verses need lyrics.

    That is the advisory doing its job. What it must not do is demand
    words for [Intro] and [Outro], which are routinely instrumental —
    the false positive seen on the first real full-song request.
    """
    advisories = [
        a
        for a in preflight(
            lyrics=TEMPLATES_BY_ID["pop"].text,
            duration_seconds=180,
            language=None,
            instrumental=False,
        )
        if a.code == "EMPTY_SECTION"
    ]
    assert len(advisories) == 1
    labels = advisories[0].detail["labels"]
    assert "Intro" not in labels and "Outro" not in labels
    assert "Verse 1" in labels and "Chorus" in labels


def test_template_text_is_one_tag_per_line():
    lines = [line for line in TEMPLATES_BY_ID["pop"].text.splitlines() if line.strip()]
    assert lines == list(TEMPLATES_BY_ID["pop"].sections)


def test_template_round_trips_through_its_own_detector():
    for template in STRUCTURE_TEMPLATES:
        detected = template_for_structure(parse_structure(template.text))
        assert detected == template.id, template.id


def test_unfamiliar_structure_detects_as_no_template():
    assert template_for_structure(parse_structure("[Verse]\na\n[Bridge]\nb")) is None


# ── Applying a template never destroys writing ────────────────────────


def test_template_fills_an_empty_sheet():
    assert apply_template("", TEMPLATES_BY_ID["minimal"]) == "[Verse]\n\n[Chorus]\n"


def test_template_replaces_a_bare_skeleton():
    # Only tags, no words: swapping shapes loses nothing.
    result = apply_template("[Verse]\n[Chorus]\n", TEMPLATES_BY_ID["pop"])
    assert result == TEMPLATES_BY_ID["pop"].text


def test_template_appends_rather_than_destroying_lyrics():
    written = "[Verse]\n창밖에 비가 내려와\n"
    result = apply_template(written, TEMPLATES_BY_ID["pop"])
    assert result.startswith(written)
    assert "창밖에 비가 내려와" in result
    assert "[Bridge]" in result


def test_template_replaces_only_when_explicitly_asked():
    written = "[Verse]\n창밖에 비가 내려와\n"
    result = apply_template(written, TEMPLATES_BY_ID["pop"], replace=True)
    assert "창밖에 비가 내려와" not in result
    assert result == TEMPLATES_BY_ID["pop"].text


def test_apply_template_does_not_mutate_its_input():
    written = "[Verse]\n창밖에 비가 내려와\n"
    snapshot = written
    apply_template(written, TEMPLATES_BY_ID["pop"])
    assert written == snapshot


def test_preamble_counts_as_content():
    assert lyrics_have_content("just a line with no tags")
    assert not lyrics_have_content("[Verse]\n[Chorus]")
    assert not lyrics_have_content("   \n  ")


def test_appending_to_untagged_lyrics_keeps_them():
    result = apply_template("plain words here", TEMPLATES_BY_ID["minimal"])
    assert "plain words here" in result
    assert "[Chorus]" in result


# ── Presets ───────────────────────────────────────────────────────────


def test_presets_cover_the_requested_set():
    names = {p.name for p in SONG_PRESETS}
    assert {"Short Demo", "Full Pop Song", "Ballad", "R&B", "Band Song", "Instrumental"} <= names


def test_every_preset_duration_is_offered_by_the_product():
    for preset in SONG_PRESETS:
        assert validate_product_duration(preset.duration), preset.id


def test_every_preset_template_resolves():
    for preset in SONG_PRESETS:
        if preset.template_id is not None:
            assert preset.template is not None, preset.id


def test_instrumental_preset_carries_no_structure():
    preset = PRESETS_BY_ID["instrumental"]
    assert preset.instrumental is True
    assert preset.template is None


def test_presets_carry_no_prompt_or_lyrics():
    # A preset shapes the frame; the writing stays the user's.
    for preset in SONG_PRESETS:
        payload = preset.to_dict()
        assert "prompt" not in payload
        assert "lyrics" not in payload


def test_full_pop_song_preset_is_a_full_song():
    preset = PRESETS_BY_ID["full_pop_song"]
    assert preset.duration >= FULL_SONG_THRESHOLD_SECONDS
    assert preset.template is not None and len(preset.template.sections) == 10


def test_template_fit_note_flags_a_crowded_song():
    note = describe_template_fit(TEMPLATES_BY_ID["pop"], 30)
    assert note is not None and "sections" in note


def test_template_fit_note_is_quiet_at_the_suggested_duration():
    for template in STRUCTURE_TEMPLATES:
        assert describe_template_fit(template, template.suggested_duration) is None, template.id


# ── Lyric budget ──────────────────────────────────────────────────────

VERSE_HEAVY = """[Verse 1]
창밖에 비가 내려와
너의 이름을 불러봐
흐릿한 유리창 너머
지난 여름이 스쳐가
낡은 사진 한 장에
우리 웃음이 남아서
조용한 방에 앉아서
지난 계절을 세어봐
다시 만날 그날까지
[Chorus]
나는 여기 있을게
"""


def test_budget_counts_lines_sections_and_syllables():
    budget = compute_lyric_budget(VERSE_HEAVY, 180)
    assert budget.section_count == 2
    assert budget.verse_count == 1
    assert budget.chorus_count == 1
    assert budget.verse_lines == 9
    assert budget.chorus_lines == 1
    assert budget.total_lines == 10
    assert budget.total_syllables > 0


def test_budget_counts_hangul_blocks_separately():
    budget = compute_lyric_budget("[Verse]\n안녕하세요 hello", 60)
    assert budget.hangul_syllables == 5  # the Latin word is excluded
    assert budget.total_syllables > budget.hangul_syllables


def test_hangul_counting_handles_decomposed_input():
    import unicodedata

    nfd = unicodedata.normalize("NFD", "안녕하세요")
    assert nfd != "안녕하세요"
    assert count_hangul_syllables(nfd) == 5


def test_hangul_counting_ignores_section_tags():
    assert count_hangul_syllables("[Verse]\n안녕") == 2


def test_budget_knows_what_a_full_song_is():
    assert not compute_lyric_budget("가사", 60).is_full_song
    assert compute_lyric_budget("가사", FULL_SONG_THRESHOLD_SECONDS).is_full_song
    assert compute_lyric_budget("가사", 240).is_full_song


def test_verse_overload_is_flagged():
    found = codes(analyze_lyric_budget(VERSE_HEAVY, 180))
    assert "VERSE_OVERLOAD" in found


def test_chorus_overload_is_flagged():
    chorus_heavy = "[Verse]\n한 줄\n[Chorus]\n" + "\n".join(f"후렴 {n}" for n in range(1, 9))
    assert "CHORUS_OVERLOAD" in codes(analyze_lyric_budget(chorus_heavy, 180))


def test_too_many_lyrics_is_flagged_for_a_full_song():
    dense = "[Verse]\n" + "\n".join("가나다라마바사아자차카타파하" for _ in range(60))
    assert "TOO_MANY_LYRICS" in codes(analyze_lyric_budget(dense, 180))


def test_too_few_lyrics_is_flagged_for_a_full_song():
    assert "TOO_FEW_LYRICS" in codes(analyze_lyric_budget("[Verse]\n한 줄", 240))


def test_section_overload_flags_a_thin_full_song():
    # Two sections spread across four minutes is ~120s each.
    assert "SECTION_OVERLOAD" in codes(analyze_lyric_budget("[Verse]\n가사\n[Chorus]\n후렴", 240))


def test_a_well_proportioned_full_song_is_quiet():
    lyrics = TEMPLATES_BY_ID["pop"].text.replace(
        "[Verse 1]", "[Verse 1]\n창밖에 비가 내려와\n너의 이름을 불러봐\n흐릿한 유리창 너머"
    )
    # Only asserting the overload family stays quiet; density depends on
    # how much of the skeleton is filled in.
    found = codes(analyze_lyric_budget(lyrics, 180))
    assert "VERSE_OVERLOAD" not in found
    assert "CHORUS_OVERLOAD" not in found
    assert "SECTION_OVERLOAD" not in found


def test_budget_is_silent_for_instrumentals():
    assert analyze_lyric_budget(VERSE_HEAVY, 240, instrumental=True) == []


def test_budget_is_silent_without_lyrics():
    assert analyze_lyric_budget("[Verse]\n[Chorus]", 240) == []


def test_budget_never_mutates_the_lyrics():
    snapshot = VERSE_HEAVY
    analyze_lyric_budget(VERSE_HEAVY, 180)
    compute_lyric_budget(VERSE_HEAVY, 180)
    assert VERSE_HEAVY == snapshot


def test_budget_detail_is_json_safe():
    for advisory in analyze_lyric_budget(VERSE_HEAVY, 180):
        payload = advisory.to_dict()
        assert isinstance(payload["detail"], dict)
        for value in payload["detail"].values():
            assert isinstance(value, (int, float, str, bool)), value


# ── Ownership split: exactly one reporter of total density ────────────


@pytest.mark.parametrize("duration", [30, 60, 119])
def test_short_requests_are_reported_by_analyze_density(duration):
    sparse = "[Verse]\n한 줄"
    assert "LYRICS_SPARSE_FOR_DURATION" in codes(analyze_density(sparse, duration))
    assert "TOO_FEW_LYRICS" not in codes(analyze_lyric_budget(sparse, duration))


@pytest.mark.parametrize("duration", [120, 180, 240])
def test_full_songs_are_reported_by_the_budget_engine(duration):
    sparse = "[Verse]\n한 줄"
    assert "LYRICS_SPARSE_FOR_DURATION" not in codes(analyze_density(sparse, duration))
    assert "TOO_FEW_LYRICS" in codes(analyze_lyric_budget(sparse, duration))


def test_density_ownership_switches_at_the_threshold():
    sparse = "[Verse]\n한 줄"
    below = codes(
        preflight(
            lyrics=sparse,
            duration_seconds=FULL_SONG_THRESHOLD_SECONDS - 1,
            language=None,
            instrumental=False,
        )
    )
    at = codes(
        preflight(
            lyrics=sparse,
            duration_seconds=FULL_SONG_THRESHOLD_SECONDS,
            language=None,
            instrumental=False,
        )
    )
    assert "LYRICS_SPARSE_FOR_DURATION" in below and "TOO_FEW_LYRICS" not in below
    assert "TOO_FEW_LYRICS" in at and "LYRICS_SPARSE_FOR_DURATION" not in at


@pytest.mark.parametrize("duration", [30, 120, 240])
def test_preflight_never_reports_total_density_twice(duration):
    dense = "[Verse]\n" + "\n".join("가나다라마바사아자차카타파하" for _ in range(60))
    found = [
        a.code
        for a in preflight(
            lyrics=dense, duration_seconds=duration, language="ko", instrumental=False
        )
    ]
    both = {"LYRICS_DENSE_FOR_DURATION", "TOO_MANY_LYRICS"} & set(found)
    assert len(both) == 1, found


# ── Intro/outro false positive fixed ──────────────────────────────────


def test_empty_intro_and_outro_do_not_warn():
    # Observed as a false positive on the first real full-song request.
    lyrics = "[Intro]\n[Verse]\n가사\n[Chorus]\n후렴\n[Outro]\n"
    assert "EMPTY_SECTION" not in codes(
        preflight(lyrics=lyrics, duration_seconds=180, language="ko", instrumental=False)
    )


def test_an_empty_verse_still_warns():
    lyrics = "[Intro]\n[Verse]\n\n[Chorus]\n후렴\n"
    assert "EMPTY_SECTION" in codes(
        preflight(lyrics=lyrics, duration_seconds=180, language="ko", instrumental=False)
    )


def test_final_chorus_counts_as_a_chorus():
    lyrics = "[Verse]\n가사\n[Final Chorus]\n후렴"
    parsed = parse_structure(lyrics)
    assert parsed.sections[1].kind is SectionKind.CHORUS
    assert "NO_CHORUS" not in codes(
        preflight(lyrics=lyrics, duration_seconds=60, language="ko", instrumental=False)
    )
