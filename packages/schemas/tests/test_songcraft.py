"""Tests for the songcraft layer: parsing, advisories, and pre-flight.

Two properties matter more than any individual assertion here, and both
are asserted explicitly rather than assumed:

1. **Nothing in this module mutates user input.** Every advisory path is
   checked against the original lyric text.
2. **Nothing in this module blocks.** Advisories are data; there is no
   code path that raises to prevent a generation.

The parameter constants are asserted against the pinned ACE-Step build
(``acestep/constants.py`` @ 6d467e4b) — if an upgrade widens or narrows
the engine's accepted values, these tests are where that shows up.
"""

from __future__ import annotations

import pytest

from luber_schemas import (
    BPM_MAX,
    BPM_MIN,
    DURATION_MAX,
    DURATION_MIN,
    DURATION_PRESETS,
    LONG_FORM_THRESHOLD_SECONDS,
    SECTION_TAG_PALETTE,
    UNEXPOSED_ENGINE_PARAMETERS,
    VALID_KEY_SCALES,
    VALID_TIME_SIGNATURE_VALUES,
    Advisory,
    AdvisoryLevel,
    SectionKind,
    analyze_density,
    audit_korean_lyrics,
    estimate_syllables,
    korean_ratio,
    parse_structure,
    preflight,
    validate_structure,
)
from luber_schemas.songcraft import (
    KOREAN_LINE_SYLLABLE_LIMIT,
    MAX_SYLLABLES_PER_SECOND,
    MIN_SECONDS_PER_SECTION,
    classify_section_label,
)


def codes(advisories: list[Advisory]) -> set[str]:
    return {a.code for a in advisories}


# ── Engine-verified parameter surface ─────────────────────────────────


def test_bpm_bounds_match_pinned_engine():
    # acestep/constants.py: BPM_MIN = 30, BPM_MAX = 300.
    assert (BPM_MIN, BPM_MAX) == (30, 300)


def test_duration_bounds_are_engine_min_and_luber_cap():
    # Upstream allows 10-600; LUBER caps at 360 (verified path only).
    assert DURATION_MIN == 10
    assert DURATION_MAX == 360
    assert DURATION_MAX < 600


def test_time_signatures_are_bare_numerators_from_engine():
    # Upstream VALID_TIME_SIGNATURES = [2, 3, 4, 6]; the value that
    # reaches the metadata block is the numerator, not "4/4".
    assert VALID_TIME_SIGNATURE_VALUES == ("2", "3", "4", "6")
    assert all("/" not in value for value in VALID_TIME_SIGNATURE_VALUES)


def test_key_scales_cover_notes_accidentals_and_modes():
    assert len(VALID_KEY_SCALES) == 7 * 3 * 2
    assert "C major" in VALID_KEY_SCALES
    assert "F# minor" in VALID_KEY_SCALES
    assert "Bb major" in VALID_KEY_SCALES


def test_key_scales_expose_ascii_accidentals_only():
    # Upstream also accepts ♯/♭; LUBER offers one spelling per key so a
    # stored value has exactly one representation.
    assert not any("♯" in k or "♭" in k for k in VALID_KEY_SCALES)


@pytest.mark.parametrize("bad", ["H major", "C dorian", "c major", "C", "", "C  major"])
def test_obviously_wrong_key_scales_are_not_offered(bad):
    assert bad not in VALID_KEY_SCALES


def test_duration_presets_are_within_bounds_and_sorted():
    assert list(DURATION_PRESETS) == sorted(DURATION_PRESETS)
    assert all(DURATION_MIN <= d <= DURATION_MAX for d in DURATION_PRESETS)
    assert LONG_FORM_THRESHOLD_SECONDS in DURATION_PRESETS


def test_unexposed_engine_parameters_each_record_a_reason():
    assert UNEXPOSED_ENGINE_PARAMETERS
    for name, reason in UNEXPOSED_ENGINE_PARAMETERS.items():
        assert name and reason.strip(), name


def test_section_tag_palette_is_bracketed_and_recognised():
    for tag in SECTION_TAG_PALETTE:
        assert tag.startswith("[") and tag.endswith("]")
        kind, _ = classify_section_label(tag[1:-1])
        assert kind is not None, tag


# ── Section parsing ───────────────────────────────────────────────────


def test_parses_tagged_sections_in_order():
    lyrics = "[Verse 1]\nline a\nline b\n[Chorus]\nhook line"
    parsed = parse_structure(lyrics)

    assert parsed.is_tagged
    assert [s.kind for s in parsed.sections] == [SectionKind.VERSE, SectionKind.CHORUS]
    assert parsed.sections[0].label == "Verse 1"
    assert parsed.sections[0].index == 1
    assert parsed.sections[0].line_number == 1
    assert parsed.sections[0].lines == ("line a", "line b")
    assert parsed.sections[1].line_number == 4


def test_untagged_lyrics_are_valid_and_land_in_preamble():
    parsed = parse_structure("just a line\nand another")
    assert not parsed.is_tagged
    assert parsed.sections == ()
    assert parsed.preamble == ("just a line", "and another")


def test_lines_before_the_first_tag_are_kept_as_preamble():
    parsed = parse_structure("stray line\n[Verse]\nsung line")
    assert parsed.preamble == ("stray line",)
    assert len(parsed.sections) == 1


def test_parsing_preserves_text_verbatim():
    lyrics = "[Verse]\n  indented line  \n\ttabbed\n"
    parsed = parse_structure(lyrics)
    assert parsed.sections[0].lines == ("  indented line  ", "\ttabbed")
    # Reassembling the section returns exactly what was typed.
    assert parsed.sections[0].text == "  indented line  \n\ttabbed"


def test_inline_brackets_are_not_section_tags():
    # An ad-lib sharing a line with lyrics is not a section marker.
    parsed = parse_structure("she said [whisper] softly\n[Chorus]\nhook")
    assert len(parsed.sections) == 1
    assert parsed.sections[0].kind is SectionKind.CHORUS
    assert parsed.preamble == ("she said [whisper] softly",)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Verse", SectionKind.VERSE),
        ("verse", SectionKind.VERSE),
        ("  VERSE  ", SectionKind.VERSE),
        ("Verse 2", SectionKind.VERSE),
        ("Pre-Chorus", SectionKind.PRE_CHORUS),
        ("PreChorus", SectionKind.PRE_CHORUS),
        ("pre chorus", SectionKind.PRE_CHORUS),
        ("Hook", SectionKind.CHORUS),
        ("Breakdown", SectionKind.BREAK),
        ("Solo", SectionKind.INSTRUMENTAL),
        ("Ending", SectionKind.OUTRO),
        ("Introduction", SectionKind.INTRO),
    ],
)
def test_section_aliases_map_to_canonical_kinds(label, expected):
    kind, _ = classify_section_label(label)
    assert kind is expected


@pytest.mark.parametrize("label", ["Drop", "후렴", "Verse!!", "12", ""])
def test_unknown_labels_classify_as_unrecognised(label):
    kind, _ = classify_section_label(label)
    assert kind is None


def test_unknown_tag_is_still_parsed_as_a_section():
    parsed = parse_structure("[Drop]\nboom")
    assert len(parsed.sections) == 1
    assert parsed.sections[0].kind is None
    assert parsed.sections[0].is_recognised is False
    assert parsed.sections[0].label == "Drop"


def test_section_ordinals_are_extracted():
    _, index = classify_section_label("Verse 3")
    assert index == 3
    _, no_index = classify_section_label("Chorus")
    assert no_index is None


def test_empty_and_whitespace_lyrics_parse_to_nothing():
    assert parse_structure("").sections == ()
    assert parse_structure("   \n  ").sections == ()


def test_has_content_distinguishes_empty_sections():
    parsed = parse_structure("[Verse]\n\n   \n[Chorus]\nreal words")
    assert parsed.sections[0].has_content is False
    assert parsed.sections[1].has_content is True


def test_count_and_kinds_helpers():
    parsed = parse_structure("[Verse]\na\n[Chorus]\nb\n[Chorus]\nc")
    assert parsed.count(SectionKind.CHORUS) == 2
    assert parsed.count(SectionKind.BRIDGE) == 0
    assert parsed.kinds == (SectionKind.VERSE, SectionKind.CHORUS, SectionKind.CHORUS)


# ── Structure advisories ──────────────────────────────────────────────


def test_untagged_lyrics_produce_no_structure_advisories():
    # Plain lyrics are a legitimate way to write a song.
    assert validate_structure(parse_structure("just some words\nno tags here")) == []


def test_unknown_tag_warns_but_says_it_is_passed_through():
    found = validate_structure(parse_structure("[Drop]\nboom\n[Chorus]\nhook"))
    advisory = next(a for a in found if a.code == "UNKNOWN_SECTION_TAG")
    assert advisory.level is AdvisoryLevel.WARNING
    assert advisory.detail["labels"] == ["Drop"]
    assert "exactly as written" in advisory.message


def test_empty_section_warns():
    found = validate_structure(parse_structure("[Verse]\n\n[Chorus]\nhook"))
    advisory = next(a for a in found if a.code == "EMPTY_SECTION")
    assert advisory.detail["labels"] == ["Verse"]


def test_missing_chorus_is_info_not_warning():
    found = validate_structure(parse_structure("[Verse]\na\n[Bridge]\nb"))
    advisory = next(a for a in found if a.code == "NO_CHORUS")
    assert advisory.level is AdvisoryLevel.INFO


def test_duplicate_section_numbering_warns():
    found = validate_structure(parse_structure("[Verse 1]\na\n[Verse 1]\nb\n[Chorus]\nc"))
    advisory = next(a for a in found if a.code == "DUPLICATE_SECTION_NUMBER")
    assert advisory.detail["labels"] == ["[Verse 1]"]


def test_distinct_numbering_does_not_warn():
    found = validate_structure(parse_structure("[Verse 1]\na\n[Verse 2]\nb\n[Chorus]\nc"))
    assert "DUPLICATE_SECTION_NUMBER" not in codes(found)


def test_intro_not_first_and_outro_not_last_are_info():
    lyrics = "[Chorus]\na\n[Intro]\nb\n[Outro]\nc\n[Verse]\nd"
    found = validate_structure(parse_structure(lyrics))
    assert {"INTRO_NOT_FIRST", "OUTRO_NOT_LAST"} <= codes(found)
    for advisory in found:
        if advisory.code in {"INTRO_NOT_FIRST", "OUTRO_NOT_LAST"}:
            assert advisory.level is AdvisoryLevel.INFO


def test_well_ordered_song_has_no_ordering_advisories():
    lyrics = "[Intro]\na\n[Verse]\nb\n[Chorus]\nc\n[Outro]\nd"
    found = codes(validate_structure(parse_structure(lyrics)))
    assert not ({"INTRO_NOT_FIRST", "OUTRO_NOT_LAST", "NO_CHORUS"} & found)


def test_instrumental_skips_vocal_structure_advisories():
    # An instrumental with empty tagged sections is normal, not a defect.
    found = codes(validate_structure(parse_structure("[Verse]\n\n[Break]\n"), instrumental=True))
    assert "EMPTY_SECTION" not in found
    assert "NO_CHORUS" not in found


def test_instrumental_with_lyrics_warns_they_will_not_be_sung():
    found = validate_structure(parse_structure("[Verse]\n노래 가사"), instrumental=True)
    advisory = next(a for a in found if a.code == "LYRICS_IN_INSTRUMENTAL")
    assert advisory.level is AdvisoryLevel.WARNING
    assert advisory.detail["labels"] == ["Verse"]


def test_non_vocal_sections_may_be_empty_without_warning():
    found = codes(validate_structure(parse_structure("[Break]\n\n[Instrumental]\n\n[Chorus]\nx")))
    assert "EMPTY_SECTION" not in found


# ── Syllable estimation ───────────────────────────────────────────────


def test_hangul_counts_one_syllable_per_block():
    assert estimate_syllables("안녕하세요") == 5


def test_latin_counts_by_vowel_group():
    assert estimate_syllables("hello") == 2
    assert estimate_syllables("the rain in spain") == 4


def test_vowelless_word_still_counts_as_one():
    assert estimate_syllables("rhythm") == 1
    assert estimate_syllables("hmm") == 1


def test_mixed_script_sums_both_systems():
    assert estimate_syllables("안녕 hello") == 2 + 2


def test_section_tags_are_not_counted_as_lyrics():
    assert estimate_syllables("[Verse]\n안녕") == 2
    assert estimate_syllables("[Chorus]") == 0


def test_punctuation_and_blank_text_count_zero():
    assert estimate_syllables("") == 0
    assert estimate_syllables("!!! ... ,,,") == 0


# ── Density advisories ────────────────────────────────────────────────


def test_dense_lyrics_warn_and_suggest_a_longer_duration():
    lyrics = "\n".join(["가나다라마바사아자차카타파하" for _ in range(12)])
    found = analyze_density(lyrics, 30)
    advisory = next(a for a in found if a.code == "LYRICS_DENSE_FOR_DURATION")
    assert advisory.level is AdvisoryLevel.WARNING
    assert advisory.detail["density_per_second"] > MAX_SYLLABLES_PER_SECOND
    assert advisory.detail["suggested_duration_seconds"] > 30


def test_sparse_lyrics_are_info_and_never_a_warning():
    found = analyze_density("안녕", 240)
    advisory = next(a for a in found if a.code == "LYRICS_SPARSE_FOR_DURATION")
    assert advisory.level is AdvisoryLevel.INFO
    assert advisory.detail["suggested_duration_seconds"] >= DURATION_MIN


def test_comfortable_density_produces_no_advisory():
    # ~60 syllables over 30s ≈ 2.7/s of singable time — inside the band.
    lyrics = "\n".join(["가나다라마바사아자차" for _ in range(6)])
    assert codes(analyze_density(lyrics, 30)) == set()


def test_empty_lyrics_produce_no_density_advisory():
    assert analyze_density("", 180) == []
    assert analyze_density("[Verse]\n[Chorus]", 180) == []


def test_instrumental_skips_density_entirely():
    dense = "\n".join(["가나다라마바사아자차카타파하" for _ in range(20)])
    assert analyze_density(dense, 30, instrumental=True) == []


def test_too_many_sections_for_the_duration_warns():
    lyrics = "".join(f"[Verse {n}]\n가나다라마\n" for n in range(1, 9))
    found = analyze_density(lyrics, 30)
    advisory = next(a for a in found if a.code == "MANY_SECTIONS_FOR_DURATION")
    assert advisory.detail["sections"] == 8
    assert advisory.detail["max_recommended"] == 30 // MIN_SECONDS_PER_SECTION


def test_section_count_within_budget_does_not_warn():
    lyrics = "[Verse]\n가나다라마바사\n[Chorus]\n아자차카타파하\n"
    assert "MANY_SECTIONS_FOR_DURATION" not in codes(analyze_density(lyrics, 120))


# ── Korean pre-flight ─────────────────────────────────────────────────


def test_korean_ratio_counts_only_letters():
    assert korean_ratio("안녕") == 1.0
    assert korean_ratio("hello") == 0.0
    assert korean_ratio("!!! ???") == 0.0
    assert korean_ratio("안녕ab") == pytest.approx(0.5)


def test_korean_lyrics_without_korean_language_warns():
    found = audit_korean_lyrics("오늘 밤 너를 생각해", language="en")
    advisory = next(a for a in found if a.code == "KOREAN_LYRICS_LANGUAGE_MISMATCH")
    assert advisory.level is AdvisoryLevel.WARNING
    assert advisory.detail["language"] == "en"


def test_unset_language_with_korean_lyrics_also_warns():
    found = audit_korean_lyrics("오늘 밤 너를 생각해", language=None)
    assert "KOREAN_LYRICS_LANGUAGE_MISMATCH" in codes(found)


def test_korean_language_without_korean_lyrics_warns():
    found = audit_korean_lyrics("tonight I think of you again", language="ko")
    advisory = next(a for a in found if a.code == "KOREAN_LANGUAGE_WITHOUT_KOREAN_LYRICS")
    assert advisory.level is AdvisoryLevel.WARNING


def test_matched_korean_lyrics_and_language_produce_no_mismatch():
    found = codes(audit_korean_lyrics("오늘 밤 너를 생각해", language="ko"))
    assert "KOREAN_LYRICS_LANGUAGE_MISMATCH" not in found
    assert "KOREAN_LANGUAGE_WITHOUT_KOREAN_LYRICS" not in found


def test_language_comparison_is_case_insensitive():
    found = codes(audit_korean_lyrics("오늘 밤 너를 생각해", language="KO"))
    assert "KOREAN_LYRICS_LANGUAGE_MISMATCH" not in found


def test_decomposed_hangul_is_recognised_as_korean():
    # macOS filesystems and some IMEs hand back NFD.
    import unicodedata

    nfd = unicodedata.normalize("NFD", "오늘 밤 너를 생각해")
    assert nfd != "오늘 밤 너를 생각해"  # guard: the fixture really is decomposed
    assert "KOREAN_LYRICS_LANGUAGE_MISMATCH" in codes(audit_korean_lyrics(nfd, language="en"))


def test_overlong_korean_line_is_info():
    long_line = "가" * (KOREAN_LINE_SYLLABLE_LIMIT + 1)
    found = audit_korean_lyrics(long_line, language="ko")
    advisory = next(a for a in found if a.code == "KOREAN_LINE_TOO_LONG")
    assert advisory.level is AdvisoryLevel.INFO
    assert advisory.detail["lines"] == [1]


def test_line_at_the_limit_does_not_warn():
    at_limit = "가" * KOREAN_LINE_SYLLABLE_LIMIT
    assert "KOREAN_LINE_TOO_LONG" not in codes(audit_korean_lyrics(at_limit, language="ko"))


def test_mixed_script_line_is_flagged_for_pronunciation():
    found = audit_korean_lyrics("오늘 밤 lonely midnight feeling", language="ko")
    advisory = next(a for a in found if a.code == "KOREAN_MIXED_SCRIPT_LINE")
    assert advisory.level is AdvisoryLevel.INFO
    assert advisory.detail["lines"] == [1]


def test_one_english_word_in_a_korean_line_is_tolerated():
    found = codes(audit_korean_lyrics("오늘 밤 baby", language="ko"))
    assert "KOREAN_MIXED_SCRIPT_LINE" not in found


def test_pure_english_lyrics_skip_korean_line_checks():
    english = "tonight I walk the quiet street and think of everything we said before"
    found = codes(audit_korean_lyrics(english, language="en"))
    assert found == set()


def test_empty_lyrics_produce_no_korean_advisories():
    assert audit_korean_lyrics("", language="ko") == []
    assert audit_korean_lyrics("[Verse]\n[Chorus]", language="ko") == []


# ── Combined pre-flight ───────────────────────────────────────────────


def test_preflight_aggregates_all_three_families():
    lyrics = "[Drop]\n" + "\n".join(["가나다라마바사아자차카타파하" for _ in range(12)])
    found = codes(preflight(lyrics=lyrics, duration_seconds=30, language="en", instrumental=False))
    assert "UNKNOWN_SECTION_TAG" in found  # structure
    assert "LYRICS_DENSE_FOR_DURATION" in found  # density
    assert "KOREAN_LYRICS_LANGUAGE_MISMATCH" in found  # korean


def test_preflight_reports_in_structure_density_korean_order():
    lyrics = "[Drop]\n" + "\n".join(["가나다라마바사아자차카타파하" for _ in range(12)])
    found = [
        a.code
        for a in preflight(lyrics=lyrics, duration_seconds=30, language="en", instrumental=False)
    ]
    assert found.index("UNKNOWN_SECTION_TAG") < found.index("LYRICS_DENSE_FOR_DURATION")
    assert found.index("LYRICS_DENSE_FOR_DURATION") < found.index("KOREAN_LYRICS_LANGUAGE_MISMATCH")


def test_clean_korean_request_is_advisory_free():
    lyrics = (
        "[Verse]\n오늘도 너를 기다려\n조용한 창가에 앉아\n"
        "[Chorus]\n다시 만날 그날까지\n나는 여기 있을게\n"
    )
    # 31 syllables over 30s ≈ 1.4/s of singable time — inside the band.
    found = preflight(lyrics=lyrics, duration_seconds=30, language="ko", instrumental=False)
    assert found == []


def test_preflight_on_plain_untagged_lyrics_does_not_demand_structure():
    lyrics = "오늘도 너를 기다려\n조용한 창가에 앉아\n다시 만날 그날까지"
    found = codes(preflight(lyrics=lyrics, duration_seconds=60, language="ko", instrumental=False))
    assert "UNKNOWN_SECTION_TAG" not in found
    assert "NO_CHORUS" not in found


def test_preflight_for_instrumental_ignores_lyric_heuristics():
    dense = "\n".join(["가나다라마바사아자차카타파하" for _ in range(20)])
    found = codes(preflight(lyrics=dense, duration_seconds=30, language=None, instrumental=True))
    assert "LYRICS_DENSE_FOR_DURATION" not in found
    assert "MANY_SECTIONS_FOR_DURATION" not in found


def test_preflight_never_mutates_the_lyrics_it_reads():
    original = "[Verse]\n  오늘 밤 lonely midnight  \n\n[Drop]\n" + "가" * 40
    snapshot = original
    preflight(lyrics=original, duration_seconds=30, language="en", instrumental=False)
    assert original == snapshot


def test_preflight_never_raises_on_hostile_input():
    # Advisories must degrade to "nothing to say", never to an exception:
    # a heuristic crash would block a generation the user is entitled to.
    for lyrics in ["", "[", "]", "[]", "[" * 500, "\x00\x01", "🎵🎶", "[Verse]" * 300]:
        for instrumental in (False, True):
            preflight(
                lyrics=lyrics,
                duration_seconds=DURATION_MIN,
                language=None,
                instrumental=instrumental,
            )


def test_advisories_serialize_to_plain_json_types():
    lyrics = "[Drop]\n" + "가" * 300
    for advisory in preflight(
        lyrics=lyrics, duration_seconds=30, language="en", instrumental=False
    ):
        payload = advisory.to_dict()
        assert set(payload) == {"code", "level", "message", "detail"}
        assert isinstance(payload["level"], str)
        assert isinstance(payload["detail"], dict)


def test_advisory_levels_are_only_info_or_warning():
    lyrics = "[Drop]\n" + "가" * 300
    found = preflight(lyrics=lyrics, duration_seconds=30, language="en", instrumental=False)
    assert {a.level for a in found} <= {AdvisoryLevel.INFO, AdvisoryLevel.WARNING}
