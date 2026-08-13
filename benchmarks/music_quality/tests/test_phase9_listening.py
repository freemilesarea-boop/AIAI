"""The Phase 9 listening package.

The design claim under test: **triage first, detail only when earned.**
At a 2/10 baseline, asking for eight rubric dimensions per track spends
listening time on precision that the quality does not support. So the
page must always ask for an overall score and the failure tags, and must
keep the detailed dimensions hidden until a track scores 5 or above.
"""

from __future__ import annotations

import json

from bench.phase9_listening import (
    DETAIL_DIMENSIONS,
    FAILURE_TAGS,
    LONG_FORM_SECTIONS,
    build_phase9_payload,
    render_phase9_page,
    write_package,
)

TRACK = {
    "case": "vocal_ballad_A_current",
    "generation_id": "d1d76e27-119a-41e8-a358-a492141efaba",
    "status": "COMPLETED",
    "duration_requested": 30,
    "lyrics": "[Verse]\n창밖에 비가 내려와\n너의 이름을 불러봐\n[Chorus]\n다시 만날 그날까지",
}
LONG_TRACK = dict(TRACK, case="gate_180s", duration_requested=180)


def test_payload_skips_unfinished_tracks():
    payload = build_phase9_payload([dict(TRACK, status="FAILED"), TRACK])
    assert [item["id"] for item in payload] == ["vocal_ballad_A_current"]


def test_payload_addresses_audio_by_generation_id():
    item = build_phase9_payload([TRACK])[0]
    assert TRACK["generation_id"] in item["audio_url"]
    assert "asset=preview" in item["audio_url"]
    assert "asset=master" in item["download_url"]


def test_expected_lines_exclude_section_tags():
    item = build_phase9_payload([TRACK])[0]
    assert item["expected_lines"] == [
        "창밖에 비가 내려와",
        "너의 이름을 불러봐",
        "다시 만날 그날까지",
    ]


def test_long_form_tracks_are_marked_and_get_sections():
    short = build_phase9_payload([TRACK])[0]
    long_form = build_phase9_payload([LONG_TRACK])[0]
    assert short["is_long_form"] is False and short["sections"] == []
    assert long_form["is_long_form"] is True
    assert long_form["sections"] == list(LONG_FORM_SECTIONS)


def test_page_asks_for_an_overall_score_first():
    page = render_phase9_page(build_phase9_payload([TRACK]))
    assert "Overall (1-10)" in page
    assert 'class="overall"' in page


def test_page_offers_every_required_failure_tag():
    page = render_phase9_page(build_phase9_payload([TRACK]))
    for code, _ in FAILURE_TAGS:
        assert f'value="{code}"' in page, code


def test_required_failure_tags_are_all_present():
    codes = {code for code, _ in FAILURE_TAGS}
    assert codes == {
        "KOREAN_LINE_OMISSION",
        "LYRIC_LINE_SKIP",
        "LYRIC_DUPLICATION",
        "TROT_LIKE_VOCAL",
        "VOCAL_STYLE_OUTDATED",
        "EXCESSIVE_SIBILANCE",
        "HIGH_END_OVERBOOST",
        "INSTRUMENT_FIDELITY_LOW",
        "STRUCTURE_COLLAPSE",
        "MELODY_DRIFT",
        "VOCAL_IDENTITY_DRIFT",
        "ENDING_FAILURE",
    }


def test_detail_block_starts_hidden():
    # The whole point: a 2/10 track must not demand eight sub-scores.
    page = render_phase9_page(build_phase9_payload([TRACK]))
    assert '<div class="detail" hidden>' in page
    assert "Number(overall.value) >= 5" in page


def test_detail_dimensions_are_present_but_gated():
    page = render_phase9_page(build_phase9_payload([TRACK]))
    for dimension in DETAIL_DIMENSIONS:
        assert dimension in page, dimension


def test_long_form_page_warns_against_judging_the_first_30_seconds():
    page = render_phase9_page(build_phase9_payload([LONG_TRACK]))
    assert "not the first 30 seconds" in page


def test_line_level_qa_offers_unknown_and_does_not_default_to_a_guess():
    page = render_phase9_page(build_phase9_payload([TRACK]))
    assert 'value="UNKNOWN"' in page
    assert 'value="SKIPPED"' in page
    assert "UNKNOWN is a real answer" in page
    # UNKNOWN is first, so the default selection is the honest one.
    options_start = page.index('__line__0"')
    assert page.index('value="UNKNOWN"', options_start) < page.index(
        'value="COMPLETE"', options_start
    )


def test_line_verdicts_match_the_api_vocabulary():
    from luber_schemas import LineVerdict

    page = render_phase9_page(build_phase9_payload([TRACK]))
    for verdict in LineVerdict:
        assert f'value="{verdict.value}"' in page, verdict


def test_failure_tags_match_the_api_vocabulary():
    from luber_schemas import FailureTag

    assert {code for code, _ in FAILURE_TAGS} == {tag.value for tag in FailureTag}


def test_korean_lyrics_survive_rendering():
    page = render_phase9_page(build_phase9_payload([TRACK]))
    assert "창밖에 비가 내려와" in page


def test_write_package_emits_page_and_manifest(tmp_path):
    page = write_package([TRACK, LONG_TRACK], tmp_path)
    assert page.is_file()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert [item["id"] for item in manifest] == ["vocal_ballad_A_current", "gate_180s"]


def test_empty_package_still_renders():
    assert "<form" in render_phase9_page([])
