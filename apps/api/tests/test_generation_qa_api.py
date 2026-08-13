"""Human QA endpoints: lyric-line completeness and full-song review.

The point of this surface is to capture a judgement no automated check
in this stack can make — whether the model actually sang each line. The
tests therefore focus on the properties that make the record
trustworthy later:

- the expected lines always come from the *submitted* lyrics, so a
  reviewer is never asked to judge a line that was never requested;
- ``UNKNOWN`` is a first-class answer, not a missing one;
- re-reviewing corrects the record instead of appending a second,
  conflicting opinion;
- recording a review never alters the generation it describes.
"""

from __future__ import annotations

import uuid

FULL_SONG = {
    "title": "QA Subject",
    "prompt": "Contemporary Korean pop ballad",
    "lyrics": (
        "[Intro]\n"
        "[Verse 1]\n창밖에 비가 내려와\n너의 이름을 불러봐\n"
        "[Chorus]\n다시 만날 그날까지\n"
        "[Outro]"
    ),
    "vocal_gender": "female",
    "duration": 180,
    "language": "ko",
    "bpm": 84,
    "key_scale": "A minor",
    "time_signature": "4",
}


async def _create(client, **overrides):
    payload = dict(FULL_SONG, **overrides)
    resp = await client.post("/v1/generations", json=payload)
    assert resp.status_code == 202, resp.text
    return resp.json()["generation_id"]


# ── Expected lines are derived from what was submitted ────────────────


async def test_expected_lines_come_from_the_submitted_lyrics(client):
    generation_id = await _create(client)
    body = (await client.get(f"/v1/generations/{generation_id}/qa")).json()

    assert [line["text"] for line in body["expected_lines"]] == [
        "창밖에 비가 내려와",
        "너의 이름을 불러봐",
        "다시 만날 그날까지",
    ]
    # Section tags are structure, not sung words.
    assert all("[" not in line["text"] for line in body["expected_lines"])


async def test_expected_lines_carry_their_section(client):
    generation_id = await _create(client)
    body = (await client.get(f"/v1/generations/{generation_id}/qa")).json()
    labels = [line["section_label"] for line in body["expected_lines"]]
    assert labels == ["Verse 1", "Verse 1", "Chorus"]


async def test_expected_lines_are_indexed_in_order(client):
    generation_id = await _create(client)
    body = (await client.get(f"/v1/generations/{generation_id}/qa")).json()
    assert [line["index"] for line in body["expected_lines"]] == [0, 1, 2]


async def test_untagged_lyrics_still_produce_expected_lines(client):
    generation_id = await _create(client, lyrics="오늘도 너를 기다려\n조용한 창가에 앉아")
    body = (await client.get(f"/v1/generations/{generation_id}/qa")).json()
    assert len(body["expected_lines"]) == 2
    assert body["expected_lines"][0]["section_label"] is None


async def test_a_fresh_generation_has_no_review(client):
    generation_id = await _create(client)
    body = (await client.get(f"/v1/generations/{generation_id}/qa")).json()
    assert body["reviewed"] is False
    assert body["overall_rating"] is None
    assert body["lyric_lines"] == []
    assert body["failure_tags"] == []


async def test_qa_for_a_missing_generation_is_404(client):
    assert (await client.get(f"/v1/generations/{uuid.uuid4()}/qa")).status_code == 404


# ── Recording a review ────────────────────────────────────────────────


async def test_records_a_triage_rating_and_failure_tags(client):
    generation_id = await _create(client)
    resp = await client.put(
        f"/v1/generations/{generation_id}/qa",
        json={
            "overall_rating": 2,
            "failure_tags": ["KOREAN_LINE_OMISSION", "TROT_LIKE_VOCAL"],
            "notes": "second line never sung",
            "reviewer": "human",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["overall_rating"] == 2
    assert set(body["failure_tags"]) == {"KOREAN_LINE_OMISSION", "TROT_LIKE_VOCAL"}
    assert body["reviewed"] is True
    assert body["notes"] == "second line never sung"


async def test_records_per_line_verdicts(client):
    generation_id = await _create(client)
    resp = await client.put(
        f"/v1/generations/{generation_id}/qa",
        json={
            "overall_rating": 3,
            "lyric_lines": [
                {
                    "line_index": 0,
                    "section_label": "Verse 1",
                    "line_text": "창밖에 비가 내려와",
                    "verdict": "COMPLETE",
                },
                {
                    "line_index": 1,
                    "section_label": "Verse 1",
                    "line_text": "너의 이름을 불러봐",
                    "verdict": "SKIPPED",
                    "note": "not sung at all",
                },
                {
                    "line_index": 2,
                    "section_label": "Chorus",
                    "line_text": "다시 만날 그날까지",
                    "verdict": "PARTIAL",
                },
            ],
        },
    )
    lines = resp.json()["lyric_lines"]
    assert [line["verdict"] for line in lines] == ["COMPLETE", "SKIPPED", "PARTIAL"]
    assert lines[1]["note"] == "not sung at all"


async def test_unknown_is_a_first_class_verdict(client):
    # On a dense mix a listener genuinely cannot always tell. Forcing a
    # guess would poison the record this exists to build.
    generation_id = await _create(client)
    resp = await client.put(
        f"/v1/generations/{generation_id}/qa",
        json={
            "lyric_lines": [
                {"line_index": 0, "line_text": "창밖에 비가 내려와", "verdict": "UNKNOWN"}
            ]
        },
    )
    assert resp.status_code == 200
    assert resp.json()["lyric_lines"][0]["verdict"] == "UNKNOWN"


async def test_duplicated_verdict_is_supported(client):
    generation_id = await _create(client)
    resp = await client.put(
        f"/v1/generations/{generation_id}/qa",
        json={
            "lyric_lines": [
                {"line_index": 0, "line_text": "창밖에 비가 내려와", "verdict": "DUPLICATED"}
            ]
        },
    )
    assert resp.json()["lyric_lines"][0]["verdict"] == "DUPLICATED"


async def test_review_can_be_rating_only(client):
    # A reviewer recording "2/10, trot-like" must not be forced to
    # adjudicate every lyric line first.
    generation_id = await _create(client)
    resp = await client.put(f"/v1/generations/{generation_id}/qa", json={"overall_rating": 2})
    assert resp.status_code == 200
    assert resp.json()["lyric_lines"] == []


async def test_rereviewing_corrects_rather_than_appends(client):
    generation_id = await _create(client)
    await client.put(
        f"/v1/generations/{generation_id}/qa",
        json={
            "overall_rating": 2,
            "lyric_lines": [
                {"line_index": 0, "line_text": "창밖에 비가 내려와", "verdict": "SKIPPED"}
            ],
        },
    )
    second = await client.put(
        f"/v1/generations/{generation_id}/qa",
        json={
            "overall_rating": 5,
            "lyric_lines": [
                {"line_index": 0, "line_text": "창밖에 비가 내려와", "verdict": "COMPLETE"}
            ],
        },
    )
    body = second.json()
    assert body["overall_rating"] == 5
    assert len(body["lyric_lines"]) == 1
    assert body["lyric_lines"][0]["verdict"] == "COMPLETE"


async def test_shortening_a_review_drops_the_stale_lines(client):
    generation_id = await _create(client)
    await client.put(
        f"/v1/generations/{generation_id}/qa",
        json={
            "lyric_lines": [
                {"line_index": 0, "line_text": "a", "verdict": "COMPLETE"},
                {"line_index": 1, "line_text": "b", "verdict": "COMPLETE"},
                {"line_index": 2, "line_text": "c", "verdict": "COMPLETE"},
            ]
        },
    )
    second = await client.put(
        f"/v1/generations/{generation_id}/qa",
        json={"lyric_lines": [{"line_index": 0, "line_text": "a", "verdict": "SKIPPED"}]},
    )
    assert [line["line_index"] for line in second.json()["lyric_lines"]] == [0]


async def test_section_verdicts_are_recorded(client):
    generation_id = await _create(client)
    resp = await client.put(
        f"/v1/generations/{generation_id}/qa",
        json={
            "section_verdicts": {
                "intro": "weak, no identity",
                "chorus": "does not land",
                "outro": "cuts off",
            }
        },
    )
    verdicts = resp.json()["section_verdicts"]
    assert verdicts["chorus"] == "does not land"
    assert set(verdicts) == {"intro", "chorus", "outro"}


async def test_unknown_section_name_is_rejected(client):
    generation_id = await _create(client)
    resp = await client.put(
        f"/v1/generations/{generation_id}/qa",
        json={"section_verdicts": {"middle_eight": "n/a"}},
    )
    assert resp.status_code == 422


async def test_unknown_failure_tag_is_rejected(client):
    generation_id = await _create(client)
    resp = await client.put(
        f"/v1/generations/{generation_id}/qa",
        json={"failure_tags": ["SOUNDS_BAD"]},
    )
    assert resp.status_code == 422


async def test_rating_outside_one_to_ten_is_rejected(client):
    generation_id = await _create(client)
    for rating in (0, 11, -1):
        resp = await client.put(
            f"/v1/generations/{generation_id}/qa", json={"overall_rating": rating}
        )
        assert resp.status_code == 422, rating


async def test_invalid_verdict_is_rejected(client):
    generation_id = await _create(client)
    resp = await client.put(
        f"/v1/generations/{generation_id}/qa",
        json={"lyric_lines": [{"line_index": 0, "line_text": "a", "verdict": "MAYBE"}]},
    )
    assert resp.status_code == 422


async def test_review_survives_a_reread(client):
    generation_id = await _create(client)
    await client.put(
        f"/v1/generations/{generation_id}/qa",
        json={
            "overall_rating": 4,
            "failure_tags": ["EXCESSIVE_SIBILANCE"],
            "lyric_lines": [
                {"line_index": 0, "line_text": "창밖에 비가 내려와", "verdict": "COMPLETE"}
            ],
        },
    )
    body = (await client.get(f"/v1/generations/{generation_id}/qa")).json()
    assert body["overall_rating"] == 4
    assert body["failure_tags"] == ["EXCESSIVE_SIBILANCE"]
    assert body["lyric_lines"][0]["verdict"] == "COMPLETE"


async def test_reviewing_does_not_alter_the_generation(client):
    generation_id = await _create(client)
    before = (await client.get(f"/v1/generations/{generation_id}")).json()
    await client.put(
        f"/v1/generations/{generation_id}/qa",
        json={"overall_rating": 1, "failure_tags": ["STRUCTURE_COLLAPSE"]},
    )
    after = (await client.get(f"/v1/generations/{generation_id}")).json()
    # A bad review does not hide, alter, or reprocess the track.
    assert before == after


async def test_review_of_a_missing_generation_is_404(client):
    resp = await client.put(f"/v1/generations/{uuid.uuid4()}/qa", json={"overall_rating": 5})
    assert resp.status_code == 404


# ── Long-form technical QA view ───────────────────────────────────────


async def test_longform_qa_reports_the_requested_frame(client):
    generation_id = await _create(client)
    body = (await client.get(f"/v1/generations/{generation_id}/longform-qa")).json()
    assert body["requested_duration"] == 180
    assert body["bpm_requested"] == 84
    assert body["key_requested"] == "A minor"
    assert body["time_signature_requested"] == "4"
    assert body["sections_requested"] == 4  # Intro, Verse 1, Chorus, Outro
    assert body["lyric_line_count"] == 3
    assert body["is_full_song"] is True


async def test_longform_qa_reports_timing_for_a_finished_run(client):
    generation_id = await _create(client)
    body = (await client.get(f"/v1/generations/{generation_id}/longform-qa")).json()
    assert body["status"] == "COMPLETED"
    assert body["actual_duration"] is not None
    assert body["generation_seconds"] is not None and body["generation_seconds"] >= 0
    assert body["real_time_factor"] is not None


async def test_short_request_is_not_a_full_song(client):
    generation_id = await _create(client, duration=30)
    body = (await client.get(f"/v1/generations/{generation_id}/longform-qa")).json()
    assert body["is_full_song"] is False


async def test_longform_qa_for_a_missing_generation_is_404(client):
    assert (await client.get(f"/v1/generations/{uuid.uuid4()}/longform-qa")).status_code == 404


async def test_longform_qa_stays_out_of_the_listener_payload(client):
    # Developer clutter must not leak into the normal experience.
    generation_id = await _create(client)
    listener_view = (await client.get(f"/v1/generations/{generation_id}")).json()
    for developer_field in (
        "real_time_factor",
        "generation_seconds",
        "sections_requested",
        "lyric_line_count",
    ):
        assert developer_field not in listener_view
