"""Running QC over audio that already exists, and reading a stored trace.

`analyze` is how thresholds get checked against real output instead of
against the fixtures they were written next to. Its most important
behaviour is the one at the bottom: a corpus where more than half the
files are rejected exits non-zero, because that means the thresholds are
wrong rather than that the songs are.
"""

from __future__ import annotations

import json

import pytest
import qc_fixtures as fx

from luber_inference_qc.cli import main


def _run(capsys, *args) -> tuple[int, dict]:
    code = main(list(args))
    captured = capsys.readouterr()
    payload = json.loads(captured.out) if captured.out.strip().startswith("{") else {}
    return code, payload


# ── analyze ──────────────────────────────────────────────────────────


def test_one_file_is_analysed_and_reported(capsys, audio_dir):
    path = fx.healthy(audio_dir / "song.wav")
    code, payload = _run(capsys, "analyze", str(path), "--duration", "12")

    assert code == 0
    assert payload["result"]["eligible"] is True
    assert payload["result"]["file"] == "song.wav"
    assert payload["qc_schema_version"]


def test_a_directory_is_summarised_with_the_findings_counted(capsys, audio_dir):
    fx.healthy(audio_dir / "a.wav")
    fx.healthy(audio_dir / "b.wav")
    fx.silent(audio_dir / "c.wav")

    code, payload = _run(capsys, "analyze", str(audio_dir), "--duration", "12")

    assert code == 0
    assert payload["files"] == 3
    assert payload["eligible"] == 2
    assert payload["rejected"] == 1
    assert payload["critical_finding_counts"]["SILENT_OUTPUT"] == 1
    assert payload["qc_seconds"]["median"] is not None


def test_a_corpus_that_is_mostly_rejected_exits_non_zero(capsys, audio_dir):
    """The thresholds are wrong, not the songs."""
    fx.silent(audio_dir / "a.wav")
    fx.silent(audio_dir / "b.wav")
    fx.healthy(audio_dir / "c.wav")

    code, _ = _run(capsys, "analyze", str(audio_dir), "--duration", "12")

    assert code == 1


def test_one_unreadable_file_does_not_stop_the_corpus(capsys, audio_dir):
    fx.healthy(audio_dir / "a.wav")
    fx.undecodable(audio_dir / "b.wav")

    code, payload = _run(capsys, "analyze", str(audio_dir), "--duration", "12")

    assert code == 0
    assert payload["files"] == 2


def test_analysing_nothing_says_so(capsys, tmp_path):
    (tmp_path / "empty").mkdir()
    assert _run(capsys, "analyze", str(tmp_path / "empty"))[0] == 2
    assert _run(capsys, "analyze", str(tmp_path / "nowhere"))[0] == 2


def test_the_limit_stops_early(capsys, audio_dir):
    for name in "abcd":
        fx.healthy(audio_dir / f"{name}.wav")
    assert _run(capsys, "analyze", str(audio_dir), "--limit", "2")[1]["files"] == 2


# ── explain ──────────────────────────────────────────────────────────


TRACE = {
    "generation_id": "gen-0001",
    "policy": {"name": "STANDARD"},
    "request_sha256": "d" * 64,
    "outcome": "SELECTED",
    "outcome_detail": "the only eligible candidate",
    "budget": {"provider_calls_used": 2, "maximum_total_provider_calls": 3},
    "finishing_outcome": "FINISHED",
    "selected_candidate_id": "cand_01",
    "selection": {"reasons": {"cand_01": "the only eligible candidate"}},
    "attempts": [
        {
            "attempt_index": 0,
            "candidate_id": "cand_00",
            "status": "REJECTED",
            "seed": 1234,
            "attribution": "USER_REQUEST",
            "findings": [{"code": "SILENT_OUTPUT", "severity": "CRITICAL", "detail": "silence"}],
        },
        {
            "attempt_index": 1,
            "candidate_id": "cand_01",
            "status": "ELIGIBLE",
            "seed": 9876,
            "attribution": "QUALITY_RETRY",
            "retry_reason": "SILENT_OUTPUT: retried with a different seed",
            "findings": [],
        },
    ],
}


@pytest.fixture
def trace_file(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(TRACE), encoding="utf-8")
    return path


def test_explain_answers_why_it_retried_and_which_one_won(capsys, trace_file):
    assert main(["explain", str(trace_file)]) == 0
    out = capsys.readouterr().out
    assert "gen-0001" in out
    assert "SILENT_OUTPUT" in out
    assert "cand_01" in out
    assert "2 of 3" in out


def test_explain_can_summarise(capsys, trace_file):
    assert main(["explain", str(trace_file), "--summary"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["retries"] == 1
    assert summary["critical_findings"] == ["SILENT_OUTPUT"]


def test_explaining_a_trace_that_is_not_there_says_so(capsys, tmp_path):
    assert main(["explain", str(tmp_path / "nowhere.json")]) == 2
