"""The record that explains a delivery, and the two things it must not carry.

**No prompt, no lyrics, no reference audio.** Phase 29's privacy rule is
that none of those leaves for telemetry, and the trace is the document
most likely to be read somewhere else — by an operator console, by a
metrics job, by whatever Phase 30 builds. The request is identified by a
digest precisely so the trace can be handled without handling the text.

**No local paths.** Candidate audio lives in a worker's directory and is
gone by the time anybody reads this. A recorded path would describe a
file that no longer exists on a machine the reader does not have.
"""

from __future__ import annotations

import json
from pathlib import Path

from luber_inference_qc import (
    Budget,
    CandidateGeneration,
    CandidateStatus,
    Finding,
    QCFinding,
    QCTrace,
    Severity,
    summarise,
)
from luber_inference_qc.policy import standard
from luber_inference_qc.trace import Outcome
from luber_inference_qc.versions import QC_SCHEMA_VERSION

PROMPT = "Dreamy Korean indie pop about a late train"
LYRICS = "[Verse]\n오늘 밤 너를 생각해"


def _candidate(index: int, **kwargs) -> CandidateGeneration:
    return CandidateGeneration(
        candidate_id=f"cand_{index:02d}",
        generation_id="gen-0001",
        attempt_index=index,
        request_sha256="d" * 64,
        raw_sha256="a" * 64,
        duration_seconds=30.0,
        audio_path=Path("/Users/someone/tmp/candidates/attempt-00.wav"),
        **kwargs,
    )


def _trace() -> QCTrace:
    trace = QCTrace(
        generation_id="gen-0001", request_sha256="d" * 64, policy=standard(), base_seed=1234
    )
    trace.add(
        _candidate(
            0,
            status=CandidateStatus.REJECTED.value,
            findings=[
                QCFinding(
                    code=Finding.SILENT_OUTPUT.value,
                    severity=Severity.CRITICAL.value,
                    detail="the file is digital silence",
                )
            ],
        )
    )
    trace.add(_candidate(1, status=CandidateStatus.ELIGIBLE.value))
    return trace


# ── privacy ──────────────────────────────────────────────────────────


def test_the_trace_carries_no_prompt_and_no_lyrics():
    """Only the digest, which is what makes the record safe to move."""
    rendered = _trace().to_json(Budget(policy=standard()))
    assert PROMPT not in rendered
    assert LYRICS not in rendered
    assert "prompt" not in rendered
    assert "lyrics" not in rendered


def test_the_trace_carries_no_local_path():
    payload = _trace().to_dict()
    rendered = json.dumps(payload)
    assert "/Users/" not in rendered
    assert "audio_path" not in rendered
    # The digest is recorded instead, which is what would let a future
    # phase recognise the bytes if it ever kept them.
    assert payload["attempts"][0]["raw_sha256"] == "a" * 64


# ── what it does record ──────────────────────────────────────────────


def test_the_trace_answers_why_this_file_was_delivered():
    trace = _trace()
    trace.outcome_detail = "the only eligible candidate"
    payload = trace.to_dict(Budget(policy=standard(), provider_calls_used=2, retry_rounds=1))

    assert payload["qc_schema_version"] == QC_SCHEMA_VERSION
    assert payload["policy"]["name"] == "STANDARD"
    assert payload["budget"]["provider_calls_used"] == 2
    assert payload["base_seed"] == 1234
    assert len(payload["attempts"]) == 2


def test_a_failure_is_recorded_rather_than_hidden():
    payload = _trace().to_dict()
    rejected = payload["attempts"][0]
    assert rejected["status"] == CandidateStatus.REJECTED.value
    assert rejected["findings"][0]["code"] == Finding.SILENT_OUTPUT.value


def test_the_trace_serialises_with_stable_key_order():
    """So a diff between two renderings is a change, not a reshuffle."""
    trace = _trace()
    assert trace.to_json() == trace.to_json()
    assert list(json.loads(trace.to_json())) == sorted(json.loads(trace.to_json()))


def test_no_musical_score_appears_anywhere_in_the_record():
    rendered = _trace().to_json()
    for word in ("musical_quality", "naturalness", "melody_quality", "commercial"):
        assert word not in rendered


def test_exhaustion_is_stated_rather_than_inferred():
    trace = _trace()
    trace.outcome = Outcome.RETRY_EXHAUSTED
    assert trace.to_dict()["exhausted"] is True


# ── the summary ──────────────────────────────────────────────────────


def test_the_summary_counts_what_an_operator_asks_about():
    trace = _trace()
    payload = trace.to_dict(Budget(policy=standard(), provider_calls_used=2, retry_rounds=1))

    summary = summarise(payload)

    assert summary["attempts"] == 2
    assert summary["retries"] == 1
    assert summary["provider_calls"] == 2
    assert summary["critical_findings"] == [Finding.SILENT_OUTPUT.value]
    assert summary["policy"] == "STANDARD"


def test_the_summary_survives_a_trace_with_nothing_in_it():
    assert summarise({})["attempts"] == 0
