"""The gate, then the ranking, and the fact that they never merge.

A single scoring function that gave invalid audio a low number would put
it in the same ordering as a working song and rely on arithmetic to keep
it from winning. The property asserted hardest below is the one that
makes that impossible: a rejected candidate never enters the ranking, so
no arithmetic mistake can promote it.
"""

from __future__ import annotations

from luber_inference_qc import (
    CandidateGeneration,
    CandidateStatus,
    Finding,
    QCFinding,
    SelectionStatus,
    Severity,
    assess_eligibility,
    select,
)


def _candidate(
    index: int,
    *,
    status=CandidateStatus.ELIGIBLE,
    score: float | None = 0.8,
    components: dict[str, float] | None = None,
    findings: list[QCFinding] | None = None,
) -> CandidateGeneration:
    return CandidateGeneration(
        candidate_id=f"cand_{index:02d}",
        generation_id="gen",
        attempt_index=index,
        request_sha256="digest",
        status=status.value,
        raw_sha256="a" * 64,
        duration_seconds=30.0,
        technical_selection_score=score,
        score_components=components or {"control_adherence": 1.0, "duration_accuracy": 1.0},
        findings=findings or [],
    )


def _finding(code: Finding, severity: Severity) -> QCFinding:
    return QCFinding(code=code.value, severity=severity.value, detail=code.value)


# ── the gate ─────────────────────────────────────────────────────────


def test_a_candidate_with_nothing_critical_may_be_delivered():
    assert assess_eligibility(_candidate(0)).eligible is True


def test_a_major_finding_costs_a_ranking_and_does_not_block_delivery():
    """ "Measurably worse" and "cannot be shipped" are different verdicts."""
    candidate = _candidate(0, findings=[_finding(Finding.CONTROL_BPM_MISMATCH, Severity.MAJOR)])
    assert assess_eligibility(candidate).eligible is True


def test_an_attempt_that_produced_no_audio_is_not_eligible():
    candidate = _candidate(0)
    candidate.raw_sha256 = None
    verdict = assess_eligibility(candidate)
    assert verdict.eligible is False
    assert "no audio was produced" in verdict.reasons[0]


def test_a_critical_finding_blocks_and_the_reason_is_recorded():
    candidate = _candidate(0, findings=[_finding(Finding.SILENT_OUTPUT, Severity.CRITICAL)])
    verdict = assess_eligibility(candidate)
    assert verdict.eligible is False
    assert any(Finding.SILENT_OUTPUT.value in reason for reason in verdict.reasons)


# ── the ranking ──────────────────────────────────────────────────────


def test_a_rejected_candidate_never_wins_however_high_its_score():
    """The property the separation exists for.

    The broken candidate is handed an impossibly good score. It still
    loses, because it never reaches the comparison.
    """
    broken = _candidate(
        0,
        status=CandidateStatus.REJECTED,
        score=99.0,
        findings=[_finding(Finding.SILENT_OUTPUT, Severity.CRITICAL)],
    )
    working = _candidate(1, score=0.01)

    selection = select([broken, working])

    assert selection.winner_candidate_id == working.candidate_id
    assert broken.candidate_id not in selection.ranking
    assert broken.selection_status == SelectionStatus.NOT_SELECTED.value
    assert Finding.SILENT_OUTPUT.value in (broken.not_selected_reason or "")


def test_fewer_major_findings_wins_before_any_score_is_compared():
    flawed = _candidate(
        0, score=0.99, findings=[_finding(Finding.CONTROL_BPM_MISMATCH, Severity.MAJOR)]
    )
    clean = _candidate(1, score=0.10)
    assert select([flawed, clean]).winner_candidate_id == clean.candidate_id


def test_closer_control_adherence_wins_before_duration():
    close = _candidate(0, components={"control_adherence": 0.9, "duration_accuracy": 0.1})
    far = _candidate(1, components={"control_adherence": 0.2, "duration_accuracy": 1.0})
    assert select([close, far]).winner_candidate_id == close.candidate_id


def test_a_tie_on_everything_measured_keeps_the_earlier_attempt():
    """Deterministic, and it is also the cheapest outcome."""
    first, second = _candidate(0), _candidate(1)
    selection = select([first, second])
    assert selection.winner_candidate_id == first.candidate_id
    assert "the earlier attempt was kept" in (second.not_selected_reason or "")


def test_the_same_candidates_always_produce_the_same_winner():
    """A selection an operator cannot reproduce is one they cannot audit."""

    def build():
        return [
            _candidate(0, score=0.5),
            _candidate(1, score=0.9),
            _candidate(2, score=0.7),
        ]

    first = select(build())
    assert select(build()).ranking == first.ranking
    assert select(list(reversed(build()))).winner_candidate_id == first.winner_candidate_id


def test_the_reason_names_the_axis_that_actually_separated_them():
    """Rather than gesturing at a score nobody can decompose."""
    winner = _candidate(0, components={"control_adherence": 1.0, "duration_accuracy": 1.0})
    loser = _candidate(1, components={"control_adherence": 0.3, "duration_accuracy": 1.0})
    selection = select([winner, loser])
    assert selection.reasons[loser.candidate_id].endswith("on control adherence")


def test_the_only_eligible_candidate_says_so():
    selection = select([_candidate(0)])
    assert selection.reasons["cand_00"] == "the only eligible candidate"


def test_nothing_eligible_produces_no_winner_rather_than_a_best_effort():
    """Budget exhaustion is a failure, not the delivery of the least
    broken candidate this engine already measured and rejected."""
    broken = _candidate(
        0,
        status=CandidateStatus.REJECTED,
        findings=[_finding(Finding.SILENT_OUTPUT, Severity.CRITICAL)],
    )
    selection = select([broken])
    assert selection.winner_candidate_id is None
    assert selection.ranking == []
