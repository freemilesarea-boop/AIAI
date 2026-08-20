"""The operator flow, end to end, through the CLI itself.

Exercised through `main()` rather than the library, because the CLI is
where the pieces are wired together and wiring is where things come
apart: a hypothesis dropped between two saves, an artifact written to a
directory nothing later reads, a state machine skipped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from evaluation_fixtures import profile, seed_registry, write_profile

from luber_evaluation.cli import main
from luber_training.entities import CheckpointKind, CheckpointStatus
from luber_training.orchestrator import Orchestrator


def _evaluation_ids(registry_root: Path) -> list[str]:
    return sorted(path.stem for path in (registry_root / "evaluations").glob("*.json"))


def _run(registry_root: Path, *argv: str) -> int:
    return main(["--registry", str(registry_root), *argv])


def _capture(capsys: Any) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(capsys.readouterr().out)
    return payload


@pytest.fixture
def profiles(tmp_path: Path) -> dict[str, Path]:
    # The candidate leaves markedly less dead air. Far enough apart to
    # clear the rate noise floor: a movement the suite cannot resolve is
    # correctly reported as inconclusive, which is not what this test is
    # about.
    return {
        "baseline": write_profile(
            tmp_path / "baseline.json", profile("baseline", silence_ratio=0.11)
        ),
        "candidate": write_profile(
            tmp_path / "candidate.json", profile("candidate", silence_ratio=0.02)
        ),
    }


def test_full_flow_create_start_qualify_verify(
    registry_root: Path,
    orchestrator: Orchestrator,
    profiles: dict[str, Path],
    capsys: Any,
) -> None:
    info = seed_registry(orchestrator, hypothesis="reduce dead air at the end of tracks")

    assert (
        _run(
            registry_root,
            "run",
            "create",
            "--candidate-id",
            info["candidate_id"],
            "--suite",
            "SMOKE",
        )
        == 0
    )
    created = _capture(capsys)
    evaluation_id = created["evaluation_id"]
    assert created["status"] == "DRAFT"
    # The hypothesis travels on the run, not only in the creation record.
    assert created["experiment_hypothesis"] == "reduce dead air at the end of tracks"

    assert (
        _run(
            registry_root,
            "run",
            "start",
            "--evaluation-id",
            evaluation_id,
            "--backend",
            "synthetic",
            "--baseline-profile",
            str(profiles["baseline"]),
            "--candidate-profile",
            str(profiles["candidate"]),
        )
        == 0
    )
    started = _capture(capsys)
    assert started["status"] == "COMPLETED"

    assert (
        _run(
            registry_root,
            "qualify",
            "--evaluation-id",
            evaluation_id,
            "--hypothesis-metric",
            "silence_ratio",
        )
        == 0
    )
    verdict = _capture(capsys)
    assert verdict["outcome"] == "QUALIFIED"
    assert verdict["hypothesis_status"] == "SUPPORTED"

    assert _run(registry_root, "verify", "--evaluation-id", evaluation_id) == 0
    assert _capture(capsys)["ok"] is True


def test_hypothesis_survives_the_run_lifecycle(
    registry_root: Path, orchestrator: Orchestrator, profiles: dict[str, Path], capsys: Any
) -> None:
    """A claim recorded at creation must still be there at the verdict.

    The gate that stops a candidate qualifying without addressing its
    own hypothesis is only as good as the hypothesis surviving to the
    point where the gate runs.
    """
    info = seed_registry(orchestrator, hypothesis="the vocals should sound more human")
    _run(registry_root, "run", "create", "--candidate-id", info["candidate_id"], "--suite", "SMOKE")
    evaluation_id = _capture(capsys)["evaluation_id"]

    _run(
        registry_root,
        "run",
        "start",
        "--evaluation-id",
        evaluation_id,
        "--backend",
        "synthetic",
        "--baseline-profile",
        str(profiles["baseline"]),
        "--candidate-profile",
        str(profiles["candidate"]),
    )
    capsys.readouterr()

    record = json.loads((registry_root / "evaluations" / f"{evaluation_id}.json").read_text())
    assert record["experiment_hypothesis"] == "the vocals should sound more human"

    _run(
        registry_root,
        "qualify",
        "--evaluation-id",
        evaluation_id,
        "--hypothesis-metric",
        "vocal_naturalness",
    )
    verdict = _capture(capsys)
    assert verdict["outcome"] == "HUMAN_REVIEW_REQUIRED"
    assert "vocal_naturalness" in verdict["human_review_required_for"]


def test_mock_checkpoint_cannot_be_evaluated(
    registry_root: Path, orchestrator: Orchestrator, capsys: Any
) -> None:
    info = seed_registry(orchestrator, checkpoint_kind=CheckpointKind.MOCK.value)
    assert (
        _run(
            registry_root,
            "run",
            "create",
            "--candidate-id",
            info["candidate_id"],
            "--suite",
            "SMOKE",
        )
        == 1
    )
    assert "MOCK" in _capture(capsys)["error"]


def test_unready_checkpoint_cannot_be_evaluated(
    registry_root: Path, orchestrator: Orchestrator, capsys: Any
) -> None:
    info = seed_registry(orchestrator, checkpoint_status=CheckpointStatus.WRITING.value)
    assert (
        _run(
            registry_root,
            "run",
            "create",
            "--candidate-id",
            info["candidate_id"],
            "--suite",
            "SMOKE",
        )
        == 1
    )
    assert "READY" in _capture(capsys)["error"]


def test_a_completed_evaluation_cannot_be_restarted(
    registry_root: Path, orchestrator: Orchestrator, profiles: dict[str, Path], capsys: Any
) -> None:
    """Identity is frozen. Re-running means a new evaluation."""
    info = seed_registry(orchestrator)
    _run(registry_root, "run", "create", "--candidate-id", info["candidate_id"], "--suite", "SMOKE")
    evaluation_id = _capture(capsys)["evaluation_id"]
    args = [
        "run",
        "start",
        "--evaluation-id",
        evaluation_id,
        "--backend",
        "synthetic",
        "--baseline-profile",
        str(profiles["baseline"]),
        "--candidate-profile",
        str(profiles["candidate"]),
    ]
    assert _run(registry_root, *args) == 0
    capsys.readouterr()
    assert _run(registry_root, *args) == 1
    assert "COMPLETED" in _capture(capsys)["error"]


def test_a_verdict_is_written_once(
    registry_root: Path, orchestrator: Orchestrator, profiles: dict[str, Path], capsys: Any
) -> None:
    """Re-deciding produces a new evaluation, never an edited verdict."""
    info = seed_registry(orchestrator)
    _run(registry_root, "run", "create", "--candidate-id", info["candidate_id"], "--suite", "SMOKE")
    evaluation_id = _capture(capsys)["evaluation_id"]
    _run(
        registry_root,
        "run",
        "start",
        "--evaluation-id",
        evaluation_id,
        "--backend",
        "synthetic",
        "--baseline-profile",
        str(profiles["baseline"]),
        "--candidate-profile",
        str(profiles["candidate"]),
    )
    capsys.readouterr()

    assert _run(registry_root, "qualify", "--evaluation-id", evaluation_id) == 0
    capsys.readouterr()
    assert _run(registry_root, "qualify", "--evaluation-id", evaluation_id) == 1
    assert "already exists" in _capture(capsys)["error"]


def test_promotion_refuses_to_approve_an_unqualified_candidate(
    registry_root: Path, orchestrator: Orchestrator, tmp_path: Path, capsys: Any
) -> None:
    """The operator adds judgement to evidence; never substitutes for it."""
    info = seed_registry(orchestrator)
    broken = write_profile(tmp_path / "broken.json", profile("broken", clipping_sample_ratio=0.08))
    clean = write_profile(tmp_path / "clean.json", profile("clean"))

    _run(registry_root, "run", "create", "--candidate-id", info["candidate_id"], "--suite", "SMOKE")
    evaluation_id = _capture(capsys)["evaluation_id"]
    _run(
        registry_root,
        "run",
        "start",
        "--evaluation-id",
        evaluation_id,
        "--backend",
        "synthetic",
        "--baseline-profile",
        str(clean),
        "--candidate-profile",
        str(broken),
    )
    capsys.readouterr()
    _run(registry_root, "qualify", "--evaluation-id", evaluation_id)
    assert _capture(capsys)["outcome"] == "REJECTED"

    assert (
        _run(
            registry_root,
            "promote",
            "--evaluation-id",
            evaluation_id,
            "--decision",
            "APPROVE_FOR_STAGING",
            "--by",
            "operator",
            "--rationale",
            "it sounded fine to me",
        )
        == 1
    )
    assert "not QUALIFIED" in _capture(capsys)["error"]

    # HOLD is permitted: it records judgement without overriding evidence.
    assert (
        _run(
            registry_root,
            "promote",
            "--evaluation-id",
            evaluation_id,
            "--decision",
            "HOLD",
            "--by",
            "operator",
            "--rationale",
            "revisit after the clipping is fixed",
        )
        == 0
    )
    review = _capture(capsys)
    assert review["decision"] == "HOLD"
    assert "not production activation" in review["note"]


def test_qualify_refuses_an_evaluation_that_did_not_complete(
    registry_root: Path, orchestrator: Orchestrator, capsys: Any
) -> None:
    info = seed_registry(orchestrator)
    _run(registry_root, "run", "create", "--candidate-id", info["candidate_id"], "--suite", "SMOKE")
    evaluation_id = _capture(capsys)["evaluation_id"]
    assert _run(registry_root, "qualify", "--evaluation-id", evaluation_id) == 1
    assert "partial evidence" in _capture(capsys)["error"]


def test_synthetic_run_cannot_be_sent_to_a_listener(
    registry_root: Path, orchestrator: Orchestrator, profiles: dict[str, Path], capsys: Any
) -> None:
    """No audio means no listening package. Nothing is invented to fill it."""
    info = seed_registry(orchestrator)
    _run(registry_root, "run", "create", "--candidate-id", info["candidate_id"], "--suite", "SMOKE")
    evaluation_id = _capture(capsys)["evaluation_id"]
    _run(
        registry_root,
        "run",
        "start",
        "--evaluation-id",
        evaluation_id,
        "--backend",
        "synthetic",
        "--baseline-profile",
        str(profiles["baseline"]),
        "--candidate-profile",
        str(profiles["candidate"]),
    )
    capsys.readouterr()

    assert _run(registry_root, "human-package", "--evaluation-id", evaluation_id) == 1
    assert "no audio" in _capture(capsys)["error"]


def test_ranking_reports_unevaluated_checkpoints_rather_than_ordering_them(
    registry_root: Path, orchestrator: Orchestrator, capsys: Any
) -> None:
    info = seed_registry(orchestrator, checkpoints=2, final_losses=(0.42, 0.19))
    assert _run(registry_root, "checkpoint", "rank", "--run-id", info["run_id"]) == 1
    payload = _capture(capsys)
    assert payload["ranked"] == []
    assert {row["checkpoint_id"] for row in payload["unranked"]} == {
        entry["checkpoint_id"] for entry in info["candidates"]
    }
    assert all("not evidence of quality" in row["reason"] for row in payload["unranked"])


def test_metrics_command_states_what_cannot_be_measured(capsys: Any, tmp_path: Path) -> None:
    assert main(["--registry", str(tmp_path / "r"), "metrics"]) == 0
    payload = _capture(capsys)
    human = [row for row in payload["metrics"] if row["mode"] == "HUMAN_REQUIRED"]
    assert human, "the catalogue must name dimensions only a listener can judge"
    assert all(row["unavailability_reason"] for row in human)
    assert payload["counts"]["HUMAN_REQUIRED"] == len(human)


def test_suite_list_reports_the_benchmark_it_can_reach(
    tmp_path: Path, repository_root: Path, capsys: Any
) -> None:
    assert (
        main(
            [
                "--registry",
                str(tmp_path / "r"),
                "--repository",
                str(repository_root),
                "suite",
                "list",
            ]
        )
        == 0
    )
    suites = {row["suite_id"]: row for row in _capture(capsys)["suites"]}
    assert suites["SYNTHETIC_SMOKE"]["available"] is True
    p20 = suites["P20_FULL"]
    assert p20["available"] is True
    # The recorded fact, not an aspiration: nobody has scored P20 yet.
    assert p20["benchmark"]["human_scores_recorded"] == 0
