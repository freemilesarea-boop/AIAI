"""The operator's own commands, driving a real worker end to end.

Exercised through `main()` rather than the library, because the CLI is
where the boundary actually is: an operator types these, and everything
below them is invisible from where the decision is made. A gate that
holds in the library and is bypassed by a flag in the CLI would hold
nowhere that matters.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pytest
from remote_fixtures import build_harness, synthetic_argv
from training_fixtures import curated_record, manifest_record

from luber_training.cli import main
from luber_training.config import TrainingConfig
from luber_training.entities import (
    CheckpointKind,
    CheckpointStatus,
    Experiment,
    ModelBaseline,
    RunStatus,
    TrainingDatasetRef,
    TrainingRun,
)
from luber_training.orchestrator import Orchestrator
from luber_training.registry import Registry

RUN_ID = "run_" + "e" * 16


def _capture(capsys: Any) -> dict[str, Any]:
    return json.loads(capsys.readouterr().out)


def _seed_registry(registry_root: Path, audio_root: Path) -> dict[str, Any]:
    """A registry holding one run ready to be staged."""
    orchestrator = Orchestrator(Registry(registry_root))
    registry = orchestrator.registry

    baseline = ModelBaseline(
        model_id="mdl_" + "a" * 16,
        provider="ace-step",
        model_family="ace-step",
        model_name="acestep-v15-turbo",
        model_version="1.5",
        upstream_commit="6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0",
        architecture="dit",
        training_strategy_support=["LORA", "LOKR"],
    )
    registry.write("models", baseline.model_id, baseline.to_dict(), overwrite=True)

    experiment = Experiment(
        experiment_id="exp_" + "b" * 16,
        name="remote-smoke",
        hypothesis="the remote pipeline works end to end",
        base_model_id=baseline.model_id,
    )
    registry.write("experiments", experiment.experiment_id, experiment.to_dict(), overwrite=True)

    run = TrainingRun(
        run_id=RUN_ID,
        experiment_id=experiment.experiment_id,
        base_model_id=baseline.model_id,
        dataset_ref=TrainingDatasetRef(
            dataset_id="ds_test",
            dataset_lock_sha256="0" * 64,
            curation_id="cur_test",
            curation_lock_sha256="1" * 64,
            curated_manifest_sha256="2" * 64,
            manifest_artifact_ref="curated_manifest.jsonl",
            selected_track_count=1,
            selected_hours=0.05,
        ),
        # CUDA is not required, so a laptop can execute the smoke path.
        # The CUDA mismatch case has its own test.
        config=TrainingConfig(epochs=1),
        execution_backend="remote-gpu",
        status=RunStatus.QUEUED.value,
    )
    registry.write("runs", run.run_id, run.to_dict(), overwrite=True)
    return {"run_id": RUN_ID, "model_id": baseline.model_id}


def _seed_run(registry_root: Path, run_id: str) -> None:
    """Add one more run record, for a run created outside the seed."""
    registry = Registry(registry_root)
    existing = registry.read("runs", RUN_ID)
    existing["run_id"] = run_id
    registry.write("runs", run_id, existing, overwrite=True)


def _dataset_build(tmp_path: Path, *, permitted: bool = True) -> dict[str, Path]:
    dataset_dir = tmp_path / "dataset-build"
    curation_dir = tmp_path / "curation-build"
    audio_root = tmp_path / "library"
    for path in (dataset_dir, curation_dir, audio_root):
        path.mkdir(parents=True, exist_ok=True)

    audio = audio_root / "trk_001.wav"
    audio.write_bytes(b"RIFF" + bytes(4096))
    digest = hashlib.sha256(audio.read_bytes()).hexdigest()

    record = manifest_record(
        "trk_001",
        permission="TRUE" if permitted else "UNKNOWN",
        rights_status="USER_OWNED" if permitted else "UNKNOWN",
        training_eligible=permitted,
        sha256=digest,
    )
    record["source"]["relative_path"] = "trk_001.wav"
    (curation_dir / "curated_manifest.jsonl").write_text(
        json.dumps(curated_record(record, action="KEEP"), sort_keys=True) + "\n", encoding="utf-8"
    )
    (curation_dir / "curation_lock.json").write_text(
        json.dumps({"curation_id": "cur_test"}), encoding="utf-8"
    )
    (curation_dir / "sampling_weights.json").write_text(
        json.dumps({"weights": {"trk_001": 1.0}}), encoding="utf-8"
    )
    (dataset_dir / "dataset_lock.json").write_text(
        json.dumps({"dataset_id": "ds_test"}), encoding="utf-8"
    )
    (dataset_dir / "dataset_manifest.jsonl").write_text(
        json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"dataset": dataset_dir, "curation": curation_dir, "audio": audio_root}


@pytest.fixture
def environment(tmp_path: Path) -> dict[str, Any]:
    harness = build_harness(tmp_path)
    registry_root = tmp_path / "training-registry"
    build = _dataset_build(tmp_path)
    _seed_registry(registry_root, build["audio"])
    return {"harness": harness, "registry": registry_root, "build": build, "tmp": tmp_path}


def _run(environment: dict[str, Any], *argv: str) -> int:
    return main(["--registry", str(environment["registry"]), *argv])


def _stage_args(environment: dict[str, Any]) -> list[str]:
    build = environment["build"]
    return [
        "--staging-root",
        str(environment["tmp"] / "remote_staging"),
        "--dataset-build",
        str(build["dataset"]),
        "--curation-build",
        str(build["curation"]),
        "--audio-root",
        str(build["audio"]),
    ]


def _connection(environment: dict[str, Any]) -> list[str]:
    return ["--transport", "local", "--worker-root", str(environment["harness"].worker_root)]


def test_the_operator_flow_up_to_dispatch(environment: dict[str, Any], capsys: Any) -> None:
    """Stage, verify, register, dispatch — on the machine we actually have.

    The dispatch is expected to be BLOCKED, and that is the point. A
    compiled plan requires CUDA; this laptop has none; preflight says so
    and no trainer starts. A test that arranged for it to pass would
    have had to label a Mac as CUDA-capable, which is the one thing this
    whole package exists to prevent.
    """
    assert (
        _run(environment, "remote", "run", "stage", "--run-id", RUN_ID, *_stage_args(environment))
        == 0
    )
    staged = _capture(capsys)
    assert staged["selected_tracks"] == 1
    assert staged["gates"]["passed"] is True

    assert (
        _run(
            environment,
            "remote",
            "run",
            "verify-staging",
            "--run-id",
            RUN_ID,
            *_stage_args(environment),
        )
        == 0
    )
    assert _capture(capsys)["ok"] is True

    assert _run(environment, "remote", "worker", "register-remote", *_connection(environment)) == 0
    registered = _capture(capsys)
    worker_id = registered["worker"]["worker_id"]
    # The probe classified it. Nothing in the CLI can override that.
    assert registered["worker"]["worker_class"] == "DEVELOPMENT_ONLY"
    assert registered["worker"]["capabilities"]["cuda_available"] is None

    _run(
        environment,
        "remote",
        "run",
        "dispatch",
        "--run-id",
        RUN_ID,
        "--worker-id",
        worker_id,
        "--allow-code-mismatch",
        *_connection(environment),
        "--staging-root",
        str(environment["tmp"] / "remote_staging"),
    )
    dispatched = _capture(capsys)
    assert dispatched["launched"] is False
    assert dispatched["preflight"]["status"] == "BLOCKED"
    assert any("cuda" in reason for reason in dispatched["preflight"]["blocking_reasons"])
    assert "No trainer was started" in dispatched["detail"]

    # Artifacts did transfer — the refusal happens after transfer and
    # before execution, which is where a capability check belongs.
    assert dispatched["transfer"]["transfer"]["ok"] is True


def test_a_worker_that_cannot_be_verified_reports_why(
    environment: dict[str, Any], capsys: Any
) -> None:
    assert _run(environment, "remote", "worker", "register-remote", *_connection(environment)) == 0
    worker_id = _capture(capsys)["worker"]["worker_id"]

    assert (
        _run(
            environment,
            "remote",
            "worker",
            "verify",
            "--worker-id",
            worker_id,
            *_connection(environment),
        )
        == 0
    )
    verified = _capture(capsys)
    assert verified["ok"] is True
    assert verified["classification"] == "DEVELOPMENT_ONLY"

    # A machine that was rebuilt keeps its id and loses its fingerprint.
    registry_path = environment["registry"] / "workers" / f"{worker_id}.json"
    record = json.loads(registry_path.read_text(encoding="utf-8"))
    record["host_identity"] = "0" * 64
    registry_path.write_text(json.dumps(record), encoding="utf-8")

    assert (
        _run(
            environment,
            "remote",
            "worker",
            "verify",
            "--worker-id",
            worker_id,
            *_connection(environment),
        )
        == 1
    )
    assert any("rebuilt" in problem for problem in _capture(capsys)["problems"])


def _await_terminal(harness: Any, run_id: str, *, timeout: float = 60.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    payload: dict[str, Any] = {}
    while time.monotonic() < deadline:
        payload = harness.client.status(run_id)
        if payload.get("state") in {"COMPLETED", "FAILED", "CANCELLED"}:
            return payload
        time.sleep(0.2)
    return payload


def test_the_cli_refuses_to_stage_unauthorised_audio(tmp_path: Path, capsys: Any) -> None:
    """There is no flag in this program that sends unlicensed audio."""
    harness = build_harness(tmp_path)
    registry_root = tmp_path / "training-registry"
    build = _dataset_build(tmp_path, permitted=False)
    _seed_registry(registry_root, build["audio"])
    environment = {"harness": harness, "registry": registry_root, "build": build, "tmp": tmp_path}

    assert (
        _run(environment, "remote", "run", "stage", "--run-id", RUN_ID, *_stage_args(environment))
        == 1
    )
    failure = _capture(capsys)
    assert failure["ok"] is False
    assert "not authorised for training" in failure["error"]

    # And nothing reached the worker.
    assert not (harness.run_root / RUN_ID).exists()


def test_the_cli_refuses_to_stage_evaluation_material(
    environment: dict[str, Any], capsys: Any
) -> None:
    protected = environment["tmp"] / "evaluation_only.txt"
    protected.write_text("# benchmark material\ntrk_001\n", encoding="utf-8")

    assert (
        _run(
            environment,
            "remote",
            "run",
            "stage",
            "--run-id",
            RUN_ID,
            *_stage_args(environment),
            "--evaluation-only",
            str(protected),
        )
        == 1
    )
    failure = _capture(capsys)
    assert "benchmark" in failure["error"]


def test_reconcile_is_idempotent_and_reports_unknown_honestly(
    environment: dict[str, Any], capsys: Any
) -> None:
    """A run the worker has never seen is NOT_PRESENT, not FAILED."""
    for _ in range(2):
        assert (
            _run(
                environment,
                "remote",
                "run",
                "reconcile",
                "--run-id",
                RUN_ID,
                *_connection(environment),
            )
            == 0
        )
        payload = _capture(capsys)
        assert payload["reconcile"]["outcome"] == "NOT_PRESENT"
        assert payload["reconcile"]["safe_to_launch"] is True
        # Nothing was applied to the run: it is still QUEUED.
        assert payload["run_status"] == RunStatus.QUEUED.value


def test_collect_registers_only_what_verified(environment: dict[str, Any], capsys: Any) -> None:
    """The `collect` verb, from a run this machine could actually execute.

    Uses a plan that does not require CUDA, because the subject here is
    collection rather than capability matching — and inventing a CUDA
    capability for the worker to satisfy a compiled plan would be the
    dishonest way to reach this code.
    """
    harness = environment["harness"]
    from remote_fixtures import build_manifest, make_plan, prepare_and_receive

    plan = make_plan(run_id="run_" + "f" * 16)
    staging = environment["tmp"] / "remote_staging" / plan.run_id
    manifest = build_manifest(staging, plan)
    prepare_and_receive(harness, plan, manifest, staging_dir=staging)

    passed, report = harness.client.preflight(plan.run_id, allow_code_mismatch=True)
    assert passed, report["blocking_reasons"]

    layout = harness.layout(plan.run_id)
    harness.worker.start(
        plan.run_id,
        argv=synthetic_argv(layout, plan.run_id, steps=2, checkpoint_every=2),
        working_directory=layout.root,
    )
    _await_terminal(harness, plan.run_id)

    # A run record so the registry can hold the checkpoint.
    _seed_run(environment["registry"], plan.run_id)

    assert (
        _run(
            environment,
            "remote",
            "run",
            "collect",
            "--run-id",
            plan.run_id,
            "--kind",
            CheckpointKind.MOCK.value,
            "--collect-root",
            str(environment["tmp"] / "collected_checkpoints"),
            *_connection(environment),
        )
        == 0
    )
    collected = _capture(capsys)
    assert collected["ok"] is True
    assert len(collected["registered"]) == 1
    assert collected["registered"][0]["status"] == CheckpointStatus.READY.value
    assert collected["registered"][0]["kind"] == CheckpointKind.MOCK.value
    assert all(item["action"].startswith("RETAIN") for item in collected["retention"])


def test_verify_remote_detects_a_tampered_artifact(
    environment: dict[str, Any], capsys: Any
) -> None:
    harness = environment["harness"]
    _run(environment, "remote", "run", "stage", "--run-id", RUN_ID, *_stage_args(environment))
    capsys.readouterr()

    from remote_fixtures import prepare_and_receive

    from luber_training.remote.manifest import RemoteArtifactManifest

    manifest = RemoteArtifactManifest.read(
        environment["tmp"] / "remote_staging" / RUN_ID / "artifact_manifest.json"
    )
    orchestrator = Orchestrator(Registry(environment["registry"]))
    prepare_and_receive(
        harness,
        orchestrator.compile_plan(RUN_ID),
        manifest,
        staging_dir=environment["tmp"] / "remote_staging" / RUN_ID,
    )

    verify_argv = [
        "remote",
        "run",
        "verify-remote",
        "--run-id",
        RUN_ID,
        *_connection(environment),
        "--staging-root",
        str(environment["tmp"] / "remote_staging"),
    ]
    assert _run(environment, *verify_argv) == 0
    assert _capture(capsys)["ok"] is True

    target = harness.run_root / RUN_ID / "dataset" / "trk_001.wav"
    original = target.read_bytes()
    target.write_bytes(b"XXXX" + original[4:])

    assert _run(environment, *verify_argv) == 1
    problems = _capture(capsys)["problems"]
    assert any("digest is" in problem for problem in problems)
