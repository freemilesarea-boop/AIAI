"""The whole remote lifecycle, on one machine, with real processes.

Control plane → stage → transfer → preflight → launch → heartbeat →
logs → metrics → checkpoint → result → download → verify → registry.

The failure cases matter more than the happy path, and the one that
matters most is double launch: every route by which a control plane
might start a second trainer against one run is tested here, because
two trainers writing into one checkpoint directory produce artifacts
that are individually well-formed and jointly meaningless.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from remote_fixtures import (
    build_harness,
    build_manifest,
    make_plan,
    prepare_and_receive,
    synthetic_argv,
)

from luber_training.entities import CheckpointKind, CheckpointStatus, RunStatus
from luber_training.orchestrator import OrchestrationError, Orchestrator
from luber_training.registry import Registry
from luber_training.remote.backend import RemoteGpuBackend
from luber_training.remote.collect import (
    CollectionError,
    collect_run,
    plan_remote_retention,
    register_collected,
)
from luber_training.remote.identity import LeaseError
from luber_training.remote.protocol import ReconcileOutcome, WorkerState, run_status_for
from luber_training.remote.result import RemoteCheckpointStatus, RemoteResult
from luber_training.remote.streams import merge_into
from luber_training.remote.transport import LocalArtifactTransport


def _wait(harness: Any, run_id: str, *, timeout: float = 30.0) -> dict[str, Any]:
    """Poll until the worker reports a terminal state."""
    deadline = time.monotonic() + timeout
    payload: dict[str, Any] = {}
    while time.monotonic() < deadline:
        payload = harness.client.status(run_id)
        if payload.get("state") in {
            WorkerState.COMPLETED.value,
            WorkerState.FAILED.value,
            WorkerState.CANCELLED.value,
        }:
            return payload
        time.sleep(0.1)
    return payload


def _start(harness: Any, plan: Any, **trainer: Any) -> dict[str, Any]:
    """Preflight and launch with the synthetic trainer."""
    passed, report = harness.client.preflight(plan.run_id, allow_code_mismatch=True)
    assert passed, report["blocking_reasons"]
    layout = harness.layout(plan.run_id)
    argv = synthetic_argv(layout, plan.run_id, **trainer)
    state = harness.worker.start(plan.run_id, argv=argv, working_directory=layout.root)
    return state.to_dict()


# ── the full success path ────────────────────────────────────────────


def test_full_remote_lifecycle(tmp_path: Path) -> None:
    harness = build_harness(tmp_path)
    plan = make_plan()
    manifest = build_manifest(harness.staging_root / plan.run_id, plan)

    prepare_and_receive(harness, plan, manifest)
    state = _start(harness, plan, steps=3, checkpoint_every=3)
    assert state["state"] in (WorkerState.RUNNING.value, WorkerState.COMPLETED.value)

    beat = harness.client.heartbeat()
    assert beat["worker_id"].startswith("wrk_")
    assert beat["active_run_id"] in (plan.run_id, None)

    final = _wait(harness, plan.run_id)
    assert final["state"] == WorkerState.COMPLETED.value
    assert final["exit_code"] == 0
    assert run_status_for(final["state"]) == RunStatus.COMPLETED.value

    logs = harness.client.logs(plan.run_id)
    assert "SYNTHETIC_TEST_FIXTURE" in logs["text"]

    metrics = harness.client.metrics(plan.run_id)
    assert metrics["count"] > 0
    # Never TRAINER: nothing here measured a model.
    assert all(event["source"] == "SIMULATED" for event in metrics["events"])

    checkpoints = harness.client.list_checkpoints(plan.run_id)
    ready = [c for c in checkpoints if c["status"] == RemoteCheckpointStatus.READY_REMOTE.value]
    assert len(ready) == 1
    assert ready[0]["sha256"]

    result = RemoteResult.from_dict(harness.client.collect_result(plan.run_id))
    assert result.exit_code == 0
    assert result.metrics_count > 0
    assert result.logs_digest

    report = collect_run(harness.transport, result, destination_root=harness.collect_root)
    assert report.ok, [item.problem for item in report.collected]
    collected = report.successful[0]
    assert collected.sha256 == ready[0]["sha256"]
    assert Path(collected.local_path or "").is_dir()

    # Nothing is deleted remotely. The instance may vanish at any time,
    # so the second copy stays until an operator says otherwise.
    decisions = plan_remote_retention(report)
    assert all(decision.action.startswith("RETAIN") for decision in decisions)


def test_collected_checkpoint_registers_as_mock_and_cannot_be_evaluated(tmp_path: Path) -> None:
    """The Phase 25 boundary holds: a fixture never becomes a candidate."""
    harness = build_harness(tmp_path)
    orchestrator = Orchestrator(Registry(tmp_path / "registry"))
    plan = make_plan()
    manifest = build_manifest(harness.staging_root / plan.run_id, plan)
    prepare_and_receive(harness, plan, manifest)
    _start(harness, plan, steps=2, checkpoint_every=2)
    _wait(harness, plan.run_id)

    result = RemoteResult.from_dict(harness.client.collect_result(plan.run_id))
    report = collect_run(harness.transport, result, destination_root=harness.collect_root)
    remote = result.checkpoints[0]

    checkpoint = register_collected(
        orchestrator,
        run_id=plan.run_id,
        collected=report.successful[0],
        remote=remote,
        # A synthetic trainer produced this. Registering it as an
        # ADAPTER would be a lie the rest of the system would believe.
        kind=CheckpointKind.MOCK.value,
    )
    assert checkpoint.status == CheckpointStatus.READY.value
    assert checkpoint.sha256 == report.successful[0].sha256

    with pytest.raises(OrchestrationError) as excinfo:
        orchestrator.create_candidate(checkpoint.checkpoint_id)
    assert "MOCK" in str(excinfo.value)


# ── one trainer, however many dispatches ─────────────────────────────


def test_double_dispatch_starts_one_trainer(tmp_path: Path) -> None:
    harness = build_harness(tmp_path)
    plan = make_plan()
    manifest = build_manifest(harness.staging_root / plan.run_id, plan)
    prepare_and_receive(harness, plan, manifest)

    first = _start(harness, plan, steps=30, step_seconds=0.1)
    assert first["state"] == WorkerState.RUNNING.value
    first_pid = first["process"]["pid"]

    # A second start against a live process returns the existing one.
    second = harness.worker.start(plan.run_id).to_dict()
    assert second["process"]["pid"] == first_pid
    assert "started nothing" in second["detail"]

    _wait(harness, plan.run_id)


def test_redispatch_with_a_different_plan_is_an_integrity_violation(tmp_path: Path) -> None:
    """One run id must mean exactly one training configuration."""
    harness = build_harness(tmp_path)
    plan = make_plan()
    manifest = build_manifest(harness.staging_root / plan.run_id, plan)
    prepare_and_receive(harness, plan, manifest)

    edited = make_plan(run_id=plan.run_id, epochs=99)
    assert edited.digest() != plan.digest()
    with pytest.raises(LeaseError) as excinfo:
        harness.worker.prepare(
            run_id=plan.run_id,
            plan_sha256=edited.digest(),
            manifest_sha256=manifest.digest(),
        )
    assert "two different training configurations" in str(excinfo.value)


def test_ambiguous_launch_reconciles_instead_of_relaunching(tmp_path: Path) -> None:
    """The acknowledgement was lost; the trainer was not.

    The control plane never learned that its launch succeeded. Calling
    dispatch again must find the running trainer and start nothing —
    this is the case that would otherwise put two trainers on one run.
    """
    harness = build_harness(tmp_path)
    plan = make_plan()
    manifest = build_manifest(harness.staging_root / plan.run_id, plan)
    prepare_and_receive(harness, plan, manifest)

    launched = _start(harness, plan, steps=30, step_seconds=0.1)
    pid = launched["process"]["pid"]

    backend = RemoteGpuBackend(harness.client, harness.transport, allow_code_mismatch=True)
    reconcile = backend.reconcile(plan.run_id)
    assert reconcile.outcome == ReconcileOutcome.RUNNING_RECOVERED.value
    assert reconcile.safe_to_launch is False

    from luber_training.entities import TrainingWorker

    worker = TrainingWorker(
        worker_id="wrk_test", name="test", backend_type="remote-gpu", host_identity="local"
    )
    result = backend.dispatch(plan, worker)
    assert result.launched is False
    assert "already has an execution" in result.detail

    assert harness.client.status(plan.run_id)["process"]["pid"] == pid
    _wait(harness, plan.run_id)


def test_control_plane_restart_rediscovers_the_run(tmp_path: Path) -> None:
    """A new process, the same run, no second trainer.

    Everything the control plane needs is on the worker and in the
    registry, so a fresh client reconstructs the picture without
    restarting anything.
    """
    harness = build_harness(tmp_path)
    plan = make_plan()
    manifest = build_manifest(harness.staging_root / plan.run_id, plan)
    prepare_and_receive(harness, plan, manifest)
    launched = _start(harness, plan, steps=30, step_seconds=0.1)
    pid = launched["process"]["pid"]

    # A completely fresh client and transport, as a restarted operator
    # process would build.
    from luber_training.remote.client import LocalWorkerClient

    reborn = LocalWorkerClient(harness.worker_root)
    transport = LocalArtifactTransport(harness.run_root)
    backend = RemoteGpuBackend(reborn, transport, allow_code_mismatch=True)

    reconcile = backend.reconcile(plan.run_id)
    assert reconcile.outcome == ReconcileOutcome.RUNNING_RECOVERED.value
    assert reconcile.process_alive is True
    assert reborn.status(plan.run_id)["process"]["pid"] == pid

    lease = harness.worker.lease(plan.run_id)
    assert lease is not None
    assert lease.training_plan_sha256 == plan.digest()
    _wait(harness, plan.run_id)


def test_worker_lost_then_recovered(tmp_path: Path) -> None:
    """Silence makes a run LOST, not FAILED — and never duplicated."""
    harness = build_harness(tmp_path)
    plan = make_plan()
    manifest = build_manifest(harness.staging_root / plan.run_id, plan)
    prepare_and_receive(harness, plan, manifest)
    _start(harness, plan, steps=30, step_seconds=0.1)

    class Unreachable:
        """A client that cannot get through. Not one that says no."""

        def status(self, run_id: str) -> dict[str, Any]:
            from luber_training.remote.client import WorkerUnreachable

            raise WorkerUnreachable("connection reset")

    backend = RemoteGpuBackend(Unreachable(), harness.transport)  # type: ignore[arg-type]
    lost = backend.status(plan)
    assert lost.status == RunStatus.LOST.value
    assert "may still be running" in lost.detail

    from luber_training.remote.backend import failure_code_for

    assert failure_code_for(backend.reconcile(plan.run_id)) == "WORKER_LOST"

    # The worker comes back. The same trainer is still there.
    recovered = RemoteGpuBackend(harness.client, harness.transport, allow_code_mismatch=True)
    report = recovered.reconcile(plan.run_id)
    assert report.outcome == ReconcileOutcome.RUNNING_RECOVERED.value
    _wait(harness, plan.run_id)


# ── stopping ─────────────────────────────────────────────────────────


def test_cancel_stops_the_trainer_and_keeps_everything(tmp_path: Path) -> None:
    harness = build_harness(tmp_path)
    plan = make_plan()
    manifest = build_manifest(harness.staging_root / plan.run_id, plan)
    prepare_and_receive(harness, plan, manifest)
    _start(harness, plan, steps=200, step_seconds=0.1)

    # Let it produce something worth preserving.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if harness.layout(plan.run_id).metrics_jsonl.is_file():
            break
        time.sleep(0.05)

    state = harness.client.cancel(plan.run_id, grace_seconds=5.0)
    assert state["state"] == WorkerState.CANCELLED.value
    assert run_status_for(state["state"]) == RunStatus.CANCELLED.value

    layout = harness.layout(plan.run_id)
    assert layout.stdout_log.is_file() and layout.stdout_log.stat().st_size > 0
    assert layout.metrics_jsonl.is_file() and layout.metrics_jsonl.stat().st_size > 0
    assert layout.result_json.is_file()

    # This harness launches the trainer in-process, so pytest is its
    # parent and an exited child lingers as a zombie until someone
    # reaps it — which `status` does. In production the launching
    # worker CLI has already exited, the trainer is reparented to init,
    # and there is no zombie to see.
    from luber_training.remote.execution import (
        ExecutionState,
        TrainerProcess,
        process_alive,
    )

    stored = ExecutionState.read(layout)
    assert stored is not None and stored.process is not None
    TrainerProcess(layout).reap(stored.process)
    assert not process_alive(stored.process.pid)


def test_cancel_after_completion_does_not_overwrite_the_outcome(tmp_path: Path) -> None:
    """A real success outranks a cancellation that arrived too late."""
    harness = build_harness(tmp_path)
    plan = make_plan()
    manifest = build_manifest(harness.staging_root / plan.run_id, plan)
    prepare_and_receive(harness, plan, manifest)
    _start(harness, plan, steps=1, step_seconds=0.01, checkpoint_every=1)
    final = _wait(harness, plan.run_id)
    assert final["state"] == WorkerState.COMPLETED.value

    after = harness.client.cancel(plan.run_id)
    assert after["state"] == WorkerState.COMPLETED.value
    assert "the earlier outcome stands" in after["detail"]


def test_sigkill_path_when_the_trainer_ignores_sigterm(tmp_path: Path) -> None:
    """Graceful first, forceful only if it has to be."""
    harness = build_harness(tmp_path)
    plan = make_plan()
    manifest = build_manifest(harness.staging_root / plan.run_id, plan)
    prepare_and_receive(harness, plan, manifest)
    _start(harness, plan, steps=400, step_seconds=0.1, ignore_sigterm=True)

    state = harness.client.cancel(plan.run_id, grace_seconds=1.0)
    assert state["state"] == WorkerState.CANCELLED.value
    assert state["process"]["cancel_signal"] == "SIGKILL"


# ── failures ─────────────────────────────────────────────────────────


def test_trainer_failure_preserves_diagnostics(tmp_path: Path) -> None:
    harness = build_harness(tmp_path)
    plan = make_plan()
    manifest = build_manifest(harness.staging_root / plan.run_id, plan)
    prepare_and_receive(harness, plan, manifest)
    _start(harness, plan, steps=5, fail_at_step=2, exit_code=3)

    final = _wait(harness, plan.run_id)
    assert final["state"] == WorkerState.FAILED.value
    assert final["exit_code"] == 3
    assert final["failure_code"] == "TRAINER_CRASH"

    layout = harness.layout(plan.run_id)
    assert layout.stderr_log.stat().st_size > 0
    assert layout.result_json.is_file()

    result = RemoteResult.from_dict(harness.client.collect_result(plan.run_id))
    assert result.exit_code == 3
    # It failed before writing anything loadable.
    assert not [c for c in result.checkpoints if c.collectable]


def test_oom_is_claimed_only_on_real_evidence(tmp_path: Path) -> None:
    harness = build_harness(tmp_path)
    plan = make_plan()
    manifest = build_manifest(harness.staging_root / plan.run_id, plan)
    prepare_and_receive(harness, plan, manifest)
    _start(harness, plan, steps=4, fail_at_step=2, simulate="oom")

    final = _wait(harness, plan.run_id)
    assert final["failure_code"] == "OOM"
    assert "out of memory" in final["detail"].lower()


def test_an_unexplained_crash_is_not_called_oom(tmp_path: Path) -> None:
    """The conservative half of the classifier, and the important half."""
    harness = build_harness(tmp_path)
    plan = make_plan()
    manifest = build_manifest(harness.staging_root / plan.run_id, plan)
    prepare_and_receive(harness, plan, manifest)
    _start(harness, plan, steps=4, fail_at_step=2, simulate="crash")

    final = _wait(harness, plan.run_id)
    assert final["failure_code"] == "TRAINER_CRASH"
    assert final["failure_code"] != "OOM"


def test_disk_full_maps_to_checkpoint_write_failed(tmp_path: Path) -> None:
    harness = build_harness(tmp_path)
    plan = make_plan()
    manifest = build_manifest(harness.staging_root / plan.run_id, plan)
    prepare_and_receive(harness, plan, manifest)
    _start(harness, plan, steps=4, fail_at_step=2, simulate="disk-full")

    final = _wait(harness, plan.run_id)
    assert final["failure_code"] == "CHECKPOINT_WRITE_FAILED"


def test_an_invalid_checkpoint_is_rejected_with_its_reason(tmp_path: Path) -> None:
    """A directory in the checkpoint folder is not a checkpoint."""
    harness = build_harness(tmp_path)
    plan = make_plan()
    manifest = build_manifest(harness.staging_root / plan.run_id, plan)
    prepare_and_receive(harness, plan, manifest)
    _start(harness, plan, steps=2, checkpoint_every=2, invalid_checkpoint=True)
    _wait(harness, plan.run_id)

    checkpoints = harness.client.list_checkpoints(plan.run_id)
    assert checkpoints
    assert all(c["status"] == RemoteCheckpointStatus.REJECTED.value for c in checkpoints)
    assert any("adapter_model.safetensors" in p for c in checkpoints for p in c["problems"])

    result = RemoteResult.from_dict(harness.client.collect_result(plan.run_id))
    report = collect_run(harness.transport, result, destination_root=harness.collect_root)
    assert report.collected == []
    assert report.skipped and report.skipped[0]["status"] == RemoteCheckpointStatus.REJECTED.value


# ── streams ──────────────────────────────────────────────────────────


def test_log_polling_returns_only_new_content(tmp_path: Path) -> None:
    harness = build_harness(tmp_path)
    plan = make_plan()
    manifest = build_manifest(harness.staging_root / plan.run_id, plan)
    prepare_and_receive(harness, plan, manifest)
    _start(harness, plan, steps=4, step_seconds=0.1)

    seen: list[str] = []
    offset = 0
    for _ in range(30):
        chunk = harness.client.logs(plan.run_id, offset=offset)
        if chunk["text"]:
            seen.append(chunk["text"])
            offset = chunk["next_offset"]
        state = harness.client.status(plan.run_id)
        if state["state"] == WorkerState.COMPLETED.value and chunk["eof"]:
            break
        time.sleep(0.1)

    combined = "".join(seen)
    lines = [line for line in combined.splitlines() if "step " in line]
    assert lines
    # Every step line appears exactly once across all polls.
    assert len(lines) == len(set(lines))


def test_metric_polling_never_duplicates_an_event(tmp_path: Path) -> None:
    harness = build_harness(tmp_path)
    plan = make_plan()
    manifest = build_manifest(harness.staging_root / plan.run_id, plan)
    prepare_and_receive(harness, plan, manifest)
    _start(harness, plan, steps=4, step_seconds=0.05)
    _wait(harness, plan.run_id)

    backend = RemoteGpuBackend(harness.client, harness.transport, allow_code_mismatch=True)
    first = backend.collect_metrics(plan)
    second = backend.collect_metrics(plan)
    third = backend.collect_metrics(plan)

    assert first
    assert second == [] and third == []

    # And the file the control plane writes holds each event once, even
    # if the same batch is merged repeatedly.
    destination = tmp_path / "control" / "metrics.jsonl"
    assert merge_into(destination, first) == len(first)
    assert merge_into(destination, first) == 0

    lines = destination.read_text(encoding="utf-8").splitlines()
    identities = {
        (json.loads(line)["step"], json.loads(line)["metric_name"]) for line in lines if line
    }
    assert len(identities) == len(lines)


# ── artifact integrity ───────────────────────────────────────────────


def test_a_corrupt_dataset_file_blocks_the_trainer(tmp_path: Path) -> None:
    """Preflight rehashes on the worker, so damage after transfer is caught."""
    harness = build_harness(tmp_path)
    plan = make_plan()
    manifest = build_manifest(harness.staging_root / plan.run_id, plan)
    prepare_and_receive(harness, plan, manifest)

    target = harness.run_root / plan.run_id / "dataset" / "track-000.wav"
    target.write_bytes(b"tampered")

    passed, report = harness.client.preflight(plan.run_id, allow_code_mismatch=True)
    assert not passed
    assert report["status"] == "FAIL"
    assert any("artifact_digests" in reason for reason in report["blocking_reasons"])

    # And the trainer cannot be started past a failed preflight.
    from luber_training.remote.worker import WorkerError

    with pytest.raises(WorkerError):
        harness.worker.start(plan.run_id, argv=["/bin/true"], working_directory=tmp_path)


def test_checkpoint_download_interruption_resumes(tmp_path: Path) -> None:
    harness = build_harness(tmp_path)
    plan = make_plan()
    manifest = build_manifest(harness.staging_root / plan.run_id, plan)
    prepare_and_receive(harness, plan, manifest)
    _start(harness, plan, steps=2, checkpoint_every=2)
    _wait(harness, plan.run_id)

    result = RemoteResult.from_dict(harness.client.collect_result(plan.run_id))
    remote = next(c for c in result.checkpoints if c.collectable)

    # A transport that dies partway through every file.
    broken = LocalArtifactTransport(harness.run_root, fail_after_bytes=16)
    failed = collect_run(broken, result, destination_root=harness.collect_root)
    assert not failed.ok
    assert not (harness.collect_root / remote.checkpoint_id).is_dir()

    # A working one resumes and finishes.
    healed = collect_run(harness.transport, result, destination_root=harness.collect_root)
    assert healed.ok
    assert healed.successful[0].sha256 == remote.sha256


def test_a_corrupt_download_never_becomes_ready(tmp_path: Path) -> None:
    harness = build_harness(tmp_path)
    orchestrator = Orchestrator(Registry(tmp_path / "registry"))
    plan = make_plan()
    manifest = build_manifest(harness.staging_root / plan.run_id, plan)
    prepare_and_receive(harness, plan, manifest)
    _start(harness, plan, steps=2, checkpoint_every=2)
    _wait(harness, plan.run_id)

    result = RemoteResult.from_dict(harness.client.collect_result(plan.run_id))
    remote = next(c for c in result.checkpoints if c.collectable)

    corrupting = LocalArtifactTransport(
        harness.run_root,
        corrupt_paths=frozenset(
            {f"{plan.run_id}/checkpoints/{remote.checkpoint_id}/adapter_model.safetensors"}
        ),
    )
    report = collect_run(corrupting, result, destination_root=harness.collect_root)
    assert not report.ok
    assert not (harness.collect_root / remote.checkpoint_id).is_dir()

    with pytest.raises(CollectionError):
        register_collected(
            orchestrator,
            run_id=plan.run_id,
            collected=report.collected[0],
            remote=remote,
            kind=CheckpointKind.MOCK.value,
        )
    assert orchestrator.registry.list_ids("checkpoints") == []

    # The remote copy is untouched, so a retry can still succeed.
    retry = collect_run(harness.transport, result, destination_root=harness.collect_root)
    assert retry.ok


def test_an_interrupted_upload_leaves_no_readable_file(tmp_path: Path) -> None:
    harness = build_harness(tmp_path)
    plan = make_plan()
    staging = harness.staging_root / plan.run_id
    manifest = build_manifest(staging, plan)
    harness.client.prepare_run(plan.run_id, plan.digest(), manifest.digest())

    broken = LocalArtifactTransport(harness.run_root, fail_after_bytes=32)
    entry = next(e for e in manifest.entries if e.role == "DATASET_AUDIO")
    from luber_training.remote.transport import TransportError

    with pytest.raises(TransportError):
        broken.upload(
            staging / entry.target_path,
            f"{plan.run_id}/{entry.target_path}",
            expected_sha256=entry.sha256,
        )

    landed = harness.run_root / plan.run_id / entry.target_path
    assert not landed.exists()
    # The partial is left for the resume, and cleanup finds it.
    assert broken.remove_temp()


def test_transfer_resume_skips_what_already_matches(tmp_path: Path) -> None:
    harness = build_harness(tmp_path)
    plan = make_plan()
    staging = harness.staging_root / plan.run_id
    manifest = build_manifest(staging, plan)

    from remote_fixtures import transfer

    first = transfer(harness, plan, manifest)
    assert first.uploaded and not first.skipped

    second = transfer(harness, plan, manifest)
    assert not second.uploaded
    assert len(second.skipped) == len(manifest.entries)
    assert second.bytes_uploaded == 0


# ── worker capability mismatch ───────────────────────────────────────


def test_a_development_worker_is_blocked_from_a_cuda_plan(tmp_path: Path) -> None:
    """The local machine is not a GPU, and preflight says so."""
    harness = build_harness(tmp_path)
    plan = make_plan(requires_cuda=True)
    manifest = build_manifest(harness.staging_root / plan.run_id, plan)
    prepare_and_receive(harness, plan, manifest)

    passed, report = harness.client.preflight(plan.run_id, allow_code_mismatch=True)
    assert not passed
    assert report["status"] in ("BLOCKED", "FAIL")
    assert any("cuda" in reason for reason in report["blocking_reasons"])

    described = harness.worker.describe()
    assert described["classification"] == "DEVELOPMENT_ONLY"


def test_cleanup_removes_scratch_and_keeps_evidence(tmp_path: Path) -> None:
    harness = build_harness(tmp_path)
    plan = make_plan()
    manifest = build_manifest(harness.staging_root / plan.run_id, plan)
    prepare_and_receive(harness, plan, manifest)
    _start(harness, plan, steps=2, checkpoint_every=2)
    _wait(harness, plan.run_id)

    layout = harness.layout(plan.run_id)
    (layout.temp_dir / "scratch.bin").write_bytes(bytes(2048))

    report = harness.client.cleanup(plan.run_id)
    assert report["removed_count"] >= 1
    assert layout.stdout_log.is_file()
    assert layout.metrics_jsonl.is_file()
    assert layout.result_json.is_file()
    assert any(layout.checkpoints_dir.iterdir())
