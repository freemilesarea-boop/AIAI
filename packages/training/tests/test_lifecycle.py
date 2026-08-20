"""The full orchestration lifecycle, and the ways it must refuse.

Every scenario the phase demands: a clean synthetic run end to end, and
then the six refusals — rights, leakage, a corrupt lock, a worker
mismatch, resume compatibility, cancellation — plus idempotency and
scale.

Not one of these trains anything.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from training_fixtures import build_locked_dataset, manifest_record

from luber_training.backends import (
    DRY_RUN,
    LocalDryRunBackend,
    RemoteGpuBackend,
    RemoteGpuBackendNotImplementedError,
    capability_check,
)
from luber_training.config import preset
from luber_training.entities import (
    Checkpoint,
    CheckpointKind,
    CheckpointStatus,
    ExperimentStatus,
    FailureCode,
    RunStatus,
)
from luber_training.gates import GateInputs
from luber_training.ids import EntityKind, new_id
from luber_training.metrics import (
    CheckpointError,
    MetricEvent,
    MetricSource,
    RetentionPolicy,
    finalize_checkpoint,
    iter_metrics,
    new_staging,
    plan_retention,
)
from luber_training.orchestrator import OrchestrationError


def gate_inputs(tmp_path: Path, records, **kwargs) -> GateInputs:
    dataset, curation = build_locked_dataset(tmp_path, records)
    return GateInputs(
        dataset_lock_path=dataset / "dataset_lock.json",
        dataset_manifest_path=dataset / "dataset_manifest.jsonl",
        curation_lock_path=curation / "curation_lock.json",
        curated_manifest_path=curation / "curated_manifest.jsonl",
        **kwargs,
    )


def clean_records(count: int = 4):
    return [manifest_record(f"trk_{i:03d}") for i in range(count)]


class TestFullSyntheticLifecycle:
    def test_end_to_end_dry_run(self, orchestrator, baseline, gpu_worker, dataset_ref, tmp_path):
        """Register, experiment, run, validate, plan, start, metrics, complete."""
        experiment = orchestrator.create_experiment(
            name="EXP-SMOKE-001",
            hypothesis="the orchestration plumbing works",
            base_model_id=baseline.model_id,
        )
        run = orchestrator.create_run(
            experiment_id=experiment.experiment_id,
            dataset_ref=dataset_ref,
            config=preset("SMOKE"),
            execution_backend=DRY_RUN,
        )
        assert run.status == RunStatus.DRAFT.value

        run, report = orchestrator.validate_run(
            run.run_id, gate_inputs(tmp_path, clean_records()), worker_id=gpu_worker.worker_id
        )
        assert report.passed, report.to_dict()
        assert run.status == RunStatus.QUEUED.value

        plan = orchestrator.compile_plan(run.run_id)
        assert plan.digest() == orchestrator.get_run(run.run_id).training_plan_sha256

        backend = LocalDryRunBackend()
        run = orchestrator.start_run(run.run_id, plan, gpu_worker, backend)
        assert run.status == RunStatus.RUNNING.value

        events = backend.collect_metrics(plan)
        orchestrator.record_metrics(run.run_id, events)
        run = orchestrator.complete_run(run.run_id)
        assert run.status == RunStatus.COMPLETED.value

        directory = Path(run.output_directory)
        for name in ("plan.json", "environment_lock.json", "run.json", "metrics.jsonl"):
            assert (directory / name).exists(), name

        experiment = orchestrator.get_experiment(experiment.experiment_id)
        assert experiment.status == ExperimentStatus.COMPLETED.value

    def test_a_dry_run_produces_no_checkpoint(
        self, orchestrator, baseline, gpu_worker, dataset_ref, tmp_path
    ):
        """The most dangerous thing this package could do is fake one."""
        experiment = orchestrator.create_experiment(
            name="E", hypothesis="h", base_model_id=baseline.model_id
        )
        run = orchestrator.create_run(
            experiment_id=experiment.experiment_id,
            dataset_ref=dataset_ref,
            config=preset("SMOKE"),
        )
        orchestrator.validate_run(
            run.run_id, gate_inputs(tmp_path, clean_records()), worker_id=gpu_worker.worker_id
        )
        plan = orchestrator.compile_plan(run.run_id)
        backend = LocalDryRunBackend()
        orchestrator.start_run(run.run_id, plan, gpu_worker, backend)
        assert backend.collect_checkpoints(plan) == []
        assert orchestrator.run_checkpoints(run.run_id) == []

    def test_dry_run_metrics_are_marked_simulated(self, dataset_ref, orchestrator, baseline):
        """A simulated number must never sit unlabelled beside a real one."""
        experiment = orchestrator.create_experiment(
            name="E", hypothesis="h", base_model_id=baseline.model_id
        )
        run = orchestrator.create_run(
            experiment_id=experiment.experiment_id, dataset_ref=dataset_ref, config=preset("SMOKE")
        )
        plan = orchestrator.compile_plan(run.run_id)
        events = LocalDryRunBackend().collect_metrics(plan)
        assert events
        assert all(event.source == MetricSource.SIMULATED.value for event in events)
        assert all(event.is_simulated for event in events)

    def test_no_simulated_loss_is_emitted(self, dataset_ref, orchestrator, baseline):
        """There is no honest number to put in `train_loss`."""
        experiment = orchestrator.create_experiment(
            name="E", hypothesis="h", base_model_id=baseline.model_id
        )
        run = orchestrator.create_run(
            experiment_id=experiment.experiment_id, dataset_ref=dataset_ref, config=preset("SMOKE")
        )
        plan = orchestrator.compile_plan(run.run_id)
        names = {event.metric_name for event in LocalDryRunBackend().collect_metrics(plan)}
        assert "train_loss" not in names

    def test_a_mock_checkpoint_can_never_become_a_candidate(
        self, orchestrator, baseline, dataset_ref
    ):
        experiment = orchestrator.create_experiment(
            name="E", hypothesis="h", base_model_id=baseline.model_id
        )
        run = orchestrator.create_run(
            experiment_id=experiment.experiment_id, dataset_ref=dataset_ref, config=preset("SMOKE")
        )
        checkpoint = orchestrator.register_checkpoint(
            Checkpoint(
                checkpoint_id=new_id(EntityKind.CHECKPOINT),
                run_id=run.run_id,
                kind=CheckpointKind.MOCK.value,
                status=CheckpointStatus.READY.value,
            )
        )
        with pytest.raises(OrchestrationError, match="MOCK"):
            orchestrator.create_candidate(checkpoint.checkpoint_id)

    def test_a_real_checkpoint_becomes_a_pending_candidate(
        self, orchestrator, baseline, dataset_ref
    ):
        """And stops there. Nothing promotes it."""
        experiment = orchestrator.create_experiment(
            name="E", hypothesis="h", base_model_id=baseline.model_id
        )
        run = orchestrator.create_run(
            experiment_id=experiment.experiment_id, dataset_ref=dataset_ref, config=preset("SMOKE")
        )
        checkpoint = orchestrator.register_checkpoint(
            Checkpoint(
                checkpoint_id=new_id(EntityKind.CHECKPOINT),
                run_id=run.run_id,
                kind=CheckpointKind.ADAPTER.value,
                status=CheckpointStatus.READY.value,
            )
        )
        candidate = orchestrator.create_candidate(checkpoint.checkpoint_id)
        assert candidate.status == "PENDING_EVALUATION"
        assert not hasattr(orchestrator, "promote")


class TestRightsFailure:
    def test_a_forbidden_track_blocks_the_run(
        self, orchestrator, baseline, gpu_worker, dataset_ref, tmp_path
    ):
        experiment = orchestrator.create_experiment(
            name="E", hypothesis="h", base_model_id=baseline.model_id
        )
        run = orchestrator.create_run(
            experiment_id=experiment.experiment_id, dataset_ref=dataset_ref, config=preset("SMOKE")
        )
        records = [*clean_records(2), manifest_record("trk_bad", permission="UNKNOWN")]
        run, report = orchestrator.validate_run(run.run_id, gate_inputs(tmp_path, records))

        assert not report.passed
        assert run.status == RunStatus.FAILED.value
        assert run.error_code == FailureCode.RIGHTS_GATE_FAILED.value

    def test_no_backend_is_reached(self, orchestrator, baseline, gpu_worker, dataset_ref, tmp_path):
        """A blocked run cannot be started at all."""
        experiment = orchestrator.create_experiment(
            name="E", hypothesis="h", base_model_id=baseline.model_id
        )
        run = orchestrator.create_run(
            experiment_id=experiment.experiment_id, dataset_ref=dataset_ref, config=preset("SMOKE")
        )
        records = [manifest_record("trk_bad", permission="FALSE")]
        orchestrator.validate_run(run.run_id, gate_inputs(tmp_path, records))
        plan = orchestrator.compile_plan(run.run_id)
        with pytest.raises(OrchestrationError, match="FAILED"):
            orchestrator.start_run(run.run_id, plan, gpu_worker, LocalDryRunBackend())


class TestLeakageFailure:
    def test_an_evaluation_only_track_blocks_the_run(
        self, orchestrator, baseline, dataset_ref, tmp_path
    ):
        experiment = orchestrator.create_experiment(
            name="E", hypothesis="h", base_model_id=baseline.model_id
        )
        run = orchestrator.create_run(
            experiment_id=experiment.experiment_id, dataset_ref=dataset_ref, config=preset("SMOKE")
        )
        records = [*clean_records(2), manifest_record("trk_p20_bench")]
        inputs = gate_inputs(tmp_path, records, evaluation_only_ids=frozenset({"trk_p20_bench"}))
        run, report = orchestrator.validate_run(run.run_id, inputs)
        assert not report.passed
        assert run.error_code == FailureCode.EVALUATION_LEAKAGE.value


class TestCorruptLock:
    def test_a_manifest_changed_after_the_lock_blocks_the_run(
        self, orchestrator, baseline, dataset_ref, tmp_path
    ):
        experiment = orchestrator.create_experiment(
            name="E", hypothesis="h", base_model_id=baseline.model_id
        )
        run = orchestrator.create_run(
            experiment_id=experiment.experiment_id, dataset_ref=dataset_ref, config=preset("SMOKE")
        )
        inputs = gate_inputs(tmp_path, clean_records(2))
        # Tamper after the freeze.
        rows = [json.loads(line) for line in inputs.dataset_manifest_path.read_text().splitlines()]
        rows.append(manifest_record("trk_added_later"))
        inputs.dataset_manifest_path.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
        )

        run, report = orchestrator.validate_run(run.run_id, inputs)
        assert not report.passed
        assert run.error_code == FailureCode.DATASET_LOCK_INVALID.value
        assert run.status == RunStatus.FAILED.value


class TestWorkerMismatch:
    def test_a_development_mac_cannot_take_a_cuda_plan(
        self, orchestrator, baseline, mac_worker, dataset_ref, tmp_path
    ):
        experiment = orchestrator.create_experiment(
            name="E", hypothesis="h", base_model_id=baseline.model_id
        )
        run = orchestrator.create_run(
            experiment_id=experiment.experiment_id, dataset_ref=dataset_ref, config=preset("SMOKE")
        )
        orchestrator.validate_run(
            run.run_id, gate_inputs(tmp_path, clean_records()), worker_id=mac_worker.worker_id
        )
        plan = orchestrator.compile_plan(run.run_id)
        check = capability_check(plan, mac_worker)
        assert not check.ok
        assert any("DEVELOPMENT_ONLY" in problem for problem in check.problems)

    def test_the_dry_run_backend_does_not_launder_the_mismatch(
        self, orchestrator, baseline, mac_worker, dataset_ref, tmp_path
    ):
        """ "It passed on dry-run" must not become evidence of GPU readiness."""
        experiment = orchestrator.create_experiment(
            name="E", hypothesis="h", base_model_id=baseline.model_id
        )
        run = orchestrator.create_run(
            experiment_id=experiment.experiment_id, dataset_ref=dataset_ref, config=preset("SMOKE")
        )
        plan = orchestrator.compile_plan(run.run_id)
        assert not LocalDryRunBackend().validate_environment(plan, mac_worker).ok

    def test_an_unprobed_worker_cannot_satisfy_cuda(
        self, orchestrator, baseline, dataset_ref, mac_worker
    ):
        """Silence is not consent: unmeasured is not "probably fine"."""
        experiment = orchestrator.create_experiment(
            name="E", hypothesis="h", base_model_id=baseline.model_id
        )
        run = orchestrator.create_run(
            experiment_id=experiment.experiment_id, dataset_ref=dataset_ref, config=preset("SMOKE")
        )
        plan = orchestrator.compile_plan(run.run_id)
        check = capability_check(plan, mac_worker)
        assert any("never reported CUDA" in problem for problem in check.problems)

    def test_unknown_vram_is_reported_as_unknown_not_as_a_pass(
        self, orchestrator, baseline, gpu_worker, dataset_ref
    ):
        experiment = orchestrator.create_experiment(
            name="E", hypothesis="h", base_model_id=baseline.model_id
        )
        run = orchestrator.create_run(
            experiment_id=experiment.experiment_id, dataset_ref=dataset_ref, config=preset("SMOKE")
        )
        plan = orchestrator.compile_plan(run.run_id)
        check = capability_check(plan, gpu_worker)
        assert check.ok
        assert any("UNKNOWN_REQUIREMENT" in note for note in check.unknown)

    def test_preflight_keeps_unknown_separate_from_pass(
        self, orchestrator, baseline, gpu_worker, dataset_ref, tmp_path
    ):
        experiment = orchestrator.create_experiment(
            name="E", hypothesis="h", base_model_id=baseline.model_id
        )
        run = orchestrator.create_run(
            experiment_id=experiment.experiment_id, dataset_ref=dataset_ref, config=preset("SMOKE")
        )
        _, report = orchestrator.validate_run(
            run.run_id, gate_inputs(tmp_path, clean_records()), worker_id=gpu_worker.worker_id
        )
        plan = orchestrator.compile_plan(run.run_id)
        preflight = orchestrator.preflight(
            run.run_id,
            plan,
            gpu_worker,
            LocalDryRunBackend(),
            report,
            require_clean_repository=False,
        )
        assert preflight.unknown
        assert preflight.checks["disk_capacity"] == "UNKNOWN"


class TestResume:
    def test_a_ready_checkpoint_can_be_resumed_with_lineage(
        self, orchestrator, baseline, dataset_ref
    ):
        experiment = orchestrator.create_experiment(
            name="E", hypothesis="h", base_model_id=baseline.model_id
        )
        first = orchestrator.create_run(
            experiment_id=experiment.experiment_id, dataset_ref=dataset_ref, config=preset("SMOKE")
        )
        checkpoint = orchestrator.register_checkpoint(
            Checkpoint(
                checkpoint_id=new_id(EntityKind.CHECKPOINT),
                run_id=first.run_id,
                kind=CheckpointKind.ADAPTER.value,
                status=CheckpointStatus.READY.value,
                epoch=1,
            )
        )
        second = orchestrator.create_run(
            experiment_id=experiment.experiment_id,
            dataset_ref=dataset_ref,
            config=preset("SMOKE"),
            parent_run_id=first.run_id,
            resume_from_checkpoint_id=checkpoint.checkpoint_id,
        )
        assert second.parent_run_id == first.run_id
        assert second.resume_from_checkpoint_id == checkpoint.checkpoint_id

    def test_a_writing_checkpoint_cannot_be_resumed(self, orchestrator, baseline, dataset_ref):
        experiment = orchestrator.create_experiment(
            name="E", hypothesis="h", base_model_id=baseline.model_id
        )
        run = orchestrator.create_run(
            experiment_id=experiment.experiment_id, dataset_ref=dataset_ref, config=preset("SMOKE")
        )
        checkpoint = orchestrator.register_checkpoint(
            Checkpoint(
                checkpoint_id=new_id(EntityKind.CHECKPOINT),
                run_id=run.run_id,
                kind=CheckpointKind.ADAPTER.value,
                status=CheckpointStatus.WRITING.value,
            )
        )
        with pytest.raises(OrchestrationError, match="only READY"):
            orchestrator.create_run(
                experiment_id=experiment.experiment_id,
                dataset_ref=dataset_ref,
                config=preset("SMOKE"),
                resume_from_checkpoint_id=checkpoint.checkpoint_id,
            )

    def test_a_mock_artifact_cannot_be_resumed(self, orchestrator, baseline, dataset_ref):
        experiment = orchestrator.create_experiment(
            name="E", hypothesis="h", base_model_id=baseline.model_id
        )
        run = orchestrator.create_run(
            experiment_id=experiment.experiment_id, dataset_ref=dataset_ref, config=preset("SMOKE")
        )
        checkpoint = orchestrator.register_checkpoint(
            Checkpoint(
                checkpoint_id=new_id(EntityKind.CHECKPOINT),
                run_id=run.run_id,
                kind=CheckpointKind.MOCK.value,
                status=CheckpointStatus.READY.value,
            )
        )
        with pytest.raises(OrchestrationError, match="no trained weights"):
            orchestrator.create_run(
                experiment_id=experiment.experiment_id,
                dataset_ref=dataset_ref,
                config=preset("SMOKE"),
                resume_from_checkpoint_id=checkpoint.checkpoint_id,
            )


class TestCancellation:
    def test_cancelling_preserves_everything_it_produced(
        self, orchestrator, baseline, gpu_worker, dataset_ref, tmp_path
    ):
        experiment = orchestrator.create_experiment(
            name="E", hypothesis="h", base_model_id=baseline.model_id
        )
        run = orchestrator.create_run(
            experiment_id=experiment.experiment_id, dataset_ref=dataset_ref, config=preset("SMOKE")
        )
        orchestrator.validate_run(
            run.run_id, gate_inputs(tmp_path, clean_records()), worker_id=gpu_worker.worker_id
        )
        plan = orchestrator.compile_plan(run.run_id)
        backend = LocalDryRunBackend()
        orchestrator.start_run(run.run_id, plan, gpu_worker, backend)

        orchestrator.record_metrics(
            run.run_id,
            [
                MetricEvent(
                    run_id=run.run_id,
                    metric_name="step_time_seconds",
                    value=1.0,
                    source=MetricSource.SIMULATED.value,
                    step=1,
                )
            ],
        )
        run = orchestrator.cancel_run(run.run_id, plan, backend)

        assert run.status == RunStatus.CANCELLED.value
        assert run.error_code == FailureCode.CANCELLED_BY_OPERATOR.value
        metrics = list(iter_metrics(Path(run.output_directory) / "metrics.jsonl"))
        assert metrics, "partial metrics must survive a cancellation"
        events = {e["event"] for e in orchestrator.registry.audit_events(run.run_id)}
        assert "RUN_CANCELLED" in events

    def test_a_cancelled_run_is_not_completed(
        self, orchestrator, baseline, gpu_worker, dataset_ref, tmp_path
    ):
        experiment = orchestrator.create_experiment(
            name="E", hypothesis="h", base_model_id=baseline.model_id
        )
        run = orchestrator.create_run(
            experiment_id=experiment.experiment_id, dataset_ref=dataset_ref, config=preset("SMOKE")
        )
        orchestrator.validate_run(
            run.run_id, gate_inputs(tmp_path, clean_records()), worker_id=gpu_worker.worker_id
        )
        plan = orchestrator.compile_plan(run.run_id)
        backend = LocalDryRunBackend()
        orchestrator.start_run(run.run_id, plan, gpu_worker, backend)
        orchestrator.cancel_run(run.run_id, plan, backend)
        with pytest.raises(OrchestrationError, match="cannot move"):
            orchestrator.complete_run(run.run_id)


class TestIdempotentLaunch:
    def test_starting_twice_starts_one_trainer(
        self, orchestrator, baseline, gpu_worker, dataset_ref, tmp_path
    ):
        experiment = orchestrator.create_experiment(
            name="E", hypothesis="h", base_model_id=baseline.model_id
        )
        run = orchestrator.create_run(
            experiment_id=experiment.experiment_id, dataset_ref=dataset_ref, config=preset("SMOKE")
        )
        orchestrator.validate_run(
            run.run_id, gate_inputs(tmp_path, clean_records()), worker_id=gpu_worker.worker_id
        )
        plan = orchestrator.compile_plan(run.run_id)

        starts: list[str] = []

        class CountingBackend(LocalDryRunBackend):
            def start(self, plan, worker):
                starts.append(plan.run_id)
                return super().start(plan, worker)

        backend = CountingBackend()
        first = orchestrator.start_run(run.run_id, plan, gpu_worker, backend)
        second = orchestrator.start_run(run.run_id, plan, gpu_worker, backend)

        assert len(starts) == 1
        assert first.status == second.status == RunStatus.RUNNING.value
        started = [
            e for e in orchestrator.registry.audit_events(run.run_id) if e["event"] == "RUN_STARTED"
        ]
        assert len(started) == 1

    def test_recompiling_a_plan_after_a_change_is_refused(
        self, orchestrator, baseline, gpu_worker, dataset_ref, tmp_path
    ):
        """A different plan means a different run, not an edited one."""
        experiment = orchestrator.create_experiment(
            name="E", hypothesis="h", base_model_id=baseline.model_id
        )
        run = orchestrator.create_run(
            experiment_id=experiment.experiment_id, dataset_ref=dataset_ref, config=preset("SMOKE")
        )
        orchestrator.compile_plan(run.run_id)

        stored = orchestrator.get_run(run.run_id)
        stored.config = preset("LORA_SMALL")
        orchestrator.save_run(stored)
        with pytest.raises(OrchestrationError, match="Create a new run"):
            orchestrator.compile_plan(run.run_id)

    def test_immutable_fields_are_frozen_after_start(
        self, orchestrator, baseline, gpu_worker, dataset_ref, tmp_path
    ):
        experiment = orchestrator.create_experiment(
            name="E", hypothesis="h", base_model_id=baseline.model_id
        )
        run = orchestrator.create_run(
            experiment_id=experiment.experiment_id, dataset_ref=dataset_ref, config=preset("SMOKE")
        )
        orchestrator.validate_run(
            run.run_id, gate_inputs(tmp_path, clean_records()), worker_id=gpu_worker.worker_id
        )
        stored = orchestrator.get_run(run.run_id)
        stored.config = preset("LORA_HIGH_QUALITY")
        with pytest.raises(OrchestrationError, match="may not change"):
            orchestrator.save_run(stored)


class TestWorkerLost:
    def test_a_vanished_worker_marks_runs_lost_not_failed(
        self, orchestrator, baseline, gpu_worker, dataset_ref, tmp_path
    ):
        """We know we lost contact. We do not know training stopped."""
        experiment = orchestrator.create_experiment(
            name="E", hypothesis="h", base_model_id=baseline.model_id
        )
        run = orchestrator.create_run(
            experiment_id=experiment.experiment_id, dataset_ref=dataset_ref, config=preset("SMOKE")
        )
        orchestrator.validate_run(
            run.run_id, gate_inputs(tmp_path, clean_records()), worker_id=gpu_worker.worker_id
        )
        plan = orchestrator.compile_plan(run.run_id)
        orchestrator.start_run(run.run_id, plan, gpu_worker, LocalDryRunBackend())

        affected = orchestrator.mark_worker_lost(gpu_worker.worker_id)
        assert run.run_id in affected
        lost = orchestrator.get_run(run.run_id)
        assert lost.status == RunStatus.LOST.value
        assert lost.error_code == FailureCode.WORKER_LOST.value

    def test_concurrency_is_capped_per_worker(
        self, orchestrator, baseline, gpu_worker, dataset_ref, tmp_path
    ):
        experiment = orchestrator.create_experiment(
            name="E", hypothesis="h", base_model_id=baseline.model_id
        )
        runs = []
        for _ in range(2):
            run = orchestrator.create_run(
                experiment_id=experiment.experiment_id,
                dataset_ref=dataset_ref,
                config=preset("SMOKE"),
            )
            _, report = orchestrator.validate_run(
                run.run_id, gate_inputs(tmp_path, clean_records()), worker_id=gpu_worker.worker_id
            )
            runs.append((run, report))

        plan = orchestrator.compile_plan(runs[0][0].run_id)
        orchestrator.start_run(runs[0][0].run_id, plan, gpu_worker, LocalDryRunBackend())

        second_plan = orchestrator.compile_plan(runs[1][0].run_id)
        preflight = orchestrator.preflight(
            runs[1][0].run_id,
            second_plan,
            gpu_worker,
            LocalDryRunBackend(),
            runs[1][1],
            require_clean_repository=False,
        )
        assert preflight.checks["worker_capacity"] == "FAIL"


class TestRemoteBackendContract:
    def test_every_execution_method_refuses(self):
        backend = RemoteGpuBackend(host_ref="host-1", credential_ref="ssh-key-name")
        for call in ("status", "cancel", "cleanup", "collect_metrics", "collect_checkpoints"):
            with pytest.raises(RemoteGpuBackendNotImplementedError):
                getattr(backend, call)(None)
        for call in ("prepare_run", "start"):
            with pytest.raises(RemoteGpuBackendNotImplementedError):
                getattr(backend, call)(None, None)

    def test_it_holds_references_never_secrets(self):
        backend = RemoteGpuBackend(host_ref="host-1", credential_ref="prod-ssh-key")
        assert backend.credential_ref == "prod-ssh-key"
        assert "BEGIN" not in backend.credential_ref

    def test_capability_matching_still_works(self, orchestrator, baseline, mac_worker, dataset_ref):
        """The one thing it can do without a transport."""
        experiment = orchestrator.create_experiment(
            name="E", hypothesis="h", base_model_id=baseline.model_id
        )
        run = orchestrator.create_run(
            experiment_id=experiment.experiment_id, dataset_ref=dataset_ref, config=preset("SMOKE")
        )
        plan = orchestrator.compile_plan(run.run_id)
        backend = RemoteGpuBackend(host_ref="h", credential_ref="c")
        assert not backend.validate_environment(plan, mac_worker).ok


class TestCheckpointFinalization:
    def test_a_valid_adapter_finalizes_atomically(self, tmp_path: Path):
        staging = new_staging(tmp_path, "ckpt_test").staging_path
        (staging / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
        (staging / "adapter_model.safetensors").write_bytes(b"weights")
        result = finalize_checkpoint(staging, tmp_path / "final")
        assert (tmp_path / "final" / "adapter_config.json").is_file()
        assert len(result.sha256) == 64
        assert result.size_bytes > 0

    def test_an_incomplete_checkpoint_never_finalizes(self, tmp_path: Path):
        """The interrupted-write case: a partial adapter is not a model."""
        staging = new_staging(tmp_path, "ckpt_partial").staging_path
        (staging / "adapter_config.json").write_text("{}", encoding="utf-8")
        with pytest.raises(CheckpointError, match="not loadable"):
            finalize_checkpoint(staging, tmp_path / "final")
        assert not (tmp_path / "final").exists()

    def test_an_empty_weights_file_is_refused(self, tmp_path: Path):
        staging = new_staging(tmp_path, "ckpt_empty").staging_path
        (staging / "adapter_config.json").write_text("{}", encoding="utf-8")
        (staging / "adapter_model.safetensors").write_bytes(b"")
        with pytest.raises(CheckpointError, match="is empty"):
            finalize_checkpoint(staging, tmp_path / "final")

    def test_an_existing_destination_is_never_overwritten(self, tmp_path: Path):
        staging = new_staging(tmp_path, "ckpt").staging_path
        (staging / "adapter_config.json").write_text("{}", encoding="utf-8")
        (staging / "adapter_model.safetensors").write_bytes(b"w")
        (tmp_path / "final").mkdir()
        with pytest.raises(CheckpointError, match="never overwritten"):
            finalize_checkpoint(staging, tmp_path / "final")

    def test_only_finalization_sets_ready(self, orchestrator, baseline, dataset_ref):
        experiment = orchestrator.create_experiment(
            name="E", hypothesis="h", base_model_id=baseline.model_id
        )
        run = orchestrator.create_run(
            experiment_id=experiment.experiment_id, dataset_ref=dataset_ref, config=preset("SMOKE")
        )
        checkpoint = orchestrator.register_checkpoint(
            Checkpoint(
                checkpoint_id=new_id(EntityKind.CHECKPOINT),
                run_id=run.run_id,
                kind=CheckpointKind.ADAPTER.value,
            )
        )
        assert checkpoint.status == CheckpointStatus.WRITING.value
        finalized = orchestrator.finalize_checkpoint_record(
            checkpoint.checkpoint_id, sha256="a" * 64, size_bytes=1024, reference="file:///x"
        )
        assert finalized.status == CheckpointStatus.READY.value

    def test_retention_plans_but_never_deletes(self):
        checkpoints = [
            {
                "checkpoint_id": f"ckpt_{i}",
                "status": "READY",
                "step": i * 100,
                "metrics_snapshot": {"train_loss": 1.0 / (i + 1)},
            }
            for i in range(6)
        ]
        plan = plan_retention(checkpoints, RetentionPolicy(keep_latest_n=2, keep_best_n=1))
        assert plan.keep
        assert plan.delete
        assert all(reason for reason in plan.reasons.values())
        assert "nothing is deleted" in plan.to_dict()["note"]


class TestRegistryDurability:
    def test_a_registry_write_is_atomic(self, registry, tmp_path):
        """No temp files survive a successful write."""
        registry.write("models", "mdl_" + "0" * 16, {"model_id": "x"})
        leftovers = list((registry.root / "models").glob(".tmp-*"))
        assert leftovers == []

    def test_the_lock_is_reentrant(self, registry):
        with registry.lock(), registry.lock():
            registry.write("models", "mdl_" + "1" * 16, {"model_id": "y"})

    def test_duplicate_ids_are_refused(self, registry):
        from luber_training.registry import ConflictError

        registry.write("models", "mdl_" + "2" * 16, {"model_id": "z"})
        with pytest.raises(ConflictError):
            registry.write("models", "mdl_" + "2" * 16, {"model_id": "z"})

    def test_an_unsafe_id_cannot_escape_the_registry(self, registry):
        from luber_training.registry import RegistryError

        with pytest.raises(RegistryError, match="unsafe"):
            registry.read("models", "../../etc/passwd")

    def test_the_audit_log_is_append_only(self, orchestrator, baseline):
        events_before = len(orchestrator.registry.audit_events())
        orchestrator.create_experiment(name="E", hypothesis="h", base_model_id=baseline.model_id)
        assert len(orchestrator.registry.audit_events()) > events_before

    def test_a_torn_audit_line_does_not_break_reading(self, registry):
        path = registry.root / "audit_log.jsonl"
        registry.append_audit("RUN_CREATED", "run_" + "0" * 16, "run")
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"partial": ')
        assert len(registry.audit_events()) == 1


class TestScale:
    def test_a_thousand_runs_stay_fast(self, orchestrator, baseline, dataset_ref):
        """Metadata only. Nothing here should be quadratic."""
        started = time.monotonic()
        experiments = [
            orchestrator.create_experiment(
                name=f"EXP-{i:03d}", hypothesis="h", base_model_id=baseline.model_id
            )
            for i in range(100)
        ]
        for index in range(1000):
            orchestrator.create_run(
                experiment_id=experiments[index % 100].experiment_id,
                dataset_ref=dataset_ref,
                config=preset("SMOKE"),
            )
        creation = time.monotonic() - started

        listing_started = time.monotonic()
        runs = orchestrator.registry.list_all("runs")
        listing = time.monotonic() - listing_started

        assert len(runs) == 1000
        assert creation < 120.0, f"creating 1000 runs took {creation:.1f}s"
        assert listing < 20.0, f"listing 1000 runs took {listing:.1f}s"

    def test_ids_do_not_collide(self):
        ids = {new_id(EntityKind.RUN) for _ in range(20_000)}
        assert len(ids) == 20_000
