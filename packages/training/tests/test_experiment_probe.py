"""The parts of the experiment probe that can be tested without a trainer.

The probe itself runs inside the trainer process and needs torch, a
model and forty minutes. What can be checked here is the logic around
that: how a validation point is shaped, when validation fires, and that
a provenance record lands beside a checkpoint with the epoch and step
filled in.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "packages" / "training" / "src" / "luber_training" / "_experiment_probe.py"


def _load():
    spec = importlib.util.spec_from_file_location("experiment_probe_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load()


def _update(step: int, epoch: int, loss: float = 1.0):
    return SimpleNamespace(kind="step", step=step, epoch=epoch, loss=loss, learning_rate=1e-4)


class _Validator:
    """Stands in for the real one; records when it was asked."""

    def __init__(self):
        self.calls = []
        self.load_error = ""

    def evaluate(self, module, *, epoch, step):
        self.calls.append((epoch, step))
        return {"epoch": epoch, "step": step, "loss": 1.0 - 0.01 * epoch}


class TestWhenValidationFires:
    def test_it_fires_once_per_epoch_boundary_and_not_mid_epoch(self):
        validator = _Validator()
        recorder = probe.Recorder(step_ceiling=100, validator=validator)
        recorder.module_ref = object()
        for step, epoch in ((1, 1), (2, 1), (3, 1), (4, 2), (5, 2), (6, 3)):
            recorder.record_step(_update(step, epoch))
            recorder.maybe_validate(_update(step, epoch))
        # Boundaries crossed: 1->2 and 2->3. The last epoch is measured
        # at the end of the run, not here.
        assert [epoch for epoch, _ in validator.calls] == [1, 2]

    def test_it_does_not_fire_before_the_first_epoch_completes(self):
        validator = _Validator()
        recorder = probe.Recorder(step_ceiling=100, validator=validator)
        recorder.module_ref = object()
        for step in (1, 2, 3):
            recorder.maybe_validate(_update(step, 1))
        assert validator.calls == []

    def test_a_recorder_with_no_validator_never_validates(self):
        recorder = probe.Recorder(step_ceiling=100)
        recorder.maybe_validate(_update(1, 1))
        recorder.maybe_validate(_update(2, 2))
        assert recorder.validation_points == []

    def test_each_point_carries_the_segment_it_came_from(self):
        validator = _Validator()
        recorder = probe.Recorder(step_ceiling=100, validator=validator, segment="B")
        recorder.module_ref = object()
        recorder.record_step(_update(1, 1))
        recorder.maybe_validate(_update(1, 1))
        recorder.record_step(_update(2, 2))
        recorder.maybe_validate(_update(2, 2))
        assert recorder.validation_points[0]["segment"] == "B"


class TestTheStepCeiling:
    def test_passing_the_ceiling_raises_inside_the_trainer(self):
        import pytest

        recorder = probe.Recorder(step_ceiling=3)
        for step in (1, 2, 3):
            recorder.record_step(_update(step, 1))
        with pytest.raises(probe.StepCeilingExceeded):
            recorder.record_step(_update(4, 1))
        assert recorder.ceiling_hit

    def test_the_points_collected_before_the_ceiling_survive(self):
        import pytest

        recorder = probe.Recorder(step_ceiling=2)
        recorder.record_step(_update(1, 1))
        recorder.record_step(_update(2, 1))
        with pytest.raises(probe.StepCeilingExceeded):
            recorder.record_step(_update(3, 1))
        assert len(recorder.points) == 3


class TestProvenanceBesideACheckpoint:
    def test_it_writes_the_supplied_record_with_the_path_filled_in(self, tmp_path):
        recorder = probe.Recorder(
            step_ceiling=100,
            provenance={"run_id": "run_1", "epoch": 0, "step": 0, "_filename": "prov.json"},
        )
        recorder.write_provenance(str(tmp_path), epoch=10, step=60)
        payload = json.loads((tmp_path / "prov.json").read_text(encoding="utf-8"))
        assert payload["run_id"] == "run_1"
        assert payload["checkpoint_path"] == str(tmp_path)
        assert payload["epoch"] == 10
        assert payload["step"] == 60
        assert "_filename" not in payload

    def test_it_records_which_checkpoints_it_wrote(self, tmp_path):
        recorder = probe.Recorder(step_ceiling=100, provenance={"run_id": "run_1"})
        recorder.write_provenance(str(tmp_path), epoch=1, step=6)
        assert recorder.checkpoints == [{"path": str(tmp_path), "epoch": 1, "step": 6}]

    def test_a_missing_directory_is_skipped_rather_than_created(self, tmp_path):
        """A provenance file beside a checkpoint that does not exist would
        describe nothing."""
        recorder = probe.Recorder(step_ceiling=100, provenance={"run_id": "run_1"})
        recorder.write_provenance(str(tmp_path / "absent"), epoch=1, step=6)
        assert not (tmp_path / "absent").exists()
        assert recorder.checkpoints == []

    def test_no_provenance_supplied_writes_nothing(self, tmp_path):
        recorder = probe.Recorder(step_ceiling=100)
        recorder.write_provenance(str(tmp_path), epoch=1, step=6)
        assert list(tmp_path.iterdir()) == []


class TestTheParameterComparison:
    def test_an_empty_intersection_reports_unknown_not_zero(self):
        """The Phase 35 trap, guarded again here."""
        recorder = probe.Recorder(step_ceiling=10)
        recorder.before = {"totally.different.name.a.b.weight": [1.0, 1.0, 4]}
        recorder.after = {"nothing.in.common.here.c.d.bias": [2.0, 2.0, 4]}
        comparison = recorder.compare()
        assert comparison["changed_tensor_count"] is None
        assert "unknown" in comparison["detail"]

    def test_wrapper_prefixes_are_stripped_before_comparing(self):
        recorder = probe.Recorder(step_ceiling=10)
        recorder.before = {"blocks.1.attn.q.lora_A.default.weight": [1.0, 1.0, 4]}
        recorder.after = {"_forward_module.blocks.1.attn.q.lora_A.default.weight": [2.0, 2.0, 4]}
        comparison = recorder.compare()
        assert comparison["changed_tensor_count"] == 1
        assert comparison["comparable_tensor_count"] == 1

    def test_no_fingerprint_pair_says_so_rather_than_guessing(self):
        recorder = probe.Recorder(step_ceiling=10)
        assert recorder.compare()["changed_tensor_count"] is None
