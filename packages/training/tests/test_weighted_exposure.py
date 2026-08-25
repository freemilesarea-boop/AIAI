"""Weighted exposure, which two phases claimed and neither had.

Phase 37 and Phase 38 wrote per-window sampling weights and trained with
``shuffle=True``. The weights changed nothing. These tests exist so that
sentence can never be true again quietly: if exposure stops tracking
weight, something here fails.

The central test is `test_a_heavier_sample_is_visited_more_often`. The
rest guard the properties a controlled experiment needs — determinism,
no silent defaults, nothing dropped.
"""

import json

import pytest

from luber_training.weighted_exposure import (
    MINIMUM_OCCURRENCES,
    DeterministicWeightedSampler,
    WeightedExposureError,
    build_exposure_plan,
    load_window_weights,
)


def _names(count: int) -> list[str]:
    return [f"track{index:02d}-w0-b0" for index in range(count)]


class TestExposureTracksWeight:
    def test_a_heavier_sample_is_visited_more_often(self):
        names = _names(4)
        weights = {names[0]: 4.0, names[1]: 2.0, names[2]: 1.0, names[3]: 1.0}
        plan = build_exposure_plan(names, weights, seed=39, length=40)
        assert plan.repeats[names[0]] > plan.repeats[names[1]]
        assert plan.repeats[names[1]] > plan.repeats[names[2]]

    def test_visits_are_roughly_proportional_to_weight(self):
        names = _names(4)
        weights = {names[0]: 4.0, names[1]: 2.0, names[2]: 1.0, names[3]: 1.0}
        plan = build_exposure_plan(names, weights, seed=39, length=80)
        # 4:2:1:1 over 80 visits is 40:20:10:10.
        assert plan.repeats[names[0]] == pytest.approx(40, abs=2)
        assert plan.repeats[names[1]] == pytest.approx(20, abs=2)
        assert plan.repeats[names[2]] == pytest.approx(10, abs=2)

    def test_the_order_really_contains_those_visits(self):
        names = _names(3)
        plan = build_exposure_plan(
            names, {names[0]: 3.0, names[1]: 1.0, names[2]: 1.0}, seed=1, length=30
        )
        for name, count in plan.repeats.items():
            assert plan.order.count(name) == count

    def test_equal_weights_expose_equally(self):
        names = _names(5)
        plan = build_exposure_plan(names, dict.fromkeys(names, 1.0), seed=7, length=25)
        assert set(plan.repeats.values()) == {5}

    def test_exposure_shares_sum_to_one(self):
        names = _names(6)
        weights = {name: 1.0 + index for index, name in enumerate(names)}
        plan = build_exposure_plan(names, weights, seed=3, length=60)
        assert sum(plan.exposure.values()) == pytest.approx(1.0)


class TestDeterminism:
    def test_the_same_seed_gives_the_same_order(self):
        names = _names(8)
        weights = {name: 1.0 + index * 0.5 for index, name in enumerate(names)}
        first = build_exposure_plan(names, weights, seed=39, length=40)
        second = build_exposure_plan(names, weights, seed=39, length=40)
        assert first.order == second.order

    def test_a_different_seed_reorders_without_changing_exposure(self):
        names = _names(8)
        weights = {name: 1.0 + index * 0.5 for index, name in enumerate(names)}
        first = build_exposure_plan(names, weights, seed=39, length=40)
        second = build_exposure_plan(names, weights, seed=40, length=40)
        assert first.order != second.order
        assert first.repeats == second.repeats

    def test_input_ordering_does_not_change_the_result(self):
        names = _names(6)
        weights = {name: 1.0 + index for index, name in enumerate(names)}
        forward = build_exposure_plan(names, weights, seed=5, length=30)
        backward = build_exposure_plan(list(reversed(names)), weights, seed=5, length=30)
        assert forward.order == backward.order


class TestItRefusesRatherThanDefaults:
    def test_a_sample_with_no_weight_raises(self):
        names = _names(3)
        with pytest.raises(WeightedExposureError, match="no weight"):
            build_exposure_plan(names, {names[0]: 1.0, names[1]: 1.0}, seed=1)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
    def test_an_unusable_weight_raises(self, bad):
        names = _names(2)
        with pytest.raises(WeightedExposureError, match="positive finite"):
            build_exposure_plan(names, {names[0]: bad, names[1]: 1.0}, seed=1)

    def test_duplicate_names_raise(self):
        with pytest.raises(WeightedExposureError, match="duplicate"):
            build_exposure_plan(["a", "a"], {"a": 1.0}, seed=1)

    def test_an_epoch_too_short_to_hold_every_sample_raises(self):
        names = _names(10)
        with pytest.raises(WeightedExposureError, match="visit"):
            build_exposure_plan(names, dict.fromkeys(names, 1.0), seed=1, length=4)

    def test_an_empty_population_raises(self):
        with pytest.raises(WeightedExposureError, match="no samples"):
            build_exposure_plan([], {}, seed=1)


class TestNothingIsDropped:
    def test_every_sample_is_visited_at_least_once(self):
        names = _names(50)
        # One sample weighted far below the rest would round to zero
        # visits without the floor.
        weights = dict.fromkeys(names, 4.0)
        weights[names[0]] = 0.01
        plan = build_exposure_plan(names, weights, seed=39)
        assert min(plan.repeats.values()) >= MINIMUM_OCCURRENCES
        assert set(plan.repeats) == set(names)

    def test_the_floor_is_recorded_when_it_binds(self):
        names = _names(50)
        weights = dict.fromkeys(names, 4.0)
        weights[names[0]] = 0.01
        plan = build_exposure_plan(names, weights, seed=39)
        assert names[0] in plan.minimum_applied

    def test_the_epoch_is_the_length_that_was_asked_for(self):
        names = _names(9)
        weights = {name: 1.0 + index for index, name in enumerate(names)}
        for length in (9, 20, 45, 100):
            plan = build_exposure_plan(names, weights, seed=2, length=length)
            assert len(plan.order) == length

    def test_a_weighted_epoch_costs_the_same_as_an_unweighted_one_by_default(self):
        names = _names(12)
        weights = {name: 1.0 + index for index, name in enumerate(names)}
        assert len(build_exposure_plan(names, weights, seed=2)) == len(names)


class TestTheSampler:
    def test_it_yields_dataset_indices_in_plan_order(self):
        names = _names(3)
        plan = build_exposure_plan(
            names, {names[0]: 2.0, names[1]: 1.0, names[2]: 1.0}, seed=4, length=12
        )
        index_of = {name: position for position, name in enumerate(names)}
        sampler = DeterministicWeightedSampler(plan, index_of)
        assert len(sampler) == 12
        assert [index_of[name] for name in plan.order] == list(sampler)

    def test_a_heavier_sample_appears_more_often_in_the_indices(self):
        names = _names(3)
        plan = build_exposure_plan(
            names, {names[0]: 6.0, names[1]: 1.0, names[2]: 1.0}, seed=4, length=24
        )
        index_of = {name: position for position, name in enumerate(names)}
        drawn = list(DeterministicWeightedSampler(plan, index_of))
        assert drawn.count(0) > drawn.count(1)

    def test_it_refuses_a_dataset_missing_a_planned_sample(self):
        names = _names(3)
        plan = build_exposure_plan(names, dict.fromkeys(names, 1.0), seed=4)
        with pytest.raises(WeightedExposureError, match="absent from the dataset"):
            DeterministicWeightedSampler(plan, {names[0]: 0, names[1]: 1})

    def test_it_is_reiterable(self):
        names = _names(4)
        plan = build_exposure_plan(names, dict.fromkeys(names, 1.0), seed=4)
        sampler = DeterministicWeightedSampler(plan, {n: i for i, n in enumerate(names)})
        assert list(sampler) == list(sampler)


class TestReadingTheWeightsThatWereAlreadyWritten:
    def test_it_reads_sampling_weights_from_a_windows_document(self, tmp_path):
        path = tmp_path / "windows_train.json"
        path.write_text(json.dumps({"sampling_weights": {"a-w0-b0": 1.5, "b-w0-b0": 0.5}}))
        assert load_window_weights(path) == {"a-w0-b0": 1.5, "b-w0-b0": 0.5}

    def test_a_document_without_weights_raises(self, tmp_path):
        path = tmp_path / "windows_train.json"
        path.write_text(json.dumps({"windows": []}))
        with pytest.raises(WeightedExposureError, match="no sampling_weights"):
            load_window_weights(path)

    def test_the_plan_serialises_what_it_did(self, tmp_path):
        names = _names(3)
        plan = build_exposure_plan(
            names, {names[0]: 2.0, names[1]: 1.0, names[2]: 1.0}, seed=8, length=12
        )
        payload = plan.to_dict()
        assert payload["epoch_length"] == 12
        assert payload["seed"] == 8
        assert payload["repeats"][names[0]] > payload["repeats"][names[1]]
        assert sum(payload["exposure"].values()) == pytest.approx(1.0, abs=1e-4)


class TestTheProbeActuallyRewiresTheLoader:
    """The half that Phase 37 and 38 were missing.

    A correct exposure plan changes nothing unless it reaches the
    DataLoader. These tests stand in fake `acestep` and `torch` modules —
    neither is installed in this package's environment, and that is
    deliberate: `luber_training` imports no torch — and check that the
    probe replaces shuffling with the planned order, and refuses loudly
    when it cannot.
    """

    @pytest.fixture
    def probe(self, monkeypatch):
        import sys
        import types

        captured = {}

        class FakeDataLoader:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.dataset = kwargs.get("dataset")
                self.batch_size = kwargs.get("batch_size", 1)
                self.num_workers = kwargs.get("num_workers", 0)
                self.pin_memory = kwargs.get("pin_memory", False)
                self.collate_fn = kwargs.get("collate_fn")
                self.drop_last = kwargs.get("drop_last", False)
                self.prefetch_factor = kwargs.get("prefetch_factor")
                self.persistent_workers = kwargs.get("persistent_workers", False)

        class FakeDataset:
            def __init__(self, stems):
                self.valid_paths = [f"/tensors/{stem}.pt" for stem in stems]

        class FakeModule:
            def __init__(self, stems):
                self.train_dataset = FakeDataset(stems)

            def train_dataloader(self):
                return FakeDataLoader(dataset=self.train_dataset, batch_size=1, num_workers=0)

        torch_mod = types.ModuleType("torch")
        utils = types.ModuleType("torch.utils")
        data = types.ModuleType("torch.utils.data")
        data.DataLoader = FakeDataLoader
        utils.data = data
        torch_mod.utils = utils
        acestep = types.ModuleType("acestep")
        training = types.ModuleType("acestep.training")
        dm = types.ModuleType("acestep.training.data_module")
        dm.PreprocessedDataModule = FakeModule
        training.data_module = dm
        acestep.training = training
        for name, mod in {
            "torch": torch_mod,
            "torch.utils": utils,
            "torch.utils.data": data,
            "acestep": acestep,
            "acestep.training": training,
            "acestep.training.data_module": dm,
        }.items():
            monkeypatch.setitem(sys.modules, name, mod)

        from luber_training import _experiment_probe

        return _experiment_probe, FakeModule, captured

    def test_the_planned_order_reaches_the_dataloader(self, probe):
        module, FakeModule, captured = probe
        stems = _names(3)
        plan = build_exposure_plan(
            stems, {stems[0]: 6.0, stems[1]: 1.0, stems[2]: 1.0}, seed=4, length=24
        )
        report = module.install_weighted_exposure(plan.order)
        loader = FakeModule(stems).train_dataloader()
        assert "sampler" in captured
        assert "shuffle" not in captured
        assert len(captured["sampler"]) == 24
        # The heaviest sample must actually be drawn more often.
        assert captured["sampler"].count(0) > captured["sampler"].count(1)
        assert report["installed"] is True
        assert report["epoch_length"] == 24
        assert loader.batch_size == 1

    def test_a_sample_absent_from_the_dataset_stops_the_run(self, probe):
        module, FakeModule, _ = probe
        stems = _names(3)
        plan = build_exposure_plan(stems, dict.fromkeys(stems, 1.0), seed=4)
        module.install_weighted_exposure((*plan.order, "not-in-the-dataset-w0-b0"))
        with pytest.raises(module.ExposureNotInstalled, match="not in the dataset"):
            FakeModule(stems).train_dataloader()

    def test_a_dataset_that_cannot_name_its_samples_stops_the_run(self, probe):
        module, FakeModule, _ = probe
        stems = _names(2)
        plan = build_exposure_plan(stems, dict.fromkeys(stems, 1.0), seed=4)
        module.install_weighted_exposure(plan.order)
        anonymous = FakeModule(stems)
        del anonymous.train_dataset.valid_paths
        with pytest.raises(module.ExposureNotInstalled, match="no valid_paths"):
            anonymous.train_dataloader()
