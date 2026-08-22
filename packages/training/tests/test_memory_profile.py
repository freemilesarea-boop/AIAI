"""Memory evidence, and the ways it must refuse to be reused.

Most of this file is about the second half. Measuring memory is the easy
part; the hard part is that a real measurement of the wrong workload
looks exactly like a real measurement of the right one, and is far more
dangerous than no measurement at all. So the tests that matter are the
refusals: a bf16 profile must not qualify fp32, a batch-1 profile must
not qualify batch 4, a two-second profile must not qualify four minutes,
and a fixture must never qualify hardware nobody owns.

Every CUDA case here runs against a literal — no NVIDIA hardware has
ever been attached to this project — and the fixture module says so.
"""

from __future__ import annotations

import dataclasses
import json
import os
import threading
import time
from pathlib import Path

import pytest
from memory_fixtures import (
    GIB,
    PRODUCTION_LATENT,
    a_cuda_profile,
    a_profile,
    a_snapshot,
    an_identity,
    requested_from,
)

from luber_hardware import ComputeDevice
from luber_training.capacity import EvidenceSource
from luber_training.capacity_policy import (
    Applicability,
    CapacityPolicy,
    CapacityQualification,
    applicability,
    qualify,
)
from luber_training.memory import (
    MemoryDomain,
    MemoryFailureKind,
    MemoryProfileIdentity,
    MemorySnapshot,
    PeakKind,
    ProfileFormatError,
    ProfileOutcome,
    ProfileStage,
    Representativeness,
    TrainingMemoryProfile,
    classify_memory_failure,
    summarise_peaks,
)
from luber_training.memory_profiler import (
    LATENT_FRAMES_PER_SECOND,
    PRODUCTION_LATENT_LENGTH,
    ProbeShape,
    load_profiles,
    render_markdown,
    write_profile,
)


def _host_bytes(total: int = 24 * GIB) -> int:
    return total


# ── 50-52. snapshots, one domain at a time ───────────────────────────


class TestSnapshots:
    def test_a_host_only_snapshot_leaves_the_accelerator_fields_unknown(self):
        snapshot = a_snapshot(ProfileStage.BASELINE.value)
        assert snapshot.value_for(MemoryDomain.HOST.value) == 512 * 1024 * 1024
        assert snapshot.value_for(MemoryDomain.APPLE_UNIFIED.value) is None
        assert snapshot.value_for(MemoryDomain.CUDA_DEVICE.value) is None

    def test_an_apple_snapshot_reports_the_driver_allocation(self):
        """Driver, not current: the caching allocator holds back the rest."""
        snapshot = a_snapshot(
            ProfileStage.FORWARD_COMPLETE.value,
            mps_current_allocated_bytes=4 * GIB,
            mps_driver_allocated_bytes=6 * GIB,
            mps_recommended_max_bytes=17 * GIB,
        )
        assert snapshot.value_for(MemoryDomain.APPLE_UNIFIED.value) == 6 * GIB
        assert snapshot.value_for(MemoryDomain.CUDA_DEVICE.value) is None

    def test_a_cuda_snapshot_is_fixture_only_and_reports_reserved(self):
        snapshot = a_snapshot(
            ProfileStage.FORWARD_COMPLETE.value,
            cuda_allocated_bytes=10 * GIB,
            cuda_reserved_bytes=12 * GIB,
            cuda_total_bytes=80 * GIB,
        )
        assert snapshot.value_for(MemoryDomain.CUDA_DEVICE.value) == 12 * GIB
        assert snapshot.value_for(MemoryDomain.APPLE_UNIFIED.value) is None

    def test_a_snapshot_survives_a_round_trip(self):
        snapshot = a_snapshot(ProfileStage.MODEL_LOADED.value, mps_driver_allocated_bytes=5 * GIB)
        assert MemorySnapshot.from_dict(snapshot.to_dict()) == snapshot

    def test_an_absent_figure_stays_none_rather_than_zero(self):
        restored = MemorySnapshot.from_dict({"stage": "BASELINE"})
        assert restored.host_rss_bytes is None
        assert restored.mps_driver_allocated_bytes is None


# ── 8, 53. peaks and evidence classes ────────────────────────────────


class TestPeaksAndEvidence:
    def test_an_apple_peak_is_sampled_and_says_so(self):
        """The pinned torch has no MPS peak counter, so it must be."""
        snapshots = [
            a_snapshot(ProfileStage.BASELINE.value, mps_driver_allocated_bytes=0),
            a_snapshot(ProfileStage.FORWARD_COMPLETE.value, mps_driver_allocated_bytes=9 * GIB),
        ]
        peaks = summarise_peaks(snapshots, device=ComputeDevice.MPS.value)
        apple = next(peak for peak in peaks if peak.domain == MemoryDomain.APPLE_UNIFIED.value)
        assert apple.kind == PeakKind.SAMPLED_PEAK.value
        assert apple.peak_bytes == 9 * GIB
        assert "lower bound" in apple.detail

    def test_a_cuda_runtime_peak_is_used_when_the_runtime_kept_one(self):
        snapshots = [a_snapshot(ProfileStage.BASELINE.value, cuda_reserved_bytes=0)]
        peaks = summarise_peaks(
            snapshots,
            device=ComputeDevice.CUDA.value,
            runtime_peaks={MemoryDomain.CUDA_DEVICE.value: 12 * GIB},
        )
        cuda = next(peak for peak in peaks if peak.domain == MemoryDomain.CUDA_DEVICE.value)
        assert cuda.kind == PeakKind.RUNTIME_PEAK.value
        assert cuda.peak_bytes == 12 * GIB

    def test_the_wrong_accelerator_domain_is_absent_rather_than_zero(self):
        peaks = summarise_peaks([a_snapshot("BASELINE")], device=ComputeDevice.MPS.value)
        assert {peak.domain for peak in peaks} == {
            MemoryDomain.HOST.value,
            MemoryDomain.APPLE_UNIFIED.value,
        }

    def test_growth_is_derived_never_measured(self):
        profile = a_profile()
        apple = profile.peak_for(MemoryDomain.APPLE_UNIFIED.value)
        assert apple is not None
        payload = apple.to_dict()
        assert payload["source"] == EvidenceSource.MEASURED.value
        assert payload["growth_source"] == EvidenceSource.DERIVED.value

    def test_an_unmeasured_peak_is_unknown_not_measured(self):
        snapshots = [a_snapshot("BASELINE", host_rss_bytes=None)]
        peaks = summarise_peaks(snapshots, device=ComputeDevice.MPS.value)
        host = next(peak for peak in peaks if peak.domain == MemoryDomain.HOST.value)
        assert host.kind == PeakKind.NOT_AVAILABLE.value
        assert host.to_dict()["source"] == EvidenceSource.UNKNOWN.value


# ── 54-55. profile identity ──────────────────────────────────────────


class TestProfileIdentity:
    @pytest.mark.parametrize(
        "field,value",
        [
            ("precision", "fp32"),
            ("device", ComputeDevice.CUDA.value),
            ("micro_batch_size", 4),
            ("lora_rank", 64),
            ("lora_alpha", 128),
            ("gradient_checkpointing", False),
            ("latent_length", 128),
            ("encoder_length", 512),
            ("model_variant", "base"),
            ("optimizer", "adafactor"),
            ("num_devices", 2),
            ("offload_encoder", True),
        ],
    )
    def test_a_memory_relevant_change_changes_the_identity(self, field, value):
        assert an_identity().digest() != an_identity(**{field: value}).digest()

    def test_gradient_accumulation_changes_the_identity_but_not_the_micro_batch(self):
        """Different things, and the profile keeps them apart.

        Accumulation multiplies the effective batch without holding more
        activations at once, so it is recorded and does not stand in for
        a larger micro batch.
        """
        base = an_identity()
        accumulated = an_identity(gradient_accumulation=8)
        assert base.digest() != accumulated.digest()
        assert accumulated.micro_batch_size == base.micro_batch_size
        assert accumulated.effective_batch_size == 8

    def test_volatile_facts_are_not_part_of_the_identity(self):
        """A profile is identified by what it measured, not by when."""
        identity = an_identity()
        before = identity.digest()
        profile_one = a_profile(identity=identity)
        profile_two = a_profile(identity=identity)
        profile_two.started_at = "2030-01-01T00:00:00+00:00"
        profile_two.finished_at = "2030-01-01T00:10:00+00:00"
        profile_two.snapshots = [
            a_snapshot("BASELINE", host_available_bytes=1 * GIB, elapsed_seconds=99.0)
        ]
        assert identity.digest() == before
        assert profile_one.identity_digest == profile_two.identity_digest

    def test_the_identity_knows_how_long_its_sequence_is_in_seconds(self):
        identity = an_identity(latent_length=PRODUCTION_LATENT)
        assert identity.latent_seconds(LATENT_FRAMES_PER_SECOND) == pytest.approx(240.0)


# ── 56, 68. applicability and staleness ──────────────────────────────


class TestApplicability:
    def test_an_identical_configuration_is_applicable(self):
        profile = a_profile()
        verdict, _ = applicability(profile, requested_from(profile.identity))
        assert verdict == Applicability.APPLICABLE.value

    @pytest.mark.parametrize(
        "field,value",
        [
            ("precision", "fp32"),
            ("device", ComputeDevice.CUDA.value),
            ("micro_batch_size", 4),
            ("lora_rank", 64),
            ("model_variant", "base"),
            ("base_model_upstream_commit", "0" * 40),
        ],
    )
    def test_a_different_configuration_is_not(self, field, value):
        profile = a_profile()
        verdict, detail = applicability(profile, requested_from(profile.identity, **{field: value}))
        assert verdict == Applicability.CONFIGURATION_MISMATCH.value
        assert field in detail

    def test_a_longer_sequence_is_not_covered_by_a_shorter_measurement(self):
        profile = a_profile(identity=an_identity(latent_length=128))
        verdict, detail = applicability(
            profile, requested_from(profile.identity, latent_length=PRODUCTION_LATENT)
        )
        assert verdict == Applicability.CONFIGURATION_MISMATCH.value
        assert "does not cover a longer run" in detail

    def test_a_shorter_sequence_is_covered_by_a_longer_measurement(self):
        """The one direction that is sound: less of the same thing."""
        profile = a_profile(identity=an_identity(latent_length=PRODUCTION_LATENT))
        verdict, _ = applicability(profile, requested_from(profile.identity, latent_length=1500))
        assert verdict == Applicability.APPLICABLE.value

    def test_an_incomplete_profile_is_not_evidence(self):
        profile = a_profile(outcome=ProfileOutcome.PROFILE_TIMEOUT.value)
        verdict, detail = applicability(profile, requested_from(profile.identity))
        assert verdict == Applicability.INCOMPLETE.value
        assert "however far it got" in detail

    def test_a_moved_runtime_makes_a_profile_stale(self):
        profile = a_profile(torch_version="2.10.0")
        verdict, detail = applicability(
            profile, requested_from(profile.identity, torch_version="2.11.0")
        )
        assert verdict == Applicability.STALE_RUNTIME.value
        assert "torch_version" in detail

    def test_a_later_date_alone_does_not_make_a_profile_stale(self):
        profile = a_profile()
        profile.finished_at = "2020-01-01T00:00:00+00:00"
        verdict, _ = applicability(profile, requested_from(profile.identity))
        assert verdict == Applicability.APPLICABLE.value

    def test_a_non_representative_profile_does_not_qualify_production(self):
        profile = a_profile(representativeness=Representativeness.NOT_REPRESENTATIVE.value)
        verdict, _ = applicability(profile, requested_from(profile.identity))
        assert verdict == Applicability.NOT_REPRESENTATIVE.value


# ── 57-61, 64. qualification ─────────────────────────────────────────


class TestQualification:
    def _qualify(self, profiles, **kwargs):
        identity = kwargs.pop("identity", an_identity())
        return qualify(
            device=kwargs.pop("device", ComputeDevice.MPS.value),
            requested=requested_from(identity),
            profiles=profiles,
            host_total_bytes=kwargs.pop("host_total_bytes", _host_bytes()),
            device_total_bytes=kwargs.pop("device_total_bytes", 24 * GIB),
            **kwargs,
        )

    def test_an_applicable_profile_within_policy_qualifies(self):
        decision = self._qualify([a_profile(peak_bytes=6 * GIB)], runs_control_plane=False)
        assert decision.qualification == CapacityQualification.QUALIFIED.value
        assert decision.permits_full_training

    def test_a_peak_below_total_but_inside_the_reserve_does_not_qualify(self):
        """The whole point of a headroom policy."""
        decision = self._qualify([a_profile(peak_bytes=22 * GIB)], runs_control_plane=False)
        assert decision.qualification == CapacityQualification.INSUFFICIENT.value
        assert not decision.permits_full_training

    def test_a_narrow_pass_is_reported_margin_low(self):
        # 15 GiB x 1.25 sampled margin = 18.75 GiB against a 24 - 3.6 =
        # 20.4 GiB budget: inside it, and above 85% of it.
        decision = self._qualify([a_profile(peak_bytes=15 * GIB)], runs_control_plane=False)
        assert decision.qualification == CapacityQualification.MARGIN_LOW.value
        assert decision.permits_full_training

    def test_no_profile_at_all_is_unverified(self):
        decision = self._qualify([])
        assert decision.qualification == CapacityQualification.UNVERIFIED.value
        assert not decision.permits_full_training
        assert decision.evidence[0].source == EvidenceSource.UNKNOWN.value

    def test_an_inapplicable_profile_is_unverified_not_qualified(self):
        """Evidence about something else is not evidence."""
        decision = self._qualify(
            [a_profile(identity=an_identity(precision="fp32"), peak_bytes=1 * GIB)]
        )
        assert decision.qualification == CapacityQualification.UNVERIFIED.value

    def test_the_control_plane_reserve_makes_a_pass_stricter(self):
        profiles = [a_profile(peak_bytes=6 * GIB)]
        shared = self._qualify(profiles, runs_control_plane=True)
        dedicated = self._qualify(profiles, runs_control_plane=False)
        host_shared = next(
            item for item in shared.domains if item.domain == MemoryDomain.HOST.value
        )
        host_dedicated = next(
            item for item in dedicated.domains if item.domain == MemoryDomain.HOST.value
        )
        assert host_shared.reserved_bytes > host_dedicated.reserved_bytes

    def test_a_sampled_peak_is_given_a_larger_margin_than_a_runtime_peak(self):
        policy = CapacityPolicy()
        assert policy.margin_for(PeakKind.SAMPLED_PEAK.value) > policy.margin_for(
            PeakKind.RUNTIME_PEAK.value
        )

    def test_the_most_conservative_applicable_profile_is_used(self):
        decision = self._qualify(
            [
                a_profile(peak_bytes=2 * GIB, profile_id="small"),
                a_profile(peak_bytes=22 * GIB, profile_id="large"),
            ],
            runs_control_plane=False,
        )
        assert decision.profile_id == "large"
        assert decision.qualification == CapacityQualification.INSUFFICIENT.value

    def test_a_cuda_fixture_cannot_qualify_an_apple_request(self):
        decision = self._qualify([a_cuda_profile()])
        assert decision.qualification == CapacityQualification.UNVERIFIED.value

    def test_an_apple_profile_cannot_qualify_a_cuda_request(self):
        """No NVIDIA hardware exists here; nothing may pretend otherwise."""
        decision = self._qualify(
            [a_profile()],
            device=ComputeDevice.CUDA.value,
            identity=an_identity(device=ComputeDevice.CUDA.value),
            device_total_bytes=80 * GIB,
        )
        assert decision.qualification == CapacityQualification.UNVERIFIED.value

    def test_a_machine_that_never_reported_its_memory_is_unverified(self):
        decision = self._qualify(
            [a_profile(peak_bytes=6 * GIB)], host_total_bytes=None, device_total_bytes=None
        )
        assert decision.qualification == CapacityQualification.UNVERIFIED.value

    def test_the_decision_names_its_policy_version(self):
        decision = self._qualify([a_profile(peak_bytes=6 * GIB)])
        assert decision.policy_version == CapacityPolicy().version

    def test_derived_figures_are_labelled_derived(self):
        decision = self._qualify([a_profile(peak_bytes=6 * GIB)], runs_control_plane=False)
        sources = {item.name: item.source for item in decision.evidence}
        assert any(
            source == EvidenceSource.MEASURED.value
            for name, source in sources.items()
            if name.startswith("measured_peak")
        )
        assert any(
            source == EvidenceSource.DERIVED.value
            for name, source in sources.items()
            if name.startswith("required_with_margin")
        )


# ── 62. representativeness ───────────────────────────────────────────


class TestRepresentativeness:
    def test_the_phase_33_canary_shape_is_not_representative(self):
        """64 frames is about two and a half seconds of audio."""
        status, detail = ProbeShape(latent_length=64, encoder_length=32).representativeness()
        assert status == Representativeness.NOT_REPRESENTATIVE.value
        assert "2s" in detail or "3s" in detail

    def test_a_production_length_shape_is_representative(self):
        status, _ = ProbeShape(latent_length=PRODUCTION_LATENT_LENGTH).representativeness()
        assert status == Representativeness.REPRESENTATIVE.value

    def test_a_middling_shape_is_partially_representative(self):
        status, _ = ProbeShape(latent_length=PRODUCTION_LATENT_LENGTH // 2).representativeness()
        assert status == Representativeness.PARTIALLY_REPRESENTATIVE.value

    def test_the_production_length_follows_from_the_vae_and_the_cap(self):
        """25 frames a second, 240 seconds. Stated so it can be checked."""
        assert LATENT_FRAMES_PER_SECOND == 48000 / (2 * 4 * 4 * 6 * 10)
        assert PRODUCTION_LATENT_LENGTH == 6000


# ── 63. terminology ──────────────────────────────────────────────────


class TestTerminology:
    def test_apple_memory_is_never_called_vram(self):
        profile = a_profile()
        rendered = render_markdown(profile).lower()
        assert "unified memory" in rendered
        assert "vram" not in rendered.replace("not vram", "")

    def test_the_evidence_marks_apple_figures_as_unified_memory(self):
        decision = qualify(
            device=ComputeDevice.MPS.value,
            requested=requested_from(an_identity()),
            profiles=[a_profile(peak_bytes=6 * GIB)],
            host_total_bytes=_host_bytes(),
            device_total_bytes=24 * GIB,
            runs_control_plane=False,
        )
        apple = [item for item in decision.evidence if "apple_unified" in item.name]
        assert apple and all(item.unified_memory for item in apple)


# ── 65. OOM taxonomy ─────────────────────────────────────────────────


class TestOomTaxonomy:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("CUDA out of memory. Tried to allocate 2.00 GiB", MemoryFailureKind.CUDA_OOM),
            ("torch.cuda.OutOfMemoryError", MemoryFailureKind.CUDA_OOM),
            (
                "MPS backend out of memory (MPS allocated: 17.00 GB)",
                MemoryFailureKind.MPS_OOM,
            ),
            ("OSError: [Errno 12] Cannot allocate memory", MemoryFailureKind.HOST_OOM),
            ("std::bad_alloc", MemoryFailureKind.HOST_OOM),
            ("something ran out of memory somewhere", MemoryFailureKind.UNKNOWN_MEMORY_FAILURE),
        ],
    )
    def test_known_signatures_are_classified(self, text, expected):
        assert classify_memory_failure(text) == expected.value

    @pytest.mark.parametrize(
        "text",
        [
            "RuntimeError: shape '[1, 2]' is invalid for input of size 6",
            "FileNotFoundError: train.py",
            "Killed",
            "",
            None,
        ],
    )
    def test_an_ordinary_failure_stays_ordinary(self, text):
        """A SIGKILL is not on any list: the OOM killer and `kill -9`
        are indistinguishable from outside the process."""
        assert classify_memory_failure(text) == MemoryFailureKind.NOT_A_MEMORY_FAILURE.value


# ── 66. the sampler stops ────────────────────────────────────────────


class TestSamplerCleanup:
    def _probe(self):
        from luber_training import _memory_probe

        return _memory_probe

    def test_it_stops_on_success(self):
        probe = self._probe()
        recorder = probe.Recorder(probe.Reader())
        sampler = probe.Sampler(recorder, 0.05)
        sampler.start()
        time.sleep(0.2)
        sampler.stop()
        assert not sampler.running

    def test_it_stops_when_the_recorder_raises(self):
        probe = self._probe()

        class Exploding(probe.Recorder):
            def record(self, stage, note=""):
                raise RuntimeError("boom")

        sampler = probe.Sampler(Exploding(probe.Reader()), 0.05)
        sampler.start()
        time.sleep(0.15)
        sampler.stop()
        assert not sampler.running

    def test_stopping_twice_is_harmless(self):
        probe = self._probe()
        sampler = probe.Sampler(probe.Recorder(probe.Reader()), 0.05)
        sampler.start()
        sampler.stop()
        sampler.stop()
        assert not sampler.running

    def test_it_leaves_no_thread_behind(self):
        probe = self._probe()
        before = {thread.name for thread in threading.enumerate()}
        sampler = probe.Sampler(probe.Recorder(probe.Reader()), 0.05)
        sampler.start()
        time.sleep(0.1)
        sampler.stop()
        after = {thread.name for thread in threading.enumerate()}
        assert "luber-memory-sampler" not in after - before

    def test_a_host_floor_crossing_is_raised_in_the_caller_not_the_sampler(self):
        """The sampler observes; the main thread decides.

        Driven with an explicit reading rather than a live one, because
        the boundary only exists where the figure does — LUBER's own
        environment has no psutil, and a check with nothing to read
        stays silent rather than guessing.
        """
        probe = self._probe()
        recorder = probe.Recorder(probe.Reader(), limits={"host_available_floor_bytes": 4 * GIB})
        recorder.check_safety({"host_available_bytes": 1 * GIB})
        assert recorder.safety_tripped
        with pytest.raises(probe.MemorySafetyAbort):
            recorder.raise_if_tripped()

    def test_an_apple_boundary_crossing_trips_on_the_runtimes_own_maximum(self):
        probe = self._probe()
        recorder = probe.Recorder(probe.Reader(), limits={"mps_recommended_max_fraction": 0.9})
        recorder.check_safety(
            {
                "mps_driver_allocated_bytes": 19 * GIB,
                "mps_recommended_max_bytes": 20 * GIB,
            }
        )
        assert recorder.safety_tripped
        assert "recommended maximum" in recorder.safety_tripped

    def test_a_reading_inside_the_boundary_does_not_trip(self):
        probe = self._probe()
        recorder = probe.Recorder(
            probe.Reader(),
            limits={"host_available_floor_bytes": 1 * GIB, "mps_recommended_max_fraction": 0.9},
        )
        recorder.check_safety(
            {
                "host_available_bytes": 8 * GIB,
                "mps_driver_allocated_bytes": 4 * GIB,
                "mps_recommended_max_bytes": 20 * GIB,
            }
        )
        assert recorder.safety_tripped is None
        recorder.raise_if_tripped()

    def test_a_missing_reading_never_trips(self):
        """No prediction: a boundary needs a number to cross."""
        probe = self._probe()
        recorder = probe.Recorder(probe.Reader(), limits={"host_available_floor_bytes": 4 * GIB})
        recorder.check_safety({"host_available_bytes": None})
        assert recorder.safety_tripped is None


# ── 67. the probe's document is parsed defensively ───────────────────


class TestStructuredEvents:
    def test_a_profile_from_another_schema_is_refused(self):
        with pytest.raises(ProfileFormatError):
            TrainingMemoryProfile.from_dict({"schema_version": "something-else/9"})

    def test_a_profile_with_no_identity_is_refused(self):
        payload = a_profile().to_dict()
        payload.pop("identity")
        with pytest.raises(ProfileFormatError):
            TrainingMemoryProfile.from_dict(payload)

    def test_unknown_fields_are_ignored_rather_than_executed(self):
        payload = a_profile().to_dict()
        payload["something_new"] = {"__class__": "os.system"}
        restored = TrainingMemoryProfile.from_dict(payload)
        assert restored.profile_id == "fixture-profile"

    def test_a_malformed_snapshot_does_not_take_the_profile_down(self):
        payload = a_profile().to_dict()
        payload["snapshots"].append({"stage": "BASELINE", "host_rss_bytes": "not a number"})
        restored = TrainingMemoryProfile.from_dict(payload)
        assert restored.snapshots[-1].host_rss_bytes is None

    def test_an_unreadable_profile_is_skipped_rather_than_fatal(self, tmp_path: Path):
        directory = tmp_path / "profiles"
        directory.mkdir()
        (directory / "broken.json").write_text("{not json", encoding="utf-8")
        (directory / "old.json").write_text(json.dumps({"schema_version": "x/1"}), encoding="utf-8")
        write_profile(a_profile(), directory)
        assert len(load_profiles(directory)) == 1

    def test_profiles_for_different_identities_coexist(self, tmp_path: Path):
        directory = tmp_path / "profiles"
        write_profile(a_profile(profile_id="bf16"), directory)
        write_profile(
            a_profile(identity=an_identity(precision="fp32"), profile_id="fp32"), directory
        )
        assert {profile.profile_id for profile in load_profiles(directory)} == {"bf16", "fp32"}


# ── 72. the profiler changes nothing to make it fit ──────────────────


class TestNoConfigMutation:
    def test_the_identity_is_taken_from_the_plan_unchanged(self):
        from preflight_fixtures import a_plan

        from luber_training.config import TrainingConfig
        from luber_training.memory_profiler import identity_for

        config = TrainingConfig(
            epochs=1, batch_size=2, rank=32, alpha=64, precision="bf16", gradient_accumulation=8
        )
        plan = a_plan(device=ComputeDevice.MPS.value, config=config)
        identity = identity_for(plan, ProbeShape(latent_length=PRODUCTION_LATENT))
        assert identity.micro_batch_size == 2
        assert identity.gradient_accumulation == 8
        assert identity.lora_rank == 32
        assert identity.precision == "bf16"
        assert identity.device == ComputeDevice.MPS.value

    def test_a_smaller_shape_is_a_different_profile_not_a_quieter_one(self):
        from preflight_fixtures import a_plan

        from luber_training.memory_profiler import identity_for, profile_id_for

        plan = a_plan(device=ComputeDevice.MPS.value)
        big = identity_for(plan, ProbeShape(latent_length=PRODUCTION_LATENT))
        small = identity_for(plan, ProbeShape(latent_length=64))
        assert big.digest() != small.digest()
        assert profile_id_for(big, plan.digest()) != profile_id_for(small, plan.digest())


def test_a_profile_says_it_says_nothing_about_quality():
    assert "music quality" in a_profile().to_dict()["note"]


def test_stage_order_is_checked_rather_than_repaired():
    profile = a_profile()
    profile.snapshots = [
        a_snapshot(ProfileStage.MODEL_LOADED.value),
        a_snapshot(ProfileStage.BASELINE.value),
    ]
    assert not profile.stages_in_order()


def test_repeated_step_stages_are_not_out_of_order():
    profile = a_profile()
    profile.snapshots = [
        a_snapshot(ProfileStage.BASELINE.value),
        a_snapshot(ProfileStage.FORWARD_COMPLETE.value),
        a_snapshot(ProfileStage.BACKWARD_COMPLETE.value),
        a_snapshot(ProfileStage.FORWARD_COMPLETE.value),
    ]
    assert profile.stages_in_order()


def test_an_identity_round_trips(tmp_path: Path):
    identity = an_identity()
    assert MemoryProfileIdentity.from_dict(identity.to_dict()).digest() == identity.digest()


# ── 69. the preflight consumes the qualification ─────────────────────


class TestPreflightIntegration:
    def _preflight(self, decision):
        from preflight_fixtures import a_request, cpu_capability

        from luber_training.capacity import capacity_report
        from luber_training.preflight import PreflightIntent, evaluate

        request = a_request(
            intent=PreflightIntent.FULL_TRAINING.value,
            capacity=capacity_report(
                cpu_capability(), device=ComputeDevice.CPU.value, free_disk_mb=200_000
            ),
        )
        return evaluate(dataclasses.replace(request, capacity_decision=decision))

    def test_a_qualified_decision_lets_full_training_reach_ready(self):
        from luber_training.preflight import PreflightStatus

        decision = qualify(
            device=ComputeDevice.MPS.value,
            requested=requested_from(an_identity()),
            profiles=[a_profile(peak_bytes=6 * GIB)],
            host_total_bytes=_host_bytes(),
            device_total_bytes=24 * GIB,
            runs_control_plane=False,
        )
        assert decision.qualification == CapacityQualification.QUALIFIED.value
        result = self._preflight(decision)
        assert result.status == PreflightStatus.READY.value, result.unverified
        assert result.capacity_qualification == CapacityQualification.QUALIFIED.value

    def test_an_unverified_decision_leaves_full_training_unverified(self):
        from luber_training.preflight import PreflightStatus

        decision = qualify(
            device=ComputeDevice.MPS.value,
            requested=requested_from(an_identity()),
            profiles=[],
            host_total_bytes=_host_bytes(),
        )
        result = self._preflight(decision)
        assert result.status == PreflightStatus.UNVERIFIED.value
        assert any("CAPACITY_UNVERIFIED" in item for item in result.unverified)

    def test_an_insufficient_decision_blocks(self):
        from luber_training.preflight import PreflightStatus

        decision = qualify(
            device=ComputeDevice.MPS.value,
            requested=requested_from(an_identity()),
            profiles=[a_profile(peak_bytes=23 * GIB)],
            host_total_bytes=_host_bytes(),
            device_total_bytes=24 * GIB,
            runs_control_plane=False,
        )
        assert decision.qualification == CapacityQualification.INSUFFICIENT.value
        result = self._preflight(decision)
        assert result.status == PreflightStatus.BLOCKED.value

    def test_a_canary_does_not_need_a_production_memory_figure(self):
        """The run that produces the figure must not require it first."""
        from preflight_fixtures import a_request, cpu_capability

        from luber_training.capacity import capacity_report
        from luber_training.preflight import PreflightIntent, PreflightStatus, evaluate

        result = evaluate(
            a_request(
                intent=PreflightIntent.CANARY.value,
                capacity=capacity_report(
                    cpu_capability(), device=ComputeDevice.CPU.value, free_disk_mb=200_000
                ),
            )
        )
        assert result.status == PreflightStatus.READY.value

    def test_a_successful_canary_alone_does_not_qualify_full_training(self):
        """Phase 33 evidence is about the mechanism, not about capacity."""
        from preflight_fixtures import CanaryEvidence, a_request, cpu_capability

        from luber_training.capacity import capacity_report
        from luber_training.preflight import PreflightIntent, PreflightStatus, evaluate

        result = evaluate(
            a_request(
                intent=PreflightIntent.FULL_TRAINING.value,
                canary=CanaryEvidence(
                    status="PASSED", mode="ACE_STEP", detail="the trainer took a step"
                ),
                capacity=capacity_report(
                    cpu_capability(), device=ComputeDevice.CPU.value, free_disk_mb=200_000
                ),
            )
        )
        assert result.status == PreflightStatus.UNVERIFIED.value
        assert result.canary_status == "PASSED"


# ── 70-71. rights are not relaxed for a measurement ──────────────────


class TestProfilingObeysRights:
    def _request(self, tmp_path: Path, dataset_dir: Path):
        from preflight_fixtures import a_plan

        from luber_training.canary import CanaryEnvelope
        from luber_training.memory_profiler import ProbeShape, ProfileRequest

        trainer = tmp_path / "trainer"
        (trainer / "checkpoints").mkdir(parents=True)
        (trainer / "train.py").write_text("", encoding="utf-8")
        interpreter = tmp_path / "python"
        interpreter.write_text("", encoding="utf-8")
        return ProfileRequest(
            plan=a_plan(device=ComputeDevice.MPS.value),
            shape=ProbeShape(latent_length=PRODUCTION_LATENT, samples=2),
            trainer_root=trainer,
            python_executable=interpreter,
            model_dir=trainer / "checkpoints",
            workspace=trainer / "workspace",
            envelope=CanaryEnvelope(max_samples=2),
            dataset_dir=dataset_dir,
        )

    def test_unauthorised_material_cannot_become_profile_input(self, tmp_path: Path):
        from luber_training.memory_profiler import profile_memory

        trainer = tmp_path / "trainer"
        dataset = trainer / "data"
        dataset.mkdir(parents=True)
        (dataset / "a.pt").write_bytes(b"")
        profile = profile_memory(self._request(tmp_path, dataset))
        assert profile.outcome == ProfileOutcome.BLOCKED.value
        assert "being small is not an authorisation" in profile.failure_reason

    def test_a_profile_of_blocked_data_is_never_evidence(self, tmp_path: Path):
        from luber_training.memory_profiler import profile_memory

        trainer = tmp_path / "trainer"
        dataset = trainer / "data"
        dataset.mkdir(parents=True)
        (dataset / "a.pt").write_bytes(b"")
        profile = profile_memory(self._request(tmp_path, dataset))
        verdict, _ = applicability(profile, requested_from(profile.identity))
        assert verdict == Applicability.INCOMPLETE.value


def test_a_timed_out_profile_is_not_a_success():
    profile = a_profile(outcome=ProfileOutcome.PROFILE_TIMEOUT.value)
    assert not profile.completed
    verdict, _ = applicability(profile, requested_from(profile.identity))
    assert verdict == Applicability.INCOMPLETE.value


# ── 51, 73. the real machine, bounded ────────────────────────────────

TRAINER_ROOT = Path.home() / "ace-step-1.5"
TRAINER_PYTHON = TRAINER_ROOT / ".venv" / "bin" / "python"
MODEL_DIR = TRAINER_ROOT / "checkpoints"

needs_trainer = pytest.mark.skipif(
    not (
        TRAINER_ROOT.is_dir()
        and (TRAINER_ROOT / "train.py").is_file()
        and TRAINER_PYTHON.is_file()
        and (MODEL_DIR / "acestep-v15-turbo").is_dir()
    ),
    reason=(
        "no ACE-Step installation with base weights was found at ~/ace-step-1.5. This test "
        "profiles the real trainer and is skipped rather than faked."
    ),
)


@needs_trainer
def test_a_real_bounded_profile_reads_the_runtimes_own_memory_figures(tmp_path: Path):
    """Runs the real trainer, briefly, and checks the numbers are real.

    Deliberately short — 64 latent frames, which the profile itself
    reports as NOT_REPRESENTATIVE — because what this test defends is
    the *instrumentation*: that the stages are observed inside the
    trainer process and that `torch.mps` answered. The representative
    measurement is an operator action, not something a test suite runs.
    """
    import shutil

    from preflight_fixtures import a_plan

    from luber_training.canary import CanaryEnvelope, default_workspace
    from luber_training.config import TrainingConfig
    from luber_training.memory_profiler import ProbeShape, ProfileRequest, profile_memory

    workspace = default_workspace(TRAINER_ROOT, "pytest-memory-profile")
    device = ComputeDevice.MPS.value if os.uname().sysname == "Darwin" else ComputeDevice.CPU.value
    plan = a_plan(
        device=device,
        config=TrainingConfig(epochs=1, rank=4, alpha=8, precision="bf16"),
    )
    shape = ProbeShape(latent_length=64, encoder_length=32, samples=2)
    try:
        profile = profile_memory(
            ProfileRequest(
                plan=plan,
                shape=shape,
                trainer_root=TRAINER_ROOT,
                python_executable=TRAINER_PYTHON,
                model_dir=MODEL_DIR,
                workspace=workspace,
                envelope=CanaryEnvelope(max_samples=2),
                timeout_seconds=900.0,
            )
        )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    assert profile.outcome == ProfileOutcome.COMPLETED.value, profile.failure_reason
    # The stages that only exist inside the trainer process.
    observed = {snapshot.stage for snapshot in profile.snapshots}
    assert {
        ProfileStage.MODEL_LOADED.value,
        ProfileStage.LORA_ATTACHED.value,
        ProfileStage.OPTIMIZER_CREATED.value,
        ProfileStage.FORWARD_COMPLETE.value,
        ProfileStage.BACKWARD_COMPLETE.value,
        ProfileStage.CHECKPOINT_COMPLETE.value,
    } <= observed

    host = profile.peak_for(MemoryDomain.HOST.value)
    assert host is not None and host.peak_bytes and host.peak_bytes > 0
    if device == ComputeDevice.MPS.value:
        apple = profile.peak_for(MemoryDomain.APPLE_UNIFIED.value)
        assert apple is not None
        # A real model was loaded, so the driver allocation is large and
        # the peak is sampled — the pinned torch has no MPS peak counter.
        assert apple.kind == PeakKind.SAMPLED_PEAK.value
        assert apple.peak_bytes and apple.peak_bytes > 1 * GIB
        assert apple.sample_count > 1
    # A short sequence is honest about being short.
    assert profile.representativeness == Representativeness.NOT_REPRESENTATIVE.value
