"""The factory end to end: determinism, cache, freeze, export, isolation.

These run the real pipeline over a real (synthetic) library. Two
properties are asserted repeatedly because everything else depends on
them: **source audio is unchanged afterwards**, and **two runs over
identical inputs produce an identical manifest**.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from conftest import tone, write_garbage, write_sidecar, write_wav

from luber_dataset.factory import manifest as manifest_io
from luber_dataset.factory.config import FactoryConfig, QualityThresholds
from luber_dataset.factory.export import ExportPolicy, export
from luber_dataset.factory.pipeline import FileAnalysis, analyse_file, run, worker_count
from luber_dataset.factory.scanner import ScannedFile
from luber_dataset.factory.schemas import TrackRecord, manifest_digest


def digests(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def single_worker(**overrides) -> FactoryConfig:
    """One worker keeps the tests quick; determinism is asserted separately."""
    return FactoryConfig(**overrides).with_overrides(workers=1)


@pytest.fixture
def owned_library(tmp_path: Path) -> Path:
    """A library the operator has declared as their own."""
    root = tmp_path / "owned"
    (root / "album").mkdir(parents=True)
    for index, frequency in enumerate((220.0, 275.0, 330.0, 415.0), start=1):
        audio = write_wav(
            root / "album" / f"track{index}.wav", tone(frequency=frequency, seed=index)
        )
        write_sidecar(
            audio,
            title=f"Track {index}",
            artist="Operator",
            album=f"Record {index}",
            rights_status="USER_OWNED",
            commercial_training_allowed="true",
            language="en",
            vocal_type="instrumental",
        )
    return root


class TestEndToEnd:
    def test_a_library_produces_canonical_records(self, library: Path, tmp_path: Path):
        result = run(library, tmp_path / "build", single_worker())
        # Four files, two of which are byte-identical, so three tracks.
        assert len(result.records) == 3
        assert len(result.duplicates) == 1

    def test_source_audio_is_unchanged_afterwards(self, library: Path, tmp_path: Path):
        """The contract. Asserted by re-hashing, not by trusting."""
        before = digests(library)
        result = run(library, tmp_path / "build", single_worker())
        assert digests(library) == before
        assert result.source_integrity_ok
        assert result.changed_sources == []

    def test_nothing_is_written_inside_the_source_tree(self, library: Path, tmp_path: Path):
        before = set(library.rglob("*"))
        run(library, tmp_path / "build", single_worker())
        assert set(library.rglob("*")) == before

    def test_a_corrupt_file_does_not_stop_the_run(self, library: Path, tmp_path: Path):
        write_garbage(library / "album_one" / "broken.wav")
        result = run(library, tmp_path / "build", single_worker())
        assert len(result.records) == 4, "the healthy tracks still came through"
        statuses = {r.track_id: (r.audio or {}).get("status") for r in result.records}
        assert "INVALID" in statuses.values()

    def test_every_rejection_has_a_reason(self, library: Path, tmp_path: Path):
        result = run(library, tmp_path / "build", single_worker())
        assert result.rejections
        for rejection in result.rejections:
            assert rejection.reasons, rejection.track_id

    def test_unknown_rights_keep_everything_out_of_training(self, library: Path, tmp_path: Path):
        """No sidecars anywhere in this library, so nothing is eligible."""
        result = run(library, tmp_path / "build", single_worker())
        assert all(not r.eligibility["training_eligible"] for r in result.records)
        assert all("RIGHTS_UNKNOWN" in r.eligibility["eligibility_reasons"] for r in result.records)

    def test_declared_rights_admit_tracks(self, owned_library: Path, tmp_path: Path):
        result = run(owned_library, tmp_path / "build", single_worker())
        assert any(r.eligibility["training_eligible"] for r in result.records)

    def test_splits_only_contain_eligible_tracks(self, owned_library: Path, tmp_path: Path):
        result = run(owned_library, tmp_path / "build", single_worker())
        for record in result.records:
            if not record.eligibility["training_eligible"]:
                assert record.split == "EXCLUDED"

    def test_no_split_leakage(self, owned_library: Path, tmp_path: Path):
        assert run(owned_library, tmp_path / "build", single_worker()).leaked_groups == []

    def test_the_summary_accounts_for_everything(self, library: Path, tmp_path: Path):
        summary = run(library, tmp_path / "build", single_worker()).summary
        for key in (
            "total_files",
            "canonical_tracks",
            "valid_audio",
            "invalid_audio",
            "exact_duplicates",
            "quality_A",
            "rejected",
            "training_eligible",
            "rights_unknown",
            "duration_total_hours",
            "sample_rate_distribution",
            "channel_distribution",
            "duration_distribution",
            "quality_flag_distribution",
            "language_distribution",
        ):
            assert key in summary, key
        assert summary["total_files"] == summary["canonical_tracks"] + len(
            run(library, tmp_path / "build2", single_worker()).duplicates
        )


class TestDeterminism:
    def test_two_runs_produce_an_identical_manifest(self, owned_library: Path, tmp_path: Path):
        """The property the whole design is arranged around."""
        first = run(owned_library, tmp_path / "a", single_worker())
        second = run(owned_library, tmp_path / "b", single_worker())
        assert manifest_digest(first.records) == manifest_digest(second.records)

    def test_the_digest_ignores_timestamps(self, owned_library: Path, tmp_path: Path):
        """Otherwise every run differs and the digest answers nothing."""
        first = run(owned_library, tmp_path / "a", single_worker())
        for record in first.records:
            record.source["source_mtime"] = 999.0
        second = run(owned_library, tmp_path / "b", single_worker())
        assert manifest_digest(first.records) == manifest_digest(second.records)

    def test_parallel_and_serial_agree(self, owned_library: Path, tmp_path: Path):
        """Workers finish in any order; the manifest must not notice."""
        serial = run(owned_library, tmp_path / "a", FactoryConfig().with_overrides(workers=1))
        parallel = run(owned_library, tmp_path / "b", FactoryConfig().with_overrides(workers=4))
        assert manifest_digest(serial.records) == manifest_digest(parallel.records)

    def test_the_written_file_is_byte_identical_across_runs(
        self, owned_library: Path, tmp_path: Path
    ):
        first = run(owned_library, tmp_path / "a", single_worker())
        second = run(owned_library, tmp_path / "b", single_worker())
        one = manifest_io.write_manifest(tmp_path / "a", first.records)
        two = manifest_io.write_manifest(tmp_path / "b", second.records)
        assert one.read_bytes() == two.read_bytes()


class TestCacheAndResume:
    def test_the_second_run_hits_the_cache(self, owned_library: Path, tmp_path: Path):
        build = tmp_path / "build"
        first = run(owned_library, build, single_worker())
        second = run(owned_library, build, single_worker())
        assert first.cache_stats["hits"] == 0
        assert second.cache_stats["hits"] > 0, second.cache_stats

    def test_a_cached_run_produces_the_same_manifest(self, owned_library: Path, tmp_path: Path):
        """A cache that changed the answer would be worse than no cache."""
        build = tmp_path / "build"
        first = run(owned_library, build, single_worker())
        second = run(owned_library, build, single_worker())
        assert manifest_digest(first.records) == manifest_digest(second.records)

    def test_changing_the_configuration_invalidates_the_cache(
        self, owned_library: Path, tmp_path: Path
    ):
        build = tmp_path / "build"
        run(owned_library, build, single_worker())
        changed = single_worker(quality=QualityThresholds(min_duration_seconds=45.0))
        second = run(owned_library, build, changed)
        assert second.cache_stats["hits"] == 0, "a threshold change must not reuse results"

    def test_force_reanalyze_ignores_the_cache(self, owned_library: Path, tmp_path: Path):
        build = tmp_path / "build"
        run(owned_library, build, single_worker())
        second = run(owned_library, build, single_worker(), force_reanalyze=True)
        assert second.cache_stats["hits"] == 0

    def test_a_corrupt_cache_is_survived(self, owned_library: Path, tmp_path: Path):
        """A damaged cache is a performance problem, never a wrong answer."""
        build = tmp_path / "build"
        run(owned_library, build, single_worker())
        (build / "cache" / "analysis_cache.json").write_text("{ truncated", encoding="utf-8")
        result = run(owned_library, build, single_worker())
        assert len(result.records) == 4

    def test_a_removed_file_is_pruned_from_the_cache(self, owned_library: Path, tmp_path: Path):
        build = tmp_path / "build"
        run(owned_library, build, single_worker())
        (owned_library / "album" / "track1.wav").unlink()
        (owned_library / "album" / "track1.json").unlink()
        result = run(owned_library, build, single_worker())
        assert len(result.records) == 3


class TestWorkerIsolation:
    def test_a_worker_never_raises(self, tmp_path: Path):
        """Whatever happens to one file, the caller gets a record."""
        broken = ScannedFile(
            source_path=str(tmp_path / "missing.wav"),
            source_filename="missing.wav",
            source_extension=".wav",
            source_size_bytes=10,
            source_mtime=0.0,
            sha256="0" * 64,
        )
        result = analyse_file(broken)
        assert isinstance(result, FileAnalysis)
        assert result.decode.get("status") == "INVALID"

    def test_one_failure_does_not_lose_the_others(self, library: Path, tmp_path: Path):
        write_garbage(library / "album_two" / "junk.wav")
        result = run(library, tmp_path / "build", FactoryConfig().with_overrides(workers=2))
        assert len(result.records) == 4

    def test_the_worker_count_is_bounded_and_safe(self):
        assert worker_count(4) == 4
        assert worker_count(0) >= 1


class TestManifestAndFreeze:
    def build(self, library: Path, output: Path):
        result = run(library, output, single_worker())
        manifest_io.write_manifest(output, result.records)
        manifest_io.write_rejections(output, result.rejections)
        manifest_io.write_duplicates(output, result.duplicates)
        manifest_io.write_review_queue(output, result.review_queue)
        manifest_io.write_summary(output, result.summary)
        return result

    def test_every_output_file_is_written(self, owned_library: Path, tmp_path: Path):
        output = tmp_path / "build"
        self.build(owned_library, output)
        for name in (
            manifest_io.MANIFEST_NAME,
            manifest_io.SUMMARY_NAME,
            manifest_io.REJECTIONS_NAME,
            manifest_io.DUPLICATES_NAME,
            manifest_io.REVIEW_NAME,
        ):
            assert (output / name).is_file(), name

    def test_the_manifest_is_versioned(self, owned_library: Path, tmp_path: Path):
        output = tmp_path / "build"
        self.build(owned_library, output)
        first = json.loads((output / manifest_io.MANIFEST_NAME).read_text().splitlines()[0])
        assert first["schema_version"] == "luber-dataset-factory/1"

    def test_every_section_is_present(self, owned_library: Path, tmp_path: Path):
        output = tmp_path / "build"
        self.build(owned_library, output)
        record = json.loads((output / manifest_io.MANIFEST_NAME).read_text().splitlines()[0])
        for section in (
            "source",
            "audio",
            "analysis",
            "music",
            "vocals",
            "text",
            "quality",
            "provenance",
            "dedup",
            "eligibility",
            "split",
        ):
            assert section in record, section

    def test_a_manifest_round_trips(self, owned_library: Path, tmp_path: Path):
        output = tmp_path / "build"
        result = self.build(owned_library, output)
        reloaded = manifest_io.read_manifest(output / manifest_io.MANIFEST_NAME)
        assert manifest_digest(reloaded) == manifest_digest(result.records)

    def test_the_review_queue_names_what_needs_deciding(self, library: Path, tmp_path: Path):
        output = tmp_path / "build"
        result = self.build(library, output)
        reasons = {item.reason for item in result.review_queue}
        assert "RIGHTS_UNKNOWN" in reasons
        for item in result.review_queue:
            assert item.recommended_action
            assert item.source_path

    def test_freezing_records_the_dataset_identity(self, owned_library: Path, tmp_path: Path):
        output = tmp_path / "build"
        result = self.build(owned_library, output)
        lock = manifest_io.freeze(output, result.records, single_worker(), dataset_id="ds-1")
        assert lock.track_count == len(result.records)
        assert lock.manifest_sha256
        assert lock.source_identity_digest
        assert (output / manifest_io.LOCK_NAME).is_file()

    def test_a_frozen_dataset_verifies_against_itself(self, owned_library: Path, tmp_path: Path):
        output = tmp_path / "build"
        result = self.build(owned_library, output)
        manifest_io.freeze(output, result.records, single_worker(), dataset_id="ds-1")
        assert manifest_io.verify_lock(output / manifest_io.LOCK_NAME, result.records) == []

    def test_verification_notices_a_changed_dataset(self, owned_library: Path, tmp_path: Path):
        """A lock that cannot fail proves nothing."""
        output = tmp_path / "build"
        result = self.build(owned_library, output)
        manifest_io.freeze(output, result.records, single_worker(), dataset_id="ds-1")
        problems = manifest_io.verify_lock(output / manifest_io.LOCK_NAME, result.records[:-1])
        assert problems

    def test_the_lock_is_stable_across_rebuilds(self, owned_library: Path, tmp_path: Path):
        """Same audio, same policy, same identity — different timestamps."""
        first = self.build(owned_library, tmp_path / "a")
        second = self.build(owned_library, tmp_path / "b")
        lock_a = manifest_io.freeze(tmp_path / "a", first.records, single_worker(), dataset_id="d")
        lock_b = manifest_io.freeze(tmp_path / "b", second.records, single_worker(), dataset_id="d")
        assert lock_a.manifest_sha256 == lock_b.manifest_sha256
        assert lock_a.source_identity_digest == lock_b.source_identity_digest


class TestTrainingExport:
    def records(self, **overrides) -> list[TrackRecord]:
        base = TrackRecord(
            track_id="trk_ok",
            source={"source_path": "/lib/a.wav", "sha256": "a" * 64},
            analysis={"duration_seconds": 200.0, "sample_rate": 44_100, "channels": 2},
            quality={"quality_tier": "A"},
            provenance={"commercial_training_allowed": "TRUE", "hard_blocks": []},
            dedup={"dedup_decision": "KEEP"},
            eligibility={"analysis_eligible": True, "training_eligible": True},
            split="TRAIN",
        )
        for key, value in overrides.items():
            setattr(base, key, value)
        return [base]

    def test_an_eligible_track_is_exported(self, tmp_path: Path):
        result = export(self.records(), tmp_path / "export")
        assert result.counts["TRAIN"] == 1

    @pytest.mark.parametrize(
        ("field", "value", "reason"),
        [
            (
                "provenance",
                {"commercial_training_allowed": "UNKNOWN", "hard_blocks": []},
                "RIGHTS_UNKNOWN",
            ),
            (
                "provenance",
                {"commercial_training_allowed": "FALSE", "hard_blocks": []},
                "RIGHTS_DENIED",
            ),
            (
                "provenance",
                {"commercial_training_allowed": "TRUE", "hard_blocks": ["SELF_MODEL_OUTPUT"]},
                "RIGHTS_HARD_BLOCK",
            ),
            ("quality", {"quality_tier": "REJECT"}, "QUALITY_REJECTED"),
            ("dedup", {"dedup_decision": "REVIEW_REQUIRED"}, "NEAR_DUPLICATE_REVIEW_REQUIRED"),
            ("dedup", {"dedup_decision": "MERGED"}, "DUPLICATE_OF_ANOTHER_TRACK"),
        ],
    )
    def test_the_defaults_exclude_what_they_should(
        self, tmp_path: Path, field: str, value: dict, reason: str
    ):
        result = export(self.records(**{field: value}), tmp_path / "export")
        assert result.counts["TRAIN"] == 0
        assert reason in result.excluded

    def test_rights_unknown_can_be_admitted_deliberately(self, tmp_path: Path):
        records = self.records(
            provenance={"commercial_training_allowed": "UNKNOWN", "hard_blocks": []}
        )
        result = export(records, tmp_path / "export", ExportPolicy(allow_rights_unknown=True))
        assert result.counts["TRAIN"] == 1

    def test_no_override_admits_hard_blocked_audio(self, tmp_path: Path):
        """An override that could clear this would make it decorative."""
        records = self.records(
            provenance={
                "commercial_training_allowed": "TRUE",
                "hard_blocks": ["SELF_MODEL_OUTPUT"],
            }
        )
        permissive = ExportPolicy(
            allow_rights_unknown=True,
            allow_review_required=True,
            allow_quality_reject=True,
            min_tier="C",
        )
        result = export(records, tmp_path / "export", permissive)
        assert result.counts["TRAIN"] == 0
        assert "RIGHTS_HARD_BLOCK" in result.excluded

    def test_all_three_files_are_written(self, tmp_path: Path):
        export(self.records(), tmp_path / "export")
        for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
            assert (tmp_path / "export" / name).is_file()

    def test_the_exporter_does_not_rescan_anything(self, owned_library: Path, tmp_path: Path):
        """It consumes records; the source tree is not touched at all."""
        output = tmp_path / "build"
        result = run(owned_library, output, single_worker())
        manifest_io.write_manifest(output, result.records)
        before = digests(owned_library)
        export(
            manifest_io.read_manifest(output / manifest_io.MANIFEST_NAME),
            tmp_path / "export",
            ExportPolicy(),
        )
        assert digests(owned_library) == before

    def test_a_row_carries_what_a_trainer_needs(self, tmp_path: Path):
        export(self.records(), tmp_path / "export")
        row = json.loads((tmp_path / "export" / "train.jsonl").read_text().splitlines()[0])
        for key in ("track_id", "audio_path", "sha256", "duration_seconds", "split"):
            assert key in row
        assert "eligibility" not in row, "a trainer must not re-decide eligibility"
