"""Scanning, identity, decode validation, and the read-only contract.

The immutability tests are the load-bearing ones. Everything else in the
factory can be re-run; audio that got overwritten during a scan is gone,
and the operator's library is not a thing this project may damage.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from factory_fixtures import (
    requires_ffmpeg,
    tone,
    transcode,
    write_garbage,
    write_truncated_wav,
    write_wav,
)

from luber_dataset.factory.decoder import DecodeStatus, decode_check, probe
from luber_dataset.factory.scanner import (
    SUPPORTED_EXTENSIONS,
    scan,
    verify_unchanged,
)


class TestRecursiveScanning:
    def test_it_finds_audio_at_every_depth(self, library: Path):
        found = {item.source_filename for item in scan(library).files}
        assert found == {"first.wav", "second.wav", "third.wav", "first_copy.wav"}

    def test_the_supported_set_covers_what_a_library_contains(self):
        for extension in (".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg"):
            assert extension in SUPPORTED_EXTENSIONS

    @pytest.mark.parametrize(
        ("name", "reason"),
        [
            (".hidden.wav", "hidden files"),
            (".DS_Store", "OS metadata"),
            ("download.wav.part", "partial downloads"),
            ("notes.txt", "sidecar text"),
            ("cover.jpg", "artwork"),
        ],
    )
    def test_it_ignores_what_is_not_a_track(self, library: Path, name: str, reason: str):
        assert not any(item.source_filename == name for item in scan(library).files), reason

    def test_a_partial_download_is_skipped_with_its_reason(self, library: Path):
        """Skipped and *recorded*: "missed it" and "ignored it" differ."""
        skipped = dict(scan(library).skipped)
        assert any(name.endswith(".part") for name in skipped)
        assert "PARTIAL_OR_TEMPORARY" in skipped.values()

    def test_an_apple_double_sidecar_is_not_mistaken_for_audio(self, tmp_path: Path):
        """`._track.wav` carries the extension and none of the audio."""
        root = tmp_path / "lib"
        write_wav(root / "track.wav", tone())
        (root / "._track.wav").write_bytes(b"resource fork")
        assert len(scan(root).files) == 1

    def test_an_empty_file_is_skipped(self, tmp_path: Path):
        root = tmp_path / "lib"
        root.mkdir()
        (root / "empty.wav").write_bytes(b"")
        result = scan(root)
        assert result.files == []
        assert dict(result.skipped)[str(root / "empty.wav")] == "EMPTY_FILE"

    def test_a_missing_root_is_an_error_not_an_empty_result(self, tmp_path: Path):
        """An empty dataset and a typo in a path must not look alike."""
        with pytest.raises(FileNotFoundError):
            scan(tmp_path / "nope")


class TestContentIdentity:
    def test_the_hash_is_stable_across_scans(self, library: Path):
        first = {item.source_path: item.sha256 for item in scan(library).files}
        second = {item.source_path: item.sha256 for item in scan(library).files}
        assert first == second

    def test_identical_bytes_produce_identical_ids(self, library: Path):
        by_name = {item.source_filename: item for item in scan(library).files}
        assert by_name["first.wav"].sha256 == by_name["first_copy.wav"].sha256
        assert by_name["first.wav"].file_id == by_name["first_copy.wav"].file_id

    def test_identity_survives_a_rename(self, tmp_path: Path):
        """Identity is content, so renaming keeps the cached analysis."""
        root = tmp_path / "lib"
        original = write_wav(root / "before.wav", tone())
        first = scan(root).files[0]
        original.rename(root / "after.wav")
        second = scan(root).files[0]
        assert first.file_id == second.file_id
        assert first.source_filename != second.source_filename

    def test_different_audio_produces_different_ids(self, library: Path):
        by_name = {item.source_filename: item for item in scan(library).files}
        assert by_name["first.wav"].file_id != by_name["second.wav"].file_id


class TestSourceImmutability:
    def test_scanning_does_not_change_a_single_byte(self, library: Path):
        before = {p: p.read_bytes() for p in library.rglob("*") if p.is_file()}
        scan(library)
        after = {p: p.read_bytes() for p in library.rglob("*") if p.is_file()}
        assert before == after

    def test_verify_unchanged_is_silent_when_nothing_moved(self, library: Path):
        assert verify_unchanged(scan(library).files) == []

    def test_verify_unchanged_catches_a_modified_source(self, library: Path):
        """The check has to be able to fail, or it proves nothing."""
        files = scan(library).files
        target = Path(files[0].source_path)
        target.write_bytes(target.read_bytes() + b"tampered")
        assert verify_unchanged(files) == [files[0].source_path]

    def test_verify_unchanged_catches_a_deleted_source(self, library: Path):
        files = scan(library).files
        Path(files[0].source_path).unlink()
        assert files[0].source_path in verify_unchanged(files)


class TestDecodeValidation:
    def test_a_healthy_wav_is_valid(self, tmp_path: Path):
        path = write_wav(tmp_path / "ok.wav", tone())
        result = decode_check(path)
        assert result.status is DecodeStatus.VALID
        assert result.sample_rate == 44_100
        assert result.channels == 2
        assert result.duration_seconds == pytest.approx(22.0, abs=0.1)
        assert result.decode_error is None

    def test_garbage_is_invalid_rather_than_an_exception(self, tmp_path: Path):
        """One corrupt file costs one record, never the run."""
        result = decode_check(write_garbage(tmp_path / "broken.wav"))
        assert result.status is DecodeStatus.INVALID
        assert result.decode_error

    def test_a_truncated_file_is_recorded_not_crashed(self, tmp_path: Path):
        result = decode_check(write_truncated_wav(tmp_path / "cut.wav", tone()))
        assert result.status in (DecodeStatus.PARTIAL, DecodeStatus.INVALID, DecodeStatus.VALID)
        assert isinstance(result.decode_error, (str, type(None)))

    def test_a_missing_file_is_invalid(self, tmp_path: Path):
        assert decode_check(tmp_path / "absent.wav").status is DecodeStatus.INVALID

    def test_partial_still_counts_as_usable_audio(self, tmp_path: Path):
        """Whether to *train* on it is a later, visible decision."""
        from luber_dataset.factory.decoder import DecodeResult

        assert DecodeResult(status=DecodeStatus.PARTIAL).usable
        assert not DecodeResult(status=DecodeStatus.INVALID).usable

    def test_bit_depth_is_absent_for_lossy_sources(self, tmp_path: Path):
        """An MP3 has no bit depth, and inventing one would be a lie."""
        source = write_wav(tmp_path / "source.wav", tone())
        encoded = transcode(
            source, tmp_path / "lossy.mp3", "-codec:a", "libmp3lame", "-b:a", "192k"
        )
        assert probe(encoded).bit_depth is None

    @requires_ffmpeg
    def test_compressed_formats_decode(self, tmp_path: Path):
        source = write_wav(tmp_path / "source.wav", tone())
        for name, args in (
            ("out.flac", ()),
            ("out.mp3", ("-codec:a", "libmp3lame", "-b:a", "192k")),
            ("out.m4a", ("-codec:a", "aac", "-b:a", "192k")),
        ):
            encoded = transcode(source, tmp_path / name, *args)
            result = decode_check(encoded)
            assert result.status is DecodeStatus.VALID, f"{name}: {result.decode_error}"
            assert result.codec

    def test_a_directory_of_mixed_health_scans_completely(self, tmp_path: Path):
        """The whole point: one bad file must not stop the others."""
        root = tmp_path / "mixed"
        write_wav(root / "good_one.wav", tone(seed=1))
        write_garbage(root / "bad.wav")
        write_wav(root / "good_two.wav", tone(seed=2))

        statuses = {
            Path(item.source_path).name: decode_check(Path(item.source_path)).status
            for item in scan(root).files
        }
        assert statuses["good_one.wav"] is DecodeStatus.VALID
        assert statuses["good_two.wav"] is DecodeStatus.VALID
        assert statuses["bad.wav"] is DecodeStatus.INVALID
