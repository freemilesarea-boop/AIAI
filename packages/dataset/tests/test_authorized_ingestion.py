"""Ingesting an operator-authorised directory, end to end.

These drive the real `scripts/dataset/ingest_pilot.py` over real WAV
files rather than a stand-in, because the things that can go wrong here
are things a stand-in would not have: authorisation escaping the scope
it was granted for, a duplicate counted twice, provenance losing which
folder a track came from, or the source material being touched.
"""

import importlib.util
import json
import math
import struct
import sys
import wave
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "dataset" / "ingest_pilot.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("ingest_pilot_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ingest = _load_script()

SCOPE = "/authorized/root/**"


def _write_wav(path: Path, *, seconds: float = 24.0, seed: int = 1) -> Path:
    """A valid 48 kHz stereo WAV the quality gate accepts.

    Long enough not to be TOO_SHORT and amplitude-modulated so the
    crest factor clears OVER_COMPRESSED. A flat sine would fail on
    dynamics and these tests would then be about the quality gate
    rather than about ingestion.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(48_000 * seconds)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(48_000)
        payload = bytearray()
        for index in range(frames):
            envelope = 0.5 + 0.5 * math.sin(2 * math.pi * 0.5 * index / 48_000) ** 8
            value = int(20_000 * envelope * math.sin(2 * math.pi * (110 * seed) * index / 48_000))
            payload += struct.pack("<hh", value, value)
        handle.writeframes(bytes(payload))
    return path


def _run(tmp_path: Path, root: Path, **extra) -> tuple[int, Path]:
    out = tmp_path / "manifest.json"
    argv = [
        "--input",
        str(root),
        "--recursive",
        "--limit",
        "0",
        "--operator-authorized-scope",
        SCOPE,
        "--operator",
        "the operator",
        "--authorized-on",
        "2026-08-23",
        "--dataset-id",
        "TEST_LIBRARY",
        "--out",
        str(out),
    ]
    for key, value in extra.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    return ingest.main(argv), out


class TestTheAuthorizationIsRecordedTruthfully:
    def test_every_track_carries_the_source_scope_and_date(self, tmp_path, capsys):
        root = tmp_path / "authorized"
        _write_wav(root / "POP" / "one.wav", seed=1)
        _write_wav(root / "POP" / "two.wav", seed=2)
        code, out = _run(tmp_path, root)
        capsys.readouterr()
        assert code == 0

        manifest = json.loads(out.read_text(encoding="utf-8"))
        assert manifest["track_count"] == 2
        for track in manifest["tracks"]:
            rights = track["rights"]
            assert rights["basis"] == "OPERATOR_AUTHORIZED_SCOPE"
            assert rights["authorization_source"] == "OPERATOR_EXPLICIT_AUTHORIZATION"
            assert rights["authorization_scope"] == SCOPE
            assert rights["authorization_recorded_at"] == "2026-08-23"

    def test_it_claims_no_licence_ownership_or_performer_agreement(self, tmp_path, capsys):
        root = tmp_path / "authorized"
        _write_wav(root / "POP" / "one.wav")
        code, out = _run(tmp_path, root)
        capsys.readouterr()
        assert code == 0

        rights = json.loads(out.read_text(encoding="utf-8"))["tracks"][0]["rights"]
        assert rights["lyrics_rights_confirmed"] is False
        assert rights["performer_rights_confirmed"] is False
        assert rights["origin_type"] == "UNKNOWN"
        assert "none is claimed here" in rights["notes"]

    def test_an_unannotated_track_has_no_invented_language(self, tmp_path, capsys):
        root = tmp_path / "authorized"
        _write_wav(root / "POP" / "one.wav")
        code, out = _run(tmp_path, root)
        capsys.readouterr()
        assert code == 0
        assert json.loads(out.read_text(encoding="utf-8"))["tracks"][0]["language"] == "unknown"


class TestScope:
    def test_material_outside_the_named_root_is_never_ingested(self, tmp_path, capsys):
        """Authorisation covers a directory, not a machine."""
        root = tmp_path / "authorized"
        _write_wav(root / "POP" / "inside.wav", seed=1)
        _write_wav(tmp_path / "elsewhere" / "outside.wav", seed=2)

        code, out = _run(tmp_path, root)
        capsys.readouterr()
        assert code == 0
        manifest = json.loads(out.read_text(encoding="utf-8"))
        assert manifest["track_count"] == 1

        staged = json.loads(out.with_suffix(".paths.json").read_text(encoding="utf-8"))
        for path in staged["tracks"].values():
            assert Path(path).is_relative_to(root)

    def test_without_an_authorized_scope_an_unannotated_track_is_excluded(self, tmp_path, capsys):
        root = tmp_path / "unauthorized"
        _write_wav(root / "one.wav")
        code = ingest.main(
            ["--input", str(root), "--limit", "0", "--out", str(tmp_path / "m.json")]
        )
        assert code == 1
        assert "no annotation/rights JSON" in capsys.readouterr().out


class TestIdentity:
    def test_a_byte_identical_duplicate_is_ingested_once(self, tmp_path, capsys):
        root = tmp_path / "authorized"
        first = _write_wav(root / "POP" / "song.wav", seed=3)
        copy = root / "Lofi" / "song-again.wav"
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(first.read_bytes())

        code, out = _run(tmp_path, root)
        printed = capsys.readouterr().out
        assert code == 0
        assert json.loads(out.read_text(encoding="utf-8"))["track_count"] == 1
        assert "byte-identical" in printed

    def test_track_ids_come_from_content_not_filenames(self, tmp_path, capsys):
        root = tmp_path / "authorized"
        audio = _write_wav(root / "POP" / "a name with spaces.wav", seed=4)
        code, out = _run(tmp_path, root)
        capsys.readouterr()
        assert code == 0

        track = json.loads(out.read_text(encoding="utf-8"))["tracks"][0]
        assert track["track_id"] == track["audio_sha256"][:16]
        assert "name" not in track["track_id"]
        assert audio.name not in json.dumps(track)

    def test_the_manifest_digest_is_the_same_on_a_second_run(self, tmp_path, capsys):
        root = tmp_path / "authorized"
        _write_wav(root / "POP" / "one.wav", seed=1)
        _write_wav(root / "Lofi" / "two.wav", seed=2)

        _run(tmp_path, root)
        first = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        _run(tmp_path, root)
        second = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        capsys.readouterr()
        assert first["content_hash"] == second["content_hash"]


class TestProvenance:
    def test_each_track_records_the_group_it_came_from(self, tmp_path, capsys):
        root = tmp_path / "authorized"
        _write_wav(root / "POP" / "one.wav", seed=1)
        _write_wav(root / "Lofi" / "two.wav", seed=2)

        code, out = _run(tmp_path, root)
        capsys.readouterr()
        assert code == 0
        groups = {t["source_group"] for t in json.loads(out.read_text(encoding="utf-8"))["tracks"]}
        assert groups == {"POP", "Lofi"}

    def test_the_source_path_map_is_marked_local_and_kept_out_of_the_manifest(
        self, tmp_path, capsys
    ):
        root = tmp_path / "authorized"
        _write_wav(root / "POP" / "one.wav")
        code, out = _run(tmp_path, root)
        capsys.readouterr()
        assert code == 0

        assert str(root) not in out.read_text(encoding="utf-8")
        side = json.loads(out.with_suffix(".paths.json").read_text(encoding="utf-8"))
        assert "never commit" in side["note"]


class TestTheSourceIsNotTouched:
    def test_ingestion_leaves_every_source_file_exactly_as_it_found_it(self, tmp_path, capsys):
        root = tmp_path / "authorized"
        paths = [
            _write_wav(root / "POP" / "one.wav", seed=1),
            _write_wav(root / "Lofi" / "two.wav", seed=2),
        ]
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}

        code, _ = _run(tmp_path, root)
        capsys.readouterr()
        assert code == 0

        for path, (payload, mtime) in before.items():
            assert path.read_bytes() == payload
            assert path.stat().st_mtime_ns == mtime
        assert sorted(p.name for p in root.rglob("*") if p.is_file()) == [
            "one.wav",
            "two.wav",
        ]


class TestFormat:
    @pytest.mark.parametrize("suffix", [".flac", ".mp3", ".m4a"])
    def test_only_the_requested_extensions_are_ingested(self, tmp_path, capsys, suffix):
        """A WAV pilot does not quietly acquire whatever else is lying about."""
        root = tmp_path / "authorized"
        _write_wav(root / "POP" / "one.wav")
        (root / "POP" / f"other{suffix}").write_bytes(b"not audio")

        code, out = _run(tmp_path, root)
        capsys.readouterr()
        assert code == 0
        assert json.loads(out.read_text(encoding="utf-8"))["track_count"] == 1
