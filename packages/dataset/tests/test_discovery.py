"""Read-only discovery, path hypotheses, and personal-data handling.

Two classes of bug bit this module during the real Desktop scan and are
pinned here: macOS returns NFD-decomposed filenames so Korean folder
names silently failed to match, and substring matching flagged every
`audio/` directory as AI music because "audio" contains "udio".
"""

import struct
import unicodedata
import wave
from pathlib import Path

import pytest

from luber_dataset.discovery import (
    AUDIO_EXTENSIONS,
    find_adjacent_lyrics,
    hypothesize_origin,
    sanitize,
    scan,
    summarize,
)


def _wav(path: Path, *, seconds: float = 1.0, rate: int = 44100, channels: int = 2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(rate * seconds)
    data = b"".join(struct.pack("<h", (i % 1000) - 500) * channels for i in range(frames))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data)
    return path


# ── Korean path matching (the NFD bug) ────────────────────────────────


@pytest.mark.parametrize(
    ("folder", "expected"),
    [
        ("AI 음원", "AI_GENERATED"),
        ("발매음원", "ORIGINAL_PROJECT"),
        ("제작 음원", "ORIGINAL_PROJECT"),
        ("original", "ORIGINAL_PROJECT"),
    ],
)
def test_korean_folder_hypotheses(folder, expected):
    assert hypothesize_origin(Path(f"/x/{folder}/track.wav"))[0] == expected


@pytest.mark.parametrize("folder", ["기성 음원", "기성음원", "빌보드", "reference"])
def test_commercial_folders_flagged_as_reference(folder):
    _, commercial = hypothesize_origin(Path(f"/x/{folder}/track.flac"))
    assert commercial is True


def test_nfd_decomposed_korean_still_matches():
    """macOS hands back NFD; the literals here are NFC."""
    decomposed = unicodedata.normalize("NFD", "AI 음원")
    assert decomposed != "AI 음원"
    assert hypothesize_origin(Path(f"/x/{decomposed}/t.wav"))[0] == "AI_GENERATED"

    decomposed_commercial = unicodedata.normalize("NFD", "기성 음원")
    assert hypothesize_origin(Path(f"/x/{decomposed_commercial}/t.flac"))[1] is True


# ── the "audio" contains "udio" bug ───────────────────────────────────


def test_audio_directory_is_not_mistaken_for_ai_music():
    assert hypothesize_origin(Path("/x/App.app/Resources/audio/voice/a.wav"))[0] == "UNKNOWN"


def test_real_ai_service_names_still_match():
    assert hypothesize_origin(Path("/x/suno/track.wav"))[0] == "AI_GENERATED"
    assert hypothesize_origin(Path("/x/udio exports/track.wav"))[0] == "AI_GENERATED"


# ── self-model detection wins ─────────────────────────────────────────


@pytest.mark.parametrize(
    "path",
    [
        "/x/luber-music-ai/data/audio/g/master.wav",
        "/x/anything/acestep/out.wav",
        "/x/raw-model-output/a.wav",
        "/x/ab-experiment/V1/a.wav",
    ],
)
def test_self_model_output_detected_anywhere(path):
    assert hypothesize_origin(Path(path))[0] == "SELF_MODEL_OUTPUT"


def test_self_model_beats_a_promising_folder_name():
    """Our own output in a 발매음원 folder is still our own output."""
    assert hypothesize_origin(Path("/x/발매음원/acestep/out.wav"))[0] == "SELF_MODEL_OUTPUT"


# ── scanning is read-only and correct ─────────────────────────────────


def test_scan_finds_audio_and_leaves_files_untouched(tmp_path):
    a = _wav(tmp_path / "AI 음원" / "one.wav")
    b = _wav(tmp_path / "기성 음원" / "two.wav")
    (tmp_path / "notes.txt").write_text("not audio", encoding="utf-8")
    before = {p: p.stat().st_mtime_ns for p in (a, b)}

    found = scan(tmp_path)

    assert {f.filename for f in found} == {"one.wav", "two.wav"}
    assert all(f.sha256 for f in found)
    assert all(f.sample_rate == 44100 and f.channels == 2 for f in found)
    # Nothing was modified.
    assert {p: p.stat().st_mtime_ns for p in (a, b)} == before


def test_scan_skips_app_bundles(tmp_path):
    _wav(tmp_path / "Some.app" / "Resources" / "ui.wav")
    _wav(tmp_path / "music" / "real.wav")
    assert {f.filename for f in scan(tmp_path)} == {"real.wav"}


def test_scan_honours_exclude_roots(tmp_path):
    _wav(tmp_path / "keep" / "a.wav")
    _wav(tmp_path / "drop" / "b.wav")
    found = scan(tmp_path, exclude_roots=(tmp_path / "drop",))
    assert {f.filename for f in found} == {"a.wav"}


def test_scan_detects_duplicates_by_content(tmp_path):
    _wav(tmp_path / "one" / "a.wav")
    _wav(tmp_path / "two" / "b.wav")  # identical content
    found = scan(tmp_path)
    assert len({f.sha256 for f in found}) == 1
    assert all(f.duplicate_group for f in found)


def test_scan_ignores_empty_files(tmp_path):
    (tmp_path / "empty.wav").write_bytes(b"")
    assert scan(tmp_path) == []


def test_every_supported_extension_is_recognised():
    assert AUDIO_EXTENSIONS == {".wav", ".flac", ".aiff", ".aif", ".mp3", ".m4a"}


# ── adjacent lyrics discovery ─────────────────────────────────────────


def test_adjacent_lyrics_found_by_basename(tmp_path):
    audio = _wav(tmp_path / "song.wav")
    (tmp_path / "song.txt").write_text("[Verse]\n가사", encoding="utf-8")
    (tmp_path / "song.lrc").write_text("[00:01.00]가사", encoding="utf-8")
    (tmp_path / "unrelated.txt").write_text("no", encoding="utf-8")

    assert find_adjacent_lyrics(audio) == ["song.lrc", "song.txt"]


def test_no_lyrics_reported_when_absent(tmp_path):
    assert find_adjacent_lyrics(_wav(tmp_path / "song.wav")) == []


# ── personal data handling ────────────────────────────────────────────


def test_sanitize_strips_the_home_prefix():
    assert sanitize("/Users/someone/Desktop/AI 음원/a.wav") == "~/Desktop/AI 음원/a.wav"


def test_sanitize_relativizes_against_a_root():
    assert (
        sanitize("/Users/someone/Desktop/AI 음원/a.wav", root=Path("/Users/someone/Desktop"))
        == "AI 음원/a.wav"
    )


def test_sanitize_output_is_nfc_normalized():
    decomposed = unicodedata.normalize("NFD", "/Users/x/Desktop/기성 음원/a.flac")
    assert sanitize(decomposed) == unicodedata.normalize("NFC", sanitize(decomposed))
    assert "기성 음원" in sanitize(decomposed)


def test_summary_contains_no_absolute_paths(tmp_path):
    _wav(tmp_path / "AI 음원" / "a.wav")
    _wav(tmp_path / "기성 음원" / "b.wav")
    stats = summarize(scan(tmp_path))

    serialized = repr(stats)
    assert "/Users/" not in serialized
    assert str(tmp_path) not in serialized
    assert stats["total_files"] == 2
    assert stats["unique_hashes"] >= 1
    assert "COMMERCIAL_REFERENCE" in stats["by_origin_hypothesis"]


def test_summary_counts_hours_and_extensions(tmp_path):
    _wav(tmp_path / "a.wav", seconds=3600.0, rate=8000)
    stats = summarize(scan(tmp_path))
    assert stats["by_extension"][".wav"] == 1
    assert stats["measured_hours"] == pytest.approx(1.0, abs=0.01)
