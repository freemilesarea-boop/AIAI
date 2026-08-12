"""Delivery-format normalization tests (real ffmpeg, real files).

These run the actual encoder rather than mocking it — the point of this
layer is that the bytes on disk have the required format, which a mock
cannot demonstrate.
"""

import shutil
import struct
import wave
from pathlib import Path

import pytest

from luber_audio_utils import (
    MASTER_BIT_DEPTH,
    MASTER_CHANNELS,
    MASTER_SAMPLE_RATE,
    PREVIEW_BITRATE_BPS,
    PREVIEW_CHANNELS,
    PREVIEW_SAMPLE_RATE,
    AudioProcessingError,
    encode_preview_mp3,
    probe_audio,
    transcode_master_wav,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe required for audio post-processing tests",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mock_generation.wav"


def _write_wav(
    path: Path, *, sample_rate: int, channels: int, sample_width: int, seconds: float = 1.0
) -> Path:
    """Write a short audible tone in an arbitrary source format."""
    frames = int(sample_rate * seconds)
    max_amplitude = 2 ** (sample_width * 8 - 1) - 1
    data = bytearray()
    for i in range(frames):
        # Simple ramp: deterministic and clearly non-silent.
        value = int(max_amplitude * 0.5 * ((i % 100) / 100 - 0.5) * 2)
        sample = value.to_bytes(sample_width, "little", signed=True)
        data += sample * channels
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sample_width)
        w.setframerate(sample_rate)
        w.writeframes(bytes(data))
    return path


# ── 1. raw audio → 48 kHz / stereo / 24-bit WAV ───────────────────────


@pytest.mark.parametrize(
    ("sample_rate", "channels", "sample_width"),
    [
        (44100, 2, 2),  # CD-rate stereo 16-bit
        (48000, 1, 2),  # mono
        (22050, 1, 1),  # low-rate 8-bit mono
        (48000, 2, 2),  # already correct rate, wrong depth
    ],
)
def test_transcode_normalizes_any_input_to_master_format(
    tmp_path, sample_rate, channels, sample_width
):
    source = _write_wav(
        tmp_path / "src.wav",
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
    )
    out = tmp_path / "master.wav"

    probe = transcode_master_wav(source, out)

    assert probe.sample_rate == MASTER_SAMPLE_RATE == 48000
    assert probe.channels == MASTER_CHANNELS == 2
    assert probe.bit_depth == MASTER_BIT_DEPTH == 24
    assert probe.codec_name == "pcm_s24le"
    assert probe.duration_seconds > 0
    assert out.stat().st_size > 0


def test_master_is_readable_as_24_bit_wav(tmp_path):
    out = tmp_path / "master.wav"
    transcode_master_wav(FIXTURE, out)

    with wave.open(str(out), "rb") as w:
        assert w.getsampwidth() == 3  # 24-bit
        assert w.getnchannels() == 2
        assert w.getframerate() == 48000
        assert w.getnframes() > 0


def test_master_audio_is_not_silent(tmp_path):
    """Format conversion must preserve the signal, not blank it."""
    out = tmp_path / "master.wav"
    transcode_master_wav(FIXTURE, out)
    with wave.open(str(out), "rb") as w:
        raw = w.readframes(min(w.getnframes(), 48000))
    peak = max(
        abs(int.from_bytes(raw[i : i + 3], "little", signed=True))
        for i in range(0, len(raw) - 2, 3)
    )
    assert peak > 100


def test_transcode_is_deterministic(tmp_path):
    """Same input must produce byte-identical output."""
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    transcode_master_wav(FIXTURE, a)
    transcode_master_wav(FIXTURE, b)
    assert a.read_bytes() == b.read_bytes()


# ── 3/4. MP3 preview ──────────────────────────────────────────────────


def test_encode_preview_mp3_meets_contract(tmp_path):
    master = tmp_path / "master.wav"
    transcode_master_wav(FIXTURE, master)
    preview = tmp_path / "preview.mp3"

    probe = encode_preview_mp3(master, preview)

    assert probe.codec_name == "mp3"
    assert probe.sample_rate == PREVIEW_SAMPLE_RATE == 48000
    assert probe.channels == PREVIEW_CHANNELS == 2
    assert probe.bitrate_bps == PREVIEW_BITRATE_BPS == 320000
    assert probe.duration_seconds > 0
    assert preview.stat().st_size > 0


def test_preview_starts_with_mp3_frame_sync(tmp_path):
    master = tmp_path / "master.wav"
    transcode_master_wav(FIXTURE, master)
    preview = tmp_path / "preview.mp3"
    encode_preview_mp3(master, preview)

    head = preview.read_bytes()[:2]
    # MPEG audio frame sync (0xFFEx) or an ID3 tag.
    assert head[0] == 0xFF or preview.read_bytes()[:3] == b"ID3"


def test_preview_duration_matches_master(tmp_path):
    master = tmp_path / "master.wav"
    master_probe = transcode_master_wav(FIXTURE, master)
    preview = tmp_path / "preview.mp3"
    preview_probe = encode_preview_mp3(master, preview)

    assert abs(preview_probe.duration_seconds - master_probe.duration_seconds) < 0.1


# ── probing / failure handling ────────────────────────────────────────


def test_probe_reports_source_properties():
    probe = probe_audio(FIXTURE)
    assert probe.sample_rate > 0
    assert probe.channels > 0
    assert probe.duration_seconds > 0
    assert probe.file_size == FIXTURE.stat().st_size


def test_probe_missing_file_raises(tmp_path):
    with pytest.raises(AudioProcessingError, match="does not exist"):
        probe_audio(tmp_path / "ghost.wav")


def test_transcode_missing_source_raises(tmp_path):
    with pytest.raises(AudioProcessingError, match="does not exist"):
        transcode_master_wav(tmp_path / "ghost.wav", tmp_path / "out.wav")


def test_transcode_rejects_non_audio_input(tmp_path):
    junk = tmp_path / "not-audio.wav"
    junk.write_bytes(b"this is definitely not audio" * 10)
    with pytest.raises(AudioProcessingError):
        transcode_master_wav(junk, tmp_path / "out.wav")


def test_encode_preview_rejects_non_audio_input(tmp_path):
    junk = tmp_path / "junk.wav"
    junk.write_bytes(struct.pack("<4s", b"RIFF") + b"\x00" * 64)
    with pytest.raises(AudioProcessingError):
        encode_preview_mp3(junk, tmp_path / "out.mp3")
