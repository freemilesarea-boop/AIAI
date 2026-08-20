"""Synthetic fixtures for the dataset factory.

Everything here is generated. Nothing is downloaded, and no test reads
the operator's music library — a test suite that depends on a particular
machine's files passes or fails for reasons that have nothing to do with
the code.

The signals are built with known properties so a test can assert what
the factory is *supposed* to find rather than whatever it happened to
find: a tone at a stated tempo, a file at a stated sample rate, two
files that are byte-identical, two that are the same audio in different
containers.
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

RATE = 44_100
#: Long enough to clear the 20 s minimum-duration gate, short enough
#: that a full suite run stays quick.
SECONDS = 22.0

FFMPEG = shutil.which("ffmpeg")
requires_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg is not installed")


def tone(
    *,
    seconds: float = SECONDS,
    rate: int = RATE,
    frequency: float = 220.0,
    seed: int = 0,
    amplitude: float = 0.4,
) -> np.ndarray:
    """A stereo signal shaped like music rather than like a test tone.

    An earlier version was a sine plus its fifth. It was rejected by the
    factory for three correct reasons that had nothing to do with what
    the tests were checking: a sine has no energy above its harmonics,
    so it read as SUSPICIOUS_BANDWIDTH; it has a crest factor near 5 dB,
    so it read as LOW_DYNAMIC_RANGE; and two sines an octave apart have
    near-identical band structure, so every pair read as a near
    duplicate. All three findings were right about the signal and wrong
    about the fixture.

    So this is broadband noise on a -4 dB/octave tilt — the spectral
    slope of a dense mix — with a per-seed band emphasis to make tracks
    spectrally distinct, tonal partials for musical content, and a slow
    amplitude envelope so the crest factor lands where real music does.
    """
    length = int(seconds * rate)
    t = np.arange(length) / rate
    rng = np.random.default_rng(seed)

    freqs = np.fft.rfftfreq(length, 1.0 / rate)
    shape = np.zeros_like(freqs)
    shape[freqs > 0] = freqs[freqs > 0] ** (-4.0 / 3.0103 / 2.0)
    # A wide per-seed emphasis, so two fixtures differ in spectral
    # balance rather than only in phase.
    centre = 400.0 * (1.0 + (seed % 7))
    shape *= 1.0 + 1.5 * np.exp(-(((freqs - centre) / (centre * 0.6)) ** 2))

    def channel(component_seed: int) -> np.ndarray:
        noise = np.fft.irfft(
            np.fft.rfft(np.random.default_rng(component_seed).standard_normal(length)) * shape,
            n=length,
        )
        noise /= float(np.abs(noise).max()) or 1.0
        partials = sum(
            np.sin(2 * np.pi * frequency * harmonic * t) / harmonic for harmonic in (1, 2, 3, 5)
        )
        return 0.7 * noise + 0.3 * partials / float(np.abs(partials).max())

    left, right = channel(seed), channel(seed + 977)
    # A slow envelope. Constant-amplitude material has an unrealistically
    # low crest factor and reads as brickwalled.
    envelope = 0.55 + 0.45 * np.abs(np.sin(2 * np.pi * 0.35 * t + rng.random()))
    pair = np.stack([left * envelope, right * envelope], axis=1)
    peak = float(np.abs(pair).max()) or 1.0
    return pair * (amplitude / peak)


def pulse(*, bpm: float, seconds: float = SECONDS, rate: int = RATE, seed: int = 0) -> np.ndarray:
    """A stereo click track at a known tempo, for the tempo estimator."""
    length = int(seconds * rate)
    signal = np.zeros(length)
    period = round(60.0 / bpm * rate)
    rng = np.random.default_rng(seed)
    burst = np.exp(-np.linspace(0, 12, 1024))
    for start in range(0, length - 1024, period):
        signal[start : start + 1024] += burst * rng.standard_normal(1024)
    peak = float(np.abs(signal).max()) or 1.0
    signal = signal * (0.5 / peak)
    return np.stack([signal, signal * 0.98], axis=1)


def write_wav(path: Path, samples: np.ndarray, *, rate: int = RATE, bit_depth: int = 16) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(int(clipped.shape[1]))
        handle.setsampwidth(bit_depth // 8)
        handle.setframerate(rate)
        if bit_depth == 16:
            data = (clipped * 32767.0).astype("<i2").tobytes()
        elif bit_depth == 24:
            packed = (clipped * 8388607.0).astype("<i4").tobytes()
            data = b"".join(packed[i : i + 3] for i in range(0, len(packed), 4))
        else:
            raise ValueError(f"unsupported fixture bit depth: {bit_depth}")
        handle.writeframes(data)
    return path


def write_sidecar(audio: Path, **fields: object) -> Path:
    path = audio.with_suffix(".json")
    path.write_text(json.dumps(fields, ensure_ascii=False), encoding="utf-8")
    return path


def transcode(source: Path, destination: Path, *args: str) -> Path:
    """Re-encode with ffmpeg, for same-audio-different-container tests."""
    if FFMPEG is None:
        pytest.skip("ffmpeg is not installed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [FFMPEG, "-nostdin", "-v", "error", "-y", "-i", str(source), *args, str(destination)],
        check=True,
        capture_output=True,
        timeout=300,
    )
    return destination


def write_truncated_wav(path: Path, samples: np.ndarray, *, keep: float = 0.4) -> Path:
    """A WAV whose header promises more audio than the file contains."""
    complete = path.with_suffix(".complete.wav")
    write_wav(complete, samples)
    raw = complete.read_bytes()
    complete.unlink()
    path.write_bytes(raw[: int(len(raw) * keep)])
    return path


def write_garbage(path: Path, *, size: int = 4096) -> Path:
    """Not audio at all, wearing an audio extension."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<I", 0xDEADBEEF) * (size // 4))
    return path


@pytest.fixture
def library(tmp_path: Path) -> Path:
    """A small tree with one of most things worth testing."""
    root = tmp_path / "library"
    (root / "album_one").mkdir(parents=True)
    (root / "album_two").mkdir(parents=True)

    write_wav(root / "album_one" / "first.wav", tone(frequency=220.0, seed=1))
    write_wav(root / "album_one" / "second.wav", tone(frequency=330.0, seed=2))
    write_wav(root / "album_two" / "third.wav", tone(frequency=440.0, seed=3))

    # Byte-identical copy of `first.wav` under another name.
    shutil.copy(root / "album_one" / "first.wav", root / "album_two" / "first_copy.wav")

    # Things the scanner must ignore.
    (root / ".hidden.wav").write_bytes(b"not scanned")
    (root / "album_one" / ".DS_Store").write_bytes(b"os metadata")
    (root / "album_one" / "notes.txt").write_text("not audio", encoding="utf-8")
    (root / "album_one" / "download.wav.part").write_bytes(b"partial")
    (root / "album_two" / "cover.jpg").write_bytes(b"image")
    return root
