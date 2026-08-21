"""Audio built to be exactly one thing wrong, and audio built to be fine.

Generated at test time, never committed. A repository that accumulated a
WAV per failure mode would be carrying tens of megabytes of binary that
nobody can review in a diff, and the generator is both smaller and more
honest — the parameters that make a file "near-silent" are visible here
rather than baked into bytes.

Determinism matters as much as content. Every signal comes from a fixed
linear congruential sequence rather than `random`, so a threshold that
holds today holds tomorrow and a failure is reproducible rather than a
coin flip somebody reruns until it passes.

The healthy signal is the important one. Most of these tests are about
*not* rejecting things, so `healthy()` has to be something a real
generation could plausibly be: harmonically rich, stereo but correlated,
with a normal peak and no long silences. If it were a sine tone every
"does not reject" assertion would be vacuous.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

SAMPLE_RATE = 44_100
CHANNELS = 2
SAMPLE_WIDTH = 2

_LCG_MODULUS = 2**31
_LCG_MULTIPLIER = 1103515245
_LCG_INCREMENT = 12345


def _noise(seed: int = 20260821):
    """A deterministic sequence in [-1, 1)."""
    state = seed
    while True:
        state = (_LCG_MULTIPLIER * state + _LCG_INCREMENT) % _LCG_MODULUS
        yield (state / (_LCG_MODULUS / 2)) - 1.0


#: A chord's worth of partials. Enough spectral spread that a rolloff
#: measurement lands where real music does rather than at 500 Hz.
_PARTIALS = (
    (110.0, 0.9),
    (220.0, 0.7),
    (330.0, 0.45),
    (440.0, 0.35),
    (660.0, 0.22),
    (880.0, 0.16),
    (1320.0, 0.10),
    (1760.0, 0.07),
    (2640.0, 0.05),
    (3520.0, 0.035),
    (5280.0, 0.02),
    (7040.0, 0.015),
    (10560.0, 0.01),
)
_PARTIAL_NORM = sum(weight for _, weight in _PARTIALS)


def write_wav(
    path: Path,
    frames: list[tuple[float, float]],
    *,
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS,
) -> Path:
    """Write float frames in [-1, 1] as 16-bit PCM."""
    payload = bytearray()
    for frame in frames:
        for index in range(channels):
            value = frame[index] if index < len(frame) else frame[0]
            clamped = max(-1.0, min(1.0, value))
            payload += struct.pack("<h", int(clamped * 32767))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(payload))
    return path


def _tone_frames(
    seconds: float,
    *,
    amplitude: float = 0.5,
    noise_level: float = 0.02,
    partials: tuple[tuple[float, float], ...] = _PARTIALS,
    tilt: float = 0.0,
    seed: int = 20260821,
) -> list[tuple[float, float]]:
    """Broadband, harmonically plausible audio.

    ``tilt`` shapes the spectrum: positive raises the high partials,
    negative lowers them. It is how the dark and bright fixtures are
    made from the same generator rather than from two hand-tuned ones.
    """
    total = int(seconds * SAMPLE_RATE)
    noise = _noise(seed)
    frames: list[tuple[float, float]] = []
    for index in range(total):
        # A gentle fade at each end, so nothing here is a click.
        fade = min(1.0, index / 2205, max(0.0, (total - index) / 2205))
        value = 0.0
        for order, (frequency, weight) in enumerate(partials):
            shaped = weight * (1.0 + tilt) ** order
            value += shaped * math.sin(2 * math.pi * frequency * index / SAMPLE_RATE)
        value = amplitude * fade * value / _PARTIAL_NORM
        left = value + noise_level * next(noise)
        right = value + noise_level * next(noise)
        frames.append((left, right))
    return frames


# ── the healthy case ─────────────────────────────────────────────────


def healthy(path: Path, seconds: float = 12.0) -> Path:
    """Audio a real generation could plausibly be. Nothing wrong with it."""
    return write_wav(path, _tone_frames(seconds))


# ── one thing wrong, each ────────────────────────────────────────────


def silent(path: Path, seconds: float = 12.0) -> Path:
    """Digital silence. Not quiet — nothing."""
    return write_wav(path, [(0.0, 0.0)] * int(seconds * SAMPLE_RATE))


def near_silent(path: Path, seconds: float = 12.0) -> Path:
    """Audible only as noise: a peak around -40 dBFS."""
    return write_wav(path, _tone_frames(seconds, amplitude=0.01, noise_level=0.0005))


def early_collapse(path: Path, seconds: float = 40.0, content_seconds: float = 12.0) -> Path:
    """Content that stops, followed by a long digital silence.

    The failure a whole-file silence ratio cannot distinguish from a
    quiet outro, which is why the collapse detector is positional.
    """
    frames = _tone_frames(content_seconds)
    padding = int((seconds - content_seconds) * SAMPLE_RATE)
    return write_wav(path, frames + [(0.0, 0.0)] * padding)


def quiet_outro(path: Path, seconds: float = 20.0) -> Path:
    """A long fade to near-nothing. Musical, and must not be rejected."""
    total = int(seconds * SAMPLE_RATE)
    base = _tone_frames(seconds)
    faded: list[tuple[float, float]] = []
    fade_start = int(total * 0.7)
    for index, (left, right) in enumerate(base):
        gain = (
            1.0
            if index < fade_start
            else max(0.0, 1.0 - (index - fade_start) / (total - fade_start))
        )
        faded.append((left * gain, right * gain))
    return write_wav(path, faded)


def severely_clipped(path: Path, seconds: float = 12.0) -> Path:
    """Distortion baked into the samples, not a peak a limiter handles."""
    frames = _tone_frames(seconds, amplitude=1.0, noise_level=0.05)
    return write_wav(path, [(left * 3.0, right * 3.0) for left, right in frames])


def peak_overshoot(path: Path, seconds: float = 12.0) -> Path:
    """Right at full scale, with almost nothing actually clipped.

    Scaled up until the peak just touches the ceiling rather than
    synthesised at amplitude 1.0: the normalised partial sum peaks well
    below full scale, and a fixture that only *claimed* to be hot would
    not exercise the check.
    """
    frames = _tone_frames(seconds, amplitude=0.6, noise_level=0.02)
    ceiling = max(max(abs(left), abs(right)) for left, right in frames)
    gain = 1.0 / ceiling
    return write_wav(path, [(left * gain, right * gain) for left, right in frames])


def dc_offset(path: Path, seconds: float = 12.0, offset: float = 0.05) -> Path:
    return write_wav(
        path, [(left + offset, right + offset) for left, right in _tone_frames(seconds)]
    )


def anti_phase(path: Path, seconds: float = 12.0) -> Path:
    """Channels inverted against each other: the mono sum cancels.

    Not a perfect inversion. At exactly -1.0 the mono sum is digital
    silence and the file reads as silent as well as out of phase, which
    conflates two findings; at 0.92 the correlation is unmistakably
    negative and there is still audio there.
    """
    return write_wav(path, [(left, -left * 0.92) for left, _ in _tone_frames(seconds)])


def mono_as_stereo(path: Path, seconds: float = 12.0) -> Path:
    """Identical channels. Narrow, phase-safe, and perfectly deliverable."""
    return write_wav(path, [(left, left) for left, _ in _tone_frames(seconds)])


def true_mono(path: Path, seconds: float = 12.0) -> Path:
    return write_wav(path, [(left,) for left, _ in _tone_frames(seconds)], channels=1)


def dark_but_valid(path: Path, seconds: float = 12.0) -> Path:
    """A dark master. A production choice, and it must stay eligible."""
    return write_wav(path, _tone_frames(seconds, tilt=-0.35))


def bright_but_valid(path: Path, seconds: float = 12.0) -> Path:
    """A bright master. Also a choice, and nothing rewards it."""
    return write_wav(path, _tone_frames(seconds, tilt=0.18, noise_level=0.05))


#: Content that stops dead below a kilohertz. Enough partials inside the
#: analyser's 200 Hz-16 kHz fit band for the slope to be measured rather
#: than fitted to silence — a single 110 Hz tone reads as a *flat* slope
#: for exactly that reason, which is not what a collapsed generation
#: looks like.
_COLLAPSED_PARTIALS = ((220.0, 1.0), (330.0, 0.5), (440.0, 0.2), (660.0, 0.05))


def spectral_collapse(path: Path, seconds: float = 12.0) -> Path:
    """Nothing above a few hundred hertz. The shape of a failed generation."""
    return write_wav(
        path,
        _tone_frames(seconds, partials=_COLLAPSED_PARTIALS, noise_level=0.0, amplitude=0.6),
    )


def channel_imbalance(path: Path, seconds: float = 12.0, ratio: float = 0.02) -> Path:
    """One channel effectively absent."""
    return write_wav(path, [(left, right * ratio) for left, right in _tone_frames(seconds)])


def truncated(path: Path, seconds: float = 3.0) -> Path:
    """Valid audio, far shorter than any plausible request."""
    return healthy(path, seconds)


def undecodable(path: Path) -> Path:
    """Bytes that are not a WAV file at all."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt this is not audio")
    return path


def empty(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def at_tempo(path: Path, bpm: float, seconds: float = 16.0) -> Path:
    """Audio with an unmistakable pulse at *bpm*.

    A click over a bed, because the tempo estimator needs onsets to find
    and a sustained chord gives it none — an "adherence" test against
    material with no pulse would be comparing two guesses.
    """
    total = int(seconds * SAMPLE_RATE)
    bed = _tone_frames(seconds, amplitude=0.25, noise_level=0.01)
    period = SAMPLE_RATE * 60.0 / bpm
    frames: list[tuple[float, float]] = []
    for index in range(total):
        phase = index % period
        # A short percussive envelope on every beat.
        hit = 0.75 * math.exp(-phase / (SAMPLE_RATE * 0.02)) if phase < SAMPLE_RATE else 0.0
        click = hit * math.sin(2 * math.pi * 2200.0 * index / SAMPLE_RATE)
        left, right = bed[index]
        frames.append((left + click, right + click))
    return write_wav(path, frames)


__all__ = [
    "CHANNELS",
    "SAMPLE_RATE",
    "anti_phase",
    "at_tempo",
    "bright_but_valid",
    "channel_imbalance",
    "dark_but_valid",
    "dc_offset",
    "early_collapse",
    "empty",
    "healthy",
    "mono_as_stereo",
    "near_silent",
    "peak_overshoot",
    "quiet_outro",
    "severely_clipped",
    "silent",
    "spectral_collapse",
    "true_mono",
    "truncated",
    "undecodable",
    "write_wav",
]
