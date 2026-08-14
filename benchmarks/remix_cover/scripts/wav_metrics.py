"""Objective test that an extension preserved its source audio.

The claim under test is specific: the first N seconds of the child are
the parent's recording, not a newly generated imitation of it. Sounding
similar is not evidence — a model asked twice for the same song produces
something that also sounds similar.

Bit-exactness is not required and would be the wrong bar. The engine
re-decodes the preserved latents through the VAE rather than copying
samples, and it re-imposes them with a boundary crossfade, so the prefix
is expected to differ slightly. What must hold is that it correlates with
the source far beyond anything an independent generation could reach.

For that reason the script also computes the same metrics against an
*unrelated* generation as a control. Without a control, a correlation
number means nothing.

Only numpy and the standard library are used — no new dependency for a
one-off measurement.
"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Return float64 samples in [-1, 1], shape (frames, channels)."""
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())

    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif width == 3:
        # 24-bit PCM: widen each 3-byte little-endian sample to int32.
        as_bytes = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        as_int = (
            as_bytes[:, 0].astype(np.int32)
            | (as_bytes[:, 1].astype(np.int32) << 8)
            | (as_bytes[:, 2].astype(np.int32) << 16)
        )
        as_int = np.where(as_int & 0x800000, as_int - 0x1000000, as_int)
        data = as_int.astype(np.float64) / 8388608.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise SystemExit(f"unsupported sample width: {width}")
    return data.reshape(-1, channels), rate


def mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1)


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-invariant SDR in dB. Higher is more faithful to the reference."""
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    energy = float(reference @ reference)
    if energy == 0.0:
        return float("nan")
    scale = float(reference @ estimate) / energy
    projection = scale * reference
    noise = estimate - projection
    denom = float(noise @ noise)
    if denom == 0.0:
        return float("inf")
    return float(10.0 * np.log10(float(projection @ projection) / denom))


def best_offset(a: np.ndarray, b: np.ndarray, search: int) -> int:
    """Integer sample lag maximising correlation, over a short window."""
    window = min(len(a), len(b), 48000 * 5)
    best, best_r = 0, -2.0
    for lag in range(-search, search + 1):
        if lag < 0:
            x, y = a[-lag : window - lag], b[:window]
        else:
            x, y = a[:window], b[lag : window + lag]
        n = min(len(x), len(y))
        if n < 1000:
            continue
        r = float(np.corrcoef(x[:n], y[:n])[0, 1])
        if r > best_r:
            best, best_r = lag, r
    return best


def compare(reference: np.ndarray, candidate: np.ndarray, rate: int) -> dict[str, float]:
    n = min(len(reference), len(candidate))
    reference, candidate = reference[:n], candidate[:n]
    offset = best_offset(reference, candidate, search=64)
    if offset > 0:
        candidate = candidate[offset:]
        reference = reference[: len(candidate)]
    elif offset < 0:
        reference = reference[-offset:]
        candidate = candidate[: len(reference)]
    n = min(len(reference), len(candidate))
    reference, candidate = reference[:n], candidate[:n]

    return {
        "samples_compared": float(n),
        "seconds_compared": n / rate,
        "sample_offset": float(offset),
        "correlation": float(np.corrcoef(reference, candidate)[0, 1]),
        "rms_error": float(np.sqrt(np.mean((reference - candidate) ** 2))),
        "max_abs_error": float(np.max(np.abs(reference - candidate))),
        "si_sdr_db": si_sdr(reference, candidate),
    }


def rms_profile(x: np.ndarray, rate: int, start: float, end: float) -> float:
    segment = x[int(start * rate) : int(end * rate)]
    return float(np.sqrt(np.mean(segment**2))) if len(segment) else 0.0


def main() -> int:
    parent_path = Path(sys.argv[1])
    child_path = Path(sys.argv[2])
    control_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    boundary = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0

    parent, rate = read_wav(parent_path)
    child, child_rate = read_wav(child_path)
    if rate != child_rate:
        raise SystemExit(f"sample-rate mismatch: {rate} vs {child_rate}")

    parent_mono, child_mono = mono(parent), mono(child)
    report: dict[str, object] = {
        "parent_seconds": len(parent_mono) / rate,
        "child_seconds": len(child_mono) / rate,
        "boundary_seconds": boundary,
        "sample_rate": rate,
    }

    prefix_len = min(len(parent_mono), int(boundary * rate))
    report["preserved_prefix"] = compare(parent_mono[:prefix_len], child_mono[:prefix_len], rate)

    if control_path is not None:
        control, control_rate = read_wav(control_path)
        if control_rate == rate:
            control_mono = mono(control)
            m = min(prefix_len, len(control_mono))
            report["control_unrelated_song"] = compare(parent_mono[:m], control_mono[:m], rate)

    # The new region must contain audio, not silence, and the seam must
    # not be a dropout.
    tail_start = boundary
    tail_end = len(child_mono) / rate
    report["levels_rms"] = {
        "child_prefix": rms_profile(child_mono, rate, 0.0, boundary),
        "child_extension": rms_profile(child_mono, rate, tail_start, tail_end),
        "boundary_before_1s": rms_profile(child_mono, rate, boundary - 1.0, boundary),
        "boundary_after_1s": rms_profile(child_mono, rate, boundary, boundary + 1.0),
    }
    extension = child_mono[int(boundary * rate) :]
    report["extension_peak"] = float(np.max(np.abs(extension))) if len(extension) else 0.0
    report["extension_silent_fraction"] = (
        float(np.mean(np.abs(extension) < 1e-4)) if len(extension) else 1.0
    )
    # Longest run of near-silence anywhere in the new region.
    if len(extension):
        quiet = np.abs(extension) < 1e-4
        runs, current = [], 0
        for q in quiet:
            current = current + 1 if q else 0
            runs.append(current)
        report["extension_longest_silence_seconds"] = max(runs) / rate
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
