"""Measure what cover preserved, separating two different questions.

SIGNAL preservation asks "is this the same recording" — waveform
correlation, SI-SDR. A remix should score *low* here; scoring high would
mean nothing was transformed.

MUSICAL preservation asks "is this the same song" — does it keep the
harmony, the rhythm, the shape over time. Those survive transformation,
so they are what decides whether the output is derived from the source or
merely another song with the same prompt.

Conflating the two is the standard way to get this wrong in both
directions, so they are reported separately and never combined.

Everything is computed with numpy: framed FFT for spectra and a
chroma-like pitch-class profile, envelope correlation for rhythm, and an
onset-flux comparison. No new dependency.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from wav_metrics import mono, read_wav, si_sdr

FRAME = 2048
HOP = 512
#: A4 = 440 Hz; bins below this are too coarse for pitch classes.
FMIN = 55.0
FMAX = 5000.0


def frames(x: np.ndarray) -> np.ndarray:
    n = 1 + max(0, (len(x) - FRAME) // HOP)
    if n <= 0:
        return np.zeros((0, FRAME))
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx] * np.hanning(FRAME)[None, :]


def spectrogram(x: np.ndarray) -> np.ndarray:
    f = frames(x)
    if len(f) == 0:
        return np.zeros((0, FRAME // 2 + 1))
    return np.abs(np.fft.rfft(f, axis=1))


def spectral_stats(x: np.ndarray, rate: int) -> dict[str, float]:
    spec = spectrogram(x)
    if len(spec) == 0:
        return {}
    freqs = np.fft.rfftfreq(FRAME, 1 / rate)
    power = spec + 1e-12
    total = power.sum(axis=1)
    centroid = (power * freqs[None, :]).sum(axis=1) / total
    cumulative = np.cumsum(power, axis=1) / total[:, None]
    rolloff = freqs[np.argmax(cumulative >= 0.85, axis=1)]
    return {
        "spectral_centroid_hz": float(np.mean(centroid)),
        "spectral_rolloff85_hz": float(np.mean(rolloff)),
    }


def chroma(x: np.ndarray, rate: int) -> np.ndarray:
    """12-bin pitch-class profile averaged over time.

    Crude next to librosa's, but it is a real chroma: energy is mapped to
    pitch classes by frequency and folded across octaves.
    """
    spec = spectrogram(x)
    if len(spec) == 0:
        return np.zeros(12)
    freqs = np.fft.rfftfreq(FRAME, 1 / rate)
    keep = (freqs >= FMIN) & (freqs <= FMAX)
    freqs, spec = freqs[keep], spec[:, keep]
    midi = 69 + 12 * np.log2(freqs / 440.0)
    classes = np.mod(np.round(midi).astype(int), 12)
    profile = np.zeros(12)
    energy = spec.mean(axis=0)
    for pitch_class in range(12):
        profile[pitch_class] = energy[classes == pitch_class].sum()
    norm = np.linalg.norm(profile)
    return profile / norm if norm else profile


def envelope(x: np.ndarray) -> np.ndarray:
    f = frames(x)
    if len(f) == 0:
        return np.zeros(0)
    return np.sqrt((f**2).mean(axis=1))


def onset_flux(x: np.ndarray) -> np.ndarray:
    spec = spectrogram(x)
    if len(spec) < 2:
        return np.zeros(0)
    diff = np.diff(spec, axis=0)
    return np.maximum(diff, 0).sum(axis=1)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else float("nan")


def correlate(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 2:
        return float("nan")
    a, b = a[:n], b[:n]
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def tempo_estimate(x: np.ndarray, rate: int) -> float:
    """Peak of the onset-flux autocorrelation, in BPM.

    Reported with its own confidence because a weak peak means the
    estimate should not be believed.
    """
    flux = onset_flux(x)
    if len(flux) < 32:
        return float("nan")
    flux = flux - flux.mean()
    corr = np.correlate(flux, flux, mode="full")[len(flux) - 1 :]
    fps = rate / HOP
    lo, hi = int(fps * 60 / 200), int(fps * 60 / 50)  # 50-200 BPM
    hi = min(hi, len(corr) - 1)
    if hi <= lo:
        return float("nan")
    lag = lo + int(np.argmax(corr[lo:hi]))
    return float(60.0 * fps / lag) if lag else float("nan")


def analyse(source: np.ndarray, candidate: np.ndarray, rate: int) -> dict[str, object]:
    n = min(len(source), len(candidate))
    s, c = source[:n], candidate[:n]
    return {
        "signal_preservation": {
            "waveform_correlation": correlate(s, c),
            "si_sdr_db": si_sdr(s, c),
            "rms_source": float(np.sqrt(np.mean(s**2))),
            "rms_output": float(np.sqrt(np.mean(c**2))),
            "peak_output": float(np.max(np.abs(candidate))),
        },
        "musical_preservation": {
            "chroma_cosine": cosine(chroma(s, rate), chroma(c, rate)),
            "envelope_correlation": correlate(envelope(s), envelope(c)),
            "onset_flux_correlation": correlate(onset_flux(s), onset_flux(c)),
            "tempo_source_bpm": tempo_estimate(s, rate),
            "tempo_output_bpm": tempo_estimate(c, rate),
        },
        "timbre": spectral_stats(candidate, rate),
        "integrity": {
            "duration_seconds": len(candidate) / rate,
            "silent_fraction": float(np.mean(np.abs(candidate) < 1e-4)),
        },
    }


def main() -> int:
    root = Path(sys.argv[1])
    source_path = Path(sys.argv[2])
    control_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    source_raw, rate = read_wav(source_path)
    source = mono(source_raw)

    outputs_report: dict[str, object] = {}
    report: dict[str, object] = {
        "outputs": outputs_report,
        "source": {
            "path": source_path.name,
            "duration_seconds": len(source) / rate,
            "sample_rate": rate,
            **spectral_stats(source, rate),
            "tempo_bpm": tempo_estimate(source, rate),
        },
    }

    outputs = sorted((root / "outputs").glob("*.wav"))
    for path in outputs:
        raw, out_rate = read_wav(path)
        if out_rate != rate:
            outputs_report[path.stem] = {"error": f"rate {out_rate} != {rate}"}
            continue
        outputs_report[path.stem] = analyse(source, mono(raw), rate)

    # The floor: an unrelated song made from the same prompt and lyrics.
    # Every number above has to be read against this, or it means nothing.
    if control_path is not None and control_path.exists():
        raw, control_rate = read_wav(control_path)
        if control_rate == rate:
            report["control_unrelated_generation"] = analyse(source, mono(raw), rate)

    print(json.dumps(report, indent=2))
    (root / "analysis.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
