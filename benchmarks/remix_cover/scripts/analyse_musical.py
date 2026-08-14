"""Time-varying musical comparison — does the cover follow the source?

The first analysis pass averaged chroma over the whole track and could
not tell a cover from an unrelated song with the same prompt: both scored
~0.98. That is a flaw in the measure, not a finding. Averaged over 30
seconds, any two pieces in the same key and genre look alike.

What distinguishes "derived from this recording" is whether the output
moves *with* the source through time: the same chord at the same moment,
the same note going up where the source goes up, energy arriving in the
same places.

So this compares sequences rather than averages:

  chroma-sequence similarity   per-frame chroma, correlated frame by frame
  pitch-contour correlation    dominant f0 per frame, over voiced frames
  self-similarity agreement    each track's own structure matrix, compared

Every number is reported against a control — an independent generation
from the same prompt and lyrics — because only the gap between them means
anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import cast

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyse_signal import FRAME, frames, spectrogram
from wav_metrics import mono, read_wav

FMIN, FMAX = 55.0, 5000.0


def chroma_sequence(x: np.ndarray, rate: int) -> np.ndarray:
    """Per-frame 12-bin pitch-class profile, L2-normalised per frame."""
    spec = spectrogram(x)
    if len(spec) == 0:
        return np.zeros((0, 12))
    freqs = np.fft.rfftfreq(FRAME, 1 / rate)
    keep = (freqs >= FMIN) & (freqs <= FMAX)
    freqs, spec = freqs[keep], spec[:, keep]
    classes = np.mod(np.round(69 + 12 * np.log2(freqs / 440.0)).astype(int), 12)
    out = np.zeros((spec.shape[0], 12))
    for pitch_class in range(12):
        out[:, pitch_class] = spec[:, classes == pitch_class].sum(axis=1)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return np.divide(out, norms, out=np.zeros_like(out), where=norms > 0)


def chroma_sequence_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-frame cosine between two chroma sequences.

    Frame-aligned: this asks whether the same harmony is present at the
    same time, which an unrelated song has no reason to satisfy.
    """
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    return float(np.mean(np.sum(a[:n] * b[:n], axis=1)))


def dominant_f0(x: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Crude per-frame pitch by spectral peak, with a voiced-frame mask."""
    spec = spectrogram(x)
    if len(spec) == 0:
        return np.zeros(0), np.zeros(0, dtype=bool)
    freqs = np.fft.rfftfreq(FRAME, 1 / rate)
    band = (freqs >= 80) & (freqs <= 1200)
    sub, sub_freqs = spec[:, band], freqs[band]
    peak_idx = np.argmax(sub, axis=1)
    f0 = sub_freqs[peak_idx]
    energy = np.sqrt((frames(x) ** 2).mean(axis=1))
    voiced = energy > max(1e-4, 0.15 * float(np.median(energy[energy > 0]) or 0))
    return f0, voiced


def pitch_contour_correlation(source: np.ndarray, other: np.ndarray, rate: int) -> dict[str, float]:
    fa, va = dominant_f0(source, rate)
    fb, vb = dominant_f0(other, rate)
    n = min(len(fa), len(fb))
    if n < 16:
        return {"correlation": float("nan"), "frames": 0.0}
    mask = va[:n] & vb[:n]
    if mask.sum() < 16:
        return {"correlation": float("nan"), "frames": float(mask.sum())}
    # Semitones: pitch relationships are logarithmic.
    a = 12 * np.log2(np.maximum(fa[:n][mask], 1e-6) / 440.0)
    b = 12 * np.log2(np.maximum(fb[:n][mask], 1e-6) / 440.0)
    if a.std() == 0 or b.std() == 0:
        return {"correlation": float("nan"), "frames": float(mask.sum())}
    return {
        "correlation": float(np.corrcoef(a, b)[0, 1]),
        "frames": float(mask.sum()),
    }


def self_similarity(x: np.ndarray, rate: int, bins: int = 48) -> np.ndarray:
    """Coarse structure matrix: how each part of a track resembles itself.

    Comparing two tracks' own structure matrices asks whether they are
    laid out the same way, independently of what they sound like.
    """
    seq = chroma_sequence(x, rate)
    if len(seq) < bins:
        return np.zeros((bins, bins))
    edges = np.linspace(0, len(seq), bins + 1).astype(int)
    pooled = np.stack([seq[edges[i] : edges[i + 1]].mean(axis=0) for i in range(bins)])
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    pooled = np.divide(pooled, norms, out=np.zeros_like(pooled), where=norms > 0)
    return pooled @ pooled.T


def structure_agreement(a: np.ndarray, b: np.ndarray) -> float:
    upper = np.triu_indices_from(a, k=1)
    x, y = a[upper], b[upper]
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def compare(source: np.ndarray, other: np.ndarray, rate: int) -> dict[str, object]:
    return {
        "chroma_sequence_similarity": chroma_sequence_similarity(
            chroma_sequence(source, rate), chroma_sequence(other, rate)
        ),
        "pitch_contour": pitch_contour_correlation(source, other, rate),
        "structure_agreement": structure_agreement(
            self_similarity(source, rate), self_similarity(other, rate)
        ),
    }


def main() -> int:
    root = Path(sys.argv[1])
    source_path = Path(sys.argv[2])
    control_paths = [Path(p) for p in sys.argv[3:]]

    source_raw, rate = read_wav(source_path)
    source = mono(source_raw)

    rows: list[tuple[str, dict[str, object]]] = []
    for path in sorted((root / "outputs").glob("*.wav")):
        raw, out_rate = read_wav(path)
        if out_rate != rate:
            continue
        rows.append((path.stem, compare(source, mono(raw), rate)))

    controls: list[tuple[str, dict[str, object]]] = []
    for path in control_paths:
        if not path.exists():
            continue
        raw, ctl_rate = read_wav(path)
        if ctl_rate != rate:
            continue
        controls.append((f"CONTROL {path.parent.name[:8]}", compare(source, mono(raw), rate)))

    # A shuffled copy of the source: the floor for "same audio, wrong
    # order". Any real time-alignment signal must beat this.
    rng = np.random.default_rng(0)
    blocks = np.array_split(source, 40)
    rng.shuffle(blocks)
    controls.append(("CONTROL source-shuffled", compare(source, np.concatenate(blocks), rate)))

    header = f"{'run':<30}{'chroma_seq':>12}{'pitch_corr':>12}{'frames':>9}{'structure':>11}"
    print(header)
    print("-" * len(header))
    for name, r in rows + controls:
        pitch = cast(dict[str, float], r["pitch_contour"])
        chroma_seq = cast(float, r["chroma_sequence_similarity"])
        structure = cast(float, r["structure_agreement"])
        print(
            f"{name:<30}{chroma_seq:>12.4f}"
            f"{pitch['correlation']:>12.4f}{pitch['frames']:>9.0f}{structure:>11.4f}"
        )

    (root / "musical.json").write_text(
        json.dumps({name: r for name, r in rows + controls}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
