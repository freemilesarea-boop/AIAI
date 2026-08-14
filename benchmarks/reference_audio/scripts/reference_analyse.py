"""Measure whether reference audio causally changed the generation.

Three comparisons, each answering a different question, and none of them
sufficient alone:

``vs reference``
    Did characteristics of the reference appear in the output? Timbre
    descriptors carry this; waveform correlation does not, because a
    style transfer shares no samples with its reference.

``vs the prompt-only control``
    Did adding a reference change anything at all? A run that matches the
    control is evidence the reference did nothing.

``pairwise``
    Did *which* reference was supplied matter, and is that difference
    larger than the difference a mere seed change produces? Without the
    seed comparison, any two runs differ and the difference proves
    nothing.

Descriptors are chosen for what they can support. Spectral centroid and
rolloff describe brightness and are the most defensible timbre signal
here. MFCC cosine describes broad spectral envelope. Chroma and onset
describe harmony and rhythm but were shown in Phase 13D to saturate on
time-averaging, so they are computed per-frame and compared as sequences.

numpy and the standard library only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import cast

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "remix_cover" / "scripts"))
from analyse_musical import chroma_sequence, chroma_sequence_similarity
from analyse_signal import FRAME, onset_flux, spectrogram, tempo_estimate
from wav_metrics import mono, read_wav, si_sdr

N_MEL = 40
N_MFCC = 13


def _mel_filterbank(rate: int, n_fft: int, n_mel: int) -> np.ndarray:
    def hz_to_mel(f: np.ndarray | float) -> np.ndarray | float:
        return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)

    def mel_to_hz(m: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    low, high = 20.0, min(rate / 2.0, 8000.0)
    points = mel_to_hz(
        np.linspace(cast(float, hz_to_mel(low)), cast(float, hz_to_mel(high)), n_mel + 2)
    )
    bins = np.floor((n_fft + 1) * points / rate).astype(int)
    filters = np.zeros((n_mel, n_fft // 2 + 1))
    for i in range(n_mel):
        left, centre, right = bins[i], bins[i + 1], bins[i + 2]
        if centre == left or right == centre:
            continue
        filters[i, left:centre] = (np.arange(left, centre) - left) / (centre - left)
        filters[i, centre:right] = (right - np.arange(centre, right)) / (right - centre)
    return filters


def mfcc(x: np.ndarray, rate: int) -> np.ndarray:
    """Mean MFCC vector — a coarse description of spectral envelope."""
    spec = spectrogram(x)
    if len(spec) == 0:
        return np.zeros(N_MFCC)
    power = spec**2
    mel = power @ _mel_filterbank(rate, FRAME, N_MEL).T
    log_mel = np.log(mel + 1e-10)
    # DCT-II across mel bands.
    n = log_mel.shape[1]
    basis = np.cos(np.pi / n * (np.arange(n) + 0.5)[None, :] * np.arange(N_MFCC)[:, None])
    coeffs = log_mel @ basis.T
    # Coefficient 0 is overall loudness; dropped so the comparison is
    # about spectral shape rather than level.
    return coeffs.mean(axis=0)[1:]


def descriptors(x: np.ndarray, rate: int) -> dict[str, float]:
    spec = spectrogram(x)
    freqs = np.fft.rfftfreq(FRAME, 1 / rate)
    power = spec + 1e-12
    total = power.sum(axis=1)
    centroid = (power * freqs[None, :]).sum(axis=1) / total
    cumulative = np.cumsum(power, axis=1) / total[:, None]
    rolloff = freqs[np.argmax(cumulative >= 0.85, axis=1)]
    flatness = np.exp(np.log(power).mean(axis=1)) / power.mean(axis=1)
    return {
        "spectral_centroid_hz": float(centroid.mean()),
        "spectral_rolloff85_hz": float(rolloff.mean()),
        "spectral_flatness": float(flatness.mean()),
        "rms": float(np.sqrt(np.mean(x**2))),
        "tempo_bpm": tempo_estimate(x, rate),
        "duration_seconds": len(x) / rate,
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denom) if denom else float("nan")


def correlate(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 2:
        return float("nan")
    a, b = a[:n], b[:n]
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compare(a: np.ndarray, b: np.ndarray, rate: int) -> dict[str, float]:
    """How alike two pieces of audio are, on several independent axes."""
    da, db = descriptors(a, rate), descriptors(b, rate)
    return {
        # Signal identity. Near zero for anything that is not a copy.
        "waveform_correlation": correlate(a[: min(len(a), len(b))], b[: min(len(a), len(b))]),
        "si_sdr_db": si_sdr(a[: min(len(a), len(b))], b[: min(len(a), len(b))]),
        # Timbre.
        "mfcc_cosine": cosine(mfcc(a, rate), mfcc(b, rate)),
        "centroid_ratio": db["spectral_centroid_hz"] / max(da["spectral_centroid_hz"], 1e-6),
        "centroid_delta_hz": db["spectral_centroid_hz"] - da["spectral_centroid_hz"],
        "rolloff_delta_hz": db["spectral_rolloff85_hz"] - da["spectral_rolloff85_hz"],
        # Harmony and rhythm, compared as sequences rather than averages.
        "chroma_sequence_similarity": chroma_sequence_similarity(
            chroma_sequence(a, rate), chroma_sequence(b, rate)
        ),
        "onset_correlation": correlate(onset_flux(a), onset_flux(b)),
    }


def main() -> int:
    root = Path(sys.argv[1])
    outputs = {p.stem: p for p in sorted((root / "outputs").glob("*.wav"))}
    references = {p.stem: p for p in sorted((root / "references").glob("*.wav"))}

    loaded: dict[str, tuple[np.ndarray, int]] = {}
    for name, path in {**outputs, **references}.items():
        raw, rate = read_wav(path)
        loaded[name] = (mono(raw), rate)

    rate = next(iter(loaded.values()))[1]
    report: dict[str, object] = {
        "descriptors": {name: descriptors(x, r) for name, (x, r) in loaded.items()},
        "comparisons": {},
    }
    comparisons = cast(dict[str, object], report["comparisons"])

    ref_a = "REFERENCE_A_electronic"
    ref_b = "REFERENCE_B_acoustic"
    control = "00_PROMPT_ONLY"

    pairs = [
        # Did the reference's character appear in the output?
        (f"{ref_a} -> 01_REFERENCE_A", ref_a, "01_REFERENCE_A"),
        (f"{ref_b} -> 02_REFERENCE_B", ref_b, "02_REFERENCE_B"),
        (f"{ref_a} -> 03_CONTRADICTORY", ref_a, "03_REFERENCE_A_CONTRADICTORY_PROMPT"),
        # Cross-check: reference A's character should not show up in the
        # run conditioned on reference B, or the effect is not causal.
        (f"{ref_a} -> 02_REFERENCE_B (cross)", ref_a, "02_REFERENCE_B"),
        (f"{ref_b} -> 01_REFERENCE_A (cross)", ref_b, "01_REFERENCE_A"),
        # Did adding a reference change anything versus no reference?
        (f"{control} vs 01_REFERENCE_A", control, "01_REFERENCE_A"),
        (f"{control} vs 02_REFERENCE_B", control, "02_REFERENCE_B"),
        # Does which reference matter?
        ("01_REFERENCE_A vs 02_REFERENCE_B", "01_REFERENCE_A", "02_REFERENCE_B"),
        # Seed noise floor: how different are two runs that differ only
        # by seed? Any reference effect must exceed this.
        ("01_REFERENCE_A vs 05_DIFFERENT_SEED", "01_REFERENCE_A", "05_REFERENCE_A_DIFFERENT_SEED"),
        # Prompt effect, measured with the reference held out.
        (f"{control} vs 04_CONTRADICTORY_ONLY", control, "04_CONTRADICTORY_PROMPT_ONLY"),
        # Prompt effect with a reference present, and the reverse.
        (
            "03_CONTRADICTORY vs 04_CONTRADICTORY_ONLY",
            "03_REFERENCE_A_CONTRADICTORY_PROMPT",
            "04_CONTRADICTORY_PROMPT_ONLY",
        ),
        (
            "01_REFERENCE_A vs 03_CONTRADICTORY",
            "01_REFERENCE_A",
            "03_REFERENCE_A_CONTRADICTORY_PROMPT",
        ),
    ]

    for label, left, right in pairs:
        if left in loaded and right in loaded:
            comparisons[label] = compare(loaded[left][0], loaded[right][0], rate)

    (root / "analysis.json").write_text(json.dumps(report, indent=2))

    print(f"{'track':<40}{'centroid':>10}{'rolloff':>10}{'flatness':>10}{'tempo':>8}")
    print("-" * 78)
    for name, d in cast(dict[str, dict[str, float]], report["descriptors"]).items():
        print(
            f"{name:<40}{d['spectral_centroid_hz']:>10.0f}{d['spectral_rolloff85_hz']:>10.0f}"
            f"{d['spectral_flatness']:>10.4f}{d['tempo_bpm']:>8.0f}"
        )

    print()
    header = f"{'comparison':<44}{'mfcc':>8}{'cent_d':>9}{'chroma':>8}{'onset':>8}{'si_sdr':>9}"
    print(header)
    print("-" * len(header))
    for label, c in cast(dict[str, dict[str, float]], comparisons).items():
        print(
            f"{label:<44}{c['mfcc_cosine']:>8.3f}{c['centroid_delta_hz']:>9.0f}"
            f"{c['chroma_sequence_similarity']:>8.3f}{c['onset_correlation']:>8.3f}"
            f"{c['si_sdr_db']:>9.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
