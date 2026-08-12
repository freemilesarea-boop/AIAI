#!/usr/bin/env python3
"""Non-destructive technical grading of discovered candidates.

Reads the discovery catalog, measures each candidate's audio, and
assigns a grade. Nothing is modified: files are opened read-only and no
audio is copied into the repository.

Grading deliberately does **not** punish a track for being loud. A
well-mastered modern pop record is loud on purpose, and that is the
sound we want the model to learn. Only measurable damage counts against
a track — real clipping, unusable dynamics, wrong format, corruption.

    uv run python scripts/dataset/grade_candidates.py

Grades:
    A       clean, full-band, stereo, correct rate — prime material
    B       usable with a minor caveat
    C       usable only if nothing better exists
    REJECT  measurable damage or wrong format
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "packages" / "dataset" / "src"))

from luber_dataset.discovery import sanitize  # noqa: E402

#: Real clipping damage, not loudness.
CLIPPING_SAMPLE_RATIO = 0.001
#: Below this the master is squashed beyond what modern loudness explains.
MIN_CREST_FACTOR_DB = 4.0
MIN_SAMPLE_RATE = 44_100
MIN_DURATION_SECONDS = 45.0


def _spectral(samples: list[int], rate: int) -> tuple[float | None, float | None]:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a dev dependency
        return None, None
    if len(samples) < 1024:
        return None, None
    data = np.asarray(samples, dtype=np.float32)
    data = data / (float(np.max(np.abs(data))) or 1.0)
    spectrum = np.abs(np.fft.rfft(data * np.hanning(len(data)).astype(np.float32)))
    freqs = np.fft.rfftfreq(len(data), 1 / rate)
    total = float(spectrum.sum()) or 1.0
    centroid = float((freqs * spectrum).sum() / total)
    # Energy above 6 kHz: the band the human evaluator called excessive.
    hf = float(spectrum[freqs >= 6000].sum() / total)
    return round(centroid, 1), round(hf, 4)


def measure(path: Path) -> dict[str, Any]:
    """Read-only measurement of a PCM WAV file."""
    result: dict[str, Any] = {"readable": False, "flags": ["UNREADABLE"]}
    try:
        with wave.open(str(path), "rb") as w:
            channels, width = w.getnchannels(), w.getsampwidth()
            rate, frames = w.getframerate(), w.getnframes()
            # A grading pass does not need a whole album in memory.
            raw = w.readframes(min(frames, rate * 90))
    except Exception:
        return result

    if rate <= 0 or channels <= 0 or frames <= 0:
        return result

    full = float(1 << (width * 8 - 1))
    samples = [
        int.from_bytes(raw[i : i + width], "little", signed=True)
        for i in range(0, len(raw) - width + 1, width)
    ]
    if not samples:
        return result

    peak = max(max(samples), -min(samples)) / full
    rms = math.sqrt(sum(s * s for s in samples) / len(samples)) / full
    peak_db = 20 * math.log10(peak) if peak > 0 else -math.inf
    rms_db = 20 * math.log10(rms) if rms > 0 else -math.inf
    crest = peak_db - rms_db if math.isfinite(rms_db) and math.isfinite(peak_db) else 0.0
    near_full = full * 0.999
    clip_ratio = sum(1 for s in samples if abs(s) >= near_full) / len(samples)
    dc = (sum(samples) / len(samples)) / full
    centroid, hf_ratio = _spectral(samples[: rate * 2] or samples, rate)

    flags: list[str] = []
    if clip_ratio > CLIPPING_SAMPLE_RATIO:
        flags.append("CLIPPING")
    if math.isfinite(rms_db) and crest < MIN_CREST_FACTOR_DB:
        flags.append("OVER_COMPRESSED")
    if rate < MIN_SAMPLE_RATE:
        flags.append("LOW_SAMPLE_RATE")
    if channels < 2:
        flags.append("MONO_SOURCE")
    if frames / rate < MIN_DURATION_SECONDS:
        flags.append("TOO_SHORT")
    if abs(dc) > 0.01:
        flags.append("DC_OFFSET")
    if math.isfinite(rms_db) and rms_db < -40:
        flags.append("TOO_QUIET")

    return {
        "readable": True,
        "duration_seconds": round(frames / rate, 2),
        "sample_rate": rate,
        "channels": channels,
        "bit_depth": width * 8,
        "peak": round(peak, 4),
        "rms_dbfs": round(rms_db, 2) if math.isfinite(rms_db) else None,
        "crest_factor_db": round(crest, 2),
        "clipping_sample_ratio": round(clip_ratio, 6),
        "dc_offset": round(dc, 5),
        "spectral_centroid_hz": centroid,
        "high_frequency_ratio": hf_ratio,
        "flags": flags,
    }


def grade(measurement: dict[str, Any]) -> str:
    if not measurement.get("readable"):
        return "REJECT"
    flags = set(measurement.get("flags") or [])
    if flags & {"UNREADABLE", "CLIPPING", "LOW_SAMPLE_RATE", "DC_OFFSET"}:
        return "REJECT"
    if flags & {"OVER_COMPRESSED", "MONO_SOURCE", "TOO_QUIET"}:
        return "C"
    if flags & {"TOO_SHORT"}:
        return "B"
    return "A"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog", type=Path, default=Path.home() / ".luber" / "discovery_catalog.json"
    )
    parser.add_argument(
        "--out", type=Path, default=Path.home() / ".luber" / "candidate_grades.json"
    )
    parser.add_argument(
        "--summary", type=Path, default=REPO_ROOT / "data" / "candidate_grades_summary.json"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.catalog.is_file():
        print(f"ABORT: no catalog at {args.catalog}; run discover_audio.py first", file=sys.stderr)
        return 2

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    # Commercial references and our own output are never training
    # candidates, so there is nothing to grade them for.
    candidates = [
        entry
        for entry in catalog
        if not entry["commercial_reference_hypothesis"]
        and entry["origin_hypothesis"] != "SELF_MODEL_OUTPUT"
        and entry["extension"] == ".wav"
    ]
    print(f"grading {len(candidates)} WAV candidates (read-only)\n")

    desktop = Path.home() / "Desktop"
    graded: list[dict[str, Any]] = []
    for entry in candidates:
        measurement = measure(Path(entry["absolute_path"]))
        graded.append({**entry, "measurement": measurement, "grade": grade(measurement)})

    counts = Counter(item["grade"] for item in graded)
    by_directory: dict[str, Counter[str]] = defaultdict(Counter)
    hypothesis_by_dir: dict[str, str] = {}
    for item in graded:
        relative = sanitize(str(Path(item["absolute_path"]).parent), root=desktop)
        by_directory[relative][item["grade"]] += 1
        hypothesis_by_dir[relative] = item["origin_hypothesis"]

    print(f"grades: {dict(counts)}\n")
    print(f"{'directory':<26}{'A':>4}{'B':>4}{'C':>4}{'REJ':>5}   hypothesis")
    for relative, tally in sorted(by_directory.items(), key=lambda kv: -sum(kv[1].values())):
        print(
            f"{relative:<26}{tally['A']:>4}{tally['B']:>4}{tally['C']:>4}"
            f"{tally['REJECT']:>5}   {hypothesis_by_dir.get(relative, '')}"
        )

    flag_counts: Counter[str] = Counter()
    for item in graded:
        for flag in item["measurement"].get("flags") or []:
            flag_counts[flag] += 1
    if flag_counts:
        print(f"\nflags: {dict(flag_counts.most_common())}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(graded, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(
            {
                "graded_candidates": len(graded),
                "grades": dict(counts),
                "flags": dict(flag_counts),
                "by_directory": {rel: dict(t) for rel, t in sorted(by_directory.items())},
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\ngrades (personal paths): {args.out}")
    print(f"sanitized summary      : {args.summary}")
    print("\nNo audio file was modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
