#!/usr/bin/env python
"""Objective measurement of the frozen Phase 20 RAW baseline.

Measures the **RAW model master** and nothing else. The Phase 14 finished
master exists and is often better, and using it here would be measuring
the equaliser rather than the model — the specific mistake this script is
written to avoid.

Reuses ``luber_audio_finishing.analyze_audio``, which already computes
levels, EBU R128 loudness, band energies, spectral slope, sibilance,
harshness, transients and stereo geometry. Adding a second analyser would
mean two definitions of "presence" in one repository.

None of these numbers is musical quality. They cannot hear a weak melody
or a trot-like phrase. What they can do is tell a listener where to look,
and catch the failures that are genuinely objective — a track that fades
early, a mix with no top end, a stereo image that collapses to mono.

    p20_analyze.py --limit 5        # quick look
    p20_analyze.py --json out.json  # full run, machine readable
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "packages" / "audio-finishing" / "src"))

from luber_audio_finishing import analyze_audio  # noqa: E402
from luber_audio_finishing.analysis import Distribution  # noqa: E402


def scalar(value: object) -> float | int | bool | None:
    """Reduce a windowed measurement to one number.

    Several metrics are distributions across analysis windows rather
    than single values. The median is the right summary here: a corpus
    description should not move because one window clipped.
    """
    if isinstance(value, Distribution):
        return round(value.p50, 4)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return None


STORAGE_ROOT = REPO_ROOT / "data"
RESULTS = Path(__file__).resolve().parents[1] / "results"


def raw_masters() -> list[tuple[str, Path]]:
    """Every RAW master on disk, addressed by generation id.

    The RAW master is written as ``master.wav``; the finished master and
    the preview live under different names, so this cannot pick one up
    by accident.
    """
    found: list[tuple[str, Path]] = []
    audio_root = STORAGE_ROOT / "audio"
    if not audio_root.is_dir():
        return found
    for directory in sorted(audio_root.iterdir()):
        candidate = directory / "master.wav"
        if candidate.is_file():
            found.append((directory.name, candidate))
    return found


def measure(path: Path) -> dict[str, float | int | bool | None]:
    analysis = analyze_audio(path)
    level, freq, sib = analysis.level, analysis.frequency, analysis.sibilance
    stereo, transient, loud = analysis.stereo, analysis.transient, analysis.loudness
    technical = analysis.technical
    raw = {
        "duration_seconds": getattr(technical, "duration_seconds", None),
        "sample_rate": getattr(technical, "sample_rate", None),
        "channels": getattr(technical, "channels", None),
        "peak_dbfs": level.peak_dbfs,
        "rms_dbfs": level.rms_dbfs,
        "crest_factor_db": level.crest_factor_db,
        "silence_ratio": level.silence_ratio,
        "clipped_samples": level.clipped_samples,
        "integrated_lufs": getattr(loud, "integrated_lufs", None),
        "loudness_range": getattr(loud, "loudness_range", None),
        "true_peak_dbtp": getattr(loud, "true_peak_dbtp", None),
        "spectral_centroid_hz": freq.spectral_centroid_hz,
        "spectral_rolloff85_hz": freq.spectral_rolloff85_hz,
        "spectral_slope_db_per_octave": freq.spectral_slope_db_per_octave,
        "air_ratio_db": freq.air_ratio_db,
        "low_mid_ratio_db": freq.low_mid_ratio_db,
        "presence_ratio_db": freq.presence_ratio_db,
        "sibilance_ratio_db": sib.sibilance_ratio_db,
        "harshness_ratio_db": sib.harshness_ratio_db,
        "onset_rate_per_second": transient.onset_rate_per_second,
        "spectral_flux": transient.spectral_flux,
        "stereo_width": stereo.width,
        "stereo_correlation": stereo.correlation,
        "side_to_mid_db": stereo.side_to_mid_db,
        "is_stereo": stereo.is_stereo,
    }
    return {key: scalar(value) for key, value in raw.items()}


def summarise(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Median and spread per metric. Median because a couple of failed
    runs should not drag the description of a corpus."""
    summary: dict[str, dict[str, float]] = {}
    keys = [
        k
        for k in rows[0]
        if isinstance(rows[0][k], (int, float)) and not isinstance(rows[0][k], bool)
    ]
    for key in keys:
        values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        if not values:
            continue
        values.sort()
        summary[key] = {
            "median": round(statistics.median(values), 3),
            "min": round(values[0], 3),
            "max": round(values[-1], 3),
            "n": len(values),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--json", dest="json_out")
    args = parser.parse_args()

    masters = raw_masters()
    if args.limit:
        masters = masters[: args.limit]
    if not masters:
        print("no RAW masters found", file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = []
    for index, (generation_id, path) in enumerate(masters, start=1):
        try:
            row = measure(path)
        except Exception as exc:
            print(f"  [{index}/{len(masters)}] {generation_id[:8]} FAILED: {type(exc).__name__}")
            continue
        annotated: dict[str, Any] = dict(row)
        annotated["generation_id"] = generation_id
        rows.append(annotated)
        print(
            f"  [{index}/{len(masters)}] {generation_id[:8]} "
            f"lufs={row['integrated_lufs']} centroid={row['spectral_centroid_hz']} "
            f"width={row['stereo_width']} silence={row['silence_ratio']}",
            flush=True,
        )

    summary = summarise(rows)
    print(f"\nmeasured {len(rows)} RAW masters")
    for key in (
        "integrated_lufs",
        "crest_factor_db",
        "spectral_centroid_hz",
        "spectral_slope_db_per_octave",
        "air_ratio_db",
        "low_mid_ratio_db",
        "presence_ratio_db",
        "sibilance_ratio_db",
        "harshness_ratio_db",
        "stereo_width",
        "stereo_correlation",
        "silence_ratio",
    ):
        if key in summary:
            s = summary[key]
            print(f"  {key:<30} median {s['median']:>9}   range {s['min']} … {s['max']}")

    if args.json_out:
        target = Path(args.json_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"items": rows, "summary": summary}, indent=2))
        print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
