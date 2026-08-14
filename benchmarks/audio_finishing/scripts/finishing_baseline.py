"""Measure a corpus of existing LUBER masters and summarise it.

The point is to test a listening report against numbers before writing a
single line of corrective DSP. "High frequencies feel rolled off" is a
claim about a distribution, and a distribution needs a corpus: one dull
track proves nothing, and a threshold picked to make one track look
broken will break every other track.

Only project-owned audio is read. Paths are taken as arguments and never
discovered by scanning; the committed records carry file names and
generation ids, never machine paths.

    uv run python scripts/finishing_baseline.py <outdir> <path> [<path> ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "packages" / "audio-finishing" / "src")
)

from luber_audio_finishing.analysis import AudioAnalysis, analyze_audio
from luber_audio_finishing.serialize import analysis_to_dict

#: Metrics summarised across the corpus, as (label, extractor path).
SUMMARY_METRICS: tuple[tuple[str, str], ...] = (
    ("duration_seconds", "technical.duration_seconds"),
    ("peak_dbfs", "level.peak_dbfs"),
    ("rms_dbfs", "level.rms_dbfs"),
    ("crest_factor_db", "level.crest_factor_db"),
    ("integrated_lufs", "loudness.integrated_lufs"),
    ("true_peak_dbfs", "loudness.true_peak_dbfs"),
    ("spectral_centroid_hz", "frequency.spectral_centroid_hz.p50"),
    ("spectral_slope_db_per_octave", "frequency.spectral_slope_db_per_octave"),
    ("air_ratio_db", "frequency.air_ratio_db.p50"),
    ("low_mid_ratio_db", "frequency.low_mid_ratio_db.p50"),
    ("presence_ratio_db", "frequency.presence_ratio_db.p50"),
    ("sibilance_ratio_db", "sibilance.sibilance_ratio_db.p50"),
    ("sibilance_peak_excess_db", "sibilance.sibilance_peak_excess_db"),
    ("harshness_ratio_db", "sibilance.harshness_ratio_db.p50"),
    ("harshness_peak_excess_db", "sibilance.harshness_peak_excess_db"),
    ("onset_rate_per_second", "transient.onset_rate_per_second"),
    ("transient_density", "transient.transient_density"),
    ("short_window_crest_db", "level.short_window_crest_db.p50"),
    ("stereo_width", "stereo.width"),
    ("stereo_full_band_width", "stereo.full_band_width"),
    ("stereo_correlation", "stereo.correlation"),
    ("low_band_correlation", "stereo.low_band_correlation"),
    ("lr_balance_db", "stereo.lr_balance_db"),
    ("side_to_mid_db", "stereo.side_to_mid_db"),
    ("low_band_side_to_mid_db", "stereo.low_band_side_to_mid_db"),
)

#: Band shares are summarised separately; they are the direct evidence
#: for or against the reported high-frequency deficit.
BAND_NAMES = ("sub", "bass", "low_mid", "mid", "presence", "brilliance", "air", "ultra_high")


def _dig(record: dict[str, Any], path: str) -> float | None:
    node: Any = record
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return float(node) if isinstance(node, (int, float)) else None


def _band_share_db(record: dict[str, Any], name: str) -> float | None:
    for band in record.get("frequency", {}).get("bands", []):
        if band.get("name") == name:
            energy = band.get("energy_db")
            return float(energy) if energy is not None else None
    return None


def _summarise(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    p10, p50, p90 = (float(v) for v in np.percentile(array, [10, 50, 90]))
    return {
        "n": float(array.size),
        "min": float(array.min()),
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "max": float(array.max()),
        "mean": float(array.mean()),
        "stdev": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def _label(path: Path) -> str:
    """A stable name that is not a filesystem path.

    Local storage lays masters out as ``audio/<generation-id>/master.wav``,
    so the parent directory is the generation id and is the useful label.
    """
    if path.name.startswith("master.") and path.parent.name:
        return path.parent.name
    return path.stem


def run(destination: Path, sources: list[Path]) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    results_path = destination / "baseline_results.jsonl"
    summary_path = destination / "baseline_summary.json"

    analyses: list[tuple[str, AudioAnalysis]] = []
    with results_path.open("w", encoding="utf-8") as handle:
        for source in sources:
            label = _label(source)
            print(f"analysing {label} ...", flush=True)
            analysis = analyze_audio(source)
            analyses.append((label, analysis))
            record = analysis_to_dict(analysis)
            record["label"] = label
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    records = [analysis_to_dict(analysis) for _, analysis in analyses]
    summary: dict[str, Any] = {
        "track_count": len(records),
        "metrics": {},
        "band_energy_db": {},
    }
    for name, path in SUMMARY_METRICS:
        values = [
            value for value in (_dig(record, path) for record in records) if value is not None
        ]
        summary["metrics"][name] = _summarise(values)
    for band in BAND_NAMES:
        values = [
            value
            for value in (_band_share_db(record, band) for record in records)
            if value is not None
        ]
        summary["band_energy_db"][band] = _summarise(values)

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {results_path.name} and {summary_path.name} for {len(records)} tracks")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    destination = Path(argv[1])
    sources = [Path(value) for value in argv[2:]]
    missing = [source for source in sources if not source.is_file()]
    if missing:
        print(f"missing audio: {', '.join(str(item) for item in missing)}")
        return 2
    return run(destination, sources)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
