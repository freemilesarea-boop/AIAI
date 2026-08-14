"""Build the RAW vs FINISHED listening package and measure both sides.

The measurements exist to prove the processing did what the plan said,
not to decide whether it sounds better. Nothing here scores quality: a
waveform difference of 40 dB and a waveform difference of 20 dB are both
consistent with an improvement and with a ruined mix, and only a
listener can tell them apart.

Audio is written outside the repository. Only the numbers are committed.

    uv run python scripts/finishing_ab.py <listening-dir> <outdir> <path> [...]
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "packages" / "audio-finishing" / "src")
)

from luber_audio_finishing import (
    AudioAnalysis,
    FinishingResult,
    analysis_to_dict,
    finish_audio,
    plan_to_dict,
)
from luber_audio_finishing.audiofile import load_audio
from luber_audio_finishing.bands import BAND_NAMES

README = """LUBER PHASE 14 — FINISHING LISTENING TEST

Each numbered pair is the same song twice: one file marked RAW and one
marked FINISHED. Please listen to both and answer the questions below.

There is no correct answer. RAW is a legitimate answer for any pair, and
so is "no useful difference". How much processing was applied differs
from pair to pair and is small in some of them by design.

One thing worth knowing before you start: the two files in a pair are
not always at exactly the same level. The finished version is level-
matched to the raw one as closely as peak safety allows, but where the
processing raised the peak of an already-full master the finished file
ends up slightly quieter. The largest difference in this set is
{max_level_delta} LU. Louder tends to sound better regardless of whether it
actually is better, so please adjust your volume between files rather
than letting level decide.

Suggested listening: a quiet room, and whatever you would normally judge
a mix on. Headphones will show stereo differences most clearly.

For each pair:

  1. Which version has better tonal balance?
  2. Which has clearer, more separated instruments?
  3. Which has better high-frequency detail?
  4. Which has less harshness or sibilance?
  5. Which has better stereo depth or width?
  6. Which preserves punch and impact better?
  7. Does the processed version sound overprocessed in any way?
  8. Overall, would you choose RAW or FINISHED?

And once you have heard all of them:

  9. Was the processing consistent across pairs, or did it help some
     songs and hurt others?
 10. Is there anything it did that you would want it to stop doing?

FILES
"""

QUESTIONS_TAIL = """
Reply with your answers per pair. Free-form is fine — "3: finished,
clearly brighter, but 5 is worse" is more useful than a score.
"""


def _delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return float(after - before)


def _band_energy(analysis: AudioAnalysis, name: str) -> float | None:
    band = analysis.frequency.band(name)
    return None if band is None else band.energy_db


def waveform_difference(raw: Path, finished: Path) -> dict[str, float | None]:
    """How much signal changed, in dB, without claiming that is quality.

    ``null_depth_db`` is the level of the difference signal relative to
    the source. A large number means the files are nearly identical; it
    says nothing about whether the change was an improvement.
    """
    left = load_audio(raw)
    right = load_audio(finished)
    frames = min(left.frames, right.frames)
    if frames == 0 or left.channels != right.channels:
        return {"null_depth_db": None, "difference_rms_dbfs": None, "frames_compared": 0.0}
    a = left.samples[:frames]
    b = right.samples[:frames]
    difference = b - a
    source_rms = float(np.sqrt(np.mean(a**2)))
    difference_rms = float(np.sqrt(np.mean(difference**2)))
    if difference_rms <= 0.0 or source_rms <= 0.0:
        return {
            "null_depth_db": None,
            "difference_rms_dbfs": None,
            "frames_compared": float(frames),
        }
    return {
        "null_depth_db": float(20.0 * np.log10(source_rms / difference_rms)),
        "difference_rms_dbfs": float(20.0 * np.log10(difference_rms)),
        "frames_compared": float(frames),
    }


def compare(result: FinishingResult) -> dict[str, Any]:
    """Before/after for every metric the plan could have moved."""
    before = result.source_analysis
    after = result.finished_analysis
    if after is None:
        raise ValueError("cannot compare a NO_ACTION result")

    bands = {
        name: {
            "before_db": _band_energy(before, name),
            "after_db": _band_energy(after, name),
            "delta_db": _delta(_band_energy(before, name), _band_energy(after, name)),
        }
        for name in BAND_NAMES
    }
    scalar: dict[str, tuple[float | None, float | None]] = {
        "spectral_centroid_hz": (
            before.frequency.spectral_centroid_hz.p50,
            after.frequency.spectral_centroid_hz.p50,
        ),
        "spectral_slope_db_per_octave": (
            before.frequency.spectral_slope_db_per_octave,
            after.frequency.spectral_slope_db_per_octave,
        ),
        "air_ratio_db": (before.frequency.air_ratio_db.p50, after.frequency.air_ratio_db.p50),
        "low_mid_ratio_db": (
            before.frequency.low_mid_ratio_db.p50,
            after.frequency.low_mid_ratio_db.p50,
        ),
        "presence_ratio_db": (
            before.frequency.presence_ratio_db.p50,
            after.frequency.presence_ratio_db.p50,
        ),
        "sibilance_ratio_p90_db": (
            before.sibilance.sibilance_ratio_db.p90,
            after.sibilance.sibilance_ratio_db.p90,
        ),
        "peak_dbfs": (before.level.peak_dbfs, after.level.peak_dbfs),
        "true_peak_dbfs": (before.loudness.true_peak_dbfs, after.loudness.true_peak_dbfs),
        "rms_dbfs": (before.level.rms_dbfs, after.level.rms_dbfs),
        "crest_factor_db": (before.level.crest_factor_db, after.level.crest_factor_db),
        "integrated_lufs": (before.loudness.integrated_lufs, after.loudness.integrated_lufs),
        "stereo_width": (before.stereo.width, after.stereo.width),
        "stereo_correlation": (before.stereo.correlation, after.stereo.correlation),
        "low_band_correlation": (
            before.stereo.low_band_correlation,
            after.stereo.low_band_correlation,
        ),
        "lr_balance_db": (before.stereo.lr_balance_db, after.stereo.lr_balance_db),
    }
    return {
        "band_energy_db": bands,
        "metrics": {
            name: {"before": pair[0], "after": pair[1], "delta": _delta(pair[0], pair[1])}
            for name, pair in scalar.items()
        },
        "safety": {
            "clipped_samples_before": before.level.clipped_samples,
            "clipped_samples_after": after.level.clipped_samples,
            "duration_before_s": before.technical.duration_seconds,
            "duration_after_s": after.technical.duration_seconds,
            "duration_delta_s": after.technical.duration_seconds
            - before.technical.duration_seconds,
            "sample_rate_before": before.technical.sample_rate,
            "sample_rate_after": after.technical.sample_rate,
            "channels_before": before.technical.channels,
            "channels_after": after.technical.channels,
        },
        "level_stage": {
            "output_gain_db": result.output_gain_db,
            "loudness_match_gain_db": result.loudness_match_gain_db,
            "peak_safety_reduction_db": result.peak_safety_reduction_db,
        },
    }


def run(listening_dir: Path, destination: Path, sources: list[Path]) -> int:
    listening_dir.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    lines: list[str] = []

    for index, source in enumerate(sources, start=1):
        label = source.parent.name if source.name.startswith("master.") else source.stem
        raw_path = listening_dir / f"{index:02d}_RAW.wav"
        finished_path = listening_dir / f"{index:02d}_FINISHED.wav"
        print(f"[{index}/{len(sources)}] finishing {label[:8]} ...", flush=True)

        result = finish_audio(source, finished_path)
        if not result.changed:
            print(f"    skipped: NO_ACTION for {label[:8]}")
            finished_path.unlink(missing_ok=True)
            continue
        shutil.copyfile(source, raw_path)

        comparison = compare(result)
        comparison["difference"] = waveform_difference(raw_path, finished_path)
        records.append(
            {
                "pair": f"{index:02d}",
                "label": label,
                "finishing_version": result.finishing_version,
                "filter_graph": result.filter_graph,
                "plan": plan_to_dict(result.plan),
                "comparison": comparison,
                "source_analysis": analysis_to_dict(result.source_analysis),
            }
        )
        flags = ", ".join(flag.value for flag in result.plan.flags) or "none"
        lines.append(f"  {index:02d}_RAW.wav / {index:02d}_FINISHED.wav   ({label[:8]})")
        print(f"    {flags}")

    (destination / "ab_results.json").write_text(
        json.dumps({"pairs": records}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # The level-difference claim is measured, not asserted: a hardcoded
    # figure would drift the moment a threshold changed.
    deltas = [
        abs(record["comparison"]["metrics"]["integrated_lufs"]["delta"] or 0.0)
        for record in records
    ]
    (listening_dir / "README.txt").write_text(
        README.format(max_level_delta=f"{max(deltas):.1f}" if deltas else "0.0")
        + "\n".join(lines)
        + "\n"
        + QUESTIONS_TAIL,
        encoding="utf-8",
    )
    print(f"\nwrote {len(records)} pairs to {listening_dir} and ab_results.json")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 4:
        print(__doc__)
        return 2
    sources = [Path(value) for value in argv[3:]]
    missing = [item for item in sources if not item.is_file()]
    if missing:
        print(f"missing audio: {', '.join(str(item) for item in missing)}")
        return 2
    return run(Path(argv[1]).expanduser(), Path(argv[2]), sources)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
