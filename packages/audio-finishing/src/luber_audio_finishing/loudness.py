"""EBU R128 loudness and true peak, measured by ffmpeg.

Implementing R128 by hand would mean reimplementing the K-weighting
filter, the 400 ms gated blocks and the 4x-oversampled true-peak
detector, and then trusting it. ffmpeg's ``ebur128`` is already a
required part of the delivery pipeline and is the reference
implementation everyone else compares against, so it is used instead.

Every field is optional. A build without ``ebur128``, an unreadable
summary, or audio too short for a single 400 ms block all produce
``None`` rather than a fabricated number, and callers must treat missing
loudness as "unknown", never as "quiet".
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Levels below this are ffmpeg's "-inf" placeholder for silence.
_SILENCE_SENTINEL_LUFS = -70.0

_INTEGRATED = re.compile(r"^\s*I:\s*(-?\d+(?:\.\d+)?)\s*LUFS", re.MULTILINE)
_RANGE = re.compile(r"^\s*LRA:\s*(-?\d+(?:\.\d+)?)\s*LU", re.MULTILINE)
_TRUE_PEAK = re.compile(r"^\s*Peak:\s*(-?\d+(?:\.\d+)?)\s*dBFS", re.MULTILINE)
_SHORT_TERM = re.compile(r"\bS:\s*(-?\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class LoudnessMeasurement:
    """R128 results. Any field may be ``None`` — see the module docstring."""

    integrated_lufs: float | None
    loudness_range_lu: float | None
    true_peak_dbfs: float | None
    short_term_p10_lufs: float | None
    short_term_p50_lufs: float | None
    short_term_p90_lufs: float | None

    @property
    def is_measured(self) -> bool:
        return self.integrated_lufs is not None


UNMEASURED = LoudnessMeasurement(
    integrated_lufs=None,
    loudness_range_lu=None,
    true_peak_dbfs=None,
    short_term_p10_lufs=None,
    short_term_p50_lufs=None,
    short_term_p90_lufs=None,
)


def _last_float(
    pattern: re.Pattern[str], text: str, *, drop_silence_sentinel: bool = False
) -> float | None:
    matches = pattern.findall(text)
    if not matches:
        return None
    try:
        value = float(matches[-1])
    except ValueError:
        return None
    # Only loudness figures carry the -70 LUFS "effectively silent"
    # placeholder. A loudness range of 0 LU and a true peak of -70 dBFS
    # are both real measurements and must survive.
    if drop_silence_sentinel and value <= _SILENCE_SENTINEL_LUFS:
        return None
    return value


def measure_loudness(path: Path) -> LoudnessMeasurement:
    """Run ``ebur128`` over a file and parse its report.

    Never raises: an unavailable or uncooperative ffmpeg yields
    :data:`UNMEASURED`, because loudness is one input to finishing and
    not a reason to refuse to analyse a file at all.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or not path.is_file():
        return UNMEASURED

    try:
        result = subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-v",
                "info",
                "-i",
                str(path),
                "-map",
                "a:0",
                "-filter:a",
                "ebur128=peak=true",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return UNMEASURED

    # ebur128 writes its running log and its summary to stderr.
    report = result.stderr or ""
    if "Summary" not in report:
        return UNMEASURED

    head, _, summary = report.rpartition("Summary")
    short_terms = [
        value
        for value in (float(raw) for raw in _SHORT_TERM.findall(head))
        if value > _SILENCE_SENTINEL_LUFS
    ]
    p10: float | None = None
    p50: float | None = None
    p90: float | None = None
    if short_terms:
        percentiles = np.percentile(np.asarray(short_terms, dtype=np.float64), [10, 50, 90])
        p10, p50, p90 = (float(value) for value in percentiles)

    return LoudnessMeasurement(
        integrated_lufs=_last_float(_INTEGRATED, summary, drop_silence_sentinel=True),
        loudness_range_lu=_last_float(_RANGE, summary),
        true_peak_dbfs=_last_float(_TRUE_PEAK, summary),
        short_term_p10_lufs=p10,
        short_term_p50_lufs=p50,
        short_term_p90_lufs=p90,
    )
