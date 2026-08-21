"""One decode per candidate, and everything measured from it.

The expensive part of QC is decoding a multi-minute WAV and running an
STFT over it. Doing that twice — once to check the audio and once for
finishing — would double the cost of the healthy path, which is the path
almost every request takes.

So: decode once, measure everything, cache by ``(sha256, versions)``.

The cache key is the content hash, never the filename. Two runs of the
same request write to two temporary paths; a filename-keyed cache would
miss both times, and a path that was reused across runs would hit with
the wrong file. The analysis version is in the key because a measurement
taken by a different version of the analyser is a different measurement,
and serving it from cache would silently mix them.

BPM and key come from the Phase 23 estimators. They are not
re-implemented: two answers to "what is the BPM of this file" would
eventually disagree, and the one that mattered would be whichever module
the caller happened to import.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from luber_audio_finishing import (
    FINISHING_VERSION,
    AudioAnalysis,
    AudioLoadError,
    LoadedAudio,
    RiskFinding,
    analyze_audio,
    evaluate_risks,
    load_audio,
)
from luber_dataset.factory.musical import estimate_key, estimate_tempo
from luber_inference_qc.collapse import CollapseMeasurement, measure_collapse
from luber_inference_qc.detectors import (
    NullVocalDetector,
    VocalAssessment,
    VocalPresenceDetector,
)
from luber_inference_qc.versions import QC_ENGINE_VERSION

#: How many measurements to keep. A worker handles one generation at a
#: time and a generation makes at most a handful of candidates, so this
#: only has to survive the run it was taken during.
CACHE_SIZE = 16


class MeasurementError(RuntimeError):
    """The file could not be measured at all.

    Distinct from "measured and found broken": this means the decode
    failed, so nothing downstream has anything to judge.
    """


@dataclass(frozen=True)
class CandidateMeasurement:
    """Everything QC knows about one file, from one decode."""

    path: Path
    sha256: str
    analysis: AudioAnalysis
    risks: tuple[RiskFinding, ...]
    collapse: CollapseMeasurement
    vocal: VocalAssessment

    #: ``None`` when the estimator's confidence was below the Phase 23
    #: threshold. Never a guess recorded as a fact.
    bpm: float | None = None
    bpm_confidence: float | None = None
    key: str | None = None
    key_mode: str | None = None
    key_confidence: float | None = None

    #: Non-finite samples make every other number here meaningless, so
    #: it is measured first and reported separately.
    non_finite_samples: int = 0

    analysis_version: str = FINISHING_VERSION
    qc_engine_version: str = QC_ENGINE_VERSION

    @property
    def duration_seconds(self) -> float:
        return self.analysis.technical.duration_seconds

    def to_dict(self) -> dict[str, Any]:
        """A summary for the trace. Not the whole analysis.

        The full `AudioAnalysis` is hundreds of numbers and belongs in a
        finishing report, not in a per-candidate record that has to stay
        readable. What is kept is what a finding cites.
        """
        level = self.analysis.level
        stereo = self.analysis.stereo
        frequency = self.analysis.frequency
        return {
            "sha256": self.sha256,
            "analysis_version": self.analysis_version,
            "duration_seconds": round(self.duration_seconds, 3),
            "sample_rate": self.analysis.technical.sample_rate,
            "channels": self.analysis.technical.channels,
            "peak_dbfs": round(level.peak_dbfs, 2),
            "rms_dbfs": round(level.rms_dbfs, 2),
            "silence_ratio": round(level.silence_ratio, 4),
            "clipped_samples": level.clipped_samples,
            "dc_offset": round(level.dc_offset, 6),
            "correlation": (
                round(stereo.correlation, 4) if stereo.correlation is not None else None
            ),
            "width": round(stereo.width, 4) if stereo.width is not None else None,
            "spectral_slope_db_per_octave": round(frequency.spectral_slope_db_per_octave, 3),
            "collapse": self.collapse.to_dict(),
            "vocal": self.vocal.to_dict(),
            "bpm": self.bpm,
            "bpm_confidence": self.bpm_confidence,
            "key": self.key,
            "key_mode": self.key_mode,
            "key_confidence": self.key_confidence,
            "non_finite_samples": self.non_finite_samples,
            "risks": [item.flag.value for item in self.risks],
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_non_finite(audio: LoadedAudio) -> int:
    import numpy as np

    return int(np.count_nonzero(~np.isfinite(audio.samples)))


class MeasurementCache:
    """Measurements by content hash, bounded.

    Not a global: a cache that outlived the run it was taken during
    would eventually serve one generation's analysis to another, and the
    only thing making that safe is the hash — which is exactly the check
    a bug would remove.
    """

    def __init__(self, size: int = CACHE_SIZE) -> None:
        self._size = size
        self._entries: OrderedDict[tuple[str, str, str], CandidateMeasurement] = OrderedDict()

    def _key(self, sha256: str) -> tuple[str, str, str]:
        return (sha256, FINISHING_VERSION, QC_ENGINE_VERSION)

    def get(self, sha256: str) -> CandidateMeasurement | None:
        key = self._key(sha256)
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
        return entry

    def put(self, measurement: CandidateMeasurement) -> None:
        key = self._key(measurement.sha256)
        self._entries[key] = measurement
        self._entries.move_to_end(key)
        while len(self._entries) > self._size:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)


def measure(
    path: Path,
    *,
    detector: VocalPresenceDetector | None = None,
    cache: MeasurementCache | None = None,
    sha256: str | None = None,
    measure_musical: bool = True,
) -> CandidateMeasurement:
    """Decode once and measure everything QC needs.

    ``measure_musical`` exists because tempo and key estimation cost a
    second pass over the samples and are only consulted when the request
    asked for a BPM or a key. Skipping them yields ``None`` rather than
    an estimate, which is the same distinction the estimators themselves
    make.
    """
    digest = sha256 or _sha256(path)
    if cache is not None:
        cached = cache.get(digest)
        if cached is not None:
            return cached

    try:
        audio = load_audio(path)
    except AudioLoadError as exc:
        raise MeasurementError(str(exc)) from exc
    except Exception as exc:  # a malformed container can fail in many ways
        raise MeasurementError(f"could not decode {path.name}: {exc}") from exc

    if audio.frames == 0:
        raise MeasurementError(f"{path.name} decoded to zero frames")

    non_finite = _count_non_finite(audio)

    try:
        analysis = analyze_audio(path)
    except Exception as exc:
        raise MeasurementError(f"could not analyse {path.name}: {exc}") from exc

    bpm = bpm_confidence = None
    key = key_mode = None
    key_confidence = None
    if measure_musical and non_finite == 0:
        mono = audio.mono()
        bpm, bpm_confidence = estimate_tempo(mono, audio.sample_rate)
        key, key_mode, key_confidence = estimate_key(mono, audio.sample_rate)

    measurement = CandidateMeasurement(
        path=path,
        sha256=digest,
        analysis=analysis,
        risks=tuple(evaluate_risks(analysis)),
        collapse=measure_collapse(audio),
        vocal=(detector or NullVocalDetector()).assess(audio),
        bpm=bpm,
        bpm_confidence=bpm_confidence,
        key=key,
        key_mode=key_mode,
        key_confidence=key_confidence,
        non_finite_samples=non_finite,
    )
    if cache is not None:
        cache.put(measurement)
    return measurement


__all__ = [
    "CACHE_SIZE",
    "CandidateMeasurement",
    "MeasurementCache",
    "MeasurementError",
    "measure",
]
