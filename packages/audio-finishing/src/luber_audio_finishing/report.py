"""The measurement report: everything observed, plus what it implies.

Analysis and risk evaluation are kept apart on purpose — one is
measurement, the other is judgement against thresholds — but almost
every caller wants both, and wants them from the same read of the file.
This is that pairing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from luber_audio_finishing.analysis import (
    AudioAnalysis,
    FrequencyMetrics,
    LevelMetrics,
    SibilanceMetrics,
    SpatialProxies,
    StereoMetrics,
    TechnicalProperties,
    TransientMetrics,
    analyze_audio,
)
from luber_audio_finishing.loudness import LoudnessMeasurement
from luber_audio_finishing.risks import RiskFinding, RiskFlag, evaluate_risks


@dataclass(frozen=True)
class AudioAnalysisReport:
    analysis: AudioAnalysis
    risk_flags: tuple[RiskFinding, ...]

    @property
    def technical(self) -> TechnicalProperties:
        return self.analysis.technical

    @property
    def level(self) -> LevelMetrics:
        return self.analysis.level

    @property
    def loudness(self) -> LoudnessMeasurement:
        return self.analysis.loudness

    @property
    def frequency(self) -> FrequencyMetrics:
        return self.analysis.frequency

    @property
    def sibilance(self) -> SibilanceMetrics:
        return self.analysis.sibilance

    @property
    def transient(self) -> TransientMetrics:
        return self.analysis.transient

    @property
    def stereo(self) -> StereoMetrics:
        return self.analysis.stereo

    @property
    def spatial(self) -> SpatialProxies:
        return self.analysis.spatial

    @property
    def flags(self) -> tuple[RiskFlag, ...]:
        return tuple(finding.flag for finding in self.risk_flags)

    def has(self, flag: RiskFlag) -> bool:
        return flag in self.flags

    def finding(self, flag: RiskFlag) -> RiskFinding | None:
        for item in self.risk_flags:
            if item.flag == flag:
                return item
        return None


def build_report(path: Path, *, measure_r128: bool = True) -> AudioAnalysisReport:
    """Measure a file and evaluate its risks in one pass."""
    analysis = analyze_audio(path, measure_r128=measure_r128)
    return AudioAnalysisReport(analysis=analysis, risk_flags=evaluate_risks(analysis))
