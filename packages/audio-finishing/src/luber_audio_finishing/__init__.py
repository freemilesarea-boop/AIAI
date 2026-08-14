"""LUBER generative audio finishing: measure, decide, render, verify.

The engine corrects presentation defects that DSP can actually correct —
tonal imbalance, thickness, harsh peaks, stereo instability, level
inconsistency — and does not attempt to improve the music. Melody,
harmony, arrangement, instrument identity and vocal identity are
properties of the generation and remain untouched; a model-quality
problem stays a model-quality problem.

The raw generation master is never overwritten. Finishing reads it and
writes elsewhere, so the source stays available for A/B comparison,
future model evaluation, and reprocessing under a later engine version.
"""

from luber_audio_finishing.analysis import (
    AudioAnalysis,
    BandMeasurement,
    Distribution,
    FrequencyMetrics,
    LevelMetrics,
    SibilanceMetrics,
    SpatialProxies,
    StereoMetrics,
    TechnicalProperties,
    TransientMetrics,
    analyze_audio,
)
from luber_audio_finishing.audiofile import AudioLoadError, LoadedAudio, load_audio
from luber_audio_finishing.bands import BAND_EDGES, BAND_NAMES, BandCoverage, band_coverage
from luber_audio_finishing.decision import (
    ActionKind,
    DeferredDecision,
    FinishingAction,
    FinishingDecisionEngine,
    FinishingPlan,
)
from luber_audio_finishing.loudness import LoudnessMeasurement, measure_loudness
from luber_audio_finishing.processor import (
    AlreadyFinishedError,
    FinishingError,
    FinishingResult,
    build_filter_graph,
    finish_audio,
    read_finishing_stamp,
)
from luber_audio_finishing.report import AudioAnalysisReport, build_report
from luber_audio_finishing.risks import RiskFinding, RiskFlag, evaluate_risks
from luber_audio_finishing.serialize import analysis_to_dict, plan_to_dict, report_to_dict
from luber_audio_finishing.version import FINISHING_VERSION, finishing_stamp

__all__ = [
    "BAND_EDGES",
    "BAND_NAMES",
    "FINISHING_VERSION",
    "ActionKind",
    "AlreadyFinishedError",
    "AudioAnalysis",
    "AudioAnalysisReport",
    "AudioLoadError",
    "BandCoverage",
    "BandMeasurement",
    "DeferredDecision",
    "Distribution",
    "FinishingAction",
    "FinishingDecisionEngine",
    "FinishingError",
    "FinishingPlan",
    "FinishingResult",
    "FrequencyMetrics",
    "LevelMetrics",
    "LoadedAudio",
    "LoudnessMeasurement",
    "RiskFinding",
    "RiskFlag",
    "SibilanceMetrics",
    "SpatialProxies",
    "StereoMetrics",
    "TechnicalProperties",
    "TransientMetrics",
    "analysis_to_dict",
    "analyze_audio",
    "band_coverage",
    "build_filter_graph",
    "build_report",
    "evaluate_risks",
    "finish_audio",
    "finishing_stamp",
    "load_audio",
    "measure_loudness",
    "plan_to_dict",
    "read_finishing_stamp",
    "report_to_dict",
]
