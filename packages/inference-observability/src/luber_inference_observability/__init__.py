"""Trend analysis over Phase 29's traces: is the system getting worse?

Phase 29 asks whether *this candidate* is technically valid. This asks
whether *the system* is trending worse, and it is a different question
with a different failure mode. Phase 29's mistake would be delivering a
broken song; this one's would be crying wolf until nobody looks — or
staying quiet through a real regression because the numbers were noisy.

Three properties everything here is built around.

**Counts travel with rates.** "2.86%" is not a fact. "12 of 420" is.

**Nothing happened is not nothing failed.** An empty window is NO_DATA,
a small one is INSUFFICIENT_DATA, and neither is NORMAL.

**Detection is arguable.** Every finding is a comparison of counted
values against written thresholds, and carries both windows, both
sample sizes and the threshold it crossed. There is no learned normal
and no anomaly score, because a verdict nobody can check is one nobody
will act on.

It detects. It changes nothing: no provider is disabled, no threshold
moved, no policy switched. The output of the worst incident this system
can raise is a sentence and the evidence for it.
"""

from luber_inference_observability.aggregation import (
    Aggregate,
    Average,
    Counters,
    Distribution,
    Metric,
    MetricStatus,
    Rate,
    aggregate,
    count,
    group,
)
from luber_inference_observability.baselines import (
    Baseline,
    BaselineKind,
    BaselineStatus,
    baselines_from,
    rolling_window,
)
from luber_inference_observability.dimensions import (
    UNKNOWN,
    Dimension,
    GroupingTooWide,
    Segment,
    TaskType,
    duration_bucket,
    task_type,
)
from luber_inference_observability.events import (
    DataQuality,
    InferenceObservation,
    observe,
    validate,
)
from luber_inference_observability.incidents import (
    Alert,
    AlertReason,
    IncidentLedger,
    IncidentPolicy,
    IncidentStatus,
    InferenceIncident,
    alerts_for,
    fingerprint,
)
from luber_inference_observability.markers import Marker, MarkerKind, derive
from luber_inference_observability.queries import (
    build_baselines,
    compare_revisions,
    compare_windows,
    evaluate,
    evaluate_segments,
    run_detection,
    summary,
    top_segments,
    trend,
)
from luber_inference_observability.regressions import (
    DEFAULT_RULES,
    Category,
    FindingType,
    Recommendation,
    RegressionFinding,
    Rule,
    Severity,
    Status,
    Thresholds,
    detect,
    regressions,
)
from luber_inference_observability.reports import health_report, render_markdown
from luber_inference_observability.storage import (
    InMemoryObservationStore,
    ObservationStore,
    verify,
)
from luber_inference_observability.versions import (
    AGGREGATION_VERSION,
    INCIDENT_POLICY_VERSION,
    OBSERVABILITY_SCHEMA_VERSION,
    PHASE29_BOUNDARY_COMMIT,
    REGRESSION_ENGINE_VERSION,
    version_block,
)
from luber_inference_observability.windows import TimeWindow, WindowSize

__all__ = [
    "AGGREGATION_VERSION",
    "DEFAULT_RULES",
    "INCIDENT_POLICY_VERSION",
    "OBSERVABILITY_SCHEMA_VERSION",
    "PHASE29_BOUNDARY_COMMIT",
    "REGRESSION_ENGINE_VERSION",
    "UNKNOWN",
    "Aggregate",
    "Alert",
    "AlertReason",
    "Average",
    "Baseline",
    "BaselineKind",
    "BaselineStatus",
    "Category",
    "Counters",
    "DataQuality",
    "Dimension",
    "Distribution",
    "FindingType",
    "GroupingTooWide",
    "InMemoryObservationStore",
    "IncidentLedger",
    "IncidentPolicy",
    "IncidentStatus",
    "InferenceIncident",
    "InferenceObservation",
    "Marker",
    "MarkerKind",
    "Metric",
    "MetricStatus",
    "ObservationStore",
    "Rate",
    "Recommendation",
    "RegressionFinding",
    "Rule",
    "Segment",
    "Severity",
    "Status",
    "TaskType",
    "Thresholds",
    "TimeWindow",
    "WindowSize",
    "aggregate",
    "alerts_for",
    "baselines_from",
    "build_baselines",
    "compare_revisions",
    "compare_windows",
    "count",
    "derive",
    "detect",
    "duration_bucket",
    "evaluate",
    "evaluate_segments",
    "fingerprint",
    "group",
    "health_report",
    "observe",
    "regressions",
    "render_markdown",
    "rolling_window",
    "run_detection",
    "summary",
    "task_type",
    "top_segments",
    "trend",
    "validate",
    "verify",
    "version_block",
]
