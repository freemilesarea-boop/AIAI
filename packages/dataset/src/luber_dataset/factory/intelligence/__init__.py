"""Dataset intelligence and auto-curation.

Phase 23 answers "is this file usable". This answers the different and
harder question: **given a large eligible dataset, is it a good dataset
to train on?**

It consumes the canonical manifest and nothing else — no rescanning, no
re-decoding, no audio touched. The Phase 23 manifest is immutable input;
every artifact here is derived and written elsewhere.

Three commitments shape the whole layer:

*Every share states its denominator.* A genre distribution built from
10% coverage is a statement about that 10%, and the accessor makes
"unknown" impossible to drop silently.

*A gap requires a declared target.* Without one, "12% Korean" is a
measurement. The default profile declares almost nothing and detects
domination only, because music datasets are not meant to be uniform.

*Rights are a hard gate, never a weight.* Curation happens strictly
after eligibility. There is no score at which unknown provenance
becomes trainable, so provenance is absent from the scoring components
entirely.
"""

from luber_dataset.factory.intelligence.curation import (
    CurationConfig,
    CurationResult,
    curate,
    read_manifest,
)
from luber_dataset.factory.intelligence.drift import DriftReport, compare, render_markdown
from luber_dataset.factory.intelligence.profile import DatasetProfile
from luber_dataset.factory.intelligence.schemas import (
    CURATION_ENGINE_VERSION,
    CURATION_SCHEMA_VERSION,
    CurationAction,
    Finding,
    Observation,
    Severity,
    TrackView,
)
from luber_dataset.factory.intelligence.targets import TargetProfile, by_name, neutral

__all__ = [
    "CURATION_ENGINE_VERSION",
    "CURATION_SCHEMA_VERSION",
    "CurationAction",
    "CurationConfig",
    "CurationResult",
    "DatasetProfile",
    "DriftReport",
    "Finding",
    "Observation",
    "Severity",
    "TargetProfile",
    "TrackView",
    "by_name",
    "compare",
    "curate",
    "neutral",
    "read_manifest",
    "render_markdown",
]
