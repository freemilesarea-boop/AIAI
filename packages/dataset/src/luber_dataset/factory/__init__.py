"""Production dataset factory: scan, analyse, dedup, gate, split, freeze.

Prepares a deterministic, auditable, resumable dataset so that GPU time
is spent training rather than discovering that a file will not decode.

Three properties the rest of the package is arranged around:

*Source audio is read-only.* Nothing here writes, renames, converts or
tags anything under the scan root, and every run re-hashes its sources
afterwards to prove it.

*Nothing is fabricated.* Where the repository has no detector — vocal
class, language, song structure, transcripts — the record says so and
carries the reason. Tempo and key are computed because they can be, and
verified against synthetic signals built at a known tempo and key.

*UNKNOWN never becomes TRUE.* The rights gate has one job and it is that
one. Unknown provenance can be analysed and can never enter a training
export without an explicit, recorded override.
"""

from luber_dataset.factory.config import (
    FACTORY_VERSION,
    SCHEMA_VERSION,
    DedupThresholds,
    FactoryConfig,
    QualityThresholds,
    SplitConfig,
)
from luber_dataset.factory.pipeline import FactoryResult, run
from luber_dataset.factory.schemas import RejectionRecord, ReviewItem, TrackRecord

__all__ = [
    "FACTORY_VERSION",
    "SCHEMA_VERSION",
    "DedupThresholds",
    "FactoryConfig",
    "FactoryResult",
    "QualityThresholds",
    "RejectionRecord",
    "ReviewItem",
    "SplitConfig",
    "TrackRecord",
    "run",
]
