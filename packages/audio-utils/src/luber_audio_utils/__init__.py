"""Audio constants shared by workers and validators.

Phase 0 pins the master-format contract only; decoding, validation and
encoding pipelines are implemented in Phase 4 (production audio
pipeline). Post-processing must stay conservative — no aggressive
limiting, normalization, EQ, or stereo widening in the MVP.
"""

MASTER_SAMPLE_RATE = 48_000
MASTER_BIT_DEPTH = 24
MASTER_CHANNELS = 2
MASTER_FORMAT = "wav"
PREVIEW_FORMAT = "mp3"

__all__ = [
    "MASTER_BIT_DEPTH",
    "MASTER_CHANNELS",
    "MASTER_FORMAT",
    "MASTER_SAMPLE_RATE",
    "PREVIEW_FORMAT",
]
