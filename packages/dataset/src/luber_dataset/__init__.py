"""LUBER training dataset tooling.

Rights validation, audio and lyric quality gates, and manifest assembly
for authorized model training. Nothing here downloads, scrapes, or
acquires audio — data is supplied by the operator with documented
rights, and this package decides whether it may be used.
"""

from luber_dataset.manifest import (
    DatasetManifest,
    Exclusion,
    TrainingRunManifest,
    build_manifest,
    summarize_quality_grade,
    validate_run_manifest,
)
from luber_dataset.quality import (
    AudioQuality,
    LyricsQuality,
    inspect_lyrics,
    inspect_training_audio,
)
from luber_dataset.rights import (
    ACCEPTABLE_STATUSES,
    RightsError,
    RightsRecord,
    RightsStatus,
    is_trainable,
    validate_rights,
)
from luber_dataset.schema import (
    ACCEPTABLE_GRADES,
    DISCOURAGED_STYLES,
    Delivery,
    PronunciationStyle,
    QualityGrade,
    TrainingTrack,
    VibratoAmount,
    VibratoCharacter,
    VocalAnnotation,
    VocalStyle,
)

__all__ = [
    "ACCEPTABLE_GRADES",
    "ACCEPTABLE_STATUSES",
    "DISCOURAGED_STYLES",
    "AudioQuality",
    "DatasetManifest",
    "Delivery",
    "Exclusion",
    "LyricsQuality",
    "PronunciationStyle",
    "QualityGrade",
    "RightsError",
    "RightsRecord",
    "RightsStatus",
    "TrainingRunManifest",
    "TrainingTrack",
    "VibratoAmount",
    "VibratoCharacter",
    "VocalAnnotation",
    "VocalStyle",
    "build_manifest",
    "inspect_lyrics",
    "inspect_training_audio",
    "is_trainable",
    "summarize_quality_grade",
    "validate_rights",
    "validate_run_manifest",
]
