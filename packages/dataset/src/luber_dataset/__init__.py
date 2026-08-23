"""LUBER training dataset tooling.

Rights validation, audio and lyric quality gates, and manifest assembly
for authorized model training. Nothing here downloads, scrapes, or
acquires audio — data is supplied by the operator with documented
rights, and this package decides whether it may be used.
"""

from luber_dataset.discovery import (
    AUDIO_EXTENSIONS,
    DiscoveredFile,
    hypothesize_origin,
    sanitize,
    scan,
    summarize,
)
from luber_dataset.manifest import (
    DatasetManifest,
    Exclusion,
    TrainingRunManifest,
    build_manifest,
    summarize_quality_grade,
    validate_run_manifest,
)
from luber_dataset.pilot_subset import (
    PILOT_SUBSET_MAX,
    PILOT_SUBSET_MIN,
    PilotSubset,
    SubsetError,
    SubsetMember,
    select_pilot_subset,
)
from luber_dataset.quality import (
    AudioQuality,
    LyricsQuality,
    inspect_lyrics,
    inspect_training_audio,
)
from luber_dataset.rights import (
    REFERENCE_ONLY_CLASSES,
    TRAINABLE_CLASSES,
    OriginType,
    RightsBasis,
    RightsError,
    RightsRecord,
    SourceClass,
    TrainingRightsStatus,
    classify,
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
    VocalPresence,
    VocalStyle,
    VocalTimbre,
)

__all__ = [
    "ACCEPTABLE_GRADES",
    "AUDIO_EXTENSIONS",
    "DISCOURAGED_STYLES",
    "PILOT_SUBSET_MAX",
    "PILOT_SUBSET_MIN",
    "REFERENCE_ONLY_CLASSES",
    "TRAINABLE_CLASSES",
    "AudioQuality",
    "DatasetManifest",
    "Delivery",
    "DiscoveredFile",
    "Exclusion",
    "LyricsQuality",
    "OriginType",
    "PilotSubset",
    "PronunciationStyle",
    "QualityGrade",
    "RightsBasis",
    "RightsError",
    "RightsRecord",
    "SourceClass",
    "SubsetError",
    "SubsetMember",
    "TrainingRightsStatus",
    "TrainingRunManifest",
    "TrainingTrack",
    "VibratoAmount",
    "VibratoCharacter",
    "VocalAnnotation",
    "VocalPresence",
    "VocalStyle",
    "VocalTimbre",
    "build_manifest",
    "classify",
    "hypothesize_origin",
    "inspect_lyrics",
    "inspect_training_audio",
    "is_trainable",
    "sanitize",
    "scan",
    "select_pilot_subset",
    "summarize",
    "summarize_quality_grade",
    "validate_rights",
    "validate_run_manifest",
]
