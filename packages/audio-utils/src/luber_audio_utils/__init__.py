"""Audio delivery utilities: format contract, transcoding, storage.

Phase 4 scope is *delivery*, not mastering — see
:mod:`luber_audio_utils.transcode` for exactly what is and is not done
to the audio.
"""

from luber_audio_utils.constants import (
    ASSET_FORMAT_CONTRACT,
    MASTER_BIT_DEPTH,
    MASTER_CHANNELS,
    MASTER_FILE_EXTENSION,
    MASTER_FORMAT,
    MASTER_MIME_TYPE,
    MASTER_SAMPLE_RATE,
    PREVIEW_BITRATE_BPS,
    PREVIEW_CHANNELS,
    PREVIEW_FILE_EXTENSION,
    PREVIEW_FORMAT,
    PREVIEW_MIME_TYPE,
    PREVIEW_SAMPLE_RATE,
)
from luber_audio_utils.factory import storage_from_settings
from luber_audio_utils.reference import (
    NormalizedReference,
    ReferenceAudioRejected,
    check_upload_size,
    inspect_upload,
    normalize_reference,
    resolve_upload_format,
    safe_display_name,
)
from luber_audio_utils.s3 import S3AudioStorage, S3StorageConfig
from luber_audio_utils.storage import (
    AudioStorage,
    AudioStorageError,
    DownloadTarget,
    LocalAudioStorage,
    finished_master_storage_key,
    generation_prefix,
    master_storage_key,
    preview_storage_key,
)
from luber_audio_utils.transcode import (
    AudioProbe,
    AudioProcessingError,
    encode_preview_mp3,
    encode_preview_mp3_async,
    probe_audio,
    transcode_master_wav,
    transcode_master_wav_async,
)
from luber_audio_utils.wav import WavInfo, WavValidationError, inspect_wav, sha256_file

__all__ = [
    "ASSET_FORMAT_CONTRACT",
    "MASTER_BIT_DEPTH",
    "MASTER_CHANNELS",
    "MASTER_FILE_EXTENSION",
    "MASTER_FORMAT",
    "MASTER_MIME_TYPE",
    "MASTER_SAMPLE_RATE",
    "PREVIEW_BITRATE_BPS",
    "PREVIEW_CHANNELS",
    "PREVIEW_FILE_EXTENSION",
    "PREVIEW_FORMAT",
    "PREVIEW_MIME_TYPE",
    "PREVIEW_SAMPLE_RATE",
    "AudioProbe",
    "AudioProcessingError",
    "AudioStorage",
    "AudioStorageError",
    "DownloadTarget",
    "LocalAudioStorage",
    "NormalizedReference",
    "ReferenceAudioRejected",
    "S3AudioStorage",
    "S3StorageConfig",
    "WavInfo",
    "WavValidationError",
    "check_upload_size",
    "encode_preview_mp3",
    "encode_preview_mp3_async",
    "finished_master_storage_key",
    "generation_prefix",
    "inspect_upload",
    "inspect_wav",
    "master_storage_key",
    "normalize_reference",
    "preview_storage_key",
    "probe_audio",
    "resolve_upload_format",
    "safe_display_name",
    "sha256_file",
    "storage_from_settings",
    "transcode_master_wav",
    "transcode_master_wav_async",
]
