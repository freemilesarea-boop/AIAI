"""Delivery-format contract for audio assets.

These values define what users receive. They are referenced by the
transcoder, the asset records written to the database, and the tests
that verify delivered bytes — so changing one changes the product
contract, not just a default.
"""

from __future__ import annotations

# ── Production master (the authoritative deliverable) ─────────────────
MASTER_SAMPLE_RATE = 48_000
MASTER_BIT_DEPTH = 24
MASTER_CHANNELS = 2
MASTER_FORMAT = "wav"
MASTER_MIME_TYPE = "audio/wav"
MASTER_FILE_EXTENSION = "wav"

# ── Preview (fast browser playback; never replaces the master) ────────
PREVIEW_FORMAT = "mp3"
PREVIEW_SAMPLE_RATE = 48_000
PREVIEW_CHANNELS = 2
PREVIEW_BITRATE_BPS = 320_000
PREVIEW_MIME_TYPE = "audio/mpeg"
PREVIEW_FILE_EXTENSION = "mp3"

#: Allowed (format, mime type, extension) triples. Serving code checks
#: against this so a stored asset can never be delivered under a
#: mismatched content type or extension.
ASSET_FORMAT_CONTRACT: dict[str, tuple[str, str]] = {
    MASTER_FORMAT: (MASTER_MIME_TYPE, MASTER_FILE_EXTENSION),
    PREVIEW_FORMAT: (PREVIEW_MIME_TYPE, PREVIEW_FILE_EXTENSION),
}
