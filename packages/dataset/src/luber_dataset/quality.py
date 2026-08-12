"""Audio and lyric quality gates for training data.

A file is not training material merely because it decodes. The Phase 5
human evaluation rejected the baseline for harsh highs, sibilance, poor
instrument fidelity, and lyric omissions — so the training set must not
contain audio exhibiting those same properties, or the LoRA would learn
to reproduce them.

Everything here is a *measurement plus a flag*. Nothing is silently
repaired: a track is either good enough to teach from or it is
excluded.
"""

from __future__ import annotations

import itertools
import math
import re
import wave
from dataclasses import dataclass, field
from pathlib import Path

# ── audio quality flags ───────────────────────────────────────────────
CLIPPING = "CLIPPING"
LOSSY_ARTIFACTS = "LOSSY_ARTIFACTS"
EXCESSIVE_NOISE = "EXCESSIVE_NOISE"
BAD_RESAMPLING = "BAD_RESAMPLING"
OVER_COMPRESSED = "OVER_COMPRESSED"
DC_OFFSET = "DC_OFFSET"
TOO_QUIET = "TOO_QUIET"
TOO_SHORT = "TOO_SHORT"
MONO_SOURCE = "MONO_SOURCE"
LOW_SAMPLE_RATE = "LOW_SAMPLE_RATE"
UNREADABLE = "UNREADABLE"

#: Share of samples at full scale that indicates real clipping damage.
CLIPPING_SAMPLE_RATIO = 0.0005
#: Crest factor below this suggests brickwalled, over-limited mastering.
MIN_CREST_FACTOR_DB = 6.0
#: Training material should be at least CD rate.
MIN_SAMPLE_RATE = 44_100
MIN_DURATION_SECONDS = 20.0
MIN_RMS_DBFS = -40.0
MAX_DC_OFFSET = 0.01

# ── lyric QA flags ────────────────────────────────────────────────────
LYRICS_EMPTY = "LYRICS_EMPTY"
LYRICS_NO_SECTIONS = "LYRICS_NO_SECTIONS"
LYRICS_DUPLICATE_LINES = "LYRICS_DUPLICATE_LINES"
LYRICS_ENCODING_CORRUPTION = "LYRICS_ENCODING_CORRUPTION"
LYRICS_SECTION_MISMATCH = "LYRICS_SECTION_MISMATCH"
LYRICS_MISSING_LINES = "LYRICS_MISSING_LINES"
LYRICS_LANGUAGE_MISMATCH = "LYRICS_LANGUAGE_MISMATCH"

_SECTION_RE = re.compile(r"^\[([^\]]+)\]\s*$")
_HANGUL_RE = re.compile(r"[가-힣]")
#: Mojibake signatures from mis-decoded Korean text.
_CORRUPTION_MARKERS = ("�", "ï¿½", "â€", "Ã¬", "Ã«")


@dataclass
class AudioQuality:
    readable: bool
    duration_seconds: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    bit_depth: int | None = None
    peak: float = 0.0
    rms_dbfs: float = -math.inf
    crest_factor_db: float = 0.0
    clipping_sample_ratio: float = 0.0
    dc_offset: float = 0.0
    flags: list[str] = field(default_factory=list)

    @property
    def acceptable(self) -> bool:
        """Anything flagged is rejected — no partial credit."""
        return self.readable and not self.flags


def inspect_training_audio(path: Path) -> AudioQuality:
    """Measure a candidate training file and flag disqualifying defects."""
    if not path.is_file() or path.stat().st_size == 0:
        return AudioQuality(readable=False, flags=[UNREADABLE])

    try:
        with wave.open(str(path), "rb") as w:
            channels = w.getnchannels()
            width = w.getsampwidth()
            rate = w.getframerate()
            frames = w.getnframes()
            raw = w.readframes(frames)
    except Exception:
        # Compressed sources must be decoded to WAV before inspection;
        # this gate deliberately refuses to guess.
        return AudioQuality(readable=False, flags=[UNREADABLE])

    if rate <= 0 or channels <= 0 or frames <= 0:
        return AudioQuality(readable=False, flags=[UNREADABLE])

    full_scale = float(1 << (width * 8 - 1))
    samples = [
        int.from_bytes(raw[i : i + width], "little", signed=True)
        for i in range(0, len(raw) - width + 1, width)
    ]
    if not samples:
        return AudioQuality(readable=False, flags=[UNREADABLE])

    peak_abs = max(max(samples), -min(samples))
    peak = peak_abs / full_scale
    rms = math.sqrt(sum(s * s for s in samples) / len(samples)) / full_scale
    rms_db = 20 * math.log10(rms) if rms > 0 else -math.inf
    peak_db = 20 * math.log10(peak) if peak > 0 else -math.inf
    crest = peak_db - rms_db if math.isfinite(rms_db) and math.isfinite(peak_db) else 0.0
    near_full = full_scale * 0.999
    clip_ratio = sum(1 for s in samples if abs(s) >= near_full) / len(samples)
    dc = (sum(samples) / len(samples)) / full_scale
    duration = frames / rate

    quality = AudioQuality(
        readable=True,
        duration_seconds=duration,
        sample_rate=rate,
        channels=channels,
        bit_depth=width * 8,
        peak=peak,
        rms_dbfs=rms_db,
        crest_factor_db=round(crest, 2),
        clipping_sample_ratio=clip_ratio,
        dc_offset=dc,
    )

    flags: list[str] = []
    if clip_ratio > CLIPPING_SAMPLE_RATIO:
        flags.append(CLIPPING)
    if math.isfinite(rms_db) and crest < MIN_CREST_FACTOR_DB:
        flags.append(OVER_COMPRESSED)
    if rate < MIN_SAMPLE_RATE:
        flags.append(LOW_SAMPLE_RATE)
    if duration < MIN_DURATION_SECONDS:
        flags.append(TOO_SHORT)
    if math.isfinite(rms_db) and rms_db < MIN_RMS_DBFS:
        flags.append(TOO_QUIET)
    if abs(dc) > MAX_DC_OFFSET:
        flags.append(DC_OFFSET)
    if channels < 2:
        flags.append(MONO_SOURCE)
    quality.flags = flags
    return quality


@dataclass
class LyricsQuality:
    line_count: int
    section_count: int
    sections: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def acceptable(self) -> bool:
        return not self.flags


def inspect_lyrics(
    text: str, *, language: str, expected_sections: int | None = None
) -> LyricsQuality:
    """QA a lyrics file.

    Line breaks and section tags are structural training signal, so they
    are checked rather than normalised away. Phase 5's human finding was
    that whole Korean sentences go missing at generation time; the
    training data must at least be internally complete and correctly
    encoded before that can be blamed on the model.
    """
    if not text.strip():
        return LyricsQuality(line_count=0, section_count=0, flags=[LYRICS_EMPTY])

    raw_lines = text.split("\n")
    sections = [m.group(1) for line in raw_lines if (m := _SECTION_RE.match(line.strip()))]
    content = [
        line.strip() for line in raw_lines if line.strip() and not _SECTION_RE.match(line.strip())
    ]

    flags: list[str] = []
    if not sections:
        flags.append(LYRICS_NO_SECTIONS)
    if expected_sections is not None and len(sections) != expected_sections:
        flags.append(LYRICS_SECTION_MISMATCH)

    # Consecutive identical lines usually mean a copy/paste fault rather
    # than an intentional repeat.
    for previous, current in itertools.pairwise(content):
        if previous == current:
            flags.append(LYRICS_DUPLICATE_LINES)
            break

    if any(marker in text for marker in _CORRUPTION_MARKERS):
        flags.append(LYRICS_ENCODING_CORRUPTION)

    if language == "ko" and content and not _HANGUL_RE.search(text):
        flags.append(LYRICS_LANGUAGE_MISMATCH)

    if not content:
        flags.append(LYRICS_MISSING_LINES)

    return LyricsQuality(
        line_count=len(content),
        section_count=len(sections),
        sections=sections,
        flags=flags,
    )
