"""Generate the deterministic mock-generation WAV fixture.

Produces a 2-second, 48kHz, stereo, 16-bit PCM audio clip containing an
audible three-note arpeggio (not silence). Uses only the Python standard
library so the fixture is reproducible anywhere:

    uv run python scripts/development/make_mock_fixture.py

The output is committed at tests/fixtures/mock_generation.wav and is what
MockGenerationProvider returns. It is a test fixture — never presented as
real AI-generated music.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

SAMPLE_RATE = 48_000
DURATION_SECONDS = 2.0
CHANNELS = 2
SAMPLE_WIDTH_BYTES = 2  # 16-bit PCM
AMPLITUDE = 0.4
NOTES_HZ = (440.0, 554.37, 659.25)  # A4, C#5, E5 — an A-major arpeggio

OUTPUT = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "mock_generation.wav"


def synthesize() -> bytes:
    total_frames = int(SAMPLE_RATE * DURATION_SECONDS)
    frames_per_note = total_frames // len(NOTES_HZ)
    samples = bytearray()
    for i in range(total_frames):
        note = NOTES_HZ[min(i // frames_per_note, len(NOTES_HZ) - 1)]
        # Short attack/release envelope per note to avoid clicks.
        pos = i % frames_per_note
        env = min(1.0, pos / 480, max(0.0, (frames_per_note - pos) / 480))
        value = AMPLITUDE * env * math.sin(2 * math.pi * note * i / SAMPLE_RATE)
        pcm = int(value * 32767)
        frame = struct.pack("<h", pcm)
        samples += frame * CHANNELS  # duplicate to both stereo channels
    return bytes(samples)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(OUTPUT), "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH_BYTES)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(synthesize())
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")  # noqa: T201


if __name__ == "__main__":
    main()
