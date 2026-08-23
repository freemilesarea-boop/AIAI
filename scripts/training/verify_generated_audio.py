#!/usr/bin/env python3
"""Phase 37 — check that generated audio is really audio.

Decodes every file an evaluation manifest points at and reports what it
found: length, peak, RMS, whether anything is non-finite, and whether
the whole file is silence. A generation that "succeeded" and wrote three
minutes of zeros is a failure nobody would notice from the exit code.

Judges nothing about the music. A person still has to listen.

    ~/ace-step-1.5/.venv/bin/python scripts/training/verify_generated_audio.py \
        data/evaluation/exp37/abc/manifest.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

#: Below this peak the file is treated as silence rather than quiet
#: music: -60 dBFS is far under anything a mix would deliver.
SILENCE_PEAK = 10 ** (-60 / 20)


def inspect(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"name": path.name, "bytes": path.stat().st_size}
    try:
        import soundfile as sf  # type: ignore[import-not-found]

        with sf.SoundFile(str(path)) as handle:
            frames, rate, channels = len(handle), handle.samplerate, handle.channels
            data = handle.read(dtype="float32")
    except Exception as exc:
        record.update(decoded=False, error=f"{type(exc).__name__}: {exc}")
        return record

    peak = float(abs(data).max()) if data.size else 0.0
    rms = float(math.sqrt(float((data.astype("float64") ** 2).mean()))) if data.size else 0.0
    finite = bool(data.size) and bool((abs(data) < float("inf")).all())
    record.update(
        decoded=True,
        sample_rate=rate,
        channels=channels,
        duration_seconds=round(frames / rate, 3) if rate else 0.0,
        peak=round(peak, 6),
        rms=round(rms, 6),
        finite=finite,
        silent=peak < SILENCE_PEAK,
        empty=frames == 0,
    )
    record["ok"] = bool(
        record["decoded"]
        and record["duration_seconds"] > 0
        and finite
        and not record["silent"]
        and not record["empty"]
    )
    return record


def main(argv: list[str]) -> int:
    manifest = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    root = Path(argv[1]).parent
    results: list[dict[str, Any]] = []
    for item in manifest["items"]:
        for side_name in item.get("sides", {}):
            directory = root / item["id"] / side_name
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.suffix.lower() in (".flac", ".wav", ".mp3", ".ogg"):
                    record = inspect(path)
                    record.update(item=item["id"], side=side_name)
                    results.append(record)

    bad = [r for r in results if not r.get("ok")]
    print(f"checked {len(results)} file(s); {len(results) - len(bad)} usable, {len(bad)} not")
    for record in bad:
        print(
            f"  FAIL {record['item']}/{record['side']}/{record['name']}: "
            f"{record.get('error') or ('silent' if record.get('silent') else 'unusable')}"
        )
    if results:
        durations = sorted(r["duration_seconds"] for r in results if r.get("decoded"))
        peaks = sorted(r["peak"] for r in results if r.get("decoded"))
        print(f"duration: min {durations[0]}s max {durations[-1]}s")
        print(f"peak    : min {peaks[0]} max {peaks[-1]}")
    (root / "audio_sanity.json").write_text(
        json.dumps({"checked": len(results), "failures": len(bad), "files": results}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"written : {root / 'audio_sanity.json'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
