#!/usr/bin/env python
"""Phase 10 inference DOE against the pinned ACE-Step engine directly.

Talks to the engine rather than through LUBER, for two reasons: the
variable under test is an engine parameter, and the output is then
unambiguously RAW_MODEL_OUTPUT with no LUBER post-processing in the
path. Every run is written to its own file alongside a manifest
recording the exact configuration and the SHA256 of the audio, so a
later processed version can never be mistaken for a model improvement.

Deliberately small. This is a designed experiment, not a sweep: one
variable at a time, fixed seed per prompt, everything else held equal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ENGINE = "http://127.0.0.1:8001"

KOREAN_LYRICS = (
    "[Verse]\n창밖에 비가 내려와\n너의 이름을 불러봐\n"
    "[Chorus]\n다시 만날 그날까지\n나는 여기 있을게\n"
)

PROMPTS: dict[str, dict[str, Any]] = {
    "ko_pop_female": {
        "prompt": "Contemporary Korean pop, warm electric piano, soft drums, female lead vocal",
        "lyrics": KOREAN_LYRICS,
        "vocal_language": "ko",
        "seed": 30100001,
    },
    "instrumental": {
        "prompt": "Warm lo-fi instrumental with electric piano, soft drums and bass",
        "lyrics": "",
        "vocal_language": "en",
        "seed": 30100002,
    },
}


def post(path: str, payload: dict[str, Any]) -> Any:
    request = urllib.request.Request(
        f"{ENGINE}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(request, timeout=120))["data"]


def run_one(
    name: str, spec: dict[str, Any], config: dict[str, Any], out_dir: Path
) -> dict[str, Any]:
    payload = {
        "prompt": spec["prompt"],
        "lyrics": spec["lyrics"],
        "vocal_language": spec["vocal_language"],
        "audio_duration": 60.0,
        "audio_format": "wav",
        "model": "acestep-v15-turbo",
        "batch_size": 1,
        "thinking": False,
        "use_cot_caption": False,
        "use_cot_language": False,
        "use_random_seed": False,
        "seed": spec["seed"],
        "bpm": 92,
        "key_scale": "C major",
        "time_signature": "4",
        **config,
    }
    started = time.time()
    task_id = post("/release_task", payload)["task_id"]
    result = None
    while time.time() - started < 1800:
        items = post("/query_result", {"task_id_list": [task_id]})
        status = items[0]["status"]
        if status == 1:
            result = json.loads(items[0]["result"])
            break
        if status == 2:
            return {
                "case": name,
                "config": config,
                "status": "FAILED",
                "error": items[0].get("result"),
            }
        time.sleep(3)
    wall = time.time() - started
    if result is None:
        return {"case": name, "config": config, "status": "TIMEOUT", "wall_seconds": wall}

    url = f"{ENGINE}{result[0]['file']}"
    audio = urllib.request.urlopen(url, timeout=300).read()
    out = out_dir / f"{name}.wav"
    out.write_bytes(audio)
    return {
        "case": name,
        "config": config,
        "status": "COMPLETED",
        "wall_seconds": round(wall, 1),
        "raw_path": str(out),
        "bytes": len(audio),
        "sha256": hashlib.sha256(audio).hexdigest(),
        "payload": payload,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", default="8,16,32")
    args = parser.parse_args()

    out_dir = Path(args.out) / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = [int(s) for s in args.steps.split(",")]

    records = []
    for prompt_id, spec in PROMPTS.items():
        for n in steps:
            case = f"{prompt_id}__steps{n}"
            print(f"  → {case}", flush=True)
            record = run_one(case, spec, {"inference_steps": n}, out_dir)
            print(f"     {record['status']} {record.get('wall_seconds', '')}s", flush=True)
            records.append(record)

    manifest = Path(args.out) / "manifest.json"
    manifest.write_text(json.dumps({"kind": "RAW_MODEL_OUTPUT", "runs": records}, indent=2))
    print(f"\nwrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
