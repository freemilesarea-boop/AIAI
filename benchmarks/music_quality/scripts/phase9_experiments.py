#!/usr/bin/env python
"""Phase 9 prompt- and lyric-conditioning experiments.

Two small, bounded A/B experiments against the real engine:

**Vocal conditioning** (3 prompts x 2 variants). Variant A is the prompt
as a user would write it; variant B appends explicit contemporary-vocal
conditioning. The question is narrow and deliberately modest: *does
prompt conditioning move the vocal at all?* Not "is B better" — that is
a listening judgement, and this script does not make it.

**Lyric formatting** (3 lyric sets x 2 variants). Variant A is the
lyrics as written; variant B carries the **same words** re-broken into
shorter lines with cleaner section segmentation. Semantic content is held
equal on purpose: the test is whether line *shape* changes omission
behaviour, not whether different words do.

Every pair shares a fixed seed, duration, BPM, key and time signature, so
the manipulated variable is the only thing that differs.

What this script produces: audio, objective measurements, and a
listening package. What it cannot produce: whether the vocal actually
sounds more contemporary, or whether fewer lines were skipped. Those
require the human listening pass this package exists to feed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "music_quality" / "scripts"))

from bench.longform import analyze_long_form, verify_controls  # noqa: E402

#: Conditioning phrases for variant B. Concepts only — no living artist
#: is referenced, by name or by paraphrase.
CONTEMPORARY_VOCAL_CONDITIONING = (
    "contemporary Korean pop vocal, restrained vibrato, clean modern phrasing, "
    "natural conversational pronunciation, modern R&B phrasing"
)

VOCAL_PROMPTS = [
    ("ballad", "Korean pop ballad with warm electric piano and soft drums"),
    ("rnb", "Korean R&B with mellow electric keys, subtle bass and brushed drums"),
    ("citypop", "Korean city pop with clean electric guitar, warm synth pads and steady drums"),
]

#: (id, variant A lyrics, variant B lyrics) — same words, different line shape.
LYRIC_SETS = [
    (
        "long_lines",
        "[Verse]\n창밖에 비가 내려와 너의 이름을 불러봐\n"
        "흐릿한 유리창 너머 지난 여름이 스쳐가\n"
        "[Chorus]\n다시 만날 그날까지 나는 여기 있을게\n",
        "[Verse]\n창밖에 비가 내려와\n너의 이름을 불러봐\n"
        "흐릿한 유리창 너머\n지난 여름이 스쳐가\n"
        "[Chorus]\n다시 만날 그날까지\n나는 여기 있을게\n",
    ),
    (
        "dense_verse",
        "[Verse]\n조용한 방에 앉아서 지난 계절을 세어봐 낡은 사진 한 장에 우리 웃음이 남아서\n"
        "[Chorus]\n바람이 불어와도 이 자리를 지킬게\n",
        "[Verse 1]\n조용한 방에 앉아서\n지난 계절을 세어봐\n"
        "[Verse 2]\n낡은 사진 한 장에\n우리 웃음이 남아서\n"
        "[Chorus]\n바람이 불어와도\n이 자리를 지킬게\n",
    ),
    (
        "untagged",
        "어둠이 걷히면 새벽이 올 거야 우리 다시 만나면 아무 말 없이 웃자\n",
        "[Verse]\n어둠이 걷히면\n새벽이 올 거야\n[Chorus]\n우리 다시 만나면\n아무 말 없이 웃자\n",
    ),
]

BASE = {
    "vocal_gender": "female",
    "language": "ko",
    "duration": 30,
    "bpm": 90,
    "key_scale": "C major",
    "time_signature": "4",
    "seed": 20260913,
}


def submit(api: str, payload: dict[str, Any]) -> str:
    request = urllib.request.Request(
        f"{api}/v1/generations",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Idempotency-Key": payload["title"]},
    )
    created: dict[str, Any] = json.load(urllib.request.urlopen(request))
    return str(created["generation_id"])


def wait(api: str, generation_id: str, timeout: int = 900) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body: dict[str, Any] = json.load(
            urllib.request.urlopen(f"{api}/v1/generations/{generation_id}")
        )
        if body["status"] in ("COMPLETED", "FAILED"):
            return body
        time.sleep(5)
    raise TimeoutError(f"generation {generation_id} did not finish in {timeout}s")


def run_case(api: str, store: Path, case_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    print(f"  → {case_id}", flush=True)
    generation = wait(api, submit(api, payload))
    record: dict[str, object] = {
        "case": case_id,
        "generation_id": generation["id"],
        "status": generation["status"],
        "prompt": generation["prompt"],
        "lyrics": generation["lyrics"],
        "duration_requested": generation["duration_requested"],
        "duration_actual": generation["duration_actual"],
        "seed": generation["seed"],
        "advisories": [a["code"] for a in generation["advisories"]],
    }
    if generation["status"] != "COMPLETED":
        record["error"] = generation["error_code"]
        return record

    started = datetime.fromisoformat(generation["started_at"])
    completed = datetime.fromisoformat(generation["completed_at"])
    record["wall_seconds"] = round((completed - started).total_seconds(), 1)

    master = next(a for a in generation["audio_assets"] if a["asset_type"] == "MASTER")
    path = store / master["storage_key"]
    record["master_path"] = str(path)
    preview = next(a for a in generation["audio_assets"] if a["asset_type"] == "PREVIEW")
    record["preview_path"] = str(store / preview["storage_key"])
    if path.is_file():
        record["analysis"] = analyze_long_form(path).to_dict()
        record["controls"] = verify_controls(
            path,
            requested_bpm=generation["bpm"],
            requested_key=generation["key_scale"],
            requested_time_signature=generation["time_signature"],
        )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8010")
    parser.add_argument("--store", required=True, help="audio storage root")
    parser.add_argument("--out", required=True, help="results JSON path")
    parser.add_argument("--experiment", choices=("vocal", "lyrics", "both"), default="both")
    args = parser.parse_args()

    store = Path(args.store)
    results: dict[str, list[dict[str, Any]]] = {"vocal_conditioning": [], "lyric_formatting": []}

    if args.experiment in ("vocal", "both"):
        print("VOCAL CONDITIONING (3 prompts x 2 variants)")
        lyrics = LYRIC_SETS[0][2]
        for name, prompt in VOCAL_PROMPTS:
            for variant, text in (
                ("A_current", prompt),
                ("B_contemporary", f"{prompt}, {CONTEMPORARY_VOCAL_CONDITIONING}"),
            ):
                case = f"vocal_{name}_{variant}"
                results["vocal_conditioning"].append(
                    run_case(
                        args.api,
                        store,
                        case,
                        {**BASE, "title": f"P9 {case}", "prompt": text, "lyrics": lyrics},
                    )
                )

    if args.experiment in ("lyrics", "both"):
        print("LYRIC FORMATTING (3 sets x 2 variants)")
        prompt = VOCAL_PROMPTS[0][1]
        for name, variant_a, variant_b in LYRIC_SETS:
            for variant, lyrics in (("A_asis", variant_a), ("B_shortlines", variant_b)):
                case = f"lyrics_{name}_{variant}"
                results["lyric_formatting"].append(
                    run_case(
                        args.api,
                        store,
                        case,
                        {**BASE, "title": f"P9 {case}", "prompt": prompt, "lyrics": lyrics},
                    )
                )

    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
