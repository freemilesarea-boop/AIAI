#!/usr/bin/env python
"""Generate Phase 20 benchmark cases through the real production path.

Deliberately not a shortcut to the provider. Every case goes through
``POST /v1/generations`` exactly as a user's would, so the audio being
scored is audio the product can actually produce — including the queue,
the worker and the storage layer. A benchmark that bypassed those would
measure a system nobody uses.

Sequential by construction: one generation at a time, because the
machine has one GPU and a benchmark that makes the Mac unusable will not
be run twice.

    p20_generate.py --cases TROT-01,KO-01        # named cases
    p20_generate.py --set KO --limit 4           # a whole set, capped
    p20_generate.py --dry-run                    # show what would run

Results are appended to ``p20_runs.jsonl`` as benchmark-id → generation-id
pairs. Nothing is overwritten: re-running a case adds a second row, and
which one counts is decided later by the manifest, not by clobbering.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

BENCHMARK = Path(__file__).resolve().parents[1] / "prompts" / "BENCHMARK_P20.json"
RUNS = Path(__file__).resolve().parents[1] / "results" / "p20_runs.jsonl"
API = "http://127.0.0.1:8000"

#: A 180s case has been observed to take ~110s. Four times the worst
#: observed run is a stall, not a slow generation.
POLL_TIMEOUT_SECONDS = 600
POLL_INTERVAL_SECONDS = 5


def load_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = json.loads(BENCHMARK.read_text())["prompts"]
    return cases


def submit(case: dict[str, Any]) -> str:
    payload = {
        "title": f"P20 {case['prompt_id']}",
        "prompt": case["prompt"],
        "lyrics": case["lyrics"],
        "vocal_gender": case["vocal_gender"],
        "duration": case["duration"],
        "language": case["language"],
        "instrumental": case["vocal_gender"] == "instrumental",
    }
    request = urllib.request.Request(
        f"{API}/v1/generations",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Idempotency-Key": str(uuid.uuid4())},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return str(json.load(response)["generation_id"])


def wait(generation_id: str) -> dict[str, Any]:
    deadline = time.time() + POLL_TIMEOUT_SECONDS
    while time.time() < deadline:
        with urllib.request.urlopen(
            f"{API}/v1/generations/{generation_id}", timeout=30
        ) as response:
            body = json.load(response)
        if body["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
            return dict(body)
        time.sleep(POLL_INTERVAL_SECONDS)
    return {"status": "TIMED_OUT", "id": generation_id}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", help="comma-separated prompt_ids")
    parser.add_argument("--set", dest="set_name", help="GEN | TROT | KO | LONG")
    parser.add_argument("--limit", type=int, help="cap the number of generations")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cases = load_cases()
    if args.cases:
        wanted = {c.strip() for c in args.cases.split(",")}
        cases = [c for c in cases if c["prompt_id"] in wanted]
    if args.set_name:
        cases = [c for c in cases if c["set"] == args.set_name]
    if args.limit:
        cases = cases[: args.limit]

    if not cases:
        print("no cases matched", file=sys.stderr)
        return 1

    if args.dry_run:
        for case in cases:
            print(
                f"  would generate {case['prompt_id']:<9} "
                f"{case['duration']:>3}s  {case['prompt'][:56]}"
            )
        print(f"{len(cases)} case(s); nothing was submitted")
        return 0

    RUNS.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    for index, case in enumerate(cases, start=1):
        started = time.time()
        print(f"[{index}/{len(cases)}] {case['prompt_id']} ({case['duration']}s) …", flush=True)
        try:
            generation_id = submit(case)
        except urllib.error.HTTPError as exc:
            print(f"  submit failed: HTTP {exc.code}", flush=True)
            failures += 1
            continue
        result = wait(generation_id)
        elapsed = round(time.time() - started, 1)
        row = {
            "benchmark_version": "p20",
            "prompt_id": case["prompt_id"],
            "set": case["set"],
            "generation_id": generation_id,
            "status": result["status"],
            "error_code": result.get("error_code"),
            "seed": result.get("seed"),
            "duration_requested": case["duration"],
            "duration_actual": result.get("duration_actual"),
            "provider": result.get("provider"),
            "model_name": result.get("model_name"),
            "model_version": result.get("model_version"),
            "elapsed_seconds": elapsed,
        }
        with RUNS.open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {result['status']} in {elapsed}s  ({generation_id})", flush=True)
        if result["status"] != "COMPLETED":
            failures += 1

    print(f"done: {len(cases) - failures}/{len(cases)} completed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
