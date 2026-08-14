"""Phase 13E calibration: what does ACE-Step's reference audio control?

Talks to the pinned ACE-Step server directly. LUBER has no reference-audio
endpoint and deliberately does not gain one in this phase — the point is
to characterise the engine before deciding whether a product feature is
warranted at all.

The design is a controlled comparison, not a benchmark. One fixed request
is the control; each run changes exactly one thing, so any difference has
a single candidate cause:

    00  prompt only, no reference          the baseline
    01  + reference A (electronic)         does a reference change anything?
    02  + reference B (acoustic)           does *which* reference matter?
    03  contradictory prompt + reference A does the prompt still win?
    04  contradictory prompt, no reference isolates the prompt's own effect
    05  reference A, different seed        is the effect bigger than noise?

Run 04 exists because without it run 03 is uninterpretable: a difference
there could come from the new prompt rather than from the reference.

There is no strength sweep. Source inspection found no scale, weight or
dropout on the reference stream — it is binary, a real reference or the
silence latent — so any "influence level" would be invented.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import httpx

ENGINE = "http://127.0.0.1:8001"
ENGINE_COMMIT = "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0"
MODEL = "acestep-v15-turbo"

#: Held constant everywhere except where a run's name says otherwise.
LYRICS = "[Verse]\n오늘도 너를 기다려\n[Chorus]\n다시 만날 그날까지"
SEED = 777777
ALT_SEED = 131313
DURATION = 30.0

PROMPT_BASE = "Warm Korean indie pop with electric piano and soft drums"
#: Deliberately at odds with reference A's aggressive electronic character.
PROMPT_CONTRADICTORY = (
    "Sparse quiet acoustic folk ballad, fingerpicked nylon guitar, no drums, no synths"
)

BASE_PAYLOAD: dict[str, object] = {
    # Reference audio conditions an otherwise ordinary text-to-music run.
    # This is not cover: no src_audio, nothing regenerated from a source.
    "task_type": "text2music",
    "lyrics": LYRICS,
    "vocal_language": "ko",
    "audio_duration": DURATION,
    "audio_format": "wav",
    "model": MODEL,
    "inference_steps": 8,
    "thinking": False,
    "use_cot_caption": False,
    "use_cot_language": False,
    "batch_size": 1,
    "use_random_seed": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    client: httpx.Client,
    out: Path,
    name: str,
    prompt: str,
    reference: Path | None,
    seed: int,
) -> dict[str, object]:
    payload = dict(BASE_PAYLOAD, prompt=prompt, seed=seed)
    fields = {
        k: ("true" if v is True else "false" if v is False else str(v)) for k, v in payload.items()
    }

    started = time.time()
    if reference is None:
        response = client.post("/release_task", data=fields, timeout=120.0)
    else:
        with reference.open("rb") as handle:
            # ``ref_audio`` is the field upstream's multipart parser reads
            # for reference audio; ``src_audio`` is the *other* mechanism
            # and would silently make this a different operation.
            response = client.post(
                "/release_task",
                data=fields,
                files={"ref_audio": (reference.name, handle, "audio/wav")},
                timeout=120.0,
            )
    envelope = response.json()
    if envelope.get("code") != 200 or envelope.get("error"):
        raise SystemExit(f"{name}: release_task failed: {envelope}")
    task_id = envelope["data"]["task_id"]
    print(f"  {name}: task {task_id}", flush=True)

    track = None
    deadline = time.time() + 900
    while time.time() < deadline:
        time.sleep(4)
        entry = client.post("/query_result", json={"task_id_list": [task_id]}, timeout=60.0).json()[
            "data"
        ][0]
        status = int(entry.get("status", 0))
        if status == 1:
            track = json.loads(entry["result"])[0]
            break
        if status == 2:
            # Recorded, never silently retried without the reference.
            raise SystemExit(f"{name}: task failed: {entry.get('result')}")
    if track is None:
        raise SystemExit(f"{name}: timed out")

    destination = out / "outputs" / f"{name}.wav"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(client.get(track["file"], timeout=300.0).content)
    elapsed = time.time() - started
    print(f"  {name}: {destination.name} in {elapsed:.0f}s", flush=True)

    return {
        "name": name,
        "engine": "ace-step",
        "engine_commit": ENGINE_COMMIT,
        "model": MODEL,
        "task_type": "text2music",
        "prompt": prompt,
        "lyrics_sha256": hashlib.sha256(LYRICS.encode()).hexdigest(),
        "reference_file": reference.name if reference else None,
        "reference_sha256": sha256(reference) if reference else None,
        "seed_requested": seed,
        "seed_actual": track.get("seed_value"),
        "parameters": payload,
        "output_file": destination.name,
        "output_sha256": sha256(destination),
        "wall_clock_seconds": round(elapsed, 1),
        "engine_reported_duration": (track.get("metas") or {}).get("duration"),
    }


def main() -> int:
    out = Path(sys.argv[1])
    reference_a = Path(sys.argv[2])
    reference_b = Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)

    plan: list[tuple[str, str, Path | None, int]] = [
        ("00_PROMPT_ONLY", PROMPT_BASE, None, SEED),
        ("01_REFERENCE_A", PROMPT_BASE, reference_a, SEED),
        ("02_REFERENCE_B", PROMPT_BASE, reference_b, SEED),
        ("03_REFERENCE_A_CONTRADICTORY_PROMPT", PROMPT_CONTRADICTORY, reference_a, SEED),
        ("04_CONTRADICTORY_PROMPT_ONLY", PROMPT_CONTRADICTORY, None, SEED),
        ("05_REFERENCE_A_DIFFERENT_SEED", PROMPT_BASE, reference_a, ALT_SEED),
    ]

    results_path = out / "results.jsonl"
    done = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["name"])

    with httpx.Client(base_url=ENGINE) as client:
        health = client.get("/health", timeout=30).json()["data"]
        if health.get("loaded_model") != MODEL:
            raise SystemExit(f"unexpected model: {health.get('loaded_model')}")
        print(f"engine ready: {health['loaded_model']}", flush=True)

        for name, prompt, reference, seed in plan:
            if name in done:
                print(f"  {name}: already done, skipping", flush=True)
                continue
            record = run(client, out, name, prompt, reference, seed)
            with results_path.open("a") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("calibration complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
