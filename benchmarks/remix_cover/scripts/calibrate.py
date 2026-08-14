"""Phase 13D calibration: what ACE-Step's cover task actually does.

Talks to the pinned ACE-Step server directly rather than through LUBER,
because the point is to characterise the engine before deciding whether a
LUBER feature is warranted. Every run is a controlled change of one
variable against a fixed source, prompt, lyrics and seed.

Writes one JSONL record per run with every engine parameter used, so the
results can be audited without rerunning anything.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import httpx

ENGINE = "http://127.0.0.1:8001"
OUT = Path(sys.argv[1])
SOURCE = Path(sys.argv[2])

SOURCE_GENERATION_ID = "a9ae6249-0d22-49d9-99c0-afeb64f88575"
ENGINE_COMMIT = "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0"
MODEL = "acestep-v15-turbo"

SOURCE_PROMPT = "Warm Korean indie pop with electric piano and soft drums"
LYRICS = "[Verse]\n오늘도 너를 기다려\n[Chorus]\n다시 만날 그날까지"

#: Fixed across every run so the only difference is the variable under
#: test. The engine still reports the seed it used; both are recorded.
SEED = 424242

#: Base payload shared by every calibration run. Written out in full on
#: each record — no hidden defaults.
BASE: dict[str, object] = {
    "task_type": "cover",
    "lyrics": LYRICS,
    "vocal_language": "ko",
    "audio_format": "wav",
    "model": MODEL,
    "inference_steps": 8,
    "thinking": False,
    "use_cot_caption": False,
    "use_cot_language": False,
    "batch_size": 1,
    "use_random_seed": False,
    "seed": SEED,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_cover(client: httpx.Client, name: str, prompt: str, strength: float) -> dict[str, object]:
    payload = dict(BASE, prompt=prompt, audio_cover_strength=strength)
    fields = {
        k: ("true" if v is True else "false" if v is False else str(v)) for k, v in payload.items()
    }

    started = time.time()
    with SOURCE.open("rb") as handle:
        response = client.post(
            "/release_task",
            data=fields,
            files={"src_audio": (SOURCE.name, handle, "audio/wav")},
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
        query = client.post("/query_result", json={"task_id_list": [task_id]}, timeout=60.0)
        entry = query.json()["data"][0]
        status = int(entry.get("status", 0))
        if status == 1:
            track = json.loads(entry["result"])[0]
            break
        if status == 2:
            raise SystemExit(f"{name}: task failed: {entry.get('result')}")
    if track is None:
        raise SystemExit(f"{name}: timed out")

    destination = OUT / "outputs" / f"{name}.wav"
    destination.parent.mkdir(parents=True, exist_ok=True)
    audio = client.get(track["file"], timeout=300.0)
    destination.write_bytes(audio.content)
    elapsed = time.time() - started
    print(f"  {name}: {destination.name} in {elapsed:.0f}s", flush=True)

    return {
        "name": name,
        "source_generation_id": SOURCE_GENERATION_ID,
        "source_sha256": sha256(SOURCE),
        "engine": "ace-step",
        "engine_commit": ENGINE_COMMIT,
        "model": MODEL,
        "task_type": "cover",
        "prompt": prompt,
        "lyrics_sha256": hashlib.sha256(LYRICS.encode()).hexdigest(),
        "seed_requested": SEED,
        "seed_actual": track.get("seed_value"),
        "parameters": payload,
        "output_file": destination.name,
        "output_sha256": sha256(destination),
        "wall_clock_seconds": round(elapsed, 1),
        "engine_reported_duration": (track.get("metas") or {}).get("duration"),
    }


# (name, prompt, audio_cover_strength) — one variable changes per group.
STYLE_KPOP = (
    "Modern polished K-pop with bright synths, tight programmed drums and layered harmonies"
)
STYLE_RNB = (
    "Contemporary R&B and neo-soul with warm electric piano, laid-back groove and lush chords"
)
STYLE_INDIE = "Indie pop with jangly guitars, live drums and a lo-fi intimate feel"

RUNS: list[tuple[str, str, float]] = [
    # Baseline: the source's own description, full strength.
    ("01_BASELINE_strength_1.00", SOURCE_PROMPT, 1.00),
    # Strength sweep: same prompt, same seed, only the dial moves.
    ("02_STRENGTH_0.75", SOURCE_PROMPT, 0.75),
    ("03_STRENGTH_0.50", SOURCE_PROMPT, 0.50),
    ("04_STRENGTH_0.25", SOURCE_PROMPT, 0.25),
    # Style transfer at full strength: only the prompt moves.
    ("05_STYLE_KPOP", STYLE_KPOP, 1.00),
    ("06_STYLE_RNB", STYLE_RNB, 1.00),
    ("07_STYLE_INDIE_POP", STYLE_INDIE, 1.00),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    results_path = OUT / "results.jsonl"
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

        for name, prompt, strength in RUNS:
            if name in done:
                print(f"  {name}: already done, skipping", flush=True)
                continue
            record = run_cover(client, name, prompt, strength)
            with results_path.open("a") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("calibration complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
