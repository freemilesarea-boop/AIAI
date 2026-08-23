#!/usr/bin/env python3
"""Phase 36 — deterministic A/B generation: base model against a LoRA.

Run by the *trainer's* interpreter, because that is where ACE-Step and
torch live. It imports nothing from LUBER and takes its whole
instruction set as JSON, so the same file works from a shell, from a
test, or from an orchestrator.

The comparison is only worth anything if the two sides differ in one
thing. So both sides get the same prompts, the same seeds, the same
duration, the same step count and the same handler instance — the LoRA
is loaded between the two passes and nothing else changes. Outputs go to
two directories that are never written twice, and a manifest records
which base output pairs with which LoRA output.

    ~/ace-step-1.5/.venv/bin/python scripts/training/ab_evaluate.py spec.json

The spec:

    {
      "project_root": "...",          # where checkpoints/ lives
      "model_variant": "acestep-v15-turbo",
      "device": "mps",
      "lora_path": "...",             # the adapter directory
      "base_dir": "...", "lora_dir": "...",
      "manifest": "...",
      "duration": 30.0, "inference_steps": 8,
      "pairs": [{"id": "...", "caption": "...", "seed": 12345}, ...]
    }

Nothing here judges the audio. It produces pairs; a human decides.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

#: Whatever the pipeline chooses to write. It saves FLAC by default and
#: the container is not the pipeline's contract, so the manifest lists
#: what is actually there rather than what a glob expected to find.
AUDIO_SUFFIXES: tuple[str, ...] = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


def audio_files(directory: Path) -> list[str]:
    """Every audio file in *directory*, by name, in a stable order."""
    return sorted(
        path.name
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )


def _generate(
    dit_handler: Any, spec: dict[str, Any], pair: dict[str, Any], save_dir: Path
) -> dict[str, Any]:
    from acestep.inference import (  # type: ignore[import-not-found]
        GenerationConfig,
        GenerationParams,
        generate_music,
    )

    params = GenerationParams(
        task_type="text2music",
        caption=str(pair["caption"]),
        # No lyrics were supplied with this library, and the training
        # conditioning was empty text for the same reason. Asking for
        # something the run never saw would compare two different
        # questions.
        lyrics="",
        instrumental=False,
        duration=float(spec.get("duration", 30.0)),
        inference_steps=int(spec.get("inference_steps", 8)),
        seed=int(pair["seed"]),
        thinking=False,
        vocal_language="unknown",
    )
    config = GenerationConfig()
    save_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = generate_music(dit_handler, None, params, config, save_dir=str(save_dir))
    return {
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "files": audio_files(save_dir),
        "error": getattr(result, "error", "") or "",
    }


def main(argv: list[str]) -> int:
    spec = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    base_dir = Path(spec["base_dir"])
    lora_dir = Path(spec["lora_dir"])
    for directory in (base_dir, lora_dir):
        if directory.exists() and any(directory.iterdir()):
            print(
                f"ABORT: {directory} already holds output; neither side is ever overwritten",
                file=sys.stderr,
            )
            return 2

    from acestep.handler import AceStepHandler  # type: ignore[import-not-found]

    dit_handler = AceStepHandler()
    dit_handler.initialize_service(
        project_root=str(spec["project_root"]),
        config_path=str(spec.get("model_variant", "acestep-v15-turbo")),
        device=str(spec.get("device", "auto")),
    )

    records: list[dict[str, Any]] = []

    # Side A first, with no adapter anywhere near the model.
    for pair in spec["pairs"]:
        target = base_dir / str(pair["id"])
        print(f"[base] {pair['id']}  caption={pair['caption']!r} seed={pair['seed']}")
        records.append(
            {"id": pair["id"], "side": "base", **_generate(dit_handler, spec, pair, target)}
        )

    loaded = dit_handler.load_lora(str(spec["lora_path"]))
    print(f"[lora] loaded: {loaded}")

    for pair in spec["pairs"]:
        target = lora_dir / str(pair["id"])
        print(f"[lora] {pair['id']}  caption={pair['caption']!r} seed={pair['seed']}")
        records.append(
            {"id": pair["id"], "side": "lora", **_generate(dit_handler, spec, pair, target)}
        )

    manifest = {
        "schema_version": "luber-ab-evaluation/1",
        "lora_path": str(spec["lora_path"]),
        "duration": spec.get("duration", 30.0),
        "inference_steps": spec.get("inference_steps", 8),
        "note": (
            "Both sides share prompt, seed, duration and step count; the adapter is the "
            "only difference. No automated judgement is made here."
        ),
        "pairs": [
            {
                "id": pair["id"],
                "caption": pair["caption"],
                "seed": pair["seed"],
                "source_group": pair.get("source_group", ""),
                "base": next(
                    (r for r in records if r["id"] == pair["id"] and r["side"] == "base"), None
                ),
                "lora": next(
                    (r for r in records if r["id"] == pair["id"] and r["side"] == "lora"), None
                ),
            }
            for pair in spec["pairs"]
        ],
    }
    Path(spec["manifest"]).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"manifest: {spec['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
