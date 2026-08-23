#!/usr/bin/env python3
"""Phase 37 — three-way deterministic generation: base, and two adapters.

Run by the trainer's interpreter. One handler, one model load, three
passes: the base model untouched, then each adapter in turn, with the
previous one unloaded first so no pass ever carries another's weights.

The comparison is only worth anything if the sides differ in one thing,
so every side gets the same prompt, seed, duration and step count, and
the outputs go to directories that are never written twice.

    ~/ace-step-1.5/.venv/bin/python scripts/training/abc_evaluate.py spec.json

The spec names `sides` as an ordered list of `{name, lora_path}` where a
null `lora_path` means the untouched base model.

Nothing here judges audio. It produces comparable sets; a person listens.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

AUDIO_SUFFIXES: tuple[str, ...] = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


def audio_files(directory: Path) -> list[str]:
    return sorted(
        path.name
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )


def lora_state(handler: Any) -> dict[str, Any]:
    return {
        "lora_loaded": bool(getattr(handler, "lora_loaded", False)),
        "use_lora": bool(getattr(handler, "use_lora", False)),
        "lora_scale": getattr(handler, "lora_scale", None),
        "active_adapter": getattr(handler, "_lora_active_adapter", None),
    }


def _generate(
    handler: Any, spec: dict[str, Any], item: dict[str, Any], save_dir: Path
) -> dict[str, Any]:
    from acestep.inference import (  # type: ignore[import-not-found]
        GenerationConfig,
        GenerationParams,
        generate_music,
    )

    params = GenerationParams(
        task_type="text2music",
        caption=str(item["caption"]),
        lyrics=str(item.get("lyrics", "")),
        instrumental=bool(item.get("instrumental", False)),
        vocal_language=str(item.get("vocal_language", "unknown")),
        bpm=item.get("bpm"),
        keyscale=str(item.get("keyscale", "")),
        timesignature=str(item.get("timesignature", "")),
        duration=float(item.get("duration", spec.get("duration", 30.0))),
        inference_steps=int(spec.get("inference_steps", 8)),
        shift=float(spec.get("shift", 3.0)),
        seed=int(item["seed"]),
        thinking=False,
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    error = ""
    try:
        outcome = generate_music(handler, None, params, GenerationConfig(), save_dir=str(save_dir))
        error = str(getattr(outcome, "error", "") or "")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "files": audio_files(save_dir),
        "error": error,
    }


def main(argv: list[str]) -> int:
    spec = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    root = Path(spec["output_root"])
    sides = spec["sides"]

    for side in sides:
        for item in spec["items"]:
            target = root / item["id"] / side["name"]
            if target.exists() and any(target.iterdir()):
                print(
                    f"ABORT: {target} already holds output; no side is overwritten", file=sys.stderr
                )
                return 2

    from acestep.handler import AceStepHandler  # type: ignore[import-not-found]

    handler = AceStepHandler()
    handler.initialize_service(
        project_root=str(spec["project_root"]),
        config_path=str(spec.get("model_variant", "acestep-v15-turbo")),
        device=str(spec.get("device", "auto")),
    )

    records: dict[str, dict[str, Any]] = {}
    for side in sides:
        name, lora_path = side["name"], side.get("lora_path")
        # Unload before loading, so a side never inherits the previous
        # adapter. Checked afterwards rather than assumed.
        try:
            handler.unload_lora()
        except Exception:
            pass
        if lora_path:
            print(f"[{name}] {handler.load_lora(str(lora_path))}")
        state = lora_state(handler)
        print(f"[{name}] adapter state: {state}")
        if lora_path and not (state["lora_loaded"] and state["use_lora"]):
            print(f"ABORT: adapter for {name} did not attach", file=sys.stderr)
            return 2
        if not lora_path and state["use_lora"]:
            print(
                f"ABORT: {name} is meant to be the untouched base model but an adapter is active",
                file=sys.stderr,
            )
            return 2

        for item in spec["items"]:
            target = root / item["id"] / name
            print(f"[{name}] {item['id']}  seed={item['seed']}")
            records.setdefault(item["id"], {})[name] = {
                **_generate(handler, spec, item, target),
                "lora_state": state,
                "lora_path": lora_path or "",
            }

    manifest = {
        "schema_version": "luber-abc-evaluation/1",
        "sides": [{"name": s["name"], "lora_path": s.get("lora_path") or ""} for s in sides],
        "inference_steps": spec.get("inference_steps", 8),
        "shift": spec.get("shift", 3.0),
        "note": (
            "Every side shares prompt, seed, duration and step count; the adapter is the "
            "only difference. No automated judgement is made here."
        ),
        "items": [
            {
                **{k: v for k, v in item.items() if k != "lyrics"},
                "lyrics_line_count": len(str(item.get("lyrics", "")).splitlines()),
                "sides": records.get(item["id"], {}),
            }
            for item in spec["items"]
        ],
    }
    Path(spec["manifest"]).parent.mkdir(parents=True, exist_ok=True)
    Path(spec["manifest"]).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"manifest: {spec['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
