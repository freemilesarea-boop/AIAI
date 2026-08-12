#!/usr/bin/env python3
"""Run the Phase 6 controlled no-training A/B experiment.

Drives ACE-Step directly with a fixed seed per prompt so exactly one
variable changes per variant. Resumable: completed cells are skipped.

    uv run python scripts/benchmark/run_ab_experiment.py \\
        --config benchmarks/music_quality/configs/phase6_ab.json --resume
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "music_quality" / "scripts"))

from bench.analysis import analyze_structure  # noqa: E402
from bench.dataset import load_dataset  # noqa: E402
from bench.experiment import (  # noqa: E402
    EngineExperiment,
    ExperimentError,
    ExperimentRecord,
    Variant,
)
from bench.metrics import measure_wav  # noqa: E402
from bench.runner import free_disk_gb  # noqa: E402

BENCH_ROOT = REPO_ROOT / "benchmarks" / "music_quality"

# Legacy compiler behaviour, reproduced verbatim so V1 is a true Phase 5
# control. The production compiler no longer does this.
_LEGACY_VOCAL = {
    "female": "female lead vocal, natural female singing voice",
    "male": "male lead vocal, natural male singing voice",
}


def legacy_compile(prompt: str, vocal_gender: str) -> str:
    if vocal_gender == "instrumental":
        return f"{prompt.strip()}, instrumental, no vocals"
    return f"{prompt.strip()}, {_LEGACY_VOCAL[vocal_gender]}"


def dedup_compile(prompt: str, vocal_gender: str, lyrics: str, language: str) -> str:
    from luber_generation_client.ace_step.compiler import AceStepPromptCompiler
    from luber_generation_client.provider import GenerationRequest
    from luber_schemas import VocalGender

    return (
        AceStepPromptCompiler()
        .compile(
            GenerationRequest(
                title="ab",
                prompt=prompt,
                lyrics=lyrics,
                vocal_gender=VocalGender(vocal_gender),
                duration_seconds=60,
                language=language,
            )
        )
        .prompt
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BENCH_ROOT / "configs" / "phase6_ab.json")
    parser.add_argument(
        "--dataset", type=Path, default=BENCH_ROOT / "prompts" / "BENCHMARK_V1.json"
    )
    parser.add_argument("--ace-step", default="http://127.0.0.1:8001")
    parser.add_argument("--output-dir", type=Path, default=BENCH_ROOT / "results")
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=REPO_ROOT / "data" / "ab-experiment",
        help="Where experiment audio is written (gitignored)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--min-free-disk-gb", type=float, default=6.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    dataset = load_dataset(args.dataset)

    variants = [Variant(**v) for v in config["variants"]]
    duration = int(config["duration"])
    results_path = args.output_dir / f"{config['experiment_id']}.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if args.resume and results_path.is_file():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") == "COMPLETED":
                done.add(f"{rec['variant_id']}::{rec['prompt_id']}")

    cells: list[tuple[Variant, dict[str, Any]]] = [
        (variant, entry)
        for variant in variants
        for entry in config["prompts"]
        if f"{variant.variant_id}::{entry['prompt_id']}" not in done
    ]
    if args.limit is not None:
        cells = cells[: args.limit]

    print(f"experiment : {config['experiment_id']}")
    print(f"variants   : {len(variants)}  prompts: {len(config['prompts'])}  duration: {duration}s")
    print(f"cells      : {len(cells)} to run, {len(done)} already done")
    print(f"free disk  : {free_disk_gb(args.audio_dir.parent):.1f} GB")

    if args.dry_run:
        for variant, entry in cells:
            print(f"  {variant.variant_id}::{entry['prompt_id']}")
        return 0

    engine = EngineExperiment(
        base_url=args.ace_step,
        model=config.get("model", "acestep-v15-turbo"),
        inference_steps=int(config.get("inference_steps", 8)),
        output_dir=args.audio_dir,
    )
    try:
        health = engine.health()
    except ExperimentError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 2
    print(f"engine     : {health.get('loaded_model')} llm={health.get('llm_initialized')}\n")

    for index, (variant, entry) in enumerate(cells, start=1):
        free = free_disk_gb(args.audio_dir.parent)
        if free < args.min_free_disk_gb:
            print(f"ABORT: free disk {free:.1f} GB below safety margin", file=sys.stderr)
            return 3

        prompt_def = dataset.by_id(entry["prompt_id"])
        seed = int(entry["seed"])
        text = (
            dedup_compile(
                prompt_def.prompt, prompt_def.vocal_gender, prompt_def.lyrics, prompt_def.language
            )
            if variant.dedup_prompt
            else legacy_compile(prompt_def.prompt, prompt_def.vocal_gender)
        )
        lyrics = "[inst]" if prompt_def.is_instrumental else prompt_def.lyrics

        payload = engine.build_payload(
            prompt=text,
            lyrics=lyrics,
            language=prompt_def.language,
            duration=duration,
            seed=seed,
            variant=variant,
            metadata=entry.get("metadata"),
        )
        record = ExperimentRecord(
            experiment_id=config["experiment_id"],
            variant_id=variant.variant_id,
            prompt_id=prompt_def.prompt_id,
            genre=prompt_def.genre,
            language=prompt_def.language,
            vocal_gender=prompt_def.vocal_gender,
            duration=duration,
            seed=seed,
            prompt_sent=text,
            lyrics_sent=lyrics,
            payload=payload,
        )

        label = f"{variant.variant_id}::{prompt_def.prompt_id}"
        print(f"[{index}/{len(cells)}] {label} ...", flush=True)
        destination = args.audio_dir / variant.variant_id / f"{prompt_def.prompt_id}.wav"
        try:
            seconds, path = engine.run(payload, destination)
        except ExperimentError as exc:
            record.status = "FAILED"
            record.error = str(exc)[:300]
            print(f"    FAILED: {record.error}", flush=True)
        else:
            record.status = "COMPLETED"
            record.generation_seconds = seconds
            record.output_path = str(path.relative_to(REPO_ROOT))
            metrics = measure_wav(path, requested_duration=float(duration))
            record.metrics = metrics.to_dict()
            record.structure = analyze_structure(path).to_dict()
            import hashlib

            record.output_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            print(
                f"    COMPLETED in {seconds}s  peak={metrics.peak:.3f} "
                f"lvlΔ={record.structure['level_drift_db']} dB "
                f"rep={record.structure['max_repetition']} flags={metrics.flags}",
                flush=True,
            )

        with results_path.open("a", encoding="utf-8") as fh:
            fh.write(record.to_json() + "\n")

    print(f"\nresults: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
