#!/usr/bin/env python3
"""Run the LUBER music quality benchmark against the real pipeline.

Generations go through the live LUBER API, so the whole Phase 4 path is
exercised. There is no mock mode by design.

Usage:

    uv run python scripts/benchmark/run_music_quality_benchmark.py \\
        --manifest benchmarks/music_quality/manifests/pilot_baseline.json \\
        --resume
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "music_quality" / "scripts"))

from bench.dataset import load_dataset  # noqa: E402
from bench.runner import (  # noqa: E402
    BASELINE,
    BenchmarkAbort,
    BenchmarkRunner,
    RunConfig,
    benchmark_id,
    free_disk_gb,
)
from bench.store import ResultStore  # noqa: E402

DEFAULT_DATASET = REPO_ROOT / "benchmarks" / "music_quality" / "prompts" / "BENCHMARK_V1.json"
DEFAULT_RESULTS = REPO_ROOT / "benchmarks" / "music_quality" / "results"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Run manifest JSON")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", default=None, help="Override the model in the manifest")
    parser.add_argument("--configuration", default=None, help="Override configuration id")
    parser.add_argument("--seed", type=int, default=None, help="Force a single seed")
    parser.add_argument(
        "--duration-tier",
        type=int,
        default=None,
        help="Only run units with this requested duration",
    )
    parser.add_argument("--limit", type=int, default=None, help="Stop after N generations")
    parser.add_argument("--resume", action="store_true", help="Skip already-completed units")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--storage-root",
        type=Path,
        default=REPO_ROOT / "data",
        help="Where the API stores audio (for measurement)",
    )
    parser.add_argument("--min-free-disk-gb", type=float, default=6.0)
    parser.add_argument("--dry-run", action="store_true", help="List units without generating")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    dataset = load_dataset(args.dataset)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    config = RunConfig(
        configuration_id=args.configuration
        or manifest.get("configuration_id", BASELINE.configuration_id),
        model=args.model or manifest.get("model", BASELINE.model),
        lm_enabled=bool(manifest.get("lm_enabled", BASELINE.lm_enabled)),
        thinking_enabled=bool(manifest.get("thinking_enabled", BASELINE.thinking_enabled)),
        inference_steps=int(manifest.get("inference_steps", BASELINE.inference_steps)),
        runtime_backend=manifest.get("runtime_backend", BASELINE.runtime_backend),
    )

    units: list[dict[str, Any]] = manifest["units"]
    if args.duration_tier is not None:
        units = [u for u in units if int(u["duration"]) == args.duration_tier]

    results_path = args.output_dir / f"{manifest['run_id']}.jsonl"
    store = ResultStore(results_path)
    done = store.completed_ids() if args.resume else set()

    pending = []
    for unit in units:
        seed = args.seed if args.seed is not None else unit.get("seed")
        seed_value = int(seed) if seed is not None else None
        bid = benchmark_id(
            str(unit["prompt_id"]), config.configuration_id, int(unit["duration"]), seed_value
        )
        if bid in done:
            continue
        pending.append((unit, seed_value, bid))
    if args.limit is not None:
        pending = pending[: args.limit]

    print(f"benchmark version : {dataset.benchmark_version}")
    print(f"configuration     : {config.configuration_id} ({config.model})")
    print(f"results           : {results_path}")
    print(f"units total       : {len(units)}  already done: {len(done)}  to run: {len(pending)}")
    print(f"free disk         : {free_disk_gb(args.storage_root):.1f} GB")

    if args.dry_run:
        for _unit, _seed, bid in pending:
            print(f"  {bid}")
        return 0

    runner = BenchmarkRunner(
        api_base=args.api,
        storage_root=args.storage_root,
        store=store,
        benchmark_version=dataset.benchmark_version,
        min_free_disk_gb=args.min_free_disk_gb,
    )
    try:
        runner.check_api()
    except BenchmarkAbort as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 2

    completed = 0
    for index, (unit, seed_value, bid) in enumerate(pending, start=1):
        prompt = dataset.by_id(str(unit["prompt_id"]))
        print(f"[{index}/{len(pending)}] {bid} ...", flush=True)
        try:
            record = runner.run_one(
                prompt, config=config, duration=int(unit["duration"]), seed=seed_value
            )
        except BenchmarkAbort as exc:
            print(f"ABORT: {exc}", file=sys.stderr)
            return 3
        flags = (record.metrics or {}).get("flags") or []
        print(
            f"    {record.status} in {record.generation_seconds}s "
            f"rtf={record.real_time_factor} flags={flags}",
            flush=True,
        )
        if record.status == "COMPLETED":
            completed += 1

    print(f"\ncompleted {completed}/{len(pending)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
