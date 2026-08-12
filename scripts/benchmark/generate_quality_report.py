#!/usr/bin/env python3
"""Render the Phase 5 baseline report from benchmark results and scores."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "music_quality" / "scripts"))

from bench.report import render_report  # noqa: E402
from bench.store import ResultStore, ScoreStore  # noqa: E402
from bench.verdict import VerdictStore  # noqa: E402

BENCH_ROOT = REPO_ROOT / "benchmarks" / "music_quality"


def _hardware() -> str:
    def sysctl(key: str) -> str:
        try:
            return subprocess.run(
                ["sysctl", "-n", key], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    memory = sysctl("hw.memsize")
    memory_gb = f"{int(memory) / 1024**3:.0f} GB" if memory.isdigit() else "unknown"
    return (
        f"- CPU: {sysctl('machdep.cpu.brand_string')}\n"
        f"- Cores: {sysctl('hw.ncpu')}\n"
        f"- Memory: {memory_gb}\n"
        f"- Backend: Apple Silicon MPS + MLX (ACE-Step macOS path)"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Results JSONL")
    parser.add_argument("--scores", type=Path, default=BENCH_ROOT / "listening" / "scores.jsonl")
    parser.add_argument(
        "--output", type=Path, default=BENCH_ROOT / "reports" / "PHASE5_BASELINE_REPORT.md"
    )
    parser.add_argument("--baseline-id", default="LUBER_BASELINE_P5_V1")
    parser.add_argument("--benchmark-version", default="v1")
    parser.add_argument("--ace-step-version", default="1.5.0")
    parser.add_argument("--ace-step-commit", default="6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0")
    parser.add_argument("--notes-file", type=Path, default=None)
    parser.add_argument(
        "--appendix-file",
        type=Path,
        default=BENCH_ROOT / "reports" / "_baseline_appendix.md",
        help="Analysis sections appended verbatim after the generated body",
    )
    parser.add_argument(
        "--verdicts", type=Path, default=BENCH_ROOT / "listening" / "verdicts.jsonl"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records = ResultStore(args.results).load()
    scores = ScoreStore(args.scores).load()
    notes = args.notes_file.read_text(encoding="utf-8") if args.notes_file else ""
    verdict = VerdictStore(args.verdicts).latest_for(args.baseline_id)

    report = render_report(
        records=records,
        scores=scores,
        baseline_id=args.baseline_id,
        benchmark_version=args.benchmark_version,
        ace_step_version=args.ace_step_version,
        ace_step_commit=args.ace_step_commit,
        hardware=_hardware(),
        notes=notes,
        verdict=verdict,
    )
    if args.appendix_file and args.appendix_file.is_file():
        report += "\n" + args.appendix_file.read_text(encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(
        f"wrote {args.output} ({len(records)} records, {len(scores)} scores, "
        f"verdict={'yes' if verdict else 'no'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
