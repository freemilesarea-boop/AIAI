"""Run QC over audio that already exists. Generates nothing.

Two jobs.

``analyze`` answers "what would QC say about this file" — for one file
or a directory of them. It is how thresholds get checked against real
output instead of against the fixtures they were written next to, and it
is the only way to find out that a rule which looks reasonable rejects a
third of the corpus.

``explain`` reads a stored trace and answers the questions an operator
actually has: why did this generation retry, why did candidate A lose,
why was B selected, how many provider calls did it cost.

Neither mutates anything. Both are safe to point at production data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from luber_inference_qc.candidate import CandidateGeneration
from luber_inference_qc.checks import RequestExpectation
from luber_inference_qc.engine import judge
from luber_inference_qc.findings import Severity
from luber_inference_qc.measurement import MeasurementCache
from luber_inference_qc.trace import summarise
from luber_inference_qc.versions import version_block

AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".m4a", ".ogg")


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str))


def _candidate(path: Path, index: int) -> CandidateGeneration:
    return CandidateGeneration(
        candidate_id=f"dryrun_{index:04d}",
        generation_id="dry-run",
        attempt_index=index,
        request_sha256="dry-run",
    )


def _expectation(args: argparse.Namespace) -> RequestExpectation:
    return RequestExpectation(
        duration_seconds=args.duration,
        bpm=args.bpm,
        key_scale=args.key,
        instrumental=args.instrumental,
    )


def _analyse_one(
    path: Path, index: int, expectation: RequestExpectation, cache: MeasurementCache
) -> dict[str, Any]:
    candidate = _candidate(path, index)
    started = time.monotonic()
    measurement = judge(candidate, path, expectation, cache=cache)
    elapsed = time.monotonic() - started
    payload: dict[str, Any] = {
        "file": path.name,
        "status": candidate.status,
        "eligible": candidate.eligible,
        "technical_selection_score": candidate.technical_selection_score,
        "score_components": candidate.score_components,
        "qc_seconds": round(elapsed, 3),
        "findings": [finding.to_dict() for finding in candidate.findings],
    }
    if measurement is not None:
        payload["measurement"] = measurement.to_dict()
    return payload


def cmd_analyze(args: argparse.Namespace) -> int:
    target = Path(args.path).expanduser()
    expectation = _expectation(args)
    cache = MeasurementCache()

    if target.is_file():
        _print({**version_block(), "result": _analyse_one(target, 0, expectation, cache)})
        return 0

    if not target.is_dir():
        print(f"no such file or directory: {target}", file=sys.stderr)
        return 2

    files = sorted(
        item
        for item in target.iterdir()
        if item.is_file() and item.suffix.lower() in AUDIO_SUFFIXES
    )
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"no audio files in {target}", file=sys.stderr)
        return 2

    results: list[dict[str, Any]] = []
    for index, path in enumerate(files):
        try:
            results.append(_analyse_one(path, index, expectation, cache))
        except Exception as exc:  # a corpus should not stop on one bad file
            results.append({"file": path.name, "status": "ERROR", "error": str(exc)})
        if args.progress:
            print(f"  {index + 1}/{len(files)} {path.name}", file=sys.stderr)

    eligible = [item for item in results if item.get("eligible")]
    rejected = [item for item in results if item.get("status") == "REJECTED"]
    errors = [item for item in results if item.get("status") == "ERROR"]

    codes: Counter[str] = Counter()
    critical_codes: Counter[str] = Counter()
    for item in results:
        for finding in item.get("findings", []) or []:
            codes[finding["code"]] += 1
            if finding["severity"] == Severity.CRITICAL.value:
                critical_codes[finding["code"]] += 1

    latencies = [item["qc_seconds"] for item in results if "qc_seconds" in item]
    latencies.sort()

    summary = {
        **version_block(),
        "corpus": str(target),
        "files": len(results),
        "eligible": len(eligible),
        "rejected": len(rejected),
        "errors": len(errors),
        "eligible_ratio": round(len(eligible) / len(results), 4) if results else 0.0,
        "rejection_ratio": round(len(rejected) / len(results), 4) if results else 0.0,
        "finding_counts": dict(codes.most_common()),
        "critical_finding_counts": dict(critical_codes.most_common()),
        "qc_seconds": {
            "min": round(latencies[0], 3) if latencies else None,
            "median": round(latencies[len(latencies) // 2], 3) if latencies else None,
            "max": round(latencies[-1], 3) if latencies else None,
            "total": round(sum(latencies), 3),
        },
    }
    if args.detail:
        summary["results"] = results
    _print(summary)

    # A rejection rate this high means the thresholds are wrong, not
    # that the corpus is. Signalled through the exit code so a dry run
    # can be wired into a check rather than only read.
    if results and len(rejected) / len(results) > 0.5:
        print(
            f"more than half the corpus was rejected ({len(rejected)}/{len(results)}); "
            "audit the thresholds rather than the songs",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    path = Path(args.trace).expanduser()
    if not path.is_file():
        print(f"no trace at {path}", file=sys.stderr)
        return 2
    trace = json.loads(path.read_text(encoding="utf-8"))

    if args.summary:
        _print(summarise(trace))
        return 0

    attempts = trace.get("attempts", []) or []
    lines: list[str] = []
    lines.append(f"generation {trace.get('generation_id')}")
    lines.append(f"  policy            {(trace.get('policy') or {}).get('name')}")
    lines.append(f"  request digest    {trace.get('request_sha256')}")
    lines.append(f"  outcome           {trace.get('outcome')}: {trace.get('outcome_detail')}")
    budget = trace.get("budget") or {}
    lines.append(
        f"  provider calls    {budget.get('provider_calls_used')} of "
        f"{budget.get('maximum_total_provider_calls')}"
    )
    lines.append(f"  finishing         {trace.get('finishing_outcome')}")
    lines.append("")

    selected = trace.get("selected_candidate_id")
    reasons = (trace.get("selection") or {}).get("reasons") or {}
    for attempt in attempts:
        mark = "→" if attempt.get("candidate_id") == selected else " "
        lines.append(
            f"{mark} attempt {attempt.get('attempt_index')} "
            f"{attempt.get('candidate_id')} "
            f"[{attempt.get('status')}] seed={attempt.get('seed')} "
            f"({attempt.get('attribution')})"
        )
        if attempt.get("retry_reason"):
            lines.append(f"    retried because   {attempt['retry_reason']}")
        for finding in attempt.get("findings", []) or []:
            if finding.get("code") == "NO_CRITICAL_FINDINGS":
                continue
            lines.append(f"    {finding['severity']:<8} {finding['code']}: {finding['detail']}")
        score = attempt.get("technical_selection_score")
        if score is not None:
            lines.append(f"    technical score   {score:.4f}")
            for name, value in sorted((attempt.get("score_components") or {}).items()):
                lines.append(f"      {name:<20} {value:.4f}")
        reason = reasons.get(attempt.get("candidate_id"))
        if reason:
            lines.append(f"    selection         {reason}")
        lines.append("")

    print("\n".join(lines))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="luber-inference-qc",
        description="Run inference quality control over existing audio. Generates nothing.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="QC one file or a directory")
    analyze.add_argument("path", help="audio file or directory of audio files")
    analyze.add_argument(
        "--duration", type=float, default=None, help="the duration that was requested"
    )
    analyze.add_argument("--bpm", type=int, default=None, help="the BPM that was requested")
    analyze.add_argument("--key", default=None, help="the key that was requested")
    analyze.add_argument(
        "--instrumental",
        action="store_true",
        default=None,
        help="an instrumental was requested",
    )
    analyze.add_argument("--limit", type=int, default=0, help="stop after this many files")
    analyze.add_argument("--detail", action="store_true", help="include every per-file result")
    analyze.add_argument("--progress", action="store_true", help="report progress to stderr")
    analyze.set_defaults(func=cmd_analyze)

    explain = sub.add_parser("explain", help="read a stored QC trace")
    explain.add_argument("trace", help="path to a trace JSON file")
    explain.add_argument("--summary", action="store_true", help="the short form, as JSON")
    explain.set_defaults(func=cmd_explain)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
