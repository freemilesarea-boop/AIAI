#!/usr/bin/env python3
"""Serve the developer-only benchmark listening tool.

Blind scoring by default, plus a blind A/B mode. Binds to localhost
only — this is an internal evaluation instrument and is deliberately not
part of the product.

    uv run python scripts/benchmark/listening_tool.py \\
        --results benchmarks/music_quality/results/pilot_baseline_p5_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "music_quality" / "scripts"))

from bench.listening import (  # noqa: E402
    build_listening_payload,
    render_ab_page,
    render_listening_page,
)
from bench.scoring import (  # noqa: E402
    ScoreValidationError,
    make_blind_pair,
    validate_artifact_tags,
    validate_scores,
)
from bench.store import ResultStore, ScoreRecord, ScoreStore  # noqa: E402

BENCH_ROOT = REPO_ROOT / "benchmarks" / "music_quality"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--scores", type=Path, default=BENCH_ROOT / "listening" / "scores.jsonl")
    parser.add_argument(
        "--ab-out", type=Path, default=BENCH_ROOT / "listening" / "ab_results.jsonl"
    )
    parser.add_argument("--evaluator", default="local-dev")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--reveal", action="store_true", help="Disable blind mode")
    parser.add_argument(
        "--ab", nargs=2, metavar=("RUN_A", "RUN_B"), help="Two results files to compare blind"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    blind = not args.reveal
    records = ResultStore(args.results).load()
    payload = build_listening_payload(records, api_base=args.api, blind=blind)
    scores = ScoreStore(args.scores)

    pairs: list[dict[str, Any]] = []
    if args.ab:
        left = {r["benchmark_id"]: r for r in ResultStore(Path(args.ab[0])).load()}
        right = {r["benchmark_id"]: r for r in ResultStore(Path(args.ab[1])).load()}
        for key in sorted(set(left) & set(right)):
            a_rec, b_rec = left[key], right[key]
            if a_rec.get("status") != "COMPLETED" or b_rec.get("status") != "COMPLETED":
                continue
            pair = make_blind_pair(f"A::{key}", f"B::{key}")

            def _preview(rec: dict[str, Any]) -> str:
                return f"{args.api}/v1/generations/{rec['generation_id']}/audio?asset=preview"

            urls = {f"A::{key}": _preview(a_rec), f"B::{key}": _preview(b_rec)}
            pairs.append(
                {
                    "pair_id": pair.pair_id,
                    "a_url": urls[pair.track_a],
                    "b_url": urls[pair.track_b],
                    "track_a": pair.track_a,
                    "track_b": pair.track_b,
                }
            )

    scoring_page = render_listening_page(payload, blind=blind).encode("utf-8")
    ab_page = render_ab_page(pairs).encode("utf-8")
    pair_index = {p["pair_id"]: p for p in pairs}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: Any) -> None:  # keep the console readable
            return

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.startswith("/ab"):
                self._send(200, ab_page, "text/html; charset=utf-8")
            else:
                self._send(200, scoring_page, "text/html; charset=utf-8")

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send(400, b'{"error":"bad json"}', "application/json")
                return

            if self.path.startswith("/ab"):
                pair = pair_index.get(str(body.get("pair_id")))
                if pair is None:
                    self._send(400, b'{"error":"unknown pair"}', "application/json")
                    return
                with Path(args.ab_out).open("a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "pair_id": pair["pair_id"],
                                "choice": body.get("choice"),
                                "confidence": body.get("confidence"),
                                "track_a": pair["track_a"],
                                "track_b": pair["track_b"],
                                "evaluator": args.evaluator,
                                "scored_at": datetime.now(UTC).isoformat(),
                            }
                        )
                        + "\n"
                    )
                self._send(200, b'{"ok":true}', "application/json")
                return

            try:
                instrumental = next(
                    (
                        item["instrumental"]
                        for item in payload
                        if item["benchmark_id"] == body.get("benchmark_id")
                    ),
                    False,
                )
                validated = validate_scores(body.get("scores") or {}, instrumental=instrumental)
                tags = validate_artifact_tags(list(body.get("artifact_tags") or []))
            except ScoreValidationError as exc:
                self._send(400, json.dumps({"error": str(exc)}).encode("utf-8"), "application/json")
                return

            scores.append(
                ScoreRecord(
                    benchmark_id=str(body.get("benchmark_id")),
                    evaluator=args.evaluator,
                    scored_at=datetime.now(UTC).isoformat(),
                    blind=bool(body.get("blind", blind)),
                    scores=validated,
                    artifact_tags=tags,
                    notes=str(body.get("notes", "")),
                )
            )
            self._send(200, b'{"ok":true}', "application/json")

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"listening tool : http://127.0.0.1:{args.port}/     ({len(payload)} tracks)")
    if pairs:
        print(f"blind A/B      : http://127.0.0.1:{args.port}/ab   ({len(pairs)} pairs)")
    print(f"blind mode     : {blind}")
    print(f"scores         : {args.scores}")
    print("Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
