#!/usr/bin/env python
"""Blind listening tool for the frozen Phase 20 human baseline.

Serves **RAW model masters only**. The listening copy is encoded straight
from `master.wav` at high bitrate; the Phase 14 finished master is never
read, because a baseline scored through the finishing pipeline measures
the equaliser as much as the model.

Blind by construction. The page is built from the benchmark id, the
prompt and the expected lyrics, and nothing else reaches it — not the
seed, the generation id, the model, the engine, the finishing status, or
any objective measurement. Showing a listener that a track measured
narrow would tell them what to hear.

    scripts/benchmark/p20_listening.py --port 8765

Scores append to ``results/p20_human_baseline.jsonl``. Progress resumes:
a case already scored is skipped on the next start, so the session can be
abandoned and picked up without losing anything.
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, ClassVar

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH = REPO_ROOT / "benchmarks" / "music_quality"
sys.path.insert(0, str(BENCH / "scripts"))

from bench.p20_rubric import (  # noqa: E402
    ARTIFACT_TAGS,
    DIMENSION_GROUPS,
    LONG_FORM_MIN_SECONDS,
    P20ScoreError,
    expected_dimensions,
    validate_scores,
    validate_tags,
)
from bench.store import ScoreRecord, ScoreStore  # noqa: E402

BENCHMARK = BENCH / "prompts" / "BENCHMARK_P20.json"
RUNS = BENCH / "results" / "p20_runs.jsonl"
SCORES = BENCH / "results" / "p20_human_baseline.jsonl"
AUDIO_ROOT = REPO_ROOT / "data" / "audio"
#: Listening copies live outside the repository — they are derived audio.
CACHE = Path.home() / ".luber" / "p20-listening"

#: The first-session set, chosen by the Phase 20H priority order: every
#: anti-trot probe, every Korean case generated, representative
#: contemporary vocal, one instrumental, one long-form.
SESSION_ONE = (
    "TROT-01",
    "TROT-02",
    "TROT-03",
    "TROT-04",
    "KO-01",
    "KO-02",
    "KO-03",
    "KO-05",
    "GEN-01",
    "GEN-06",
    "GEN-10",
    "LONG-01",
)


def load_cases() -> dict[str, dict[str, Any]]:
    cases = json.loads(BENCHMARK.read_text())["prompts"]
    return {c["prompt_id"]: c for c in cases}


def load_runs() -> dict[str, str]:
    """benchmark id → generation id, latest completed run wins."""
    mapping: dict[str, str] = {}
    if not RUNS.is_file():
        return mapping
    for line in RUNS.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "COMPLETED":
            mapping[row["prompt_id"]] = row["generation_id"]
    return mapping


def raw_master(generation_id: str) -> Path:
    return AUDIO_ROOT / generation_id / "master.wav"


def listening_copy(benchmark_id: str, source: Path) -> Path:
    """A transparent MP3 of the RAW master, cached by benchmark id.

    Encoded from `master.wav` and nothing else. 320 kbps: this is a
    listening convenience for the browser, not a processing step, and it
    must not become an audible variable in the scores.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    target = CACHE / f"{benchmark_id}.mp3"
    if target.is_file() and target.stat().st_mtime >= source.stat().st_mtime:
        return target
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(source), "-codec:a", "libmp3lame", "-b:a", "320k", str(target)],
        capture_output=True,
        check=True,
        timeout=300,
    )
    return target


def build_queue() -> list[dict[str, Any]]:
    cases, runs = load_cases(), load_runs()
    queue = []
    for benchmark_id in SESSION_ONE:
        case, generation_id = cases.get(benchmark_id), runs.get(benchmark_id)
        if case is None or generation_id is None:
            continue
        source = raw_master(generation_id)
        if not source.is_file():
            continue
        instrumental = case["vocal_gender"] == "instrumental"
        queue.append(
            {
                "benchmark_id": benchmark_id,
                "prompt": case["prompt"],
                "lyrics": case["lyrics"],
                "instrumental": instrumental,
                "korean": case["language"] == "ko" and not instrumental,
                "duration": float(case["duration"]),
                # Kept server-side only. Never rendered.
                "_generation_id": generation_id,
                "_source": source,
            }
        )
    return queue


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>LUBER P20 blind listening</title><style>
 body{font:15px/1.55 -apple-system,system-ui,sans-serif;max-width:920px;margin:0 auto;
      padding:28px 20px 80px;background:#0e0e11;color:#e8e8ea}
 h1{font-size:17px;margin:0 0 4px} .muted{color:#9a9aa2;font-size:13px}
 .card{background:#17171c;border:1px solid #26262e;border-radius:10px;padding:18px;margin:16px 0}
 pre{white-space:pre-wrap;font:13px/1.6 ui-monospace,monospace;color:#c9c9d2;margin:6px 0 0}
 audio{width:100%;margin-top:10px}
 .grp{margin:18px 0 6px;font-weight:600;font-size:13px;color:#b9b9c4;
      border-bottom:1px solid #26262e;padding-bottom:5px}
 .dim{display:flex;align-items:center;gap:10px;padding:5px 0}
 .dim label{flex:1;font-size:13px} .dim input{width:64px;background:#0e0e11;color:#e8e8ea;
      border:1px solid #33333d;border-radius:6px;padding:6px 8px;font-size:14px}
 .tags{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
 .tags label{font-size:11px;background:#1d1d24;border:1px solid #2e2e38;border-radius:20px;
      padding:4px 9px;cursor:pointer}
 textarea{width:100%;min-height:70px;background:#0e0e11;color:#e8e8ea;border:1px solid #33333d;
      border-radius:8px;padding:9px;font:13px/1.5 inherit}
 button{background:#6c5cff;color:#fff;border:0;border-radius:8px;padding:11px 20px;
      font-size:14px;cursor:pointer;margin-right:8px}
 button.ghost{background:#26262e}
 .err{color:#ff8080;font-size:13px;margin-top:10px;white-space:pre-wrap}
 .done{text-align:center;padding:60px 0}
</style></head><body>
<h1>Blind listening — benchmark p20</h1>
<div class="muted">__PROGRESS__</div>
__BODY__
</body></html>"""


def render(item: dict[str, Any] | None, index: int, total: int, error: str = "") -> str:
    if item is None:
        body = (
            '<div class="done card"><h1>Session complete</h1>'
            '<p class="muted">All available cases have been scored. '
            "Scores are saved; you can close this page.</p></div>"
        )
        return PAGE.replace("__PROGRESS__", f"{total} of {total} scored").replace("__BODY__", body)

    dims = expected_dimensions(
        instrumental=item["instrumental"],
        korean=item["korean"],
        duration_seconds=item["duration"],
    )
    parts = [
        f'<div class="card"><h1>{html.escape(item["benchmark_id"])}</h1>',
        f'<div class="muted">Prompt</div><pre>{html.escape(item["prompt"])}</pre>',
    ]
    if item["lyrics"]:
        parts.append(
            '<div class="muted" style="margin-top:12px">Expected lyrics — '
            "score completeness against exactly this</div>"
            f"<pre>{html.escape(item['lyrics'])}</pre>"
        )
    parts.append(
        f'<audio controls preload="none" src="/audio/{html.escape(item["benchmark_id"])}.mp3">'
        "</audio></div>"
    )
    parts.append('<form method="post" action="/save"><div class="card">')
    for group, names in DIMENSION_GROUPS:
        applicable = [n for n in names if n in dims]
        if not applicable:
            continue
        parts.append(f'<div class="grp">{group}</div>')
        for name in applicable:
            label = name.replace("_", " ")
            parts.append(
                f'<div class="dim"><label for="{name}">{label}</label>'
                f'<input id="{name}" name="s_{name}" type="number" min="1" max="10" '
                f'inputmode="numeric" required></div>'
            )
    parts.append('<div class="grp">Artifact tags</div><div class="tags">')
    for tag in ARTIFACT_TAGS:
        parts.append(f'<label><input type="checkbox" name="tag" value="{tag}"> {tag}</label>')
    parts.append(
        '</div><div class="grp">Notes</div>'
        '<textarea name="notes" placeholder="Anything the rubric has no field for"></textarea>'
    )
    parts.append(f'<input type="hidden" name="benchmark_id" value="{item["benchmark_id"]}">')
    if error:
        parts.append(f'<div class="err">{html.escape(error)}</div>')
    parts.append(
        '<div style="margin-top:16px"><button type="submit">Save &amp; next</button>'
        '<button class="ghost" type="submit" formaction="/skip">Skip</button></div>'
    )
    parts.append("</div></form>")
    return PAGE.replace(
        "__PROGRESS__", f"track {index + 1} of {total} · scored in this set: {index}"
    ).replace("__BODY__", "".join(parts))


class Handler(BaseHTTPRequestHandler):
    # Class-level shared state: one server, one session, one queue.
    queue: ClassVar[list[dict[str, Any]]] = []
    store: ScoreStore
    skipped: ClassVar[set[str]] = set()
    evaluator: str = "owner"

    def _scored(self) -> set[str]:
        return {r["benchmark_id"] for r in self.store.load()}

    def _pending(self) -> list[dict[str, Any]]:
        done = self._scored() | self.skipped
        return [i for i in self.queue if i["benchmark_id"] not in done]

    def _send(self, body: str, status: int = 200) -> None:
        payload = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: Any) -> None:  # quieter console
        return

    def do_GET(self) -> None:
        if self.path.startswith("/audio/"):
            benchmark_id = Path(self.path).stem
            item = next((i for i in self.queue if i["benchmark_id"] == benchmark_id), None)
            if item is None:
                self.send_error(404)
                return
            data = listening_copy(benchmark_id, item["_source"]).read_bytes()
            # Range support matters here: without it a browser cannot
            # seek, and scrubbing back to re-hear a phrase is most of
            # what scoring a two-minute track involves.
            span = self.headers.get("Range", "")
            start, end = 0, len(data) - 1
            partial = False
            if span.startswith("bytes="):
                first, _, last = span[6:].partition("-")
                try:
                    if first:
                        start = int(first)
                        end = int(last) if last else end
                    elif last:
                        start = max(0, len(data) - int(last))
                    partial = 0 <= start <= end < len(data)
                except ValueError:
                    partial = False
            chunk = data[start : end + 1] if partial else data
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Accept-Ranges", "bytes")
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
            return
        pending = self._pending()
        total = len(self.queue)
        self._send(render(pending[0] if pending else None, total - len(pending), total))

    def _form(self) -> dict[str, list[str]]:
        from urllib.parse import parse_qs

        length = int(self.headers.get("Content-Length", "0"))
        return parse_qs(self.rfile.read(length).decode(), keep_blank_values=True)

    def do_POST(self) -> None:
        form = self._form()
        benchmark_id = form.get("benchmark_id", [""])[0]
        item = next((i for i in self.queue if i["benchmark_id"] == benchmark_id), None)
        if item is None:
            self.send_error(400, "unknown benchmark id")
            return

        if self.path == "/skip":
            self.skipped.add(benchmark_id)
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()
            return

        raw = {k[2:]: v[0] for k, v in form.items() if k.startswith("s_") and v[0] != ""}
        try:
            scores = validate_scores(
                raw,
                instrumental=item["instrumental"],
                korean=item["korean"],
                duration_seconds=item["duration"],
            )
            tags = validate_tags(form.get("tag", []))
        except P20ScoreError as exc:
            pending = self._pending()
            total = len(self.queue)
            self._send(render(item, total - len(pending), total, error=str(exc)), status=400)
            return

        self.store.append(
            ScoreRecord(
                benchmark_id=benchmark_id,
                evaluator=self.evaluator,
                scored_at=datetime.now(UTC).isoformat(),
                blind=True,
                scores=scores,
                artifact_tags=tags,
                notes=form.get("notes", [""])[0].strip(),
            )
        )
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--evaluator", default="owner")
    parser.add_argument(
        "--prepare-only", action="store_true", help="encode listening copies and exit"
    )
    args = parser.parse_args()

    queue = build_queue()
    if not queue:
        print("no generated P20 cases available to score", file=sys.stderr)
        return 1

    print(f"listening set: {len(queue)} case(s)")
    for item in queue:
        kind = (
            "instrumental"
            if item["instrumental"]
            else ("korean vocal" if item["korean"] else "vocal")
        )
        long_form = " long-form" if item["duration"] >= LONG_FORM_MIN_SECONDS else ""
        listening_copy(item["benchmark_id"], item["_source"])
        print(f"  {item['benchmark_id']:<9} {int(item['duration']):>3}s  {kind}{long_form}")
    print(f"listening copies (RAW-encoded): {CACHE}")

    if args.prepare_only:
        return 0

    Handler.queue = queue
    Handler.store = ScoreStore(SCORES)
    Handler.evaluator = args.evaluator
    already = len({r["benchmark_id"] for r in Handler.store.load()})
    if already:
        print(f"resuming: {already} case(s) already scored")

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\n  http://127.0.0.1:{args.port}\n")
    print(f"scores append to {SCORES}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped; progress is saved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
