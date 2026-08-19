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
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LUBER P20 blind listening</title><style>
 body{font:15px/1.55 -apple-system,system-ui,sans-serif;max-width:920px;margin:0 auto;
      padding:0 20px 120px;background:#0e0e11;color:#e8e8ea}
 h1{font-size:17px;margin:0 0 4px} .muted{color:#9a9aa2;font-size:13px}
 .card{background:#17171c;border:1px solid #26262e;border-radius:10px;padding:18px;margin:16px 0}
 pre{white-space:pre-wrap;font:13px/1.6 ui-monospace,monospace;color:#c9c9d2;margin:6px 0 0}
 audio{width:100%;margin-top:10px}
 /* Sticky so progress and the save control are reachable without
    scrolling a 41-field form. */
 .bar{position:sticky;top:0;z-index:20;background:#0e0e11;border-bottom:1px solid #26262e;
      padding:12px 0 10px;margin:0 -20px;padding-left:20px;padding-right:20px}
 .bar-row{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
 .count{font-size:14px;font-weight:600}
 .count.ok{color:#4ade80} .count.partial{color:#e8e8ea}
 .grow{flex:1}
 .grp{margin:18px 0 6px;font-weight:600;font-size:13px;color:#b9b9c4;
      border-bottom:1px solid #26262e;padding-bottom:5px;
      display:flex;justify-content:space-between;align-items:baseline;gap:8px}
 .grp .sec{font-weight:500;font-size:12px;color:#9a9aa2}
 .grp .sec.ok{color:#4ade80}
 .dim{display:flex;align-items:center;gap:10px;padding:5px 0;border-radius:6px}
 .dim.missing{background:#3a1d1d;box-shadow:0 0 0 1px #7f3535 inset}
 .dim label{flex:1;font-size:13px} .dim input{width:64px;background:#0e0e11;color:#e8e8ea;
      border:1px solid #33333d;border-radius:6px;padding:6px 8px;font-size:14px}
 .dim.missing input{border-color:#c05656}
 .tags{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
 .tags label{font-size:11px;background:#1d1d24;border:1px solid #2e2e38;border-radius:20px;
      padding:4px 9px;cursor:pointer}
 textarea{width:100%;min-height:70px;background:#0e0e11;color:#e8e8ea;border:1px solid #33333d;
      border-radius:8px;padding:9px;font:13px/1.5 inherit}
 button{background:#6c5cff;color:#fff;border:0;border-radius:8px;padding:11px 20px;
      font-size:14px;cursor:pointer}
 button:disabled{background:#33333d;color:#8a8a94;cursor:not-allowed}
 button.ghost{background:#26262e;color:#e8e8ea}
 .banner{border-radius:8px;padding:11px 13px;margin:12px 0;font-size:13px;white-space:pre-wrap;
      display:none}
 .banner.err{background:#3a1d1d;border:1px solid #7f3535;color:#ffb4b4;display:block}
 .banner.ok{background:#16301f;border:1px solid #2f6b45;color:#86efac;display:block}
 .banner.info{background:#1d1d2b;border:1px solid #3a3a55;color:#bfc4ff;display:block}
 .done{text-align:center;padding:60px 0}
 @media (max-width:480px){
   body{padding:0 14px 140px}
   .bar{margin:0 -14px;padding-left:14px;padding-right:14px}
 }
</style></head><body>
<div class="bar">
  <div class="bar-row">
    <div><strong style="font-size:14px">Blind listening — p20</strong>
      <div class="muted" id="progress">__PROGRESS__</div></div>
    <div class="grow"></div>
    <div class="count partial" id="counter">Scored: 0 / 0</div>
    <button type="submit" form="scoreform" id="save" disabled>Save &amp; next</button>
  </div>
</div>
<div id="topbanner" class="banner"></div>
__BODY__
<script>__SCRIPT__</script>
</body></html>"""


#: Shown in draft keys so a future baseline cannot collide with this one.
BASELINE_ID = "luber-baseline-p20-v1"

CLIENT_SCRIPT = r"""
// Client-side guard rails. The server remains authoritative: it
// revalidates every submission and this script cannot loosen it. What
// this adds is knowing *before* clicking, and never losing typed work.
(function () {
  const form = document.getElementById("scoreform");
  if (!form) return;

  const required = JSON.parse(document.getElementById("required-dims").textContent);
  const benchmarkId = form.dataset.benchmark;
  const baseline = form.dataset.baseline;
  const draftKey = `luber.p20h.draft.${baseline}.${benchmarkId}`;
  const counter = document.getElementById("counter");
  const saveButton = document.getElementById("save");
  const banner = document.getElementById("topbanner");
  const completedCount = Number(form.dataset.completed);
  const totalCount = Number(form.dataset.total);

  const scoreInput = (name) => form.querySelector(`[name="s_${name}"]`);
  const filled = (name) => {
    const el = scoreInput(name);
    return el && el.value !== "" && Number(el.value) >= 1 && Number(el.value) <= 10;
  };

  function show(kind, message) {
    banner.className = "banner " + kind;
    banner.textContent = message;
    if (kind === "err") window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function refresh() {
    const done = required.filter(filled).length;
    const complete = done === required.length;
    counter.textContent = complete
      ? `Scored: ${done} / ${required.length} · Ready to save`
      : `Scored: ${done} / ${required.length}`;
    counter.className = "count " + (complete ? "ok" : "partial");
    saveButton.disabled = !complete;

    // Per-section progress, using the rubric's own grouping.
    form.querySelectorAll("[data-section]").forEach((header) => {
      const names = JSON.parse(header.dataset.section);
      const n = names.filter(filled).length;
      const badge = header.querySelector(".sec");
      badge.textContent =
        n === names.length ? `${n} / ${names.length} ✓` : `${n} / ${names.length}`;
      badge.className = "sec" + (n === names.length ? " ok" : "");
    });
  }

  // ---- draft autosave -------------------------------------------------
  function collectDraft() {
    const scores = {};
    required.forEach((name) => {
      const el = scoreInput(name);
      if (el && el.value !== "") scores[name] = el.value;
    });
    return {
      scores,
      tags: [...form.querySelectorAll('input[name="tag"]:checked')].map((t) => t.value),
      notes: form.querySelector('[name="notes"]').value,
    };
  }

  // Once a score is confirmed persisted, this track's draft must stay
  // gone. Without the flag the unload flush below re-creates it from the
  // still-populated form as the page navigates away, and the next visit
  // "restores" a draft for a track that is already saved.
  let persisted = false;

  function saveDraft() {
    if (persisted) return;
    try {
      localStorage.setItem(draftKey, JSON.stringify(collectDraft()));
    } catch {
      /* private mode: the form still works, just without recovery */
    }
  }

  function restoreDraft() {
    let raw = null;
    try {
      raw = localStorage.getItem(draftKey);
    } catch {
      return false;
    }
    if (!raw) return false;
    try {
      const draft = JSON.parse(raw);
      Object.entries(draft.scores || {}).forEach(([name, value]) => {
        const el = scoreInput(name);
        if (el) el.value = value;
      });
      (draft.tags || []).forEach((tag) => {
        const el = form.querySelector(`input[name="tag"][value="${tag}"]`);
        if (el) el.checked = true;
      });
      if (draft.notes) form.querySelector('[name="notes"]').value = draft.notes;
      return Object.keys(draft.scores || {}).length > 0 || (draft.tags || []).length > 0;
    } catch {
      return false;
    }
  }

  form.addEventListener("input", () => {
    saveDraft();
    refresh();
  });
  form.addEventListener("change", () => {
    saveDraft();
    refresh();
  });
  // Autosave makes a confirm dialog unnecessary; this is just a flush.
  window.addEventListener("beforeunload", saveDraft);
  window.addEventListener("pagehide", saveDraft);

  // ---- submit ---------------------------------------------------------
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const missing = required.filter((name) => !filled(name));
    form.querySelectorAll(".dim").forEach((row) => row.classList.remove("missing"));
    if (missing.length) {
      missing.forEach((name) => scoreInput(name)?.closest(".dim")?.classList.add("missing"));
      show("err", `Not saved — ${missing.length} score(s) still needed: ${missing.join(", ")}`);
      const first = scoreInput(missing[0]);
      first?.scrollIntoView({ block: "center", behavior: "smooth" });
      first?.focus();
      return;
    }

    saveButton.disabled = true;
    saveButton.textContent = "Saving…";
    try {
      const response = await fetch("/save", { method: "POST", body: new FormData(form) });
      if (!response.ok) {
        // Server rejected it. The draft stays: nothing the user typed is
        // discarded before persistence is confirmed.
        const detail = await response.text();
        const reason = (detail.match(/data-error="([^"]*)"/) || [])[1] || `HTTP ${response.status}`;
        show("err", `Score was NOT saved. Your draft is preserved.\n${reason}`);
        saveButton.disabled = false;
        saveButton.textContent = "Save & next";
        return;
      }
      // Persisted. Only now is the draft safe to remove.
      persisted = true;
      try {
        localStorage.removeItem(draftKey);
      } catch {
        /* nothing to clean up */
      }
      show("ok", `Score saved — ${completedCount + 1} / ${totalCount} complete`);
      setTimeout(() => window.location.assign("/"), 700);
    } catch (error) {
      show(
        "err",
        "Score was NOT saved. Your draft is preserved.\nThe server could not be reached."
      );
      saveButton.disabled = false;
      saveButton.textContent = "Save & next";
    }
  });

  document.getElementById("skip")?.addEventListener("click", () => {
    const body = new FormData();
    body.append("benchmark_id", benchmarkId);
    fetch("/skip", { method: "POST", body }).then(() => window.location.assign("/"));
  });

  if (restoreDraft()) show("info", "Draft restored — your earlier scores for this track are back.");
  refresh();
})();
"""


def render(
    item: dict[str, Any] | None,
    index: int,
    total: int,
    error: str = "",
    completed: int | None = None,
) -> str:
    """Draw the current track.

    ``index`` is the position in the queue; ``completed`` is how many
    scores are actually persisted. They differ once a track is skipped,
    and conflating them would tell the listener they had finished work
    they had not — the count has to come from the store.
    """
    if item is None:
        body = (
            '<div class="done card"><h1>Session complete</h1>'
            '<p class="muted">All available cases have been scored. '
            "Scores are saved; you can close this page.</p></div>"
        )
        done = total if completed is None else completed
        return (
            PAGE.replace("__PROGRESS__", f"Completed {done} / {total}")
            .replace("__BODY__", body)
            .replace("__SCRIPT__", "")
        )

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
    parts.append(
        # novalidate: the browser's own tooltip lands on whichever field it
        # picks, often off-screen on a 41-field form — the exact failure
        # this page exists to remove. The handler below reports every
        # missing field at the top and scrolls to the first one.
        f'<form id="scoreform" method="post" action="/save" novalidate '
        f'data-benchmark="{html.escape(item["benchmark_id"])}" '
        f'data-baseline="{BASELINE_ID}" '
        f'data-completed="{index if completed is None else completed}" '
        f'data-total="{total}">'
        '<div class="card">'
    )
    for group, names in DIMENSION_GROUPS:
        applicable = [n for n in names if n in dims]
        if not applicable:
            continue
        parts.append(
            f"<div class=\"grp\" data-section='{json.dumps(applicable)}'>"
            f'<span>{group}</span><span class="sec">0 / {len(applicable)}</span></div>'
        )
        for name in applicable:
            label = name.replace("_", " ")
            parts.append(
                f'<div class="dim"><label for="{name}">{label}</label>'
                f'<input id="{name}" name="s_{name}" type="number" min="1" max="10" '
                f'inputmode="numeric" required></div>'
            )
    parts.append(
        '<div class="grp"><span>Artifact tags</span>'
        '<span class="muted">optional</span></div><div class="tags">'
    )
    for tag in ARTIFACT_TAGS:
        parts.append(f'<label><input type="checkbox" name="tag" value="{tag}"> {tag}</label>')
    parts.append(
        '</div><div class="grp"><span>Notes</span><span class="muted">optional</span></div>'
        '<textarea name="notes" placeholder="Anything the rubric has no field for"></textarea>'
    )
    parts.append(f'<input type="hidden" name="benchmark_id" value="{item["benchmark_id"]}">')
    parts.append(
        '<div style="margin-top:16px">'
        '<button class="ghost" type="button" id="skip">Skip this track</button></div>'
    )
    parts.append("</div></form>")
    # The required list drives the counter and the client-side guard.
    parts.append(
        f'<script type="application/json" id="required-dims">{json.dumps(list(dims))}</script>'
    )
    if error:
        parts.append(f'<div data-error="{html.escape(error)}"></div>')

    return (
        PAGE.replace(
            "__PROGRESS__",
            f"Track {index + 1} / {total} · Completed "
            f"{index if completed is None else completed} / {total}",
        )
        .replace("__BODY__", "".join(parts))
        .replace("__SCRIPT__", CLIENT_SCRIPT)
    )


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
        scored = len(self._scored())
        self._send(
            render(
                pending[0] if pending else None,
                total - len(pending),
                total,
                completed=scored,
            )
        )

    def _form(self) -> dict[str, list[str]]:
        """Parse either an ordinary form post or a fetch() FormData body."""
        import cgi
        from urllib.parse import parse_qs

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        content_type = self.headers.get("Content-Type", "")
        if content_type.startswith("multipart/form-data"):
            environ = {
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": str(length),
            }
            import io

            parsed = cgi.FieldStorage(fp=io.BytesIO(raw), environ=environ, keep_blank_values=True)
            out: dict[str, list[str]] = {}
            for key in parsed.keys():
                values = parsed[key]
                out[key] = [v.value for v in values] if isinstance(values, list) else [values.value]
            return out
        return parse_qs(raw.decode(), keep_blank_values=True)

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
            self._send(
                render(
                    item, total - len(pending), total, error=str(exc), completed=len(self._scored())
                ),
                status=400,
            )
            return

        # A refresh or a back-navigation must not append the same track
        # twice. Append-only auditability is kept: nothing is rewritten,
        # the second write is simply refused.
        if benchmark_id in self._scored():
            self._send(
                render(
                    item,
                    len(self._scored()),
                    len(self.queue),
                    error=f"{benchmark_id} is already scored; it was not saved again.",
                    completed=len(self._scored()),
                ),
                status=409,
            )
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
        # 200 rather than a redirect: the client submits with fetch and
        # needs to distinguish a confirmed save from a rejection before
        # it clears the local draft.
        self._send("saved", status=200)


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
