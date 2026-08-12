"""Developer-only listening tool: blind scoring and blind A/B.

Deliberately separate from the product UI. This is an evaluation
instrument, not a feature — it is never linked from the app and never
shipped to users.

Generates a self-contained static page plus a tiny local HTTP handler
that appends scores to the JSONL store.
"""

from __future__ import annotations

import json
from typing import Any

from bench.scoring import ARTIFACT_TAGS, RUBRIC_DIMENSIONS, VOCAL_ONLY_DIMENSIONS

PAGE_TITLE = "LUBER benchmark listening (developer tool)"


def build_listening_payload(
    records: list[dict[str, Any]],
    *,
    api_base: str = "http://127.0.0.1:8000",
    blind: bool = True,
) -> list[dict[str, Any]]:
    """Reduce result records to what the evaluator may see.

    In blind mode the configuration, model, and seed are withheld — the
    evaluator must not be able to infer which system produced a track.
    """
    items: list[dict[str, Any]] = []
    for record in records:
        if record.get("status") != "COMPLETED" or not record.get("generation_id"):
            continue
        instrumental = record.get("vocal_gender") == "instrumental"
        item: dict[str, Any] = {
            "benchmark_id": record["benchmark_id"],
            "audio_url": f"{api_base}/v1/generations/{record['generation_id']}/audio?asset=preview",
            "download_url": f"{api_base}/v1/generations/{record['generation_id']}/audio?asset=master&download=true",
            "prompt": record.get("prompt", ""),
            "lyrics": record.get("lyrics", ""),
            "duration_requested": record.get("duration_requested"),
            "instrumental": instrumental,
            "dimensions": [
                d for d in RUBRIC_DIMENSIONS if not (instrumental and d in VOCAL_ONLY_DIMENSIONS)
            ],
        }
        if not blind:
            item["revealed"] = {
                "configuration_id": record.get("configuration_id"),
                "model": record.get("model"),
                "seed": record.get("seed"),
                "genre": record.get("genre"),
                "language": record.get("language"),
            }
        items.append(item)
    return items


def render_listening_page(
    payload: list[dict[str, Any]], *, blind: bool, post_url: str = "/score"
) -> str:
    """Self-contained HTML scoring page."""
    data = json.dumps(payload, ensure_ascii=False)
    tags = json.dumps(list(ARTIFACT_TAGS))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{PAGE_TITLE}</title>
<style>
 body{{font:14px/1.5 -apple-system,system-ui,sans-serif;background:#0b0b0d;color:#eee;
      margin:0;padding:24px;max-width:920px;margin-inline:auto}}
 h1{{font-size:18px}} .muted{{color:#888}}
 .card{{background:#141417;border:1px solid #26262b;border-radius:12px;padding:20px;margin:16px 0}}
 pre{{white-space:pre-wrap;background:#0f0f12;padding:12px;border-radius:8px;
      border:1px solid #26262b;max-height:220px;overflow:auto}}
 audio{{width:100%;margin:12px 0}}
 .grid{{display:grid;grid-template-columns:1fr auto;gap:6px 12px;align-items:center}}
 .grid label{{font-size:13px}}
 input[type=number]{{width:70px;background:#0f0f12;color:#eee;border:1px solid #333;
      border-radius:6px;padding:4px 6px}}
 .tags{{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0}}
 .tags label{{font-size:11px;background:#1b1b20;border:1px solid #2d2d33;border-radius:999px;
      padding:3px 8px;cursor:pointer}}
 textarea{{width:100%;background:#0f0f12;color:#eee;border:1px solid #333;border-radius:8px;
      padding:8px;min-height:60px}}
 button{{background:#7c3aed;color:#fff;border:0;border-radius:8px;padding:10px 18px;
      font-weight:600;cursor:pointer;margin-right:8px}}
 button.secondary{{background:#26262b}}
 .blindnote{{background:#1d1a08;border:1px solid #4d3f0a;color:#e8d48b;padding:8px 12px;
      border-radius:8px;font-size:12px}}
</style></head><body>
<h1>{PAGE_TITLE}</h1>
<p class="muted">Developer evaluation instrument. Not part of the product.</p>
{'<p class="blindnote">BLIND MODE — model, configuration, and seed are hidden until you save.</p>' if blind else ""}
<div id="root"></div>
<script>
const ITEMS = {data};
const TAGS = {tags};
const BLIND = {str(blind).lower()};
let idx = 0;

function render() {{
  const root = document.getElementById('root');
  if (idx >= ITEMS.length) {{
    root.innerHTML = '<div class="card"><b>All tracks scored.</b> ' + ITEMS.length + ' total.</div>';
    return;
  }}
  const it = ITEMS[idx];
  root.innerHTML = `
    <div class="card">
      <div class="muted">Track ${{idx + 1}} / ${{ITEMS.length}}${{BLIND ? '' : ' — ' + it.benchmark_id}}</div>
      <audio controls preload="none" src="${{it.audio_url}}"></audio>
      <div class="muted">Requested duration: ${{it.duration_requested}}s${{it.instrumental ? ' · instrumental' : ''}}</div>
      <h3>Prompt</h3><pre>${{escapeHtml(it.prompt)}}</pre>
      ${{it.lyrics ? '<h3>Lyrics</h3><pre>' + escapeHtml(it.lyrics) + '</pre>' : ''}}
      <h3>Scores (1-10)</h3>
      <div class="grid">
        ${{it.dimensions.map(d => `<label for="s_${{d}}">${{d.replace(/_/g,' ')}}</label>
           <input id="s_${{d}}" type="number" min="1" max="10" step="1">`).join('')}}
      </div>
      <h3>Artifacts</h3>
      <div class="tags">
        ${{TAGS.map(t => `<label><input type="checkbox" value="${{t}}" class="tag"> ${{t}}</label>`).join('')}}
      </div>
      <h3>Notes</h3><textarea id="notes"></textarea>
      <div style="margin-top:14px">
        <button onclick="save()">Save &amp; next</button>
        <button class="secondary" onclick="skip()">Skip</button>
      </div>
    </div>`;
}}

function escapeHtml(s) {{
  return String(s).replace(/[&<>]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;'}})[c]);
}}

function save() {{
  const it = ITEMS[idx];
  const scores = {{}};
  let missing = false;
  for (const d of it.dimensions) {{
    const v = parseInt(document.getElementById('s_' + d).value, 10);
    if (!Number.isInteger(v) || v < 1 || v > 10) {{ missing = true; break; }}
    scores[d] = v;
  }}
  if (missing) {{ alert('Every dimension needs an integer score from 1 to 10.'); return; }}
  const tags = [...document.querySelectorAll('.tag:checked')].map(e => e.value);
  fetch('{post_url}', {{
    method: 'POST', headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{
      benchmark_id: it.benchmark_id, blind: BLIND, scores, artifact_tags: tags,
      notes: document.getElementById('notes').value
    }})
  }}).then(r => {{
    if (!r.ok) {{ alert('Save failed'); return; }}
    idx++; render();
  }}).catch(() => alert('Save failed'));
}}

function skip() {{ idx++; render(); }}
render();
</script></body></html>
"""


def render_ab_page(pairs: list[dict[str, Any]], *, post_url: str = "/ab") -> str:
    """Blind A/B page. Neither side is labelled with its configuration."""
    data = json.dumps(pairs, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LUBER blind A/B (developer tool)</title>
<style>
 body{{font:14px/1.5 -apple-system,system-ui,sans-serif;background:#0b0b0d;color:#eee;
      margin:0;padding:24px;max-width:820px;margin-inline:auto}}
 .card{{background:#141417;border:1px solid #26262b;border-radius:12px;padding:20px;margin:16px 0}}
 audio{{width:100%;margin:8px 0}}
 button{{background:#7c3aed;color:#fff;border:0;border-radius:8px;padding:10px 18px;
      font-weight:600;cursor:pointer;margin-right:8px}}
 select{{background:#0f0f12;color:#eee;border:1px solid #333;border-radius:6px;padding:6px}}
</style></head><body>
<h1>Blind A/B comparison</h1>
<p style="color:#888">Neither track is labelled. Choose on sound alone.</p>
<div id="root"></div>
<script>
const PAIRS = {data};
let i = 0;
function render() {{
  const root = document.getElementById('root');
  if (i >= PAIRS.length) {{ root.innerHTML = '<div class="card"><b>Done.</b></div>'; return; }}
  const p = PAIRS[i];
  root.innerHTML = `<div class="card">
    <div style="color:#888">Pair ${{i + 1}} / ${{PAIRS.length}}</div>
    <h3>Track A</h3><audio controls preload="none" src="${{p.a_url}}"></audio>
    <h3>Track B</h3><audio controls preload="none" src="${{p.b_url}}"></audio>
    <h3>Which is better?</h3>
    <select id="choice"><option>A</option><option>B</option><option>tie</option></select>
    <select id="confidence"><option>Low</option><option>Medium</option><option>High</option></select>
    <div style="margin-top:14px"><button onclick="save()">Save &amp; next</button></div>
  </div>`;
}}
function save() {{
  const p = PAIRS[i];
  fetch('{post_url}', {{
    method: 'POST', headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{
      pair_id: p.pair_id, choice: document.getElementById('choice').value,
      confidence: document.getElementById('confidence').value
    }})
  }}).then(() => {{ i++; render(); }});
}}
render();
</script></body></html>
"""
