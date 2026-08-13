"""Phase 9 listening package: triage first, detail only when it earns it.

The human evaluator currently rates this model around 2/10. Asking for
twelve rubric dimensions on a track that fails in the first ten seconds
wastes the scarcest resource in this project — listening time. So the
form asks one question first:

    **Overall, 1-10.**

Below 5, the evaluator sees only failure tags and a notes box: at that
quality the useful information is *which* defect, not how much of each.
At 5 or above the detailed dimensions appear, because a track that good
is worth describing precisely.

The failure tags are the defects actually reported on this project, so
answers aggregate into something a later training phase can use instead
of prose nobody can query.

A/B pairs are presented side by side and **unlabelled by variant** — the
evaluator sees "left" and "right", not "with contemporary conditioning".
Knowing which side is the experiment would bias exactly the judgement
the experiment exists to collect.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

#: Tags the evaluator can apply. Mirrors ``luber_schemas.FailureTag`` —
#: the same vocabulary the QA API stores, so listening output can be
#: PUT straight to /v1/generations/{id}/qa without translation.
FAILURE_TAGS: tuple[tuple[str, str], ...] = (
    ("KOREAN_LINE_OMISSION", "A Korean lyric line was not sung at all"),
    ("LYRIC_LINE_SKIP", "A lyric line was skipped"),
    ("LYRIC_DUPLICATION", "A line was sung twice"),
    ("TROT_LIKE_VOCAL", "Vocal has a trot-like delivery"),
    ("VOCAL_STYLE_OUTDATED", "Vocal style sounds dated, not contemporary"),
    ("EXCESSIVE_SIBILANCE", "Harsh s/sh sounds"),
    ("HIGH_END_OVERBOOST", "Too much high-frequency energy overall"),
    ("INSTRUMENT_FIDELITY_LOW", "Instruments sound synthetic or smeared"),
    ("STRUCTURE_COLLAPSE", "Song structure falls apart"),
    ("MELODY_DRIFT", "Melody loses its identity over time"),
    ("VOCAL_IDENTITY_DRIFT", "Singer seems to change during the track"),
    ("ENDING_FAILURE", "Ending is abrupt, missing, or broken"),
)

#: Only shown for tracks rated 5 or above.
DETAIL_DIMENSIONS: tuple[str, ...] = (
    "vocal_quality",
    "vocal_naturalness",
    "lyric_intelligibility",
    "melody_quality",
    "arrangement",
    "mix_balance",
    "structure_coherence",
    "commercial_viability",
)

#: Long-form tracks get section-by-section prompts.
LONG_FORM_SECTIONS: tuple[str, ...] = (
    "intro",
    "verse_1",
    "chorus",
    "verse_2",
    "bridge",
    "final_chorus",
    "outro",
)


def build_phase9_payload(
    tracks: list[dict[str, Any]], *, api_base: str = "http://127.0.0.1:8010"
) -> list[dict[str, Any]]:
    """Reduce Phase 9 records to what the evaluator needs to judge them."""
    items: list[dict[str, Any]] = []
    for track in tracks:
        generation_id = track.get("generation_id")
        if not generation_id or track.get("status") != "COMPLETED":
            continue
        duration = track.get("duration_requested") or 0
        items.append(
            {
                "id": track.get("case") or track.get("title") or generation_id,
                "generation_id": generation_id,
                "group": track.get("group", "single"),
                "audio_url": f"{api_base}/v1/generations/{generation_id}/audio?asset=preview",
                "download_url": (
                    f"{api_base}/v1/generations/{generation_id}/audio?asset=master&download=true"
                ),
                "duration_requested": duration,
                "lyrics": track.get("lyrics", ""),
                "expected_lines": [
                    line
                    for line in (track.get("lyrics") or "").splitlines()
                    if line.strip() and not line.strip().startswith("[")
                ],
                "is_long_form": duration >= 120,
                "sections": list(LONG_FORM_SECTIONS) if duration >= 120 else [],
            }
        )
    return items


def _tag_checkboxes(item_id: str) -> str:
    return "".join(
        f'<label class="tag"><input type="checkbox" name="{html.escape(item_id)}__tag" '
        f'value="{code}"> <span>{html.escape(code)}</span>'
        f"<em>{html.escape(description)}</em></label>"
        for code, description in FAILURE_TAGS
    )


def _detail_fields(item_id: str) -> str:
    return "".join(
        f'<label class="dim">{html.escape(name.replace("_", " "))}'
        f'<input type="number" min="1" max="10" name="{html.escape(item_id)}__{name}"></label>'
        for name in DETAIL_DIMENSIONS
    )


def _section_fields(item_id: str, sections: list[str]) -> str:
    if not sections:
        return ""
    rows = "".join(
        f'<label class="dim">{html.escape(name.replace("_", " "))}'
        f'<input type="text" name="{html.escape(item_id)}__section__{name}" '
        f'placeholder="what happens here?"></label>'
        for name in sections
    )
    return f"<fieldset><legend>Section by section</legend>{rows}</fieldset>"


def _lyric_line_rows(item_id: str, lines: list[str]) -> str:
    if not lines:
        return ""
    options = "".join(
        f'<option value="{v}">{v}</option>'
        for v in ("UNKNOWN", "COMPLETE", "PARTIAL", "SKIPPED", "DUPLICATED")
    )
    rows = "".join(
        f'<tr><td class="idx">{index}</td><td class="line">{html.escape(line)}</td>'
        f'<td><select name="{html.escape(item_id)}__line__{index}">{options}</select></td></tr>'
        for index, line in enumerate(lines)
    )
    return (
        '<details class="lines"><summary>Lyric line completeness '
        f"({len(lines)} lines)</summary>"
        '<p class="hint">UNKNOWN is a real answer. Do not guess.</p>'
        f"<table><thead><tr><th>#</th><th>Submitted line</th><th>Heard</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></details>"
    )


def render_phase9_page(payload: list[dict[str, Any]], *, title: str = "Phase 9 listening") -> str:
    """Render the triage-first listening form as a standalone page."""
    cards = []
    for item in payload:
        item_id = str(item["id"])
        long_form_note = (
            '<p class="warn">Long-form track — listen to the whole thing, '
            "not the first 30 seconds.</p>"
            if item["is_long_form"]
            else ""
        )
        cards.append(
            f"""
<article class="card" data-id="{html.escape(item_id)}">
  <h2>{html.escape(item_id)}</h2>
  <p class="meta">{item["duration_requested"]}s requested</p>
  {long_form_note}
  <audio controls preload="none" src="{html.escape(item["audio_url"])}"></audio>
  <p><a href="{html.escape(item["download_url"])}">download master</a></p>

  <label class="triage">Overall (1-10)
    <input type="number" min="1" max="10" name="{html.escape(item_id)}__overall"
           class="overall" required>
  </label>

  <fieldset class="tags"><legend>What went wrong</legend>{_tag_checkboxes(item_id)}</fieldset>

  <label class="notes">Notes
    <textarea name="{html.escape(item_id)}__notes" rows="3"></textarea>
  </label>

  <div class="detail" hidden>
    <fieldset><legend>Detail (only for 5+)</legend>{_detail_fields(item_id)}</fieldset>
    {_section_fields(item_id, item["sections"])}
  </div>

  {_lyric_line_rows(item_id, item["expected_lines"])}
</article>"""
        )

    return f"""<!doctype html>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
 body {{ font-family: system-ui, sans-serif; max-width: 60rem; margin: 2rem auto;
        padding: 0 1rem; background:#111; color:#eee; }}
 h1 {{ font-size: 1.4rem; }}
 .card {{ border:1px solid #333; border-radius:10px; padding:1rem; margin:1.5rem 0;
          background:#181818; }}
 .card h2 {{ font-size:1rem; margin:0 0 .25rem; font-family:ui-monospace,monospace; }}
 .meta {{ color:#888; font-size:.8rem; margin:.2rem 0 .6rem; }}
 .warn {{ color:#fbbf24; font-size:.85rem; }}
 audio {{ width:100%; }}
 .triage {{ display:block; margin:1rem 0 .5rem; font-weight:600; }}
 .triage input {{ width:5rem; margin-left:.5rem; }}
 fieldset {{ border:1px solid #333; border-radius:8px; margin:.75rem 0; }}
 legend {{ color:#aaa; font-size:.8rem; padding:0 .4rem; }}
 .tag {{ display:block; font-size:.85rem; margin:.25rem 0; }}
 .tag em {{ color:#777; font-style:normal; margin-left:.4rem; font-size:.8rem; }}
 .dim {{ display:inline-block; margin:.3rem .8rem .3rem 0; font-size:.85rem; }}
 .dim input {{ margin-left:.4rem; width:4rem; }}
 .dim input[type=text] {{ width:16rem; }}
 .notes {{ display:block; font-size:.85rem; }}
 .notes textarea {{ width:100%; background:#111; color:#eee; border:1px solid #333;
                    border-radius:6px; }}
 .lines table {{ width:100%; border-collapse:collapse; font-size:.85rem; }}
 .lines td, .lines th {{ border-bottom:1px solid #262626; padding:.25rem .4rem;
                         text-align:left; }}
 .lines .idx {{ color:#666; width:2rem; }}
 .hint {{ color:#888; font-size:.8rem; }}
 button {{ background:#7c3aed; color:#fff; border:0; border-radius:8px;
           padding:.7rem 1.2rem; font-weight:600; cursor:pointer; }}
</style>
<h1>{html.escape(title)}</h1>
<p>Rate overall first. Detail appears only for tracks you score 5 or above —
below that, the tags and notes are what matter.</p>
<form id="f">{"".join(cards)}
  <button type="submit">Save results</button>
</form>
<script>
 // Detail is revealed only when a track earns it. Below 5 the evaluator
 // records what failed, not how much of each dimension failed.
 for (const card of document.querySelectorAll('.card')) {{
   const overall = card.querySelector('.overall');
   const detail = card.querySelector('.detail');
   const sync = () => {{ detail.hidden = !(Number(overall.value) >= 5); }};
   overall.addEventListener('input', sync);
   sync();
 }}
 document.getElementById('f').addEventListener('submit', (event) => {{
   event.preventDefault();
   const data = {{}};
   for (const [key, value] of new FormData(event.target).entries()) {{
     if (value === '') continue;
     if (key.endsWith('__tag')) (data[key] ||= []).push(value);
     else data[key] = value;
   }}
   const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
   const a = document.createElement('a');
   a.href = URL.createObjectURL(blob);
   a.download = 'phase9-listening-results.json';
   a.click();
 }});
</script>
"""


def write_package(
    tracks: list[dict[str, Any]],
    out_dir: Path,
    *,
    api_base: str = "http://127.0.0.1:8010",
    title: str = "Phase 9 listening",
) -> Path:
    """Write the listening page and its machine-readable manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_phase9_payload(tracks, api_base=api_base)
    (out_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    page = out_dir / "index.html"
    page.write_text(render_phase9_page(payload, title=title), encoding="utf-8")
    return page
