# Phase 8 — Advanced Controls, Songcraft, and Observability

> **What this phase changes:** how precisely a user can *ask* for a track,
> and how completely we can *see* what was asked for.
>
> **What this phase does not change:** how good the resulting music is.
> Phase 8 adds controllability and observability. It does not improve the
> underlying learned musical quality of the model by one decibel. A track
> generated with `bpm=128, key_scale=F# minor` is the same model doing the
> same thing with more specific conditioning — not a better model. Closing
> the quality gap to Suno 4.5-class output remains a model, data, and
> fine-tuning problem, and nothing in this phase should be read as
> progress against it.

Pinned engine for every claim below: **ACE-Step 1.5 @
`6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`** (see
[ACE_STEP_UPSTREAM_AUDIT.md](ACE_STEP_UPSTREAM_AUDIT.md)).

---

## 1. Advanced controls

Three musical parameters are exposed. All three are **optional**, all
three default to **unset**, and unset means *the model decides* — which
is exactly what happened before Phase 8 existed.

| Control | Wire field | Values offered | Source of those values |
|---|---|---|---|
| BPM | `bpm` (int) | 30–300 | `acestep/constants.py`: `BPM_MIN = 30`, `BPM_MAX = 300` |
| Key / Scale | `key_scale` (str) | 42 values, `"<note><accidental> <mode>"` e.g. `"F# minor"` | `acestep/constants.py`: `VALID_KEYSCALES` = notes × accidentals × modes |
| Time Signature | `time_signature` (str) | `"2"`, `"3"`, `"4"`, `"6"` | `acestep/constants.py`: `VALID_TIME_SIGNATURES = [2, 3, 4, 6]` |

### Capability evidence

These are not guesses about what the engine might accept. Each was read
out of the pinned source:

- **The request fields exist and are named this way.**
  `acestep/api/http/release_task_models.py:37-39` declares
  `bpm: Optional[int]`, `key_scale: str = ""`, `time_signature: str = ""`.
  `acestep/api/http/release_task_param_parser.py:20-24` accepts those
  spellings (plus `keyscale`/`keyScale`/`key` aliases we do not use).

- **They reach the DiT, with or without the LM.**
  `acestep/inference.py:826-831` places `"bpm"`, `"key_scale"` and
  `"time_signature"` directly into `dit_generate_kwargs`, unconditionally.
  The `user_metadata` block at `inference.py:651-670` that also consumes
  them sits inside `if use_lm:` — so that path is LM-only, but the DiT
  path is not. LUBER runs DiT-only (`thinking=false`), and the controls
  still apply.

- **Time signature is the bare numerator, not a fraction.**
  `acestep/constrained_logits_processor.py:206` constrains the model's
  `timesignature` vocabulary to the integers `[2, 3, 4, 6]`. Sending
  `"4/4"` would be off-vocabulary. The UI shows `4 (4/4)` as a label and
  sends `4`.

- **Key/scale is stored one way only.** Upstream `KEYSCALE_ACCIDENTALS`
  includes the Unicode `♯`/`♭` alongside ASCII `#`/`b`. LUBER offers only
  the ASCII forms, so one key has exactly one representation in our data.
  A Unicode accidental is rejected at the edge rather than silently
  becoming a second spelling of the same key.

### Default behaviour, precisely

An unset control is **omitted from the provider payload entirely** — not
sent as `""`, not sent as a house default. Upstream treats `""` as "not
specified", so an empty string would work, but it would be
indistinguishable in the request trace from a deliberate choice. Omission
keeps "the user chose nothing" and "the user chose something" separable
after the fact.

The consequence is the property the whole phase rests on: **a request
with no advanced controls produces a byte-identical provider payload to
Phase 7.** This is enforced by a frozen golden payload captured from the
committed Phase 7 provider at `c4ea7ea`, asserted both at the
`_build_payload` level and over the wire
(`packages/generation-client/tests/test_ace_step_advanced_controls.py`).

### Engine parameters deliberately *not* exposed

Recorded in `UNEXPOSED_ENGINE_PARAMETERS`
(`packages/schemas/src/luber_schemas/songcraft.py`) so the reason lives
next to the absence:

| Parameter | Why not |
|---|---|
| `guidance_scale` | auto-corrected to 1.0 by the engine for turbo checkpoints; a control the engine overwrites is a lie |
| `infer_method` | ode/sde sampler choice; Phase 6 A/B showed no user-visible win, and sde worsened end-level drift |
| `lm_negative_prompt` | the only negative-prompt field upstream, and it is LM-only; the DiT has none and the LM is disabled on this host |
| `thinking` | requires the 1.7B LM, disabled on this host |
| `use_adg` | base-model only; not applicable to `acestep-v15-turbo` |

---

## 2. Song structure editor

The lyrics field is still a plain textarea, and **plain untagged lyrics
remain completely valid**. What Phase 8 adds around it:

- **A section-tag palette** (`SECTION_TAG_PALETTE`) — clicking a tag
  inserts it on its own line at the caret. This is the only way the
  editor ever writes into the lyrics, and it requires an explicit click.
- **A structure outline** showing how the backend parsed the tags,
  including tags it did *not* recognise (shown in amber, passed through
  as written) and tagged sections with no content.

Rules the editor follows:

- **Structure is never forced.** No validation requires a tag. Untagged
  lyrics report "the lyrics are sent as one block. That is fine; tags are
  optional."
- **Lyrics are never rewritten automatically.** Nothing reflows,
  re-tags, normalises whitespace, or "fixes" the text. What is stored and
  what is sent to the engine is byte-for-byte what was typed — asserted
  in `apps/api/tests/test_generations_phase8_api.py`
  (`test_advisories_do_not_alter_the_submitted_lyrics`) and in the web
  suite (`never rewrites lyrics on its own`).
- **A tag on a line with lyrics is not a section tag.** `she said
  [whisper] softly` is an ad-lib, not a marker; only a line consisting
  solely of `[...]` is treated as a section.

Recognised tags, aliases (`Hook` → chorus, `Breakdown` → break, `Solo` →
instrumental, …) and ordinals (`[Verse 2]` → verse, index 2) are defined
in `_SECTION_ALIASES`.

---

## 3. Pre-flight advisories

### Semantics

An advisory is **a note beside the lyrics, never a gate**. Specifically:

- Advisories have two levels, `info` and `warning`. Neither blocks.
- Nothing in the advisory system can prevent a generation. The create
  endpoint computes advisories, records them, returns them, and proceeds.
- The only rejections in the create path are objectively invalid API
  input: a BPM outside the engine's range, a key/scale the engine does
  not accept, a duration outside bounds, a missing title, a parent that
  does not exist or is not the caller's, a variation label with no
  parent. A heuristic *opinion* is never one of them.
- Advisories are computed from the lyrics but never modify them.

### Families

| Family | Codes |
|---|---|
| Structure | `UNKNOWN_SECTION_TAG`, `EMPTY_SECTION`, `NO_CHORUS`, `DUPLICATE_SECTION_NUMBER`, `INTRO_NOT_FIRST`, `OUTRO_NOT_LAST`, `LYRICS_IN_INSTRUMENTAL` |
| Duration / density | `LYRICS_DENSE_FOR_DURATION`, `LYRICS_SPARSE_FOR_DURATION`, `MANY_SECTIONS_FOR_DURATION` |
| Korean pre-flight | `KOREAN_LYRICS_LANGUAGE_MISMATCH`, `KOREAN_LANGUAGE_WITHOUT_KOREAN_LYRICS`, `KOREAN_LINE_TOO_LONG`, `KOREAN_MIXED_SCRIPT_LINE` |

The density and structure thresholds **are judgement calls**, not engine
facts. They are named constants (`MIN_SYLLABLES_PER_SECOND`,
`MAX_SYLLABLES_PER_SECOND`, `SINGABLE_FRACTION`,
`MIN_SECONDS_PER_SECTION`, `KOREAN_LINE_SYLLABLE_LIMIT`,
`KOREAN_DOMINANCE_THRESHOLD`) so they can be tuned against listening
evidence rather than argued about in prose. They produce warnings only.

### Korean specifics

- Text is normalised to NFC before analysis — macOS and some IMEs hand
  back NFD, and a decomposed jamo sequence must not read as non-Hangul.
- Syllable estimation counts Hangul and CJK per character (one block =
  one syllable) and Latin by vowel group. It over-counts silent vowels
  and under-counts diphthongs; it is only ever used to separate "far too
  many words" from "far too few".

### One implementation, two surfaces

`POST /v1/generations/preflight` runs the *same* `preflight()` the create
path runs and creates nothing — no row, no job, no provider call. The
editor calls it (debounced, single-flight) as the user types. This is
deliberate: a browser-side reimplementation of the heuristics could
disagree with the stored advisories, and the two disagreeing would be
worse than having no live preview at all. Asserted by
`test_preflight_agrees_with_what_create_records`.

A failed pre-flight is silent. It clears the panel; it never becomes a
user-facing error, and it never blocks Generate.

---

## 4. Request trace

`request_trace` answers exactly one question after the fact: **what did
LUBER actually send to the generation provider?**

- Written by `GenerationService._record_trace` **before** the provider
  runs, so a *failed* generation is as inspectable as a successful one.
- Best-effort: if building or storing the trace raises, the failure is
  logged and the generation proceeds. A diagnostic must never fail the
  thing the user asked for.
- `NULL` means "not recorded" — the row predates Phase 8, the provider
  implements no trace, or the run never reached the provider. It never
  means "empty".
- Providers that do not implement `describe_request` inherit a concrete
  base returning `{}`, and the service stores nothing. Phase 7-era
  providers are unaffected.

### What it must never contain

API keys, credentials, authorization headers, secrets, raw environment
variables, hostnames, base URLs, or local filesystem paths. The
ACE-Step implementation reads only the compiled prompt/lyrics, the
conditioning it added or skipped, the model name, engine version,
inference steps, and the payload.

This is tested, not just asserted in prose: the provider is configured
with a distinctive API key and internal hostname, the trace is
serialized, and the test fails if either string — or `api_key`,
`base_url`, `Authorization`, `Bearer`, `token`, `password`, or the temp
directory path — appears anywhere in it
(`test_describe_request_never_leaks_credentials_or_host`,
`test_describe_request_never_leaks_local_paths`,
`test_persisted_ace_step_trace_carries_no_secrets`).

### What it does contain

`provider`, `model`, `engine_version`, `inference_steps`,
`original_prompt`, `compiled_prompt`, `original_lyrics`,
`compiled_lyrics`, `vocal_language`, `instrumental`,
`added_conditioning`, `skipped_conditioning`, and the full `payload`.
The original-vs-compiled pairing is the useful part: it shows what the
prompt compiler added on the user's behalf.

---

## 5. Lineage and "Generate again"

### Semantics

- `parent_generation_id` — the generation this one was started from.
  Optional. `ON DELETE SET NULL`: deleting an original does **not**
  delete what was made from it, and the child keeps all of its own
  settings.
- `variation_label` — a short human label for the relationship.
  **Rejected without a parent** (422): a label naming a lineage that does
  not exist is not silently dropped, because the caller believes it is
  recording something.
- A parent must be one the caller may access. The check reuses
  `caller_may_access`, the same ownership rule the audio endpoints use —
  no second authorization system was invented. An inaccessible parent
  returns a response **identical** to a non-existent one, so lineage
  cannot be used as an existence oracle for other people's work.

### What "Generate again" actually is

Clicking **Generate again** on a completed track:

1. copies that generation's settings (title, prompt, lyrics, vocal,
   language, duration, BPM, key/scale, time signature) into a fresh
   draft;
2. records the previous generation as the new one's parent;
3. lets the user change anything before submitting.

### What it is explicitly *not*

**It is not audio-to-audio variation.** No audio from the parent is
reused, referenced, or conditioned on. The pinned engine does expose
audio-conditioned task types (`cover`, `repaint`, `lego`, `extract`,
`reference_audio`), and LUBER has wired **none** of them — their
semantics, quality, and stability are unverified against this
deployment. Implementing a "variation" button that silently re-rolled a
text-to-music generation while implying audio similarity would be
pretending to a capability we have not built. Two identical submissions
differ only by the engine's own sampling.

This is the lineage and repeat-generation *foundation*. True variation
needs the audio-conditioned engine paths wired and verified, and that is
a later phase.

---

## 6. Verification performed

| Gate | Result |
|---|---|
| Python tests (`pytest`) | 606 passed, 1 skipped (baseline was 423 passed, 1 skipped) |
| `ruff check` / `ruff format --check` | clean |
| `mypy .` | clean, 82 source files |
| Web tests (`vitest`) | 95 passed (baseline 59) |
| `tsc --noEmit` | clean |
| `eslint` | clean |
| `next build` | succeeds |
| `alembic upgrade head` on PostgreSQL 15 | clean, single head `0004` |
| `alembic downgrade 0003` → `upgrade head` | clean round trip, data preserved |
| `ON DELETE SET NULL` on PostgreSQL | verified live: child survives, parent link nulled, own settings intact |

### Migration parity

`alembic check` against the migrated PostgreSQL schema reports **no drift
on the `generations` table** — the Phase 8 migration matches the ORM
exactly (7 columns, types, nullability, the self-FK and its index).

It does report one pre-existing mismatch on `audio_assets`: the database
constraint is named `uq_audio_assets_generation_id_asset_type` while the
model declares `uq_audio_assets_generation_id`. The columns are
identical; only the name differs. **This is baseline, not Phase 8** —
running the same check against the committed Phase 7 code
(`c4ea7ea`) on a fresh database produces the identical failure. It is
left untouched rather than silently repaired inside a Phase 8 changeset.

---

## 7. Known limitations

1. **Phase 8 does not improve musical quality.** Restated because it is
   the single most important caveat: this is controllability and
   observability work.

2. **Duration is still capped at 60s in the UI.** `DURATION_PRESETS` in
   `songcraft.py` lists `(30, 60, 120, 180, 240)` with a comment claiming
   240s is "the longest verified in Phase 8". No evidence of that
   verification exists in this repository, so the form continues to offer
   only 30s and 60s. Exposing an unverified duration would be inventing a
   capability. Raise the presets when a real end-to-end run on the target
   hardware backs them.

3. **SQLite does not enforce the lineage foreign key.** The Python test
   suites build their schema with `Base.metadata.create_all` on SQLite,
   which ignores foreign keys unless `PRAGMA foreign_keys=ON`. The API
   tests therefore assert only that a child *survives* its parent's
   deletion; the `SET NULL` itself is verified against PostgreSQL (§6).

4. **`variation_label` has no UI.** The field is accepted, validated,
   stored, and returned, and "Generate again" does not set it. It is
   backend-ready for a later labelling feature.

5. **The advisory thresholds are unvalidated against listening tests.**
   They come from the Phase 5 benchmark set and reasoning, not from a
   study correlating density with perceived quality.

6. **Syllable estimation is approximate**, especially for Latin text.

7. **No authentication.** `caller_may_access` compares an `X-User-Id`
   header against `user_id`, and generations created through the API
   still carry no owner. The lineage ownership check is real and enforced
   against owned rows today, but it inherits the placeholder identity
   source. Phase 8 deliberately added no auth.

8. **Advanced controls are not verified to *audibly* work.** We have
   proven the parameters reach the DiT conditioning path in the pinned
   source and that LUBER sends them correctly. We have not run a
   listening study confirming that asking for `bpm=128` yields a track at
   128 BPM. That is an engine-behaviour question, and the honest status
   is: plumbed and evidenced, not perceptually validated.

---

## 8. What Phase 8 does NOT solve

- Musical quality, arrangement coherence, mix quality, or vocal realism.
- The gap to Suno 4.5-class output.
- Long-form (>60s) generation on this deployment.
- Audio-to-audio variation, cover, repaint, or stem extraction.
- Style transfer or reference-audio conditioning.
- Authentication, accounts, billing, or subscriptions — explicitly out of
  scope.
- Any form of mastering. LUBER's audio pipeline transcodes and does
  nothing else; that is unchanged.
