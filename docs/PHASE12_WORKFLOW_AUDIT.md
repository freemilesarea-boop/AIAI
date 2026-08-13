# Phase 12 — Product workflow audit

Read-only inspection of the code at `49426d0`, before any Phase 12 edit.
Its purpose is to stop Phase 12 from rebuilding things that already work.

The target workflow is:

```
CREATE → GENERATE → LISTEN → ORGANIZE → ITERATE → DOWNLOAD
```

Everything below is measured against that loop.

---

## CURRENT CAPABILITIES

### Data model (`packages/database`, migration head 0006)

| Table | What exists |
|---|---|
| `generations` | title, prompt, lyrics, vocal_gender, duration_requested/actual, **seed**, language, instrumental, bpm, key_scale, time_signature, request_trace, advisories, **parent_generation_id** (`ON DELETE SET NULL`), variation_label, **project_id** (`ON DELETE SET NULL`), status, provider, model_name, model_version, idempotency_key (unique), timestamps, error_code/message |
| `generation_jobs` | one queue job per generation, attempt counters, worker id |
| `audio_assets` | one row per (generation, asset_type); MASTER wav + PREVIEW mp3 both exist today |
| `generation_qa`, `lyric_line_qa` | Phase 9 human QA — developer surface, not product |
| `projects` | id, name, user_id (reserved), created_at, updated_at |

`user_id` is nullable on both `generations` and `projects` and is reserved
for the authentication phase. Nothing writes it yet.

**Confirmed by query, not assumption:** every completed generation in the
E2E database has both a `MASTER/wav` and a `PREVIEW/mp3` asset (23 of
each). An MP3 download therefore needs no new transcoding pipeline.

### API (`apps/api`)

- `POST /v1/generations` — creates **one** generation, one job, enqueues it.
  Honours `Idempotency-Key`, records advisories, validates a supplied
  `parent_generation_id` against the same ownership rule used for audio.
- `POST /v1/generations/preflight` — advisory-only, side-effect free.
- `GET /v1/generations` — limit/offset list, newest first.
- `GET /v1/generations/{id}` — one generation.
- `GET /v1/generations/{id}/audio?asset=master|preview&download=` — serves
  bytes or redirects to a signed URL; filename built by
  `build_download_filename`.
- `DELETE /v1/generations/{id}` — deletes DB rows, then storage.
- `GET /v1/generations/{id}/lineage` — parent + children.
- `GET|PUT /v1/generations/{id}/qa`, `GET .../longform-qa` — developer QA.
- `POST|GET|PATCH|DELETE /v1/projects`, `GET /v1/projects/{id}/generations`,
  `PUT /v1/generations/{id}/project`.

### Frontend (`apps/web`)

- App shell with Create / Library / Projects, mobile drawer, desktop sidebar.
- **One** `<audio>` element in `PlayerProvider`, mounted above the router;
  `PlayerBar` is a pure control surface. Playback survives navigation.
- Create: Simple/Custom tabs, presets, structure editor, prompt chips,
  advanced controls, preflight advisories, a **single** tracked job with
  polling and refresh recovery, and a session "Recent generations" list
  scoped by ids in `localStorage`.
- Library: fetch 100, client-side search **by title only**, status filter
  tabs, newest/oldest sort.
- Projects: single page with a list on one side and the opened project
  beside it; create / rename / delete / assign / remove.
- Song detail: listener-facing brief + lyrics + lineage, diagnostics behind
  an Advanced disclosure.
- `SongCard` shows deterministic gradient artwork, play, WAV, status.

### Already correct — do not rebuild

- Provider abstraction (`frontend → API → GenerationService →
  MusicGenerationProvider → AceStepProvider`).
- Single global audio element and its lifetime.
- Deleting a project keeps its music (`ON DELETE SET NULL`).
- Status vocabulary is already humanised in `StatusPill` and
  `lib/generationStatus.ts`; no worker jargon reaches the UI.
- Download filename sanitisation exists server-side.
- Seed is plumbed end-to-end: `request.seed` → `use_random_seed=false` +
  `seed` in the ACE-Step payload → `seed_used` → persisted on completion.

---

## MISSING CAPABILITIES

Grouped by the phase requirement that names them.

| # | Gap |
|---|---|
| 2 | No rename. No delete from the UI at all. No confirmation anywhere. Lineage is not explicitly re-pointed on delete — on PostgreSQL the FK nulls the child, on SQLite (tests) it does not, so the two disagree. |
| 3 | No favourites — no column, no endpoint, no control, no filter. |
| 4 | "Generate again" exists but is fused with prefill; there is no separate "Duplicate settings" that opens Create *without* recording lineage. |
| 5 | Seed is never shown as a control. Song detail shows `Seed` only as an Advanced-adjacent field. No Random/Fixed choice, no reuse. |
| 6 | One CREATE produces exactly one song. No result-count control. |
| 7 | No grouping concept of any kind. |
| 8 | Recent generations exist but the Create page tracks **one** job; a second submit replaces the first. |
| 9 | `GenerationForm` is `disabled` for the whole inference duration (`busy = phase === "submitting" \|\| "tracking"`). The page is blocked for minutes. |
| 10 | Download filename is `midnight-window.wav` — sanitised but not the requested `LUBER - {Title}.wav` form. |
| 11 | PREVIEW mp3 assets exist and are streamed, but never offered as a download. |
| 12 | No metadata editing at all. |
| 13/14 | Projects has no dedicated route; state is transient, so a refresh loses the opened project. No sort. Delete has no confirmation. |
| 15 | No selection mode, no bulk actions. |
| 16 | Search is title-only; no prompt search; only two sort options; no favourites filter. |
| 19 | No `cover_art_url`. |
| 21 | No toast/feedback mechanism. |
| 22 | No dialog component. |
| 23 | `/projects` state is React-only and does not survive a refresh. |

---

## DATA MODEL CHANGES REQUIRED

Three columns on `generations`, all nullable-or-defaulted, all actually read:

| Column | Type | Why |
|---|---|---|
| `favorite` | `BOOLEAN NOT NULL DEFAULT false` | Req. 3. Must be server state, not localStorage. |
| `generation_group_id` | `UUID NULL`, indexed | Req. 6/7. One CREATE → N sibling rows. Application metadata only. |
| `cover_art_url` | `TEXT NULL` | Req. 19. Read by the UI to choose artwork vs placeholder. Never written in Phase 12. |

No `generation_groups` table. A group is the set of rows sharing an id; a
table would add a lifecycle (orphan rows, cascade rules) that buys nothing
at this size. This is the "do not overengineer it" instruction taken
literally.

Nothing else changes. In particular no provenance column becomes mutable.

---

## API CHANGES REQUIRED

| Method | Path | Purpose |
|---|---|---|
| `PATCH` | `/v1/generations/{id}` | Rename + favourite. Body accepts **only** `title` and `favorite`; any other key is a 422, which is how provenance immutability is enforced rather than merely documented. |
| `POST` | `/v1/generations` | Gains `result_count: 1\|2` and `seed`. Creates N independent rows/jobs sharing a `generation_group_id`. Response gains `generation_group_id` and `generations[]`; the existing `generation_id` field is retained and points at the first result so the Phase 3–11 contract does not break. |
| `GET` | `/v1/generations/groups/{group_id}` | Refresh recovery for a two-result submission. |
| `POST` | `/v1/generations/bulk-delete` | `{ids[]}` → count. |
| `POST` | `/v1/generations/bulk-project` | `{ids[], project_id\|null}` → count. |
| `PUT` | `/v1/generations/{id}/project` | Return `GenerationResponse` instead of today's odd single-item `GenerationListResponse`. |
| `DELETE` | `/v1/generations/{id}` | Unchanged externally; the repository gains an explicit child re-point so SQLite and PostgreSQL agree. |

Seed policy for `result_count = 2` with a **fixed** seed: the first result
uses the given seed, the second is generated with a fresh engine seed.
Two identical seeds would produce two identical songs, which defeats the
entire point of asking for alternatives. The UI states this in one line
rather than letting the user discover it.

---

## FRONTEND CHANGES REQUIRED

New:

- `components/ui/Toast.tsx` — provider + `useToast()`, mounted in the root
  layout beside `PlayerProvider`.
- `components/ui/ConfirmDialog.tsx` — focus-trapped, Escape-closable,
  destructive variant. Replaces every `window.confirm`.
- `hooks/useGenerationQueue.ts` — replaces `useGenerationJob`. Holds **many**
  jobs, polls each independently, persists their ids for refresh recovery.
- `components/SongActions.tsx` — the one place rename/delete/favourite/
  duplicate/project-assignment are expressed, so Library, Projects and
  Song detail cannot drift apart.
- `app/projects/[id]/page.tsx` — real route, refresh-safe.
- `lib/download.ts` — `LUBER - {Title}.wav` naming, shared.

Changed:

- Create: result-count selector, seed control, a queue of live job cards,
  a CREATE button disabled only during the POST.
- Library: prompt search, favourites filter, four sorts, selection mode.
- Projects: list route + detail route, sort, confirmation on delete.
- Song detail: favourite, rename, seed display, duplicate settings, MP3.
- `SongCard`: favourite heart, cover art hook, action slot.

---

## MIGRATION PLAN

1. `0007_favorites_groups_cover_art.py`, `down_revision = "0006"`.
2. `upgrade`: three `add_column`s + one index on `generation_group_id`.
   `favorite` gets `server_default=sa.false()` so the column can be added
   `NOT NULL` to a populated table, matching the lesson from 0005 where a
   `NOT NULL` column without a server default passed SQLite-built tests and
   failed on real PostgreSQL.
3. `downgrade`: drop index, drop the three columns.
4. Validate on real PostgreSQL: `upgrade head` → `downgrade -1` →
   `upgrade head`, then assert exactly one head and `current == 0007`.

---

## TEST PLAN

**Backend** (`apps/api/tests`, `packages/database/tests`)

- rename changes only the title; a request carrying `prompt`, `seed`,
  `lyrics`, `model_name` or `provider` is rejected 422.
- favourite toggles, persists, and round-trips through `GET`.
- delete: parent removed → child survives with a `NULL` parent; jobs and
  assets go; DB state is committed before storage deletion is attempted.
- two-result: one POST → two rows, two jobs, one shared group id, two
  distinct seeds, independent statuses.
- partial failure: fail one sibling, assert the other still reports
  `COMPLETED` and the group read returns both.
- fixed seed + 2 results → first honours the seed, second differs.
- idempotent replay of a two-result submission returns the same group.
- bulk delete / bulk assign counts and 404-free behaviour on unknown ids.
- project lifecycle incl. delete-with-songs leaves songs unfiled.
- `cover_art_url` is present and `null` in the response schema.
- migration 0007 up/down on PostgreSQL.

**Frontend** (`apps/web`)

Result-count selector, group rendering, two independent queue cards,
partial-failure presentation, favourite toggle, rename flow, delete
confirmation (including Escape), duplicate settings prefill without
lineage, seed Random/Fixed, library filter/search/sort composition,
selection mode + bulk actions, projects list and detail, toast feedback,
player integration from every surface, refresh recovery, and the
mobile-critical layout behaviours.

**Deliberately changed existing test:** `create-page.test.tsx` currently
asserts *"prevents duplicate submission while a generation is in flight"*.
Requirement 9 explicitly reverses that product decision. The test is
rewritten to assert the guarantee that still holds — a double click sends
one POST — while a second, separate submission during inference is now
expected to succeed. That is a specification change, not a weakened test.

---

## OUTCOME

Written after implementation. Every gap listed above was closed; what
follows is only the part a reader could not infer from the plan.

### Decisions taken during the work

**A pinned seed applies to the first result only.** Giving both siblings
the same seed asks the engine for the same song twice, which defeats the
purpose of requesting alternatives. Verified against the real engine: a
`seed=777` single result came back with `seed=777`; a two-result
submission produced `3405275941` and `1981867754`.

**Download filenames keep Unicode.** Phase 3 slugged titles to ASCII,
which turned every Korean title into `luber-track-1a2b3c4d.wav`. Titles
now survive as `LUBER - 오늘 밤.wav`; Starlette emits `filename*=utf-8''`,
which current browsers decode. Characters no filesystem accepts are
still stripped and dot-runs collapsed, so `..` cannot appear.

**A group is a shared id, not a table.** `generation_group_id` has no
foreign key and no `generation_groups` row. A group is the set of
generations carrying the id — nothing owns it, so nothing can orphan it.

**Deletion re-points descendants explicitly.** PostgreSQL would do this
via `ON DELETE SET NULL`, but SQLite (where unit tests run) does not
enforce it, so the two disagreed. The repository now nulls children
itself and both behave identically.

### Defects found and fixed by QA rather than by tests

1. **Library overflowed at 390px.** Adding the Favorites tab made five
   filters, which do not fit; the whole document scrolled sideways
   (`scrollWidth 447` vs `clientWidth 390`) on every Library view. `Tabs`
   now scrolls inside its own container and uses tighter padding below
   `sm`.
2. **Bulk toolbar broke onto three rows** because the shared `inputClass`
   `w-full` beat the `w-auto` override — utility order in a class string
   does not decide precedence.
3. **Selection checkboxes were 16px.** The label is now the target, at
   40px, with the control still visually small.
4. **Song titles were hard-clipped with no ellipsis** — `truncate` on an
   `inline-flex` container does nothing; it belongs on the text.
5. **Preset and template buttons were 30px tall**, below a comfortable
   thumb target, matching the correction already applied to `Chip`.

### Pre-existing issue fixed in passing

`alembic check` had been reporting permanent phantom drift on
`audio_assets`: the ORM constraint was unnamed, so autogenerate invented
a name that differed from the one migration 0003 created. Naming it in
the model makes the check meaningful again. No migration was needed —
the database already had the name.

### Measurement note

The QA tool's overflow detector was taught to ignore elements inside a
horizontal scroller, because content extending past the viewport there
is the intent rather than a defect. Its touch-target check measures a
checkbox's wrapping label, since that is the region a thumb actually
hits. Both changes were made *after* the corresponding defects were
fixed, not to make them disappear.

Headless Chrome's autoplay policy rejects `play()` from a synthetic
click, which made the player render "This track could not be played" in
QA. That was confirmed to be a harness artifact, not a product defect,
by reading the media element directly: `readyState 4`, `error null`,
`paused false`.
