# Legacy ownership — classification and migration plan

Written from the live schema and the live rows before anything was
changed. Its job is to decide which tables own data directly, which
inherit it, and who the 55 pre-authentication generations belong to.

**Authentication exists. Ownership metadata exists after this migration.
Authorization does not.** Nothing in Part 2 makes anyone's songs
private — every route still serves every row to every caller. That is
Part 3.

---

## 1. State before the migration

| Table | Rows | Ownership today |
|---|---|---|
| `users` | 0 | — |
| `sessions` | 0 | — |
| `generations` | 55 | `user_id` nullable, **no FK**, all 55 NULL |
| `audio_assets` | 119 | none, and none needed — see §3 |
| `projects` | 0 | `user_id` nullable, **no FK** |
| `reference_audio` | 4 | **no column** |
| `generation_jobs` | 55 | none needed |
| `generation_qa` / `lyric_line_qa` | 0 | none needed |

No user exists, so no account can already have claimed anything. Lineage
fingerprint before: **2 edges**, graph digest
`bec8228feab31656766c987ac1591217`. P20 benchmark hash verified matching
its manifest.

## 2. Who owns the legacy data

Not the first person to sign up. Not a test account. Not the
administrator's own login. Those all produce the same failure: a rule
nobody stated, discovered only after a stranger owns the Phase 20
corpus.

Instead, one internal anchor:

| | |
|---|---|
| Email | `legacy-system@internal.luber` |
| UUID | `e3c4d3cd-d86f-52f2-91b7-2b97f5011653` |
| `password_hash` | `NULL` |

The UUID is **deterministic** — `uuid5(NAMESPACE_DNS, email)` — so every
database that runs this migration produces the same anchor, and the
migration is idempotent with respect to it. It is written literally into
the migration rather than computed there, so the value is auditable in
the diff.

`.luber` is not a real TLD, so the address can never receive mail and can
never be verified.

### Why it cannot be authenticated, and why that needs no new mechanism

Two existing behaviours already close both doors, which is the whole
reason this design is small:

- **Login** — `luber_api.routes.auth.login` refuses when
  `not user.password_hash`, before any verification. A NULL hash is
  rejected with the same generic 401 as any other failure.
- **Signup** — `users.email` is unique. A signup with this address hits
  the constraint and returns 409. It cannot overwrite the row, cannot
  attach a password to it, and cannot convert it into an account.

So the anchor is unreachable by password *and* unclaimable by
registration, using only what Part 1 already built. Adding a "reserved
account" flag would be a second mechanism enforcing what the first
already enforces — and a second thing to get wrong. Both properties are
tested rather than assumed.

## 3. Which tables own, and which inherit

Every foreign key in the schema was read to establish this:

```
users
 ├── generations        (user_id — direct)
 │    ├── audio_assets      → generation_id
 │    ├── generation_jobs   → generation_id
 │    ├── generation_qa     → generation_id
 │    ├── lyric_line_qa     → generation_id
 │    └── generations       → parent_generation_id  (lineage)
 ├── projects           (user_id — direct)
 └── reference_audio    (user_id — direct, added here)
```

**Direct ownership: three tables.** `generations`, `projects`,
`reference_audio` are each created by a user as a distinct act.

**Everything else inherits through `generation_id`.** `audio_assets` in
particular gets **no** `user_id`, and that is a deliberate refusal: a
duplicated owner column is a second source of truth that can disagree
with the first. An asset belongs to whoever owns its generation, full
stop, and 0 of the 119 assets lack a generation to inherit from.

`generations.project_id` and `generations.reference_audio_id` are
associations, not ownership. Part 3 must check that both sides belong to
the same user; that is an authorization rule, not a schema one.

## 4. What the migration does

One transactional migration, `0014`:

1. **Insert the anchor** if absent — `ON CONFLICT DO NOTHING` on the
   fixed UUID, so re-running changes nothing.
2. **Add `reference_audio.user_id`**, nullable at first.
3. **Backfill** `generations.user_id`, `projects.user_id` and
   `reference_audio.user_id` **only where NULL**. An existing non-NULL
   owner is never overwritten.
4. **Verify** no NULL ownership remains, and fail the migration if any
   does rather than proceeding to a constraint that would throw anyway.
5. **Add foreign keys** to `users.id` on all three, and indexes on each
   `user_id`.
6. **Set NOT NULL** on all three, which is only reachable once step 4
   has proved it.

Ordering matters: the constraint comes after the backfill and after the
check, so the migration either completes or leaves the database exactly
as it found it.

### What it must not touch

Lineage (`parent_generation_id`), storage keys, SHA-256 digests,
durations, audio metadata, finishing traces, benchmark files, the rubric,
the taxonomy, and any audio byte. The migration writes one column on
three tables and inserts one row.

## 5. Downgrade

Drops the constraints, indexes, FKs and the `reference_audio.user_id`
column, and clears the backfilled `user_id` values back to NULL.

**It deletes no product data**, and deliberately **leaves the anchor
user in place**. Removing it would require deciding what happens to rows
that still reference it, and the only safe answers are "cascade" — which
destroys generations — or "orphan" — which breaks the FK. Leaving one
unauthenticatable row behind costs nothing and removes the entire
question.

## 6. After this migration

`generations.user_id` becomes NOT NULL, so **every future generation
must supply an owner**. Nothing in Part 2 makes the API do that: the
create path still runs unauthenticated, and until Part 3 supplies the
owner from the session, an authenticated-looking product does not exist.
The invariant is now enforced by the database, which is what makes Part 3
a change of behaviour rather than a change of hope.

## 7. Insecure surfaces that remain, for Part 3

Recorded here so Part 3 has a list rather than a search:

- `get_caller_user_id` (`routes/generations.py`) still reads the
  forgeable `X-User-Id` header. It cannot authenticate — Part 1 proved
  that — but product routes still consult it.
- Library, song detail, audio delivery, lineage, edit routes, delete,
  bulk delete, projects and reference audio all serve any caller.
- Reference audio can still be attached to a generation across owners.
- Downloads are reachable by anyone who knows a generation UUID.

None of these are fixed here. Fixing them piecemeal is what Part 3 is
structured to avoid.
