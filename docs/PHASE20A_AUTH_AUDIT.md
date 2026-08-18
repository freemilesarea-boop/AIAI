# Authentication and data ownership — audit

Read from the repository and the live database before any authentication
code was written. The headline is that the ground is **partly prepared
and entirely unenforced**: identity exists as a shape in the schema, and
nothing anywhere checks it.

---

## 1. Answers to the required questions

| Question | Answer |
|---|---|
| Existing user model | **YES**, but incomplete |
| Existing session system | **NO** |
| Existing ownership fields | `generations.user_id`, `projects.user_id` — nullable, **no FK**. `reference_audio` has none |
| Current data visibility | **Global.** Every route serves every row to every caller |
| Migration required | **Likely — certain**, in fact |

## 2. What exists

**`users` table** (migration `0001_create_users.py`): `id`, `email`
(unique, indexed), `display_name`, `created_at`. **Zero rows.** It has
never been used.

Critically, it has **no `password_hash`**. The table can identify a user
but cannot authenticate one.

**Ownership columns**, both nullable and both without a foreign key:

| Table | `user_id` | Rows today |
|---|---|---|
| `generations` | present, nullable, no FK, no index | 55 |
| `projects` | present, nullable, no FK, no index | 0 |
| `reference_audio` | **absent** | 4 |
| `audio_assets` | absent, and correctly so — owned via its generation | 119 |

**`get_caller_user_id`** (`routes/generations.py:247`) reads an
`X-User-Id` header and returns it as a UUID, or `None` if malformed. Its
own docstring calls it "a placeholder for the authentication phase". It
is threaded through many route signatures already, which is genuinely
useful: the *shape* of an identity-carrying request is established.

It is also trivially forgeable. Any caller may claim any id. Nothing
downstream compares it to anything, so today it is inert rather than
dangerous — but it must not survive into an authenticated build.

## 3. What does not exist

- **No password hashing of any kind.** Not argon2, bcrypt, passlib or
  itsdangerous — none is installed. A dependency will have to be added,
  and the audit's recommendation is `argon2-cffi` (Argon2id), which is
  the maintained reference implementation and needs no wrapper.
- **No session concept.** No table, no Redis keys, no cookie handling.
  `set_cookie` appears nowhere in the API.
- **No rate limiting.** No middleware, no 429 outside the provider's own
  busy signal. Redis is available and is the natural place for it.
- **No ownership enforcement anywhere.** Every generation, project,
  reference and audio asset is readable, playable, downloadable and
  deletable by anyone who can reach the API.

## 4. CORS and cookies

`main.py` already configures `CORSMiddleware` with
`allow_credentials=True` and an explicit `settings.cors_origins` list —
not a wildcard. That is the correct shape for cookie authentication and
needs no change in principle.

The frontend client does **not** currently send credentials:
`apps/web/src/lib/api.ts` never sets `credentials`. Every `fetch` will
need `credentials: "include"`, or the cookie will simply not travel —
web on :3000 and API on :8000 are different origins.

## 5. The legacy data problem

This is the part that needs a decision rather than code.

55 generations and 4 reference audio rows exist, all created before
authentication, all unowned. They include the entire Phase 20 benchmark
corpus and the frozen `luber-baseline-p20-v1` baseline.

**What must not happen:** assigning them to whoever signs up first. In a
production deployment that hands one stranger the whole corpus, and the
rule "first signup inherits everything" is invisible until it has already
happened.

**Recommended policy** — an explicit bootstrap owner:

1. Create one clearly-named development user in the migration itself
   (`legacy@local.invalid`, or similar reserved-domain address that can
   never receive mail and can never be registered).
2. Assign every existing `generations.user_id` and the new
   `reference_audio.user_id` to that user, in the same migration.
3. Give it **no usable password hash**, so it cannot be logged into. It
   is an ownership anchor, not an account.
4. Make `user_id` `NOT NULL` only *after* backfill, so the constraint is
   provable rather than hoped for.
5. Document that a real deployment starts with an empty database and
   never runs the backfill branch.

The alternative — leaving ownership nullable and adding an explicit
"claim" flow — is more faithful to production but adds a UI and a
security surface for a problem that exists only on this one machine.

## 6. Scope of the enforcement change

Ownership is not one check. Every route below currently serves data
without asking who is calling:

| Area | Routes | Enforcement needed |
|---|---|---|
| Library | list generations | filter at query level, not in React |
| Song detail | get generation | 404 rather than 403, to avoid confirming existence |
| Audio | `/audio` master and preview | ownership before bytes |
| Lineage | `/lineage` | traversal must not cross an owner boundary |
| Edits | extend, replace-range, cover, generate-again | owner derived from parent, never from the request |
| Delete | single and bulk | ownership before the 409 descendant check |
| Projects | list, get, create, rename, delete, assign | scoped; assignment requires both sides owned |
| Reference audio | upload, list, use in generation | scoped; a cross-user reference must be refused without confirming it exists |

The lineage rule deserves emphasis: a derived generation must inherit its
parent's owner **server-side**. Phase 17 made lineage durable; a
cross-user parent would make it a data leak.

## 7. CSRF

Web and API are separate origins in development (`:3000` and `:8000`),
which means `SameSite=Lax` will **not** attach the cookie to
cross-origin requests at all — including legitimate ones from the web
app. This has to be resolved deliberately, and the choice affects CSRF
posture:

- **Same-origin via a proxy** (web proxies `/v1` to the API): `Lax`
  works, CSRF risk is minimal, and no token machinery is needed. The
  cleanest option, and it also removes the CORS credential question.
- **Cross-origin with `SameSite=None; Secure`**: requires HTTPS even
  locally, and reintroduces real CSRF exposure needing an explicit
  token.

The audit's recommendation is the proxy. It is the smallest defensible
strategy and avoids building token machinery to defend against a problem
created by an arrangement chosen for convenience.

## 8. What this implies for sequencing

The work divides into four parts that are each independently testable,
and the middle one is irreversible:

1. **Auth core** — password hashing, user creation, sessions, cookies,
   the `require_current_user` dependency, rate limiting. Adds routes;
   changes no existing behaviour.
2. **Migration** — `password_hash`, `sessions`, `reference_audio.user_id`,
   FKs, indexes, legacy backfill. **Touches the live database holding the
   Phase 20 corpus.**
3. **Enforcement** — scoping every route in §6. Changes the behaviour of
   the entire existing product at once.
4. **Frontend** — provider, `/signup`, `/login`, logout, route
   protection, `credentials: "include"`.

Part 3 is the one that breaks everything if it is half-done: an API where
some routes require a session and others do not is worse than either
extreme, because the product appears to work while leaking.

---

*Read-only audit. No schema, route, component or dependency was changed
to produce it, and the Phase 20 baseline is untouched — benchmark hash
verified identical.*
