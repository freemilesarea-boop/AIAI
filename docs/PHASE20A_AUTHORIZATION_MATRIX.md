# Authorization matrix

Every route the API serves, what it must require, and what it must
answer. Built by enumerating the live OpenAPI schema — 32 operations, no
route inferred from a filename.

**This is the specification for Part 3, not a description of today.** The
"current" column is what the running code does; the rest is what it must
do. Nothing here is implemented yet.

---

## Status today

| | |
|---|---|
| Authentication | works (Part 1) |
| Ownership recorded in the database | yes (Part 2) |
| **Ownership enforced by routes** | **no** |

26 product operations currently serve any caller. `caller_may_access`
exists but treats the legacy anchor as public, so every one of the 55
historical generations is readable, playable and downloadable by anyone
who can reach the API.

## The matrix

`AUTH` = requires a valid session · `SCOPE` = query filtered by
`current_user.id` · `ORIGIN` = trusted-origin check (unsafe methods
only) · `ANON` = expected status without a session · `CROSS` = expected
status for another user's resource.

### Public — must stay anonymous

| Method | Path | AUTH | SCOPE | ORIGIN | ANON | CROSS |
|---|---|---|---|---|---|---|
| GET | `/health` | no | — | — | 200 | — |
| GET | `/ready` | no | — | — | 200/503 | — |
| GET | `/openapi.json`, `/docs`, `/redoc` | no | — | — | 200 | — |

Liveness and readiness must not require a session: a probe has no cookie,
and an auth failure would look like an outage.

### Auth — anonymous by necessity

| Method | Path | AUTH | SCOPE | ORIGIN | ANON | CROSS |
|---|---|---|---|---|---|---|
| POST | `/v1/auth/signup` | no | — | **yes** | 201 | — |
| POST | `/v1/auth/login` | no | — | **yes** | 200/401 | — |
| POST | `/v1/auth/logout` | no | — | **yes** | 204 | — |
| GET | `/v1/auth/me` | **yes** | self | — | **401** | — |

Origin checking is already applied to this router (Part 1).

### Generations — all must become authenticated and scoped

| Method | Path | AUTH | SCOPE | ORIGIN | ANON | CROSS |
|---|---|---|---|---|---|---|
| POST | `/v1/generations` | yes | owner = current user | yes | 401 | — |
| GET | `/v1/generations` | yes | **list + total** | — | 401 | n/a |
| POST | `/v1/generations/preflight` | yes | — | yes | 401 | — |
| GET | `/v1/generations/{id}` | yes | yes | — | 401 | **404** |
| PATCH | `/v1/generations/{id}` | yes | yes | yes | 401 | **404** |
| DELETE | `/v1/generations/{id}` | yes | yes | yes | 401 | **404** |
| GET | `/v1/generations/{id}/audio` | yes | **via generation** | — | 401 | **404** |
| GET | `/v1/generations/{id}/lineage` | yes | whole tree | — | 401 | **404** |
| POST | `/v1/generations/{id}/extend` | yes | parent owned | yes | 401 | **404** |
| POST | `/v1/generations/{id}/replace-range` | yes | parent owned | yes | 401 | **404** |
| POST | `/v1/generations/{id}/cover` | yes | parent owned | yes | 401 | **404** |
| PUT | `/v1/generations/{id}/project` | yes | **both sides** | yes | 401 | **404** |
| GET/PUT | `/v1/generations/{id}/qa` | yes | yes | yes (PUT) | 401 | **404** |
| GET | `/v1/generations/{id}/longform-qa` | yes | yes | — | 401 | **404** |
| GET | `/v1/generations/groups/{group_id}` | yes | yes | — | 401 | **404** |
| POST | `/v1/generations/bulk-delete` | yes | **own rows only** | yes | 401 | silently skipped |
| POST | `/v1/generations/bulk-project` | yes | **both sides** | yes | 401 | silently skipped |

### Projects

| Method | Path | AUTH | SCOPE | ORIGIN | ANON | CROSS |
|---|---|---|---|---|---|---|
| GET | `/v1/projects` | yes | list | — | 401 | n/a |
| POST | `/v1/projects` | yes | owner = current user | yes | 401 | — |
| GET | `/v1/projects/{id}` | yes | yes | — | 401 | **404** |
| PATCH | `/v1/projects/{id}` | yes | yes | yes | 401 | **404** |
| DELETE | `/v1/projects/{id}` | yes | yes | yes | 401 | **404** |
| GET | `/v1/projects/{id}/generations` | yes | **both sides** | — | 401 | **404** |

### Reference audio

| Method | Path | AUTH | SCOPE | ORIGIN | ANON | CROSS |
|---|---|---|---|---|---|---|
| POST | `/v1/reference-audio` | yes | owner = current user | yes | 401 | — |
| GET | `/v1/reference-audio/limits` | yes | — | — | 401 | — |

Attaching a reference at generation time is not its own route, but it is
an ownership boundary: the reference and the generation must both belong
to the caller, and a foreign reference id must fail as though it does not
exist.

## Rules the matrix encodes

**401 anonymous, 404 cross-user.** A 403 would confirm that a UUID
belongs to somebody. Absent and not-yours must be indistinguishable, so a
random UUID, another user's UUID and a legacy UUID all answer 404.

**Totals are scoped.** A user with two songs sees `total=2`, not
`total=57`. A count leaks the size of the corpus and is easy to forget
because the list itself already looks correct.

**Scope in the query, not after it.** `WHERE id = :id AND user_id =
:user` rather than load-then-compare. The second form has the row in
memory before the check, which is one early return away from a leak.

**Assets inherit.** No `user_id` on `audio_assets`; authorization
resolves through the owning generation. No asset may be reachable because
its storage key or asset UUID is known.

**Bulk operations filter, they do not fail.** A request mixing own and
foreign ids operates on the caller's rows only and reports nothing about
the others — refusing the whole batch would confirm the foreign ids
exist.

**Descendants inherit the actor.** Extend, replace, cover and generate-
again set the child's owner from the session, after proving the parent
belongs to the caller. Never from the parent row alone.

## Legacy corpus

The 55 historical generations, 119 assets and 4 references belong to the
internal anchor, which cannot log in. After Part 3 they are unreachable
through every authenticated product route, for every user.

That is the correct outcome and it is intended. They are not reassigned
to anybody. P20 verification uses direct database and storage tooling,
which is what the benchmark scripts already do.

## What Part 3 must change

1. **Router-level dependencies** — `require_current_user` and
   `enforce_trusted_origin` on the generations, projects and
   reference-audio routers, so a new route is protected by default
   rather than by remembering.
2. **Owner-scoped repository methods** — roughly 15 of the 42 existing
   methods gain a `user_id` parameter and a `WHERE` clause.
3. **Delete three things**: `get_caller_user_id`, `caller_may_access`,
   and the `LEGACY_OWNER_ID` fallbacks in `GenerationRepository`
   (generations, projects, reference audio). Removing the fallback is
   what stops new data being attributed to the anchor.
4. **Update the tests.** 382 client calls across 14 product test files
   are currently anonymous and will all return 401. They must be given
   real sessions — not a testing bypass, which would leave the suite
   unable to prove the boundary it exists to prove.
5. **Add adversarial coverage** — two users, cross-user 404 on every
   resource type, `X-User-Id` spoof, legacy corpus invisible, fresh user
   sees an empty library.

Step 3 is the atomic part. Removing the fallback while any route still
creates data without a session produces rows the database rejects; adding
scoping to some routes and not others produces a product that looks
private and is not. They land together or not at all.
