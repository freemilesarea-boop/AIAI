# Authentication architecture

How a person proves who they are, and what the product does with that.

**Parts 1 and 2 of Phase 20A.** Authentication works end to end, and
every row of product data now has an owner in the database.
**Authorization still does not exist** — every generation, project and
reference is still visible to every caller, exactly as before. That is
deliberate: enforcement is Part 3, and shipping it half-done would be
worse than not starting, because the product would appear to isolate
users while leaking.

Nobody's songs are private yet.

---

## Implemented

### Passwords

Argon2id via `argon2-cffi`, at the library's own defaults. It generates
its own salts and encodes its own parameters into the stored string;
nothing in this repository writes crypto.

Policy is **length only** — at least 10 characters, at most 1024, spaces
and any Unicode allowed. No composition rules: requiring an uppercase
letter and a symbol reliably produces `Password1!`, which is a pattern
attackers already have. Long passwords are **never truncated**, which is
what the 1024 ceiling is for — a bound on request size, not a silent
trim that would make a passphrase weaker than its owner believes.

A successful login is the only moment the plaintext exists, so it is
where `check_needs_rehash` runs and an aged hash is upgraded.

### Sessions

Opaque, server-side, in PostgreSQL. 256 bits from `secrets.token_urlsafe`.

The browser holds the raw token. **The server stores only SHA-256 of
it.** A dump of the `sessions` table therefore contains no usable
credentials. SHA-256 is correct here precisely because it is fast — the
input is OS randomness, so there is no dictionary to run and a slow hash
would buy nothing.

Expiry is enforced **in the query**, not after it, so a session cannot
authenticate for the instant between being loaded and being checked. The
`users` join means a deleted user's session stops working immediately as
well.

Sessions live in the database rather than in memory, so they survive an
API restart — verified by restarting the API and reusing the cookie.

### Cookie

| | |
|---|---|
| Name | `luber_session` |
| HttpOnly | true — an XSS bug must not also be session theft |
| SameSite | `Lax` |
| Secure | `false` in development, `true` in production (`is_production`) |
| Path | `/` |
| Domain | not set — scoped to the exact host |
| Lifetime | 14 days, `session_lifetime_seconds` |

`Secure` is environment-aware for a concrete reason: a `Secure` cookie
is silently dropped over plain HTTP, so hardcoding it would make local
login appear to succeed and every subsequent request arrive anonymous.

### Same-origin proxy

The browser never talks to `:8000`. It calls same-origin `/api/...`
paths and Next rewrites them to the backend (`API_PROXY_TARGET`,
default `http://127.0.0.1:8000`).

**Why `SameSite=None` was rejected as the normal architecture.** With
the browser calling the API's own port directly, every request is
cross-site, so the cookie would need `SameSite=None`. That requires
`Secure` — and therefore HTTPS even on localhost — and it discards the
protection Lax provides against cross-site state-changing requests,
which would then have to be rebuilt with CSRF tokens. The proxy costs
one rewrite rule and removes all of that. It is also the shape
production takes anyway, with one public origin in front of both.

### CSRF

Two layers, and the reasoning matters more than the count.

`SameSite=Lax` is the primary control: the browser does not attach the
cookie to cross-site POST, PUT, PATCH or DELETE. Because all legitimate
traffic is same-origin through the proxy, Lax costs nothing.

`enforce_trusted_origin` is defence in depth, applied to the whole auth
router. Unsafe methods carrying an `Origin` the deployment does not
serve are refused with 403. A **missing** `Origin` is allowed —
non-browser clients do not send one and have no ambient cookie to abuse,
so refusing them would break curl and server-to-server calls without
addressing the threat CSRF describes.

No CSRF tokens. With same-origin traffic and Lax cookies they would be
machinery defending against an exposure this architecture does not have.
The auth routes are the entire surface a session cookie currently
authenticates; Part 3 extends origin validation as product routes begin
reading the cookie.

### Rate limiting

Fixed window in Redis, per client address, applied to signup and login
separately: 10 attempts per 15 minutes (`auth_rate_limit_attempts`,
`auth_rate_limit_window_seconds`). A successful login clears the
counter, so two typos then the right password costs nothing.

**If Redis is unavailable, requests are allowed.** Stated plainly
because it is the uncomfortable choice: the alternative locks every user
out during a Redis outage. Passwords remain Argon2id and sessions remain
server-side, so what is lost is the brute-force ceiling, not
authentication — and the outage is already loud through `/ready`.

### Account enumeration

Login answers `401 "Email or password is incorrect."` for an unknown
address and for a wrong password, with identical status and body.
Distinguishing them would make the endpoint a membership oracle for any
address someone cares to test.

Signup necessarily reports a duplicate (409) — a user has to be told why
they cannot register — which is the one place the product accepts the
trade.

### Current-user dependency

`require_current_user` is the only way a route learns who is calling.
Cookie parsing lives in one module; every failure — no cookie, unknown
token, expired session, deleted user — collapses to the same `None`, so
the cookie cannot be used to probe.

**`X-User-Id` cannot authenticate.** That header is a pre-auth
placeholder, forgeable by anyone, and is still read by product routes
that have no ownership checks yet. It never reaches the auth dependency,
and a test pins that. Part 3 removes it.

### Session cleanup

`luber_health.py --prune-sessions` deletes expired rows. Expired
sessions already fail to authenticate, so this reclaims space rather
than revoking access, and the query cannot touch a user or any product
data. No scheduler was introduced.

---

## Legacy ownership

Pre-authentication data belongs to one internal anchor rather than to
any person.

| | |
|---|---|
| Email | `legacy-system@internal.luber` |
| UUID | `e3c4d3cd-d86f-52f2-91b7-2b97f5011653` — `uuid5(NAMESPACE_DNS, email)`, so it is identical on every database |
| `password_hash` | `NULL` |
| Owns | 55 generations, 4 reference audio rows |

It cannot become an account, and needs no special mechanism to prevent
it. Login refuses a NULL hash before verifying anything; signup hits the
unique email and returns 409, so the row cannot be claimed or given a
password. Both are tested, including the case-variant address.

Migration `0014` inserts the anchor, adds `reference_audio.user_id`,
backfills only rows whose owner is NULL, verifies none remain, then adds
the foreign keys, indexes and NOT NULL. Ordering is the safety property:
the constraint is only reachable after the backfill is proved.

**Direct ownership is three tables** — `generations`, `projects`,
`reference_audio`. Everything else reaches an owner through
`generation_id`. `audio_assets` deliberately has no `user_id`: a second
owner column is a second source of truth that can disagree with the
first.

### The bridge, and how to remove it

`user_id` is NOT NULL, but no code supplies an owner until Part 3. Two
temporary pieces close that gap, and both are meant to be deleted:

- `GenerationRepository` defaults a missing owner to `LEGACY_OWNER_ID`.
  An unauthenticated create genuinely has no user behind it, and the
  anchor records that rather than hiding it.
- `caller_may_access` treats the anchor exactly as it used to treat
  `NULL` — as "pre-authentication, readable". Without this the migration
  would have switched the product off: every row owned, nobody able to
  authenticate as the anchor, every anonymous read a 404.

Grepping `LEGACY_OWNER_ID` lists everything Part 3 has to replace with
the session user.

## Not implemented

None of these exist, and no UI suggests they do:

- **Ownership enforcement** — Part 3. Every generation, project,
  reference and audio asset is currently readable by any caller, and
  `X-User-Id` is still consulted by product routes.
- **Signup and login pages** — the API exists; the UI is Part 4.
- **Route protection** — no frontend route requires a session yet.
- **Email verification** — an address is never confirmed.
- **Password reset** — a forgotten password cannot be recovered. No
  "forgot password" link exists, rather than a dead one.
- **OAuth / social login**, **MFA**, **billing**, **roles/teams**.

## Schema

Migration `0013`, additive only:

- `users.password_hash` — nullable. A user without one cannot be logged
  into, which Part 2 relies on for its ownership anchor.
- `sessions` — `id`, `token_hash` (unique), `user_id` (FK, cascade),
  `created_at`, `expires_at`, indexed on expiry and user.

It touches no ownership column and no existing row. Applied to the live
database with all 55 generations, 119 assets and 4 references intact.
