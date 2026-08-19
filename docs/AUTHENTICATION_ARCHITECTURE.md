# Authentication architecture

How a person proves who they are, and what the product does with that.

**Phase 20A, complete.** A person can create an account in a browser,
use a private workspace, and be certain nobody else can reach it.
Authentication, ownership and enforcement are all in place; what follows
describes the system that exists, not a plan.

Songs are private. The boundary is the API: every product route requires
a session and every query is scoped to the caller, so what the browser
chooses to render cannot widen access.

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

## The boundary, in one place

| Question | Answer |
|---|---|
| Who is calling? | The server-side session, resolved from an HttpOnly cookie. Nothing else — `X-User-Id` is read nowhere |
| What can they see? | Only rows whose `user_id` matches, filtered in SQL rather than after loading |
| What happens to a guest? | 401 on every product route |
| What happens to a foreign UUID? | 404, byte-identical to one that does not exist |
| Where is the check? | `require_current_user` on the router, `GenerationRepository(owner=…)` on the query |

Frontend route protection exists for UX and request hygiene. It prevents
a guest's page from rendering and firing private requests; it is not the
security boundary and does not need to be.

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

## Browser experience

`AuthProvider` asks `/v1/auth/me` on every load. There is no token to
persist, so nothing about the session is kept client-side. Three states:
`loading` means "not asked yet" and `unauthenticated` means "asked, and
nobody" — collapsing them would flash the signed-out UI on first paint.

`RequireAuth` renders a placeholder rather than the page until the
session resolves. Rendering children first would fire their private
requests on a guest's behalf and briefly paint a page shaped like
somebody's library.

**401 recovery.** A 401 from a *product* request ends the session
through the same path a manual sign-out uses: private storage cleared,
in-memory state dropped, redirect to login carrying a safe return
destination. Only 401 — a 403 is an origin refusal and a 404 is somebody
else's resource, and treating either as an expired session would sign
people out for touching the wrong thing.

**Return destinations** are treated as attacker-controlled. Absolute
URLs, scheme-relative `//host`, `javascript:`, backslash variants and
control characters are refused rather than sanitised: a value that needs
cleaning to be safe is a value to reject.

**Private state does not outlive a session.** Two localStorage keys
(`luber.activeGenerationId`, `luber.recentGenerations`) hold song ids and
titles, and the global player holds a track title. All are cleared on
sign-out and on expiry. The player learns about it through a small
`session-events` broadcast, because `AuthProvider` sits above
`PlayerProvider` and cannot call into it directly.

**Logout is local-first-safe.** If the network call fails the local
session is discarded anyway: leaving someone apparently signed in on a
shared machine is the worse outcome, and the server session expires on
its own regardless.

**Auth forms post.** `method="post"` is a safety net rather than a
feature — the submit handler prevents default, so it is never used. But
a form with no method defaults to GET, and a submit escaping the React
handler would put the password in the URL, the history and every access
log in between.

## Security headers

`X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`,
`Referrer-Policy: strict-origin-when-cross-origin`, and a
`Permissions-Policy` denying camera, microphone and geolocation.

Deliberately absent: **Content-Security-Policy** and **HSTS**. A CSP
tight enough to be worth having needs the app's real script, style and
media origins measured under production; a loose one is decoration. HSTS
is meaningless until the deployment terminates TLS. Both are deployment
work, recorded rather than guessed at.

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
