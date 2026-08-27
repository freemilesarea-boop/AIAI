# BOORDA production deployment — Vercel + Railway + Neon + Upstash + Gabia

The non-GPU production topology, what each platform runs, and the exact
configuration each needs. No secret value appears in this file or in the
repository; every one is set in a platform's own environment form.

GPU inference is **not** part of this. Production music generation is
NOT READY and is covered under "What does not work yet".

```
                       Gabia (registrar + DNS)
                    ┌──────────┴──────────┐
        boorda.kr ──▶ Vercel              │
                      Next.js web         │  api.boorda.kr
                        │                 └──────▶ Railway
                        │  /api/* rewrite          FastAPI
                        └─────────────────────────▶  │
                                                     ├─▶ Neon PostgreSQL
   PayApp ──── server-to-server ────────────────────▶│
   callbacks   https://api.boorda.kr/v1/billing/…    └─▶ Upstash Redis
                                                            ▲
                                          Railway worker ───┘  (not deployed
                                          Railway cron  ─────┘   — see below)
```

## Why the API has its own hostname

The browser keeps the same-origin path it already has: `next.config.ts`
rewrites `/api/:path*` to `API_PROXY_TARGET`, so the session cookie stays
`SameSite=Lax` on every request including POSTs, and no credentialed
cross-origin traffic is needed. That arrangement is deliberate — see
`docs/AUTHENTICATION_ARCHITECTURE.md`.

PayApp's callbacks are different. They carry no cookie, so they gain
nothing from same-origin, and routing them through Vercel would make a
payment notification depend on the frontend deployment being healthy. A
payment provider posting into a frontend outage is not a failure mode
worth having, so callbacks go straight to `api.boorda.kr`.

Both facts hold at once: `PAYAPP_PUBLIC_BASE_URL` only affects the URLs
we hand PayApp, and `API_PROXY_TARGET` only affects the browser.

## Services

| Platform | Service | Builds from | Command |
|---|---|---|---|
| Vercel | web | `apps/web` | Next.js build |
| Railway | `boorda-api` | `infra/docker/api.Dockerfile` | image `CMD`, binds `$PORT` |
| Railway | `boorda-worker` | `infra/docker/worker.Dockerfile` | `arq luber_generation_worker.worker.WorkerSettings` |
| Railway | `boorda-billing-reconcile` | `infra/docker/worker.Dockerfile` | `python scripts/ops/billing_reconcile.py`, cron |

Railway config-as-code lives in `infra/railway/*.json`. Point each
service's *Config-as-code* setting at the matching file.

Do **not** create PostgreSQL or Redis inside Railway. Neon and Upstash
are the chosen managed services.

## Neon: the pooled endpoint, measured rather than assumed

Production uses Neon's **pooled** endpoint (`…-pooler.…neon.tech`), in
`postgresql+asyncpg://` form:

```
postgresql+asyncpg://<user>:<password>@<endpoint>-pooler.<region>.aws.neon.tech/<db>?ssl=require
```

Two conversions are required and are easy to miss. The connection string
Neon's console hands out is `postgresql://…?sslmode=require&channel_binding=require`:
the scheme must become `postgresql+asyncpg://`, or SQLAlchemy loads the
synchronous psycopg2 driver; and `sslmode`/`channel_binding` are libpq
parameters that asyncpg does not accept — `ssl=require` replaces both.

**On prepared statements.** `create_async_engine_from_url` passes no
`statement_cache_size`, so asyncpg prepares and caches statements. The
long-standing advice is that this cannot work behind PgBouncer in
transaction mode, because a client is not pinned to one server
connection between transactions.

That advice does not match what this deployment measures. Against the
pooled endpoint:

| | |
|---|---|
| Load | 40 concurrent workers × 25 transactions |
| Result | 1000 / 1000 succeeded |
| Errors | 0 |
| Distinct server backends observed | 15 |

Each transaction ran the same parameterised statement, and the work was
spread across fifteen different backend PIDs — so connections genuinely
were reassigned, and the cached statements survived it. Neon's pooler
supports protocol-level prepared statements (PgBouncer 1.21+), which is
what makes this hold.

Scope of that claim, stated precisely: it is one measurement, against
Neon's pooler, at this concurrency, with this driver pairing
(SQLAlchemy 2.x + asyncpg). It is not a general guarantee that asyncpg
works behind every PgBouncer.

The direct endpoint (the same host without `-pooler`) was measured under
the identical load and also passed, so it remains a safe fallback. If
prepared-statement errors ever do appear, the switch is
`connect_args={"statement_cache_size": 0}` in
`packages/database/src/luber_database/engine.py` — deliberately not
carried now, since nothing measured needs it.

`pool_pre_ping=True` is already set, which is what makes a Neon endpoint
that has scaled to zero reconnect rather than error.

## Upstash: no code change needed

Verified against the installed versions rather than assumed:

| | |
|---|---|
| `arq` 0.28.0 | `RedisSettings.from_dsn("rediss://…")` → `ssl=True` |
| `redis` 5.3.1 | `Redis.from_url("rediss://…")` → `SSLConnection` |

Both the API's Redis client and every ARQ worker accept the TLS scheme
directly. Use Upstash's standard Redis TCP/TLS endpoint:

```
rediss://default:<password>@<endpoint>.upstash.io:6379
```

Not the REST API — ARQ needs a long-lived connection and the code has no
REST transport.

## Environment

Secrets are set in each platform's environment form. Never in git, never
in `.env.example`, never echoed into logs.

### Railway — `boorda-api`

| Variable | Value | Secret |
|---|---|---|
| `ENVIRONMENT` | `production` | |
| `DATABASE_URL` | Neon pooled string, `postgresql+asyncpg://…?ssl=require` | ● |
| `REDIS_URL` | Upstash `rediss://…` | ● |
| `PAYAPP_USERID` | operator's | ● |
| `PAYAPP_LINKKEY` | operator's | ● |
| `PAYAPP_LINKVAL` | operator's | ● |
| `PAYAPP_PUBLIC_BASE_URL` | `https://api.boorda.kr` | |
| `PAYAPP_RETURN_BASE_URL` | `https://boorda.kr` | |
| `CORS_ORIGINS` | `["https://boorda.kr"]` | |
| `GENERATION_PROVIDER` | `ace_step` | |
| `STORAGE_PROVIDER` | `local` until a bucket exists | |

`ENVIRONMENT=production` is what turns on `Secure` session cookies —
`cookie_is_secure()` returns `settings.is_production`. Setting it wrong
means either cookies dropped over plain HTTP, or cookies sent without
`Secure` over HTTPS.

Sessions need **no signing secret**: the token is random and only its
SHA-256 hash is stored.

### Railway — `boorda-worker` and `boorda-billing-reconcile`

`ENVIRONMENT`, `DATABASE_URL`, `REDIS_URL` — the same Neon and Upstash
values. No PayApp secrets: neither service talks to PayApp.

The reconciliation job reads `DATABASE_URL` only.

### Vercel — web

| Variable | Value |
|---|---|
| `API_PROXY_TARGET` | `https://api.boorda.kr` |

That is the whole list. Vercel gets **no** `DATABASE_URL`, no `REDIS_URL`
and no PayApp secret — nothing rendered server-side needs one, and a
secret on the frontend platform is a secret in one more place than it has
to be.

## Migrations

Run 0019 against Neon **before** any billing use, from one place only.
Set it as the API service's pre-deploy command in Railway:

```
uv run --no-sync alembic -c packages/database/alembic.ini upgrade head
```

One service, so two deployments cannot race the same migration. The
worker and the cron job must not run migrations.

Verify afterwards that all four billing invariants exist:
`uq_one_open_checkout_per_user`, `uq_billing_events_provider_fingerprint`,
`uq_billing_payments_provider_payment`,
`uq_subscriptions_provider_subscription`.

## Reconciliation schedule

`17 3 * * *` — once a day, at an offset minute rather than on the hour,
so it does not pile onto whatever else fires at 03:00.

Daily suits what the job detects. `RENEWAL_GRACE_HOURS` is 48, so a
renewal is not even considered missing until two days after its period
ended; checking more often than daily would find nothing new. The job is
also idempotent about its own output — it does not re-flag a subscription
that already has an open anomaly of the same kind.

`restartPolicyType: NEVER` — a cron job that restarts on exit is a cron
job that runs continuously.

## What does not work yet

**Production music generation: NOT READY.** Two independent reasons.

*No GPU.* `GENERATION_PROVIDER=ace_step` points at
`ACE_STEP_BASE_URL`, which defaults to `127.0.0.1:8001` — the operator's
Mac. Nothing on Railway serves it.

*No object storage.* `STORAGE_PROVIDER=local` writes under `data/` on a
container filesystem that does not survive a redeploy. S3-compatible
storage is already implemented (`S3AudioStorage`, presigned URLs) and
needs configuration, not code: `STORAGE_PROVIDER=s3`, `STORAGE_BUCKET`,
`STORAGE_REGION`, `STORAGE_ENDPOINT` (for R2), `STORAGE_ACCESS_KEY_ID`,
`STORAGE_SECRET_ACCESS_KEY`.

**Consequence, stated plainly:** the API will accept
`POST /v1/generations` and reserve an allowance slot. With no worker
deployed the row stays `QUEUED` forever, and an unsettled reservation
holds its slot by design. So a real user pressing 만들기 on this
deployment spends allowance and receives nothing.

There is no switch that refuses generation — `provider_from_settings`
offers only `mock` and `ace_step`, and `mock` would fabricate songs,
which is worse than refusing. Adding a disabled state is product work,
not infrastructure, and is deliberately out of scope here.

**Therefore this deployment must not be advertised or opened to real
users** until GPU and object storage are connected. It exists to carry
one operator-supervised PayApp payment test, where billing is the subject
and generation is not exercised.

**Email:** Resend is not implemented anywhere in the repository. No
billing correctness depends on email — payment confirmation is the
`pay_state=4` notification, not a message to anyone.
