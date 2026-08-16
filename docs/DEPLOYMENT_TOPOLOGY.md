# Deployment topology

The shape a production LUBER deployment should take, and what changes
from the single Mac it runs on today. Nothing here is provisioned. No
vendor is chosen, because nothing in this design needs one.

---

## 1. Shape

```
                    Internet
                       │
              ┌────────▼────────┐
              │  reverse proxy  │   TLS, rate limiting
              └───┬─────────┬───┘
        ┌─────────▼──┐   ┌──▼──────────┐
        │    Web     │   │     API     │      public
        └────────────┘   └──┬───┬───┬──┘
   ─────────────────────────┼───┼───┼──────── trust boundary
                 ┌──────────┘   │   └────────┐
           ┌─────▼─────┐  ┌─────▼─────┐  ┌───▼──────┐
           │ PostgreSQL│  │   Redis   │  │  object  │
           └─────▲─────┘  └─────▲─────┘  │  storage │
                 │              │        └───▲──────┘
              ┌──┴──────────────┴────────────┘
              │  generation worker  │
              └──────────┬──────────┘
                  ┌──────▼───────┐
                  │ ACE-Step GPU │              private
                  └──────────────┘
```

The worker is the only component that talks to the GPU service, and the
GPU service talks to nothing — it is called, it answers.

## 2. Public and private

**Public:** the reverse proxy only. Web and API sit behind it.

**Never public:** PostgreSQL, Redis, the generation worker, ACE-Step,
object storage credentials. None of them authenticate callers today, and
none of them should be asked to. In particular ACE-Step accepts any
request that reaches it and will spend a GPU-minute on it — reachability
*is* authorisation, which is exactly why it must not be reachable.

The worker has no inbound port at all. It reaches out to Redis,
PostgreSQL, storage and the GPU, and nothing reaches it. That is worth
preserving; it means the component doing the expensive work has no attack
surface of its own.

This phase opens no ports and configures no networking.

## 3. Restart ownership

| Component | Owner in production | Restart policy |
|---|---|---|
| Reverse proxy | platform / systemd | always |
| Web | platform, or `next start` under systemd | always |
| API | `uvicorn` under systemd (or a managed runtime) | always |
| Generation worker | systemd unit, one per host | `Restart=on-failure`, `RestartSec=30` |
| PostgreSQL | managed service, or systemd | managed |
| Redis | managed service, or systemd | managed |
| ACE-Step | systemd on the GPU host | `Restart=on-failure`, generous `RestartSec` — model load is slow |

The macOS LaunchAgent in `deploy/launchd/` is the development analogue of
the worker's systemd unit. Its choices translate directly:
`KeepAlive{SuccessfulExit:false}` → `Restart=on-failure`,
`ThrottleInterval=30` → `RestartSec=30`.

## 4. Persistent data

Three things are irreplaceable, and only three:

| Data | Where | Lost if |
|---|---|---|
| Generation rows, lineage, assets metadata | PostgreSQL | the database is lost — **backups required** |
| Audio objects (RAW, FINISHED, PREVIEW) | object storage | the bucket is lost — audio cannot be reproduced |
| Reference audio uploads | object storage | as above, within their grace period |

Everything else is disposable. Redis holds the queue: losing it strands
`QUEUED` rows that must be resubmitted, which is an inconvenience, not a
data loss. Worker and API filesystems hold nothing that matters.

The pairing that does matter: the database and the bucket must be backed
up **together**. A row pointing at an object that a restore did not
include is precisely the `missing_object` case `luber_health.py` reports,
and it is unrecoverable rather than merely inconsistent.

## 5. Scaling

**Workers scale horizontally.** ARQ's per-job claim already guarantees
exactly-once pickup across any number of them, and the Phase 19 lock is
per host by design so it cannot stand in the way. Each host runs one
worker; add hosts to add throughput. The real ceiling is GPU capacity,
not the queue.

**The API scales horizontally** — it holds no generation state.

**The web tier scales horizontally** or moves to a platform.

**PostgreSQL and Redis scale vertically**, as usual, and neither is close
to a limit at this size.

**ACE-Step is the constraint.** One model process serves one generation
at a time and answers HTTP 429 when its queue fills — surfaced to users
as `PROVIDER_BUSY`. Growth means more GPU hosts behind a load balancer,
with `ACE_STEP_BASE_URL` pointing at it. Nothing in the worker changes.

## 6. Health boundaries

Each tier is probed for what it is responsible for:

- Reverse proxy → `GET /health` on API and web. Liveness only; a 503
  from `/ready` must not remove the API from rotation, because an API
  that cannot queue can still serve every existing song.
- Orchestrator → `GET /ready` for deployment gating, where "cannot reach
  its dependencies" *should* block a rollout.
- Worker → no endpoint. It is supervised by process liveness, and its
  real health is queue depth plus stuck-generation counts, both reported
  by `luber_health.py`.
- ACE-Step → its own HTTP surface, reachable only from the worker.

## 7. Moving ACE-Step to a GPU host

Today ACE-Step runs on the same Mac as everything else at
`http://127.0.0.1:8001`, using MLX on Apple Silicon. On an NVIDIA host:

**Configuration.** `ACE_STEP_BASE_URL` becomes a private address rather
than loopback. It is already an environment variable, so no code changes.

**Latency.** Loopback becomes a network hop. Irrelevant against a
generation measured in tens of seconds, but the *upload* stops being
free: cover and edit requests send source audio as multipart, and a
50 MB master over a slow link is no longer instant. Keep the worker and
the GPU host on the same private network.

**Timeouts.** `ACE_STEP_REQUEST_TIMEOUT` (60s) covers submission, and
that is the one now carrying a real upload — worth re-measuring rather
than assuming. `ACE_STEP_GENERATION_TIMEOUT` (1800s) covers inference and
should not shrink: it is a liveness backstop, and a slow-but-working
engine must never be reported as a dead one. The worker's `job_timeout`
is derived from it and follows automatically.

**Storage.** The worker downloads the engine's output and writes the
assets, so only the worker needs bucket credentials. The GPU host needs
none. Keep it that way.

**Model startup.** Loading weights takes far longer than serving a
request. The GPU service's supervisor needs a `RestartSec` and a startup
grace period sized for the load, or a health check will declare a booting
model dead and restart it forever.

**Supervision.** The GPU service gets its own systemd unit on its own
host. It is not the worker's job to start it — and the worker
deliberately does not refuse to run when it is missing.

**Health.** The proxy must not expose it. The worker reaches it directly;
nothing else does.

## 8. Configuration

Every deployment-relevant setting is already an environment variable with
a development default (`packages/shared/src/luber_shared/settings.py`),
and `.env.example` documents them without carrying a real credential.
What changes per environment:

| Variable | Local | Production |
|---|---|---|
| `DATABASE_URL` | local PostgreSQL | managed instance, private |
| `REDIS_URL` | `localhost:6379` | managed instance, private |
| `ACE_STEP_BASE_URL` | `127.0.0.1:8001` | GPU host, private |
| `STORAGE_PROVIDER` | `local` | `s3` |
| `STORAGE_BUCKET` | unset | the bucket |
| `LUBER_LOG_DIR` | `~/.luber/log` | wherever the platform collects logs |

Secrets arrive from the platform's secret store, not from a file in the
repository.

## 9. Startup and shutdown order

**Up:** PostgreSQL → Redis → object storage (managed, always) →
ACE-Step → API and worker → web.

**Down**, and the order matters more here:

1. Stop accepting submissions — take the API out of rotation, or stop it.
   Nothing new should enter a queue that is being torn down.
2. Let the worker drain, or stop it with `SIGTERM` and accept that the
   in-flight generation is re-queued rather than lost.
3. Stop ACE-Step.
4. Stop Redis and PostgreSQL last.

Reversing steps 1 and 2 is the mistake to avoid: draining a worker while
the API is still accepting work is a queue that never empties.
