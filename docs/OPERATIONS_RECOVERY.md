# Operations — recovery procedures

Commands for a LUBER development stack that has gone wrong. Every one of
them has been run against this machine; nothing here is aspirational.

Run from the repository root with the project venv. Start by asking what
is actually broken:

```bash
.venv/bin/python scripts/development/luber_health.py
```

Exit status is 0 when every check passes and 1 when any fails, so it is
safe in a cron or a pre-deploy gate. It never mutates anything — there is
no repair flag, by design.

---

## Normal startup

The four processes, in dependency order. Redis and PostgreSQL are
expected to be running already.

```bash
# API
.venv/bin/uvicorn luber_api.main:app --host 127.0.0.1 --port 8000

# Generation worker — exactly one
.venv/bin/arq luber_generation_worker.worker.WorkerSettings

# Web
cd apps/web && pnpm dev
```

ACE-Step runs from its own checkout and virtualenv:

```bash
~/ace-step-1.5/.venv/bin/acestep-api --host 127.0.0.1 --port 8001
```

---

## API restart

Safe at any time, including mid-generation. The API holds no generation
state: the row is in PostgreSQL and the job is in Redis, and the worker
is not talking to the API. A generation in flight completes normally.

```bash
kill "$(lsof -nP -iTCP:8000 -sTCP:LISTEN | awk 'NR==2{print $2}')"
.venv/bin/uvicorn luber_api.main:app --host 127.0.0.1 --port 8000
```

Confirm: `curl -s localhost:8000/ready` → `{"status":"ready"}`.

---

## Web restart

Also safe mid-generation — the browser reads job state from the API on
every load, so a reload recovers an in-flight generation.

```bash
kill "$(lsof -nP -iTCP:3000 -sTCP:LISTEN | awk 'NR==2{print $2}')"
cd apps/web && pnpm dev
```

**Never run `pnpm build` while `pnpm dev` is running.** Both write
`.next`, and the result is a corrupted cache that fails at runtime with
no useful message. Stop dev first; if the cache is already corrupt:

```bash
kill <dev pid>; rm -rf apps/web/.next; cd apps/web && pnpm dev
```

---

## Worker restart

Stopping a worker mid-generation is survivable but not free: the
in-flight run is abandoned, recorded as
`FAILED / GENERATION_INTERRUPTED`, re-queued by ARQ, and re-run from the
beginning by the next worker. That is a second full inference. Prefer to
wait for the queue to drain.

```bash
# Drain first, if you can afford to wait.
redis-cli -p 6379 zcard luber:generation      # 0 means nothing pending

kill -TERM "$(pgrep -f 'arq .*luber_generation_worker')"
.venv/bin/arq luber_generation_worker.worker.WorkerSettings
```

`SIGTERM`, not `SIGKILL`: `SIGTERM` lets ARQ record the cancellation and
re-queue the job.

---

## Duplicate workers

Two workers on one queue will not process the same job twice — ARQ claims
each job in Redis before starting it — but they will run two generations
at once against one GPU. Detect and fix:

```bash
pgrep -fl 'arq .*luber_generation_worker'     # expect exactly one line
kill -TERM <the extra pid>
```

`luber_health.py` reports this as a failed check.

---

## Redis restart

The generation queue lives in Redis. Losing it while jobs are pending
loses those jobs — the rows survive in PostgreSQL, stranded at `QUEUED`.

**The worker does not survive Redis going away.** It raises
`redis.exceptions.ConnectionError` and exits; it does not reconnect.
Expect to restart it afterwards.

```bash
redis-cli -p 6379 zcard luber:generation      # 0 before proceeding
redis-cli -p 6379 save                        # snapshot to dump.rdb

redis-cli -p 6379 shutdown nosave
redis-server --port 6379 --dir /Users/theblank/luber-redis-data \
             --dbfilename dump.rdb --daemonize yes
redis-cli -p 6379 ping                        # PONG

.venv/bin/arq luber_generation_worker.worker.WorkerSettings
```

Pass the same `--dir` and `--dbfilename` the running instance reported,
or the restarted server will come up against a different data directory.
Check them before stopping it:

```bash
redis-cli -p 6379 config get dir dbfilename appendonly
```

While Redis is down the API stays up and read-only: `/health` is 200,
`GET /v1/generations` works, and `POST /v1/generations` returns **503
`QUEUE_FAILED`** with the row marked `FAILED` rather than left waiting.
`/ready` returns 503 and names the unavailable dependency.

---

## A generation stuck in QUEUED

Queue wait is normally under a second, so more than a few minutes means
nothing is consuming.

```bash
pgrep -fl 'arq .*luber_generation_worker'     # is there a worker?
redis-cli -p 6379 zcard luber:generation      # is the job actually queued?
```

- **No worker** → start one; the job is picked up within a second.
- **Worker running, queue depth 0, row still QUEUED** → the job was lost
  from Redis (a restart without a snapshot). The row will never run.
  Nothing re-enqueues it automatically; the user's recourse is to
  generate again.

---

## A generation stuck in GENERATING

Since Phase 18 this should not persist: a cancelled run records
`GENERATION_INTERRUPTED`, and a slow one hits the provider's timeout and
records `GENERATION_TIMEOUT`. If you see one anyway:

```bash
.venv/bin/python scripts/development/luber_health.py     # flags >15min
pgrep -fl 'arq .*luber_generation_worker'
curl -s localhost:8001/docs -o /dev/null -w '%{http_code}\n'   # engine alive?
```

A run legitimately in progress holds a worker at high CPU. If the worker
is gone and the row still says `GENERATING`, the row predates this
phase's fix; ARQ will not re-run a job it has already finished with.
Correcting such a row is a manual `UPDATE` and therefore a deliberate
decision, not a routine operation — confirm the worker is genuinely not
running first.

---

## Provider (ACE-Step) outage

```bash
curl -s localhost:8001/docs -o /dev/null -w '%{http_code}\n'   # 200 = alive
```

Generations submitted during an outage fail with a stable code rather
than hanging: `MODEL_LOAD_FAILED` when the engine is unreachable,
`PROVIDER_BUSY` when its own queue is full (HTTP 429). Nothing needs
cleaning up — no partial asset is published on a failed run. Restart the
engine from its own checkout and generate again.

The LUBER stack does not need restarting for a provider outage.

---

## Storage inconsistency

```bash
.venv/bin/python scripts/development/luber_health.py --json
```

`asset_consistency` reports four distinct problems. They are not equally
serious:

| Report | Meaning | Action |
|---|---|---|
| `completed_without_delivery_asset` | The UI says ready and there is nothing to play | Investigate first — this is the one users hit |
| `missing_object` | A row points at an object that is gone | Regenerate; do not delete the row blindly |
| `duplicate_roles` | Two assets share one role | Should be impossible (upsert by role); investigate before touching |
| `orphan_objects` | An object with no row | Usually an in-flight generation. Harmless; re-check before acting |

Nothing here deletes anything. Deleting storage is not a recovery
procedure — a wrong deletion destroys audio that cannot be reproduced,
while a stale object costs disk.

Unused reference audio is a separate, deliberate lifecycle with its own
grace period; use its dry run rather than deleting objects by hand.

---

## When to do nothing

- An `orphan_objects` count that matches the number of generations
  currently running.
- A single `PROVIDER_BUSY` failure — the engine was saturated and the
  same request will succeed.
- A generation that has been `GENERATING` for under two minutes; the
  observed maximum is 110 seconds.
