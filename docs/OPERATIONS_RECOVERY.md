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
ACESTEP_NO_INIT=false ~/ace-step-1.5/.venv/bin/acestep-api --host 127.0.0.1 --port 8001
```

**`ACESTEP_NO_INIT=false` is not optional.** Without it the server starts
in lazy-load mode: it binds the port, answers `/health` with 200, and
loads no model until the first generation arrives. LUBER checks readiness
before it submits, sees `models_initialized: false`, and fails the
generation with `MODEL_LOAD_FAILED` — so every song fails against an
engine that looks alive. Measured: the engine received only health and
model-list probes and never a generation request.

Eager start costs about a minute of loading and roughly 9 GB of weights
resident. That is the trade: a slow start, or a fast start that cannot
generate.

Ready means `models_initialized` is true and `loaded_model` names the
configured model:

```bash
curl -s localhost:8001/health | python3 -m json.tool
# data.status            = "ok"
# data.models_initialized = true
# data.loaded_model       = "acestep-v15-turbo"   ← must match ACE_STEP_MODEL
```

`loaded_lm_model` may be `null`; the lyric LM loads separately and text
to music does not wait on it. Check the health fields rather than the
HTTP status — a 200 alone does not mean the engine can generate.

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

.venv/bin/python scripts/development/luber_runtime.py restart
```

The tool sends `SIGTERM`, never `SIGKILL`: `SIGTERM` lets ARQ record the
cancellation and re-queue the job. It finds the process through the
worker's own lock file and verifies the PID's command line before
signalling it, so it cannot hit an unrelated Python process.

Force-killing is appropriate in exactly one case: the worker has ignored
`SIGTERM` past the timeout *and* you have confirmed it is not mid-
generation. `stop` prints the `kill -9` command with the right PID rather
than doing it for you.

---

## Duplicate workers

Two workers on one queue will not process the same job twice — ARQ claims
each job in Redis before starting it — but they will run two generations
at once against one GPU. Detect and fix:

Since Phase 19 a second worker on this machine cannot start: each one
takes an exclusive `flock` on `~/.luber/run/generation-worker.lock` at
startup and a duplicate exits with status **3** and a message naming the
holder. Verify:

```bash
.venv/bin/python scripts/development/luber_runtime.py status
```

`luber_health.py` also reports duplication as a failed check. If it ever
does fire, one of the two was started in a way that bypassed the lock —
find it through `status`, not through a pattern kill.

The lock is per machine, which is deliberate: a future deployment running
one worker on each of several hosts is legitimate, and a global lock
would forbid it. Cross-host exactly-once processing is ARQ's per-job
claim, not this.

---

## Redis restart

The generation queue lives in Redis. Losing it while jobs are pending
loses those jobs — the rows survive in PostgreSQL, stranded at `QUEUED`.

**The worker does not survive Redis going away.** It raises
`redis.exceptions.ConnectionError` and exits; it does not reconnect.
Expect to restart it afterwards.

```bash
redis-cli -p 6379 zcard luber:generation      # 0 before proceeding

# Read the running server's own settings rather than assuming them, so
# the restart comes back on the same data.
REDIS_DIR=$(redis-cli -p 6379 config get dir | tail -1)
REDIS_DB=$(redis-cli -p 6379 config get dbfilename | tail -1)
redis-cli -p 6379 save                        # snapshot to $REDIS_DB

redis-cli -p 6379 shutdown nosave
redis-server --port 6379 --dir "$REDIS_DIR" --dbfilename "$REDIS_DB" --daemonize yes
redis-cli -p 6379 ping                        # PONG

.venv/bin/arq luber_generation_worker.worker.WorkerSettings
```

The `config get` reads above matter: start Redis without them and it
comes up on its compiled-in default directory, which is **not** where
this deployment's data lives. Capture them before the shutdown, not
after — a stopped server cannot be asked.

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
curl -s localhost:8001/health | grep -o '"models_initialized":[a-z]*'  # engine READY?
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
curl -s localhost:8001/health | grep -o '"models_initialized":[a-z]*'  # true = ready
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

---

## Supervised services

The generation worker is the only service this repository supervises. It
is the one whose unattended death silently stops the product; everything
else is either already supervised by the system or something you start
deliberately and want to watch.

| Service | Owner | Auto-restart |
|---|---|---|
| PostgreSQL | Homebrew LaunchAgent | yes, at login |
| Redis | system package, started manually | **no** |
| ACE-Step | manual (heavy, started on purpose) | no |
| API | manual during development | no |
| Web | manual during development | no |
| Generation worker | this repository | only once the agent is installed |

### State, restart, logs

```bash
.venv/bin/python scripts/development/luber_runtime.py status
.venv/bin/python scripts/development/luber_runtime.py start
.venv/bin/python scripts/development/luber_runtime.py stop
.venv/bin/python scripts/development/luber_runtime.py restart

tail -f ~/.luber/log/generation-worker.err.log
```

Logs go to `~/.luber/log` (override with `LUBER_LOG_DIR`). They are
append-only and **not rotated** — see the limitation note below.

### Turning auto-restart on

Not installed by default. Installing it registers a persistent user
LaunchAgent that survives logout and reboot, so it asks first:

```bash
# Look at exactly what would be installed:
.venv/bin/python scripts/development/luber_runtime.py plist

# Install and load it (prompts for confirmation):
.venv/bin/python scripts/development/luber_runtime.py plist --install
```

### Turning it off

```bash
# Temporarily — stop the worker without launchd bringing it back:
launchctl unload ~/Library/LaunchAgents/com.luber.generation-worker.plist

# Permanently:
.venv/bin/python scripts/development/luber_runtime.py plist --uninstall
```

`--uninstall` removes only the agent file. Logs, the lock file, Redis
data and the database are left alone.

### What the agent does when things go wrong

`KeepAlive` is set to restart on *unsuccessful* exit only, so a clean
`stop` stays stopped. `ThrottleInterval` is 30s, which bounds the one
crash loop that can really happen: with Redis down the worker exits at
startup, launchd waits 30s, tries again, and keeps doing so until Redis
returns — then the next attempt succeeds on its own. Two failures a
minute is cheap, and the failures are visible in the error log rather
than hidden.

A worker that cannot reach PostgreSQL refuses to start for the same
reason and recovers the same way. ACE-Step is deliberately *not*
checked: a generation submitted while the engine is down fails
truthfully with its own code, so refusing to run the queue would turn a
per-job failure into an outage.
