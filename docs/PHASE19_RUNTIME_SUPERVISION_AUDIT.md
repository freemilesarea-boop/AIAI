# Runtime supervision — ownership audit

Who starts each LUBER service today, what happens when it dies, and
which of those answers is unacceptable. Read from the running machine
before anything was changed.

---

## 1. What is actually running

Every process inspected live: PID, parent, working directory, start time.

| Service | Port | PID/PPID | Started by | Restarts itself | Survives shell exit | Survives logout | Survives reboot |
|---|---|---|---|---|---|---|---|
| PostgreSQL 15 | 5432 | 1188 / 1 | Homebrew LaunchAgent `homebrew.mxcl.postgresql@15` | **yes** (`KeepAlive`) | yes | no (user agent) | **yes** (`RunAtLoad`) |
| Redis | 6379 | 6969 / 1 | manual `redis-server --daemonize` | no | yes | no | no |
| ACE-Step | 8001 | 52076 / 52075 | `bash start_api_server_macos_pinned.sh` → `uv run` | no | yes (reparented) | no | no |
| LUBER API | 8000 | 2837 / 1 | manual `nohup uvicorn` | no | yes | no | no |
| Generation worker | — | 19350 / 1 | manual `nohup arq` | no | yes | no | no |
| LUBER web | 3000 | 17961 / 17955 | `pnpm dev` (live parent) | no | **no** | no | no |

`brew services list` confirms the split: `postgresql@15 started`,
`redis none`. PostgreSQL is the only service on this machine with an
owner. Everything else is running because someone typed a command, and
`ppid 1` on most of them means the shell that typed it is long gone.

The gap Phase 18 named is visible here as a table row: the generation
worker has no owner, and its unattended death silently stops the product
while every other service keeps answering.

## 2. Choosing a supervisor

| Option | Verdict |
|---|---|
| **launchd user agent** | **Chosen.** Native, no install, no root, `KeepAlive` + `ThrottleInterval` give bounded restart, log paths are part of the config. |
| launchd *system* daemon | Rejected. Needs root and runs before login for a development service that needs neither. |
| Homebrew services | Right answer for Redis, wrong shape for LUBER: `brew services` supervises formulae, not a repository's own process. |
| Repository supervisor process | Rejected. Writing a process babysitter means writing restart, backoff, log handling and its own supervision — all of which launchd already has. |
| tmux / screen | Rejected. Recovers nothing; it only keeps a terminal alive. |
| `nohup` (status quo) | Rejected. This audit is what it produced. |

No third-party process manager is installed. The system already has one.

## 3. Which services this repository supervises

Deliberately one: **the generation worker**.

- **PostgreSQL** — already a Homebrew LaunchAgent. A second supervisor
  would fight it. Left alone.
- **Redis** — see §7. Not taken over, for a specific reason.
- **ACE-Step** — manual on purpose. It holds ~1.2 GB resident and is the
  most expensive thing on the machine; a developer should decide when it
  runs, and it lives in its own checkout with its own virtualenv.
- **API / web** — manual on purpose. These are the processes a developer
  restarts most and watches most; daemonising `next dev` in particular
  would hide the output it exists to print. Production is a different
  question, answered in `DEPLOYMENT_TOPOLOGY.md`.

## 4. The single-worker guarantee

Two workers cannot process the same job — ARQ claims each job in Redis
under a `WATCH`/`MULTI` transaction — but they can run two inferences at
once on hardware sized for one.

| Mechanism | Verdict |
|---|---|
| `ps` + grep | Rejected. Racy, and one typo from killing an unrelated process. |
| PID file | Rejected on its own. A PID file goes stale on `SIGKILL` and needs validation logic that can itself be wrong. |
| **`flock` on a lock file** | **Chosen.** The kernel releases it when the holder dies *however* it dies, so there is no stale state and no cleanup path. |
| Database advisory lock | Rejected. Global, so it forbids a legitimate multi-host deployment; also makes worker startup depend on a session staying open. |
| Redis lock | Rejected for the same reason, and it would need the very dependency whose loss kills the worker. |
| launchd single-instance | Necessary but not sufficient: launchd guarantees one *supervised* instance, and does nothing about a worker started by hand alongside it. |

The lock is taken inside the worker's own startup, so "running" means
"holds the lock", and a duplicate exits **3** with a message naming the
holder. Verified: a raw `arq …` launched beside a running worker exited 3
and left the original untouched; after `SIGKILL` the lock was immediately
reclaimable.

**Scope, stated plainly.** This is a per-host guarantee. One worker on
each of five machines is a legitimate future topology and this lock
cannot prevent it, because each host has its own file. Cross-host
exactly-once processing is ARQ's per-job claim and nothing here changes
that.

## 5. Startup dependencies

Real dependencies, not assumed ones:

| Service | Needs at startup | Does *not* need |
|---|---|---|
| API | PostgreSQL | Redis (starts and serves reads; `POST` returns 503 `QUEUE_FAILED`), ACE-Step |
| Worker | Redis (ARQ), PostgreSQL | **ACE-Step** |
| Web | nothing | API (renders, shows errors) |

The worker now verifies PostgreSQL explicitly with a `select 1` and
refuses to start if it cannot reach it — ARQ already fails loudly on
Redis, but PostgreSQL is not touched until the first job, so a bad
database URL used to look healthy right until it dropped a real
generation.

ACE-Step is deliberately not checked. A generation submitted while the
engine is down fails truthfully with a stable code, so refusing to run
the queue would convert a recoverable per-job failure into an outage of
every job — including the ones that would have succeeded by the time
they ran.

Startup order follows from the table: PostgreSQL → Redis → API and
worker (either order) → web. ACE-Step can be started at any point before
the first generation.

Shutdown is the reverse, with one rule: stop accepting submissions
before draining the worker, so nothing is queued into a system that is
going away. Locally that is just "stop the API first"; the production
version is in `DEPLOYMENT_TOPOLOGY.md`.

## 6. Restart policy and crash loops

`KeepAlive` is `{SuccessfulExit: false}`, not a bare `true`. A clean exit
is a deliberate stop and relaunching it would make `stop` impossible.

`ThrottleInterval` is 30s against launchd's default of 10s. The crash
loop that can really happen is a worker starting while Redis is down: it
exits, waits 30s, tries again, and keeps doing so until Redis returns —
at which point the next attempt simply succeeds. Two attempts a minute
costs nothing and every failure is written to the error log rather than
hidden. This is bounded by the platform, not by a hand-written retry
loop.

## 7. Redis

Phase 18 established that Redis loss kills the worker. The obvious fix is
`brew services start redis`, and it has a catch worth stating rather than
discovering later.

This deployment's Redis runs with `dir` set to a directory under the
user's home — chosen at some earlier point, and holding the queue's
snapshot. Homebrew's Redis reads `/opt/homebrew/etc/redis.conf` and would
come up on its own default data directory. Starting `brew services redis`
today would therefore supervise a Redis pointed at *different data*.

Two honest options, neither taken here because both change machine state
beyond this repository:

1. Set `dir` in Homebrew's `redis.conf` to the existing directory, then
   `brew services start redis`. Supervision without moving data.
2. Accept Homebrew's directory. Simplest, but abandons the current
   snapshot — and the queue is the one thing in Redis worth keeping.

Recommendation: option 1, as a deliberate step the user takes. Wrapping
Redis in a LUBER-owned agent was rejected: it is a system package with
its own supported lifecycle, and a second owner is how two supervisors
end up fighting over one port.

## 8. Logging

Supervised output goes to `~/.luber/log/generation-worker.{out,err}.log`,
resolved from `$HOME` (or `LUBER_LOG_DIR`) — never a committed absolute
path. The lock lives in `~/.luber/run`, honouring `XDG_RUNTIME_DIR` where
it is set so the Linux path is identical.

**Not rotated.** launchd does not rotate, and adding a `newsyslog` entry
is a machine-wide change this phase does not make. The worker logs one
line per job, so growth is slow, but this is a real limitation and not a
solved problem.

## 9. Health versus readiness

Three things that answer different questions:

| | Question | Fails when |
|---|---|---|
| `GET /health` | Is this process alive? | The API is down |
| `GET /ready` | Can it do its job? | A dependency it needs is missing — returns 503 naming it |
| `luber_health.py` | Is the whole system sound? | Any service, stuck job, or database/disk disagreement |

`/health` deliberately stays 200 when Redis is down. It is a liveness
probe, and a supervisor that restarts the API because Redis is missing
would be restarting the wrong process. `/ready` is the one that tells the
truth about serving.

---

*Audit performed before any change. The supervision described here is
generated from a repository template and is **not installed**; the
command to install it is in `OPERATIONS_RECOVERY.md`.*
