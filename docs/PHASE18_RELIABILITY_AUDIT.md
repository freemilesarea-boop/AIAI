# Reliability — failure-mode audit

Every boundary a generation crosses, what happens when it breaks, and
whether anything is lost. Written against the running stack (PostgreSQL
5432, Redis 6379, ACE-Step 8001, API 8000, web 3000, one ARQ worker) and
verified by interrupting it, not by reading it.

Three weaknesses were found. Two are fixed in this phase; the third is a
supervision gap that no code change can close.

---

## 1. The path

```
browser → API → PostgreSQL (row)      ← durable record
              → Redis (arq zset)      ← durable queue
       worker ← Redis
              → ACE-Step :8001        ← inference
              → local storage         ← RAW master
              → finishing             ← FINISHED master
              → preview transcode
              → PostgreSQL (assets, status)
       browser ← API (poll) → audio route → player / download
```

The API never runs inference. It writes a row, enqueues an id, and
returns. Everything after that is the worker's.

## 2. Boundaries

### Browser → API
- **Failure** API down or restarting.
- **Behaviour** The browser's fetch fails; the page keeps rendering. Job
  state is re-read from the API on the next poll.
- **Recovery** Automatic. Verified: a generation submitted, the API
  killed mid-run and restarted, the generation completed untouched.
- **Data loss** None. The API holds no generation state in memory.
- **Duplicate work** None — a restart does not re-enqueue.
- **Stuck state** None.
- **Observability** `/health` (liveness) and `/ready` (dependencies).
- **User sees** A stalled poll, then normal state.

### API → PostgreSQL
- **Failure** Commit fails while writing the row.
- **Behaviour** The request raises before enqueue; no id reaches Redis.
- **Data loss** None; nothing was promised.
- **User sees** A 5xx with a stable code.

### API → Redis (enqueue)
- **Failure** Redis unreachable.
- **Behaviour** Verified by stopping Redis: `POST /v1/generations` →
  **503 `QUEUE_FAILED`**, and the row that had already been written is
  marked `FAILED/QUEUE_FAILED` in the same request. `/ready` reports
  `{"redis":"unavailable"}` with 503 while `/health` stays 200.
- **Data loss** None. No orphan QUEUED row is left waiting for a worker
  that will never see it — the compensating write is what prevents it.
- **Stuck state** None.
- **User sees** "The generation service is busy and could not accept
  your request."

### Redis → worker
- **Failure** Redis dies while the worker is idle.
- **Behaviour** Verified: the worker raises `redis.ConnectionError` and
  **the process exits**. It does not reconnect.
- **Data loss** None — the queue is an RDB-backed zset and survives.
- **Recovery** Requires an external restart. See §4.

### Worker → ACE-Step
- **Failure** Provider unreachable, OOM, model load failure, or a full
  queue.
- **Behaviour** `GenerationProviderError` carries a stable `ErrorCode`;
  the generation lands in `FAILED` with that code, and no asset is
  published.
- **Queue full** ACE-Step answers **HTTP 429 "Server busy: queue is
  full"** (`release_task_route.py:74`, covered by its own test). This
  used to fall through to `UNKNOWN_GENERATION_ERROR` — a timing
  condition reported as a broken song. Now classified from the status
  code as **`PROVIDER_BUSY`**.
- **Duplicate work** None; a provider failure is terminal, not retried.
- **User sees** Copy specific to the code; for `PROVIDER_BUSY`, that the
  engine is busy and their settings are fine.

### Worker → storage → finishing → preview
- **Failure** Any write or transcode failure.
- **Behaviour** Raises inside the try block → `FAILED` with
  `UPLOAD_FAILED` / `ENCODING_FAILED`. `COMPLETED` is written only after
  every asset is stored *and* recorded.
- **Data loss** The RAW master is never overwritten by finishing; a run
  that produces no finished master retracts the stale row rather than
  leaving it to win delivery selection.
- **Duplicate work** Assets are upserted by role against deterministic
  storage keys, so a re-run replaces rather than accumulates. Verified
  on a real interrupted-then-retried generation: one row per role.
- **Covered by** `test_postprocess_failure.py`,
  `test_finishing_integration.py`.

### Worker cancelled mid-generation
The one that was broken. See §3.

### API → player / download
- **Failure** Missing object behind a present row.
- **Behaviour** The audio route resolves by generation id and role;
  storage keys never appear in a request.
- **Verified** `200` with `accept-ranges: bytes`; `206` with a correct
  `content-range` for both a leading and a mid-file range; `attachment`
  disposition under `?download=true`; served bytes SHA-256-identical to
  the stored object and to the recorded hash. Only `master` and
  `preview` are addressable — `RAW_MASTER` is not reachable by any
  client value.

## 3. Finding: a cancelled job left the row claiming GENERATING

`asyncio.CancelledError` derives from `BaseException`, so
`GenerationService.execute`'s `except Exception` never saw it. Anything
that cancelled the task — the worker being stopped, or ARQ's own
`job_timeout` — left the row at `GENERATING` with no error recorded and
no process behind it.

Reproduced: a generation was taken to `GENERATING`, the worker sent
`SIGTERM`, and the row stayed `GENERATING` with `error_code = NULL`.

Two triggers, with different endings:

| Trigger | Before |
|---|---|
| Worker stopped | Row stuck `GENERATING`; ARQ re-queues, and a returning worker silently repairs it |
| ARQ `job_timeout` (300s) | Row stuck `GENERATING` **permanently** — a timeout is not a cancellation as far as ARQ's retry rule is concerned, so nothing ever runs again |

The second was reachable. ARQ's default `job_timeout` is 300s while the
ACE-Step provider's own liveness backstop is at least 1800s, so the queue
always cut a slow generation first — and cut it in the one way that
recorded nothing.

**Fixed two ways.** `execute` now catches `CancelledError`, records
`FAILED/GENERATION_INTERRUPTED`, and re-raises so ARQ keeps its retry
semantics; a retry calls `mark_started`, which clears the error and moves
the row on. And `job_timeout` is now set from
`ace_step_generation_timeout + 600`, putting the queue's limit *outside*
the provider's so a slow engine produces a truthful `GENERATION_TIMEOUT`
instead of a silent stall.

`max_tries` is also pinned to 2. Each attempt is a full inference run;
the ARQ default of 5 means a repeatedly-interrupted job can burn five
times the compute of the song it is trying to produce.

## 4. Finding: the worker does not survive Redis, and nothing restarts it

The worker process exits on Redis loss and there is no supervisor —
`launchd`, `brew services` and a container runtime are all absent here,
and Redis itself is a bare orphaned `redis-server` (ppid 1). Recovery is
manual.

This is not a code defect and is not fixed by code. It is documented in
`OPERATIONS_RECOVERY.md`, and `luber_health.py` reports worker absence —
and worker *duplication*, which is the other half of the same gap.

## 5. Finding: a retry could re-run a finished generation

ARQ's retry is keyed on the job, not on what the job already achieved.
The window between `mark_completed` and ARQ recording success is small,
but a cancellation inside it would re-run a generation that had already
delivered audio — handing the user a different song in place of the one
they had.

**Fixed:** `execute` returns immediately when the row is already
`COMPLETED`.

## 6. Duplicate workers

ARQ claims each job under an `arq:in-progress:<id>` key set inside a
`WATCH`/`MULTI` transaction before the job starts, so two workers cannot
pick up the same job concurrently — the second sees the key and skips.
The claim's TTL is `job_timeout + 10s`, which after the change above sits
well beyond any real generation, so it cannot lapse mid-run and let a
second worker in.

Exactly-once *pickup* is therefore guaranteed by the queue. Exactly-once
*side effects* are guaranteed separately, by asset upsert-by-role and
deterministic storage keys — which is what makes the retry path safe.

Two workers on one queue is still worth flagging, because it doubles
inference capacity against a single-GPU machine. `luber_health.py`
reports it as a failure.

## 7. Idempotency

`Idempotency-Key` is required on create. A repeated key returns the
original generation rather than creating a second one, which covers a
double-click and a network retry alike. `result_count = 2` is a
different thing — two deliberate songs from one request, sharing a
`generation_group_id` — and is not affected.

## 8. State machine

```
QUEUED ──► STARTING ──► GENERATING ──► POST_PROCESSING ──► UPLOADING ──► COMPLETED
   │           │             │                │                │
   └───────────┴─────────────┴────────────────┴────────────────┴──► FAILED
```

- `QUEUED` is written by the **API**; every other transition is written
  by the **worker**, in `GenerationService.execute`. Nothing else writes
  status except the API's two compensating `mark_failed` calls (enqueue
  failure, and one edit path).
- Each transition is a single committed `UPDATE`. There is no
  multi-row status write, so no transition is partially applied.
- `CANCELLED` exists in the enum and is terminal, but no code path
  writes it: user-initiated cancellation is not a product feature yet.
- **Regressions** are possible in principle — `mark_started` will move
  any row to `STARTING` — but the only caller is now guarded by the
  COMPLETED check in §5.
- **Permanently stuck**: was possible via §3, is not now. A cancelled
  run records `FAILED`; a slow run hits the provider's timeout and
  records `GENERATION_TIMEOUT`.

## 9. Stale thresholds

Measured over the 38 completed generations in this deployment before
this phase:

| | queue wait | run (start → complete) |
|---|---|---|
| mean | 0.3s | 54.7s |
| max | 0.5s | 110.5s |

By requested duration, a 180-second song averaged 68.9s and peaked at
110.5s — inference time is dominated by step count, not by song length.

`luber_health.py` therefore flags a non-terminal generation older than
**15 minutes** (≈8× the worst observed run) and a still-`QUEUED` one
older than **5 minutes**, since queue wait is bounded by inference rather
than by the queue and a long wait means nothing is consuming. Both only
report. Nothing auto-fails a generation: the cost of a wrong "stuck" is
destroying a healthy run, and the cost of a late flag is an operator
looking twice.

## 10. Consistency, as of this audit

`luber_health.py` against live data:

```
database               40 generations
asset_consistency      83 assets · 0 missing objects · 0 duplicate roles ·
                       0 undeliverable · 0 orphan objects
lineage_consistency    0 self-parent · 0 missing parent ·
                       0 root with edit_kind · 0 in a cycle
reference_consistency  4 references · 2 in use · 0 used but missing
```

No legacy anomalies. Nothing was repaired to reach this state — the audit
is read-only and found nothing to repair.

## 11. Concurrency

`max_jobs = 1` per worker, so one worker runs one generation at a time
and additional submissions wait in Redis. With one worker there is no
simultaneous inference and no OOM path from concurrency; a second worker
would double GPU pressure on a single-GPU machine, which is why
duplication is reported as a fault rather than a capacity option.

Backpressure is explicit at both ends: Redis holds the backlog, and when
ACE-Step's own queue saturates it answers 429, now surfaced as
`PROVIDER_BUSY`. Nothing retries a busy provider automatically — bounded
retry on a saturated single-GPU engine would deepen the queue it is
waiting on.

---

*Findings verified by interrupting the running system. §3, §5 and the
429 classification are fixed in this phase; §4 is documented, not
fixable in code.*
