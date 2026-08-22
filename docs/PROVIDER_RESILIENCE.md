# Provider resilience

How the system behaves when the engine that makes the music stops
answering — and, just as much, what it refuses to do about it.

The short version: when a provider is failing, stop calling it, say so,
and refuse the requests it cannot serve. Never quietly serve something
else.

---

## 1. Five things that all look like "a retry"

Before this phase the system had four separate mechanisms that could
cause a second provider call, and no vocabulary that told them apart.
Left alone they multiply: two ARQ tries × three quality attempts ×
however many providers is eighteen inferences for one failed request.

| Layer | Owner | Question it answers | Bounded by |
|---|---|---|---|
| Transport retry | the provider client | did the HTTP call arrive? | none today — the ACE-Step client does not retry |
| Quality retry | Phase 29 `CandidateGenerationController` | was the audio good enough? | `maximum_total_provider_calls` |
| Circuit breaking | Phase 31 | should we be calling this provider at all? | not a retry; it *removes* attempts |
| Provider failover | Phase 31 | is there somewhere else this exact request can go? | `maximum_providers_per_generation` |
| User regeneration | the person | do I want a different song? | the person |

**Phase 31 adds no loop of its own.** Phase 29 still decides *whether*
there is another attempt; Phase 31 decides *where* that attempt goes.
This is the single most important property in the design, and there is a
test named after it
(`test_the_quality_retry_budget_is_still_the_only_attempt_budget`).

---

## 2. The circuit

One circuit per **provider and task type**. `ace_step:TEXT_TO_MUSIC` and
`ace_step:COVER` are separate, because a cover path broken by a model
that cannot repaint is not a reason to stop generating music from text.

### States

- **CLOSED** — traffic flows. (Closed is the healthy one: a closed
  circuit conducts.)
- **OPEN** — traffic is refused without calling the provider. The
  request fails immediately with `PROVIDER_BUSY` instead of waiting out
  a timeout that is already known to be coming.
- **HALF_OPEN** — the cooldown has elapsed. A small, leased number of
  requests are admitted as **probes** to find out whether the provider
  is back. Everything else is still refused.

### Opening

Two independent kinds of evidence, because each alone gets a common case
wrong:

- **Rolling failure rate** — more than `failure_rate_threshold` (0.5) of
  a `window` (5 minutes) once there are at least `minimum_samples` (10).
  Catches a provider failing most of the time under real traffic.
  Requires traffic.
- **Consecutive failures** — `consecutive_failure_threshold` (5) in a
  row. Catches a provider that is *completely* down on a quiet
  deployment, where a rate would never gather enough samples.

Rate limits count at `rate_limit_weight` (0.5). A provider declining
politely is less broken than one that cannot answer at all.

### Recovering

`open_duration` (30s), doubling per consecutive open up to
`maximum_open_duration` (10 minutes). A provider that comes back and
immediately falls over again is not asked again in another thirty
seconds.

Then HALF_OPEN: `probe_concurrency` (1) request at a time, holding a
**lease** (`probe_lease`, 5 minutes) rather than a promise to return the
slot — a worker killed mid-probe must not wedge the circuit forever.
`probe_successes_to_close` (2) consecutive successes close it. One good
answer from a provider that was down is not yet evidence it is up.

---

## 3. What counts as the provider's fault

The most damaging thing a circuit breaker can do is open on evidence
that was never about the provider. Ten users uploading unusable
reference tracks must not take the engine offline for everybody else.

| Category | Counts toward the circuit? |
|---|---|
| `PROVIDER_TIMEOUT`, `PROVIDER_UNAVAILABLE`, `PROVIDER_TRANSPORT` | yes |
| `PROVIDER_INTERNAL_ERROR` | yes |
| `PROVIDER_RATE_LIMIT` | yes, at half weight |
| `PROVIDER_AUTH_FAILED` | yes — and never retried; it reproduces |
| `REQUEST_REJECTED` (a bad request) | **no** |
| `QUALITY_REJECTED` (Phase 29 said the audio was bad) | **no** |
| `CANCELLED` (the user changed their mind) | **no** |
| `INTERNAL_ERROR` (our bug) | **no** |

The last three are the ones worth stating out loud. A provider that
answers promptly with audio we judge too quiet is *working*; counting
that as unavailability would open a circuit against a healthy engine and
turn a quality problem into an outage.

Classification happens where the evidence still exists. The ACE-Step
client collapses 429 and 401 into `MODEL_LOAD_FAILED`, so
`classify_exception` reads the status code off the exception before that
distinction is lost. The user-facing error code is unchanged.

---

## 4. Routing, failover, and what is never done

Routing picks a provider per attempt. Failover moves a request to a
different one — and is **DISABLED by default**.

**Failover is permitted only when the second provider can represent the
request unchanged.** Capabilities are read off the provider objects
themselves (`supports_reference_audio`, `supports_edit(kind)`,
`supports_audio_to_audio()`), never from a list somebody maintains by
hand.

Never, under any failure, silently:

- reduce the requested duration
- drop the reference track
- drop or alter the lyrics
- flip instrumental/vocal
- drop BPM or key
- change the task type

If the request cannot be served as it was made, it **fails**, and the
failure names the missing capability. A user who gets a song back is
entitled to assume it is the song they asked for.

**Budget.** `maximum_providers_per_generation` (2) distinct providers per
generation. Separate from — and multiplied by nothing against — Phase
29's attempt budget.

**Explicit choices are not overridden.** Nothing in the product lets a
caller name a provider today; the rule is in place for when it does.

---

## 5. Degraded mode

Degraded means **fewer things work, and we say which**. It never means
quietly producing something different.

`readiness()` derives, per capability, which providers could serve it and
what their circuits are doing:

- **AVAILABLE** — at least one provider closed.
- **DEGRADED** — the only candidates are probing their way back. Most
  requests will be refused.
- **UNAVAILABLE** — nothing can serve it now.
- **NOT_CONFIGURED** — nothing ever could here. Distinct on purpose:
  a deployment without a cover-capable model is not in an incident.

This is a *third* answer, separate from `/health` (is the process up?)
and `/ready` (are the dependencies reachable?). The API can be alive,
Postgres and Redis fine, and every provider circuit open. A load
balancer must not remove the API for that: the API is working; the thing
it calls is not.

---

## 6. Many workers, one opinion

`max_jobs = 1` per generation worker, so several worker processes is the
normal way to scale. Circuit state is therefore **in the database**, not
in a process.

Every transition is a compare-and-set on a `revision` column — a
conditional `UPDATE … WHERE circuit_key = :k AND revision = :expected`,
with `rowcount` as the answer. The read-modify-write version of this is
not a compare-and-set at all: under a 16-thread burst it produced
several "the circuit opened" transitions for one opening. There is a
test that runs exactly that burst.

The loser of a race re-reads and reconsiders, which is usually the right
answer rather than a conflict to resolve: the circuit it was about to
open is already open. Recording evidence therefore never raises into a
generation; **operator actions do**, because a manual open that silently
did not happen is the worst outcome available here.

---

## 7. Operator surface

**Read** — the console at `/ops/inference/circuits` shows circuit state,
the policy those numbers are measured against, capability readiness, and
the full transition log.

**Write** — the CLI, and only the CLI:

```
python -m luber_provider_resilience status
python -m luber_provider_resilience open  ace_step --task TEXT_TO_MUSIC --operator you --reason "..."
python -m luber_provider_resilience close ace_step --task TEXT_TO_MUSIC --operator you --reason "..."
python -m luber_provider_resilience reset ace_step --task TEXT_TO_MUSIC --operator you
```

The console is a **non-production deployment switch** — refused outright
when the environment is production. An incident that needs a circuit
forced open is, by definition, happening in production. A button there
would be an incident tool that is absent during incidents. The CLI runs
wherever the database is reachable, and its mutations are audited into
the same transition table the console reads.

See `docs/PROVIDER_INCIDENT_RUNBOOK.md`.

---

## 8. Metrics and traces

Counters: `requests_rejected_circuit_open`, `provider_failover_total`,
`probe_admitted_total`, `probe_refused_total`, `circuit_opened_total`,
`circuit_closed_total`.

Every generation that routed carries a `resilience` block in its QC
trace: the routing decisions, the attempts with their categories and
circuit transitions, the providers touched, and a plain-language
narrative. It is additive — a generation that did not route stores
`null`, and Phase 29's trace is otherwise untouched.

**Privacy.** No prompt, no lyrics, no title, no user id, and no
credential appears in any resilience record. This is structural: circuit
rows and console responses have no field one could occupy. Failure
*categories* are stored; provider error *messages* are not, because that
is where an API key would be echoed back.

---

## 9. Relationships and limits

**Phase 29 (QC / adaptive retry).** Owns the attempt budget. Phase 31
never adds an attempt. A quality rejection never counts as provider
evidence.

**Phase 30 (observability).** Deliberately *not* wired to the circuit.
Phase 30's evidence is slow and quality-oriented — trend windows,
baselines, regression detection. A circuit is a fast availability
device. Letting a regression detector open circuits would make an
audio-quality finding stop traffic minutes later, for reasons nobody
watching the dashboard would connect. The two are read side by side by a
person.

**Phase 27 (remote execution).** Covers **training and checkpoint**
workflows. Generation still runs on the local ARQ generation-worker
path. There is no remote generation, and nothing here should be read as
implying otherwise.

### Known limits

1. **One production provider.** With only ACE-Step configured, failover
   has nowhere to go. The circuit breaker is still worth having — fast,
   honest refusal instead of a queue of timeouts — but production
   failover does not exist until a second equivalent provider does. No
   fake provider was invented to demonstrate it; `mock` returns a
   committed fixture and is explicitly refused as a routing target,
   because an outage that silently delivered the same two seconds of
   audio to everybody would be worse than the outage.
2. **Thresholds are code, not configuration.** `CircuitPolicy` defaults
   are what the worker uses. Tuning them is a deploy.
3. **No cross-region or per-tenant circuits.** One circuit per provider
   and task, globally.
4. **Probe admission is per circuit, not per worker.** A worker that
   dies holding a probe lease costs the deployment one lease duration
   before another probe is admitted.
