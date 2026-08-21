# Inference Observability

*Phase 30. Is inference quality getting worse over time, what changed,
for which requests, and on which provider revision — answered from
Phase 29's traces, without ever touching a prompt.*

---

## The question this answers, and the one it does not

Phase 29 asks whether **this candidate** is technically valid. This asks
whether **the system** is trending worse. They are different questions
with different failure modes: Phase 29's mistake would be delivering a
broken song, and this one's would be crying wolf until nobody looks — or
staying silent through a real regression because the numbers were noisy.

It **detects**. It changes nothing. No provider is disabled, no
threshold moved, no policy switched, no service restarted. The output of
the worst incident this system can raise is a sentence, the evidence for
it, and a list of things a human might look at.

## Three rules everything is built on

**Counts travel with rates.** "2.86%" is not a fact — it could be 12 of
420 or 2 of 70, and those need different responses. Every rate carries
its numerator and denominator, and every renderer prints them.

**Nothing happened is not nothing failed.** An empty window is `NO_DATA`.
A window with samples but too few is `INSUFFICIENT_DATA`. Neither is
`NORMAL`, because "we cannot tell" and "it is fine" are different
answers and only one of them lets somebody stop looking.

**Detection is arguable.** Every finding is a comparison of counted
values against written thresholds, and carries both windows, both sample
sizes, both counts and the threshold it crossed. There is no learned
normal, no anomaly score, and no black-box model — a verdict nobody can
check is one nobody will act on.

## Where the data comes from

The canonical source is **`generations` rows plus the
`inference_qc_trace` column** Phase 29 added in migration 0015. Not
logs: logs rotate, are sampled under load, and are not transactional
with the row they describe, so an analytics system depending on them
cannot answer "how many failed last Tuesday" once they age out.

### The privacy boundary is structural

`generations` holds `prompt`, `lyrics` and `title` as ordinary columns,
and `request_trace` — which reads like a diagnostic blob — contains
`original_prompt` and `original_lyrics` in full. An analytics layer
querying that table directly would be one `SELECT *` from putting
lyrics on a dashboard.

So Phase 30 reads a **projection**: `inference_observations`, one row
per generation, holding counts, latencies, outcome codes and
low-cardinality dimensions. There is no column a prompt could occupy, no
response model with a field for one, and the ingester names every source
field it reads. Three independent reasons, so none of them has to be
remembered.

Also absent: `user_id`. No metric in this phase is per-user, and a
column nobody needs is a column that can leak. `generation_id` is kept
as a pseudonymous drilldown handle, and `request_sha256` is kept as
identity — never as a grouping dimension, because grouping by it
produces one bucket per request.

Neither analytics table has a foreign key to `generations`. Deleting a
song must not delete the record that it was generated: a row that
vanished with a user's deletion would silently change last month's
counts.

## What is measured

| Dimension | Source |
| --- | --- |
| provider | `generations.provider` |
| provider revision | `model_name@model_version`; UNKNOWN when null |
| task type | `edit_kind` + `reference_audio_id` — columns a client cannot forge |
| requested duration + bucket | `duration_requested` |
| language | `generations.language`, **explicit metadata only** — never inferred from prompt text |
| instrumental, BPM requested, key requested | explicit request columns |
| QC policy, engine, retry-policy, finishing versions | the Phase 29 trace |
| LUBER revision | written only by incremental ingestion; see below |

| Counter / metric | Source |
| --- | --- |
| requests, completed, failed, cancelled | `generations.status` |
| provider calls, candidates, retry rounds | trace `budget` |
| first-candidate accepted | `selected_candidate_id` → `attempt_index == 0` |
| retry exhausted | trace `exhausted` |
| critical and soft findings | `attempts[].findings[]`, split by severity |
| provider latency, QC latency | trace `timings` |
| total latency | `completed_at − started_at` |
| **delivery latency** | total − candidate phase; see below |

### Two honest gaps

**Finishing latency is not stored.** Phase 22's `FinishingRecord`
carries no timing and `produce_delivery_assets` does not measure its
stages. What is recorded is `delivery_latency_seconds` — the measured
span between the candidate phase ending and completion, which covers
post-processing, finishing, encoding and upload together — named for
exactly that. A field called "finishing" that held four stages would be
worse than no field. Adding a stopwatch to a Phase 22 artefact for a
Phase 30 convenience was rejected.

**The LUBER revision is only knowable at ingest.** Nothing records the
repository revision on a generation row, and a backfill running today
cannot know which commit produced a generation last week. So
`luber_revision` is written **only by incremental ingestion**, where the
ingesting process is the process that produced the generation; backfill
writes UNKNOWN. Deployment correlation is served primarily by explicit
markers.

## Windows

Half-open `[start, end)` intervals in UTC, so adjacent windows tile
without double-counting the instant they share. Named sizes: `5m`,
`1h`, `24h`, `7d`, plus arbitrary start/end for a question nobody
anticipated. Every aggregate, finding, incident and report carries the
window it came from.

## Rates and distributions

Rates: success, failure, first-candidate accept, quality retry, retry
exhaustion, provider failure, provider timeout, invalid audio, early
collapse, duration failure, severe clipping, silent output, spectral
collapse. Averages: provider calls and candidates per generation — the
cost signal.

Latency is reported as **P50 / P90 / P95 / P99 / max / count**, never as
a mean alone: the mean of a bimodal distribution describes neither mode,
and the request an operator cares about is at P95. Quantiles are
nearest-rank, so the value returned is a measurement that actually
happened rather than an interpolation nothing experienced.

### Denominators, and what is excluded from them

- **Cancellations** are excluded from every quality rate and from
  delivery success. A user changing their mind is not the model getting
  worse, and a spike after a UI change would otherwise read as one.
- **Generations without a candidate trace** (before `460642e`) are
  excluded from every candidate-derived rate. Their retries are
  *unknown*, not zero. The exclusion is reported in `coverage`, not
  silent — a denominator that quietly shrank is a number nobody can
  reconcile.
- **Soft findings never count as failures.** A harshness advisory and
  invalid audio live in separate maps, in the API, in the report and on
  the dashboard.

## Baselines

A baseline is a reference value for one metric in one segment, with the
window it was measured over and the sample count behind it.

**Rolling** — the previous 7 days, ending **1 hour before** the current
window. The gap matters: a baseline running right up to the current
window would absorb the first hours of a regression into "normal" and
then report that nothing changed.

**Frozen** — a fixed interval somebody named: a known-good week, the
revision before a rollout. Immutable, because recomputing a reference
point makes every historical comparison against it unreproducible.

**Cold start** — a revision without enough history is
`BASELINE_BUILDING`, not `NORMAL`. But it is not left invisible either:
a revision with no history is compared against **its peers in the same
window**, so a bad rollout is visible on the day it ships rather than a
week later. That is a weaker claim and is labelled as one — same period,
different code, but requests are not randomised between revisions.

## Regression detection

Four guards, in order, each because of a specific way naive detection
embarrasses itself:

1. **Sample size first**, before any arithmetic. Below the policy's
   minimum the answer is `INSUFFICIENT_DATA`.
2. **Absolute *and* relative delta**, both required. 0.1% → 0.2% is a
   100% relative increase and operationally nothing; a policy firing on
   relative change alone spends its credibility on rounding.
3. **Direction is per metric.** Acceptance falling and retries rising
   are the same news, and the direction lives with the metric so a new
   policy cannot get it backwards.
4. **Severity is earned.** CRITICAL means stop what you are doing.

Findings: `FIRST_CANDIDATE_ACCEPTANCE_DROP`,
`QUALITY_RETRY_RATE_INCREASE`, `RETRY_EXHAUSTION_INCREASE`,
`GENERATION_FAILURE_INCREASE`, `INVALID_AUDIO_INCREASE`,
`EARLY_COLLAPSE_INCREASE`, `DURATION_FAILURE_INCREASE`,
`SEVERE_CLIPPING_INCREASE`, `SILENT_OUTPUT_INCREASE`,
`SPECTRAL_COLLAPSE_INCREASE`, `PROVIDER_FAILURE_INCREASE`,
`PROVIDER_TIMEOUT_INCREASE`, `PROVIDER_CALL_INCREASE`,
`LATENCY_REGRESSION`.

Each is categorised **AVAILABILITY**, **QUALITY** or **EFFICIENCY**. A
provider timing out is not a model producing bad songs; keeping the
taxonomy explicit is what stops an operator being sent to inspect audio
when the machine is unreachable.

### What a finding says

> early_collapse_rate moved from 3/412 (0.73%) to 11/96 (11.46%) for
> duration_bucket=181_240.

What it does **not** say is why. "Early collapse rose after the deploy"
is a correlation an operator can act on; "the deploy caused early
collapse" is a claim this system has no evidence for.

## Incidents

One row per logical regression, updated rather than multiplied. A
detector running every five minutes would otherwise produce 288 alerts a
day for one problem.

**Fingerprint** — derived from finding type, category, **metric** and
segment. Never from the value, which moves, or the window, which always
does. The metric is included because one finding type can cover several:
total latency and provider latency both raise `LATENCY_REGRESSION` and
are different problems.

**Recovery** — three consecutive clean evaluations, not one. A metric
sitting near its threshold crosses and uncrosses with ordinary variance,
and an OPEN/RESOLVED cycle is worse than no signal. An evaluation that
*could not tell* — `NO_DATA`, `INSUFFICIENT_DATA` — is not evidence of
recovery: traffic going quiet is exactly when a regression is easiest to
miss.

**Escalation** is one-way within an occurrence. Severity rises with the
evidence and does not fall on a single better sample.

**Status** — `OPEN` → `ACKNOWLEDGED` → `RESOLVED` / `DISMISSED`.
Acknowledging records that a human has seen it and **does not suppress
measurement**: evidence keeps accumulating and an acknowledged incident
that worsens escalates. Dismissal **requires a reason** and deletes
nothing.

A resolved incident that recurs opens a *new* one rather than
reanimating the old, because the old record is the history of what
happened last time.

## Alerting

Phase 30 produces `Alert` records and does nothing with them: no Slack,
no email, no webhook. An integration is a decision about where an
operator's attention gets interrupted, and that is a phase of its own.
What is fixed now is the *shape*, so the eventual notifier does not also
have to invent one.

An incident alerts when it **opens**, when it **escalates**, and when it
**resolves** — not on the hundred evaluations in between.

## The operator console

`/ops/inference`, API at `/v1/ops/inference/…`. Its own section rather
than a tab inside the training console: training spends money on rented
hardware and this spends nothing, and putting them behind one nav would
put an operator investigating a retry spike one click from a button that
starts a GPU.

Security is **reused, not rebuilt** — the same `OPS_CONSOLE_ENABLED`
switch, the same operator token, the same production refusal, the same
same-origin proxy that holds the token server-side. Phase 30 changed the
proxy to allowlist namespaces (`training`, `inference`) so the credential
still lives in exactly one place; an unlisted namespace 404s.

Screens: health (cards, retry / failure / latency trends, QC findings,
worst segments, provider filter), incidents (list and detail with
evidence and operator actions), providers (revisions and an A/B
comparison), generations (list and safe per-generation trace).

Charts break at a gap rather than interpolating, and draw thin buckets
hollow. A line drawn through a quiet night shows a recovery that never
happened; a spike from three requests and one from three hundred must
not look identical.

Polling is 60s. Inference analytics does not need sub-second updates.

## Ingestion

**Idempotent by construction.** `generation_id` is the projection's
primary key, so ingesting the same generation twice replaces a row
rather than adding one. Nothing needs to check first, so nothing can get
the check wrong.

**Incremental** — resumes from a watermark (the newest observation), so
a scheduled run does not rescan the table. The generation worker also
records each finished generation as it completes, in a call that cannot
fail the job: losing an analytics row is a gap in a chart, while raising
there would turn a delivered song into a failed one.

**Backfill** — reads from the beginning. Running it twice changes no
count.

**Failures and cancellations are ingested too.** A system that observed
only successes would report perfect health during an outage.

**Late finalisation** — Phase 29 writes its trace as the run proceeds, so
a generation observed mid-flight and again at the end produces two
projections of one row. The second replaces the first; the counts do not
double because the key did not change.

## Data quality

Telemetry can be wrong, and an observability system that crashes on a
malformed row goes blind exactly when something upstream has broken. Each
check names an impossibility — negative latency, a completed generation
with no candidates, calls below candidates, an impossible retry count, a
missing schema version, a selected attempt out of range — and a row
carrying one is still stored, still counted where it can be, and flagged.

`observability verify` reports duplicates, forbidden fields, schema
mismatches, invalid counters, naive timestamps and duplicate incident
fingerprints. It returns a report rather than raising, because the
interesting question is usually "how many rows are like this".

## The CLI

```bash
uv run python -m luber_inference_observability ingest          # incremental
uv run python -m luber_inference_observability backfill        # from the beginning
uv run python -m luber_inference_observability summary --window 24h
uv run python -m luber_inference_observability regressions --window 1h
uv run python -m luber_inference_observability detect          # evaluates and updates incidents
uv run python -m luber_inference_observability incidents
uv run python -m luber_inference_observability incident show <id>
uv run python -m luber_inference_observability incident acknowledge <id> --operator alex
uv run python -m luber_inference_observability incident dismiss <id> --operator alex --reason "..."
uv run python -m luber_inference_observability report --window 7d --output ./inference_health_report
uv run python -m luber_inference_observability providers --left acestep@v1 --right acestep@v2
uv run python -m luber_inference_observability deployment --at 2026-08-21T12:00:00+00:00
uv run python -m luber_inference_observability segments --window 7d
uv run python -m luber_inference_observability verify
```

`regressions` and `detect` exit non-zero when something crossed, so a
scheduled run can be a check rather than only a thing to read.

## Configuration

| Setting | Default | Effect |
| --- | --- | --- |
| `observability_enabled` | `true` | `false` stops the worker recording observations; the scheduled `ingest` still works |
| `luber_revision` | `""` | Recorded on observations this process writes. Not defaulted to anything plausible — a guess in a column that reads like a fact is worse than UNKNOWN |
| `OPS_CONSOLE_ENABLED` / `OPS_OPERATOR_TOKEN` | Phase 28's | Gate the console; unchanged |

## Storage and scale

Two tables in the existing database. No ClickHouse, no Elasticsearch: at
this volume they would be operational burden bought with no query the
current design cannot serve.

Aggregation loads a window's rows and counts them in Python. That is a
real constraint — a window whose rows do not fit in memory cannot be
aggregated this way — and it is the right trade here: a week of traffic
is thousands of rows of scalars, and pushing every metric into SQL would
put the definition of "retry rate" in two places. The suite holds this
honest with a 100,000-observation test; if it stops being true the
answer is a pre-aggregation table, not a quiet regression.

Three indexes — time, provider+time, revision+time — matching the only
three shapes a query takes. Indexing every dimension would slow
ingestion for queries that scan the window's rows anyway.

**Retention** is defined and not automated. `delete_observations` exists
so a policy has something to call; nothing invokes it, because a
background job that quietly pruned last quarter would make a
year-on-year comparison impossible without anybody noticing.

## Limitations

- Finishing latency is not measured (see above).
- The LUBER revision is UNKNOWN for backfilled rows.
- Revision comparison is observational, not randomised; traffic mix can
  differ between revisions.
- Aggregation is in-memory per window.
- No notification integration.
- Grouping is capped at two dimensions. A wider split divides the
  samples until no bucket can support a finding.
- Musical quality is still not measured, and this phase does not change
  that. Every metric here is technical or operational.

## What later phases may add, and what is not automatic now

A future phase may add notification, provider failover or a circuit
breaker. **None of them is automatic in Phase 30.** This system detects
and explains; disabling a provider, changing a policy or moving a QC
threshold are operator decisions with costs it cannot weigh.

---

*See also:*
[INFERENCE_INCIDENT_RUNBOOK.md](INFERENCE_INCIDENT_RUNBOOK.md) for what
to do when one of these fires,
[INFERENCE_QUALITY_CONTROL.md](INFERENCE_QUALITY_CONTROL.md) for the
Phase 29 engine that produces the traces,
[PHASE30_OBSERVABILITY_AUDIT.md](PHASE30_OBSERVABILITY_AUDIT.md) for the
pre-implementation audit, and
[TRAINING_CONSOLE.md](TRAINING_CONSOLE.md) for the operator security
boundary this reuses.
