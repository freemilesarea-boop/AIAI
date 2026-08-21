# Inference Incident Runbook

*An incident fired. What to look at, what the evidence can and cannot
tell you, and what you are allowed to do about it.*

For how detection works, see
[INFERENCE_OBSERVABILITY.md](INFERENCE_OBSERVABILITY.md). This document
is the half you read at 2am.

---

## Before anything else

**Three questions, in this order.**

1. **Is the data current?** The console shows a banner when the
   projection is behind, and `ingest-status` answers it directly. A
   dashboard fed by a stale projection renders perfectly and lies
   completely.
2. **Is the window big enough?** A finding exists only when the sample
   minimums were met, but a *chart* will happily draw three requests.
   Read the counts, not the percentages.
3. **Is it availability or quality?** Every incident carries a category.
   `AVAILABILITY` means the provider did not answer — go to the provider,
   the GPU host, or Phase 27's remote execution. `QUALITY` means it
   answered and what it produced was worse. Sending somebody to inspect
   audio when the machine is unreachable wastes the first hour.

```bash
uv run python -m luber_inference_observability incidents
uv run python -m luber_inference_observability incident show <id>
uv run python -m luber_inference_observability summary --window 1h
```

## What the evidence is, and what it is not

Every finding carries both windows, both sample counts, both numerator
and denominator, the deltas, and the threshold it crossed. That is
enough to disagree with it, which is the point.

What it never carries is a cause. A marker near a spike is a
correlation. A revision that looks worse than its peer was not
randomised against it. **Do not conclude, from this system alone, that a
deployment caused anything.** Use it to decide where to look.

There are no prompts, lyrics or titles anywhere in this system, so
"which prompts triggered it" is not a question it can answer. The
segment dimensions — duration bucket, task type, language, revision —
are what narrowing looks like here.

---

## Symptom: retry rate up, acceptance down

`QUALITY_RETRY_RATE_INCREASE` and `FIRST_CANDIDATE_ACCEPTANCE_DROP` are
the same news arriving twice: more first candidates are being rejected.
Users are still getting songs — the retry is invisible to them — but
every retry is a full generation somebody is paying for.

**Look at:** which finding is driving it.

```bash
uv run python -m luber_inference_observability summary --window 1h
```

The `findings.critical` map is the answer. Group by it before doing
anything, because the response is completely different per code.

| Dominant finding | What it usually means | Where to look |
| --- | --- | --- |
| `SILENT_OUTPUT`, `NEAR_SILENT` | The model is producing nothing | The loaded checkpoint. A mismatched or partially-copied checkpoint produces silence |
| `EARLY_COLLAPSE` | Generations stop before their requested length | Requested durations — check the duration segment before blaming the model |
| `DURATION_SHORT` / `DURATION_LONG` | The provider is not honouring the requested length | The provider's duration parameter mapping. Configuration far more often than model |
| `SEVERE_CLIPPING` | Output distorted in the samples | Inference gain/normalisation. Not the finishing engine — Phase 22 runs after QC |
| `SPECTRAL_COLLAPSE` | Output is one tone | A badly broken model. Roll the checkpoint back |
| `PROVIDER_TIMEOUT` | Capacity, not quality | See the outage section |

**Do not** widen a Phase 29 threshold to make the graph go down. The
retry rate is the symptom; the thresholds are corpus-calibrated and
guarded by tests. Loosening one makes the chart look better and ships
the broken audio.

---

## Symptom: early-collapse spike

**Check the duration segment first.** Collapse concentrates by length
more than by anything else, and an overall rate that moved a little
often hides one bucket that moved enormously.

```bash
uv run python -m luber_inference_observability segments \
  --window 1h --group-by provider,duration_bucket --metric early_collapse_rate
```

If it is confined to one bucket — say `181_240` — that is a model or
parameter problem at that length, not a general degradation. If it is
uniform across buckets, look at the checkpoint.

**Gather:** the segment ranking, three generation traces from the
affected bucket, and the provider revision they ran on.

```bash
uv run python -m luber_inference_observability incident show <id>
```

**Do not conclude** that long requests are "too hard". The same model
served them last week; what changed is the question.

---

## Symptom: provider timeout or failure spike

This is an **availability** incident. The model may be perfectly fine.

**Look at:** whether the provider is reachable at all, whether the GPU
host is up, whether a queue is backed up, and whether the failure code
is one that reproduces (`MODEL_LOAD_FAILED`, `OUT_OF_MEMORY`) or one
that might not (`GENERATION_TIMEOUT`, `PROVIDER_BUSY`).

Phase 29 already distinguishes these: a provider that never produced
audio keeps its own error code and is **not** reported as a quality
failure. If you are seeing `QUALITY_CHECK_FAILED` you are not looking at
an outage.

**Gather:** provider logs, the host's state, and the failure-code
histogram from `summary`.

**Safe operator options:** none of them are in this console. Restarting a
worker, failing over, or draining a queue are operations elsewhere.

---

## Symptom: latency regression

**Read which stage moved.** Two incidents can both be
`LATENCY_REGRESSION` and mean different things:

- `total_latency_seconds` rose but `provider_latency_seconds` did not →
  the slowdown is after generation: post-processing, encoding, upload,
  or storage.
- `provider_latency_seconds` rose → the model or the host.
- `qc_latency_seconds` rose → QC itself, which normally costs about 0.9%
  of the audio's duration. A rise here usually means longer audio, not
  slower code.

Compare at the same quantile. The finding says which one it used.

**Do not** compare a P95 against a remembered average.

---

## Symptom: a duration segment regressed but the system looks fine

This is the case segment detection exists for, and it is easy to
dismiss because every headline number is healthy.

```bash
uv run python -m luber_inference_observability segments --window 24h \
  --group-by provider,duration_bucket --minimum-samples 30
```

**Note what is *not* listed.** The ranking drops segments below the
minimum sample count and reports how many. A short list means most
segments are small, not that only these exist.

---

## Symptom: a new provider revision looks bad

A revision that shipped today has no history, so the rolling baseline
cannot judge it. It is instead compared against **its peers in the same
window** — same period, different code.

```bash
uv run python -m luber_inference_observability providers \
  --left acestep@v1 --right acestep@v2 --window 24h --markdown
```

**What this is:** two measurements over one period.

**What it is not:** a controlled experiment. Requests are not randomised
between revisions. If the new revision is serving a different traffic
mix — more long requests, more covers — it can look worse for reasons
that have nothing to do with its weights. Check the segment breakdown
before concluding.

**If the revision is genuinely bad:** rolling it back is a deployment
operation. This console will not do it and does not offer to.

---

## Symptom: everything spiked right after a deploy

```bash
uv run python -m luber_inference_observability deployment \
  --at 2026-08-21T12:00:00+00:00 --hours 24
```

This reports before and after. It says, in the response, that it is a
correlation.

**What else changed at that time?** A rollout, a traffic shift, an
upstream model swap, a bad batch of long requests and a cron job all
look identical from inside this data. The marker is a place to start
looking, not a verdict — and a rollback of an innocent change costs a
deploy and leaves the real cause running.

---

## Incident hygiene

**Acknowledge** when you have seen it and are working on it. It does not
suppress anything: evidence keeps accumulating and an acknowledged
incident that worsens escalates and says so.

```bash
uv run python -m luber_inference_observability incident acknowledge <id> --operator you
```

**Dismiss** when it is real but not worth acting on — a load test, a
known migration, a deliberate experiment. A reason is required, and it
is kept forever.

```bash
uv run python -m luber_inference_observability incident dismiss <id> \
  --operator you --reason "load test, ticket LUB-482"
```

**Do not dismiss to tidy the list.** The next person to see this
fingerprint reads your reason first, and "cleaning up" is the one that
wastes their hour.

**Resolution is automatic** and requires three consecutive clean
evaluations. If an incident will not resolve, the metric is still
crossing — or traffic has gone quiet, which is deliberately not counted
as recovery.

---

## When the numbers themselves look wrong

**Telemetry problems** appear in `summary` under `data_quality` and on
the console. They mean a row carried something impossible: negative
latency, more candidates than provider calls, a completed generation
with no candidates.

```bash
uv run python -m luber_inference_observability verify
```

A row with a problem is still stored and still counted where it can be —
going blind exactly when something upstream broke would be worse.

**A window straddling `460642e`** contains generations with no candidate
trace. Their retries are unknown, not zero, and they are excluded from
candidate-derived rates. The console says so; if a denominator looks
smaller than the request count, this is why.

**If nothing has been ingested:**

```bash
uv run python -m luber_inference_observability backfill
```

Running it twice changes no count.

---

## What this system will never do for you

It will not disable a provider, lower a duration cap, change a retry
budget, move a QC threshold, switch a model or restart a service. Every
one of those has a cost it cannot weigh, and a detector that acted on
the same evidence it is asking you to check would be making your
decision with less information than you have.

The recommendations on an incident — `INVESTIGATE_PROVIDER`,
`COMPARE_PROVIDER_REVISION`, `CHECK_RECENT_DEPLOYMENT`,
`INSPECT_DURATION_SEGMENT`, `CHECK_SAMPLE_GENERATIONS`,
`CONSIDER_TEMPORARY_OPERATOR_POLICY_CHANGE` — are things to look at.
None of them is performed.

If you do decide to change Phase 29's behaviour during an incident, the
controls are in
[GENERATION_RELIABILITY_RUNBOOK.md](GENERATION_RELIABILITY_RUNBOOK.md):
`INFERENCE_QC_POLICY=CONSERVATIVE` narrows what justifies a retry while
keeping the gate, and `INFERENCE_QC_ENABLED=false` is a full bypass that
will ship audio this system would have refused.

---

*See also:*
[INFERENCE_OBSERVABILITY.md](INFERENCE_OBSERVABILITY.md),
[GENERATION_RELIABILITY_RUNBOOK.md](GENERATION_RELIABILITY_RUNBOOK.md),
[OPERATIONS_RECOVERY.md](OPERATIONS_RECOVERY.md).
