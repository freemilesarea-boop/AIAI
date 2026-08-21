# Generation Reliability Runbook

*What to do when generations start failing quality control, when the
retry rate climbs, or when somebody asks why a particular song came out
the way it did.*

For how the engine works, see
[INFERENCE_QUALITY_CONTROL.md](INFERENCE_QUALITY_CONTROL.md). This
document is the operational half.

---

## The one tool

Phase 29's operator surface is the CLI. It runs QC over audio that
already exists, generates nothing, mutates nothing, and is safe to point
at production data.

```bash
# What would QC say about this file?
uv run python -m luber_inference_qc analyze song.wav --duration 180 --bpm 120

# What does it say about a whole directory?
uv run python -m luber_inference_qc analyze /path/to/output --progress

# Why did this generation do what it did?
uv run python -m luber_inference_qc explain trace.json
uv run python -m luber_inference_qc explain trace.json --summary
```

`analyze` over a directory exits **1** when more than half the files are
rejected. That exit code is the signal to audit the thresholds, not the
songs.

### Getting a trace out of the database

The trace lives on `generations.inference_qc_trace` as JSON text.

```bash
psql "$DATABASE_URL" -At -c \
  "select inference_qc_trace from generations where id = '<uuid>'" \
  > /tmp/trace.json
uv run python -m luber_inference_qc explain /tmp/trace.json
```

The trace carries no prompt, no lyrics, no reference audio and no local
paths — only a request digest — so it can be moved to wherever the
investigation is happening without moving user content with it. Keep it
that way: paste traces, not prompts.

---

## Symptom: the retry rate has climbed

**What it means.** More first candidates are being rejected. Users are
still getting songs — the retry is invisible to them — but every retry
is a full generation somebody is paying for.

**Read this first.** Summarise a sample of recent traces:

```bash
uv run python -m luber_inference_qc explain /tmp/trace.json --summary
```

`critical_findings` in the summary names what actually failed. Group by
it before doing anything else, because the response is completely
different per code.

| Dominant finding | What changed | What to do |
| --- | --- | --- |
| `SILENT_OUTPUT`, `NEAR_SILENT` | The model is producing nothing | Check the checkpoint that is loaded; a mismatched or partially-copied checkpoint produces silence |
| `EARLY_COLLAPSE` | Generations stop before their requested length | Look at requested durations — a spike in long requests against a model trained on short ones does this |
| `DURATION_SHORT` / `DURATION_LONG` | The provider is not honouring the requested length | Check the provider's duration parameter mapping; this is a configuration failure far more often than a model failure |
| `SEVERE_CLIPPING` | Output is distorted in the samples | Check inference gain/normalisation settings, not the finishing engine — Phase 22 runs after this |
| `SPECTRAL_COLLAPSE` | Output is a single tone | This is a badly broken model. Roll back the checkpoint |
| `PROVIDER_TIMEOUT` | The provider is slow or unreachable | This is a capacity problem wearing a quality costume; see below |
| `CONTROL_BPM_MISMATCH` | Requested tempos are not being met | Recorded, never rejected on its own. Investigate at leisure |

**What not to do.** Do not widen a threshold to make the retry rate go
down. The retry rate is a symptom; the thresholds are calibrated against
the corpus and validated by
`packages/inference-qc/tests/test_thresholds.py`. Loosening one makes
the graph look better and ships the broken audio.

---

## Symptom: generations are failing with `QUALITY_RETRY_EXHAUSTED`

Every attempt the budget allowed was made and every one was rejected.

1. `explain` the trace. Read the per-attempt findings.
2. If **every attempt failed on the same code**, the model is
   reproducing a defect deterministically. The retry budget is not the
   problem and raising it will not help — it will spend more money
   confirming the same answer. Fix or roll back the model.
3. If the attempts failed on **different** codes, output is unstable.
   That is a genuine candidate for `EXPERIMENTAL_MULTI_CANDIDATE` while
   investigating, at three times the inference cost. Do not leave it on.

## Symptom: generations are failing with `QUALITY_CHECK_FAILED`

Nothing further would have helped: either the policy does not retry what
went wrong, or a non-retryable failure occurred, or the profile is
`STRICT_REPRODUCIBLE`.

The trace's `outcome_detail` says which, in the planner's own words.

## Symptom: failures with a provider error code, not a QC code

`MODEL_LOAD_FAILED`, `GENERATION_TIMEOUT`, `OUT_OF_MEMORY`,
`PROVIDER_BUSY` and friends mean the provider never produced audio. This
is **not** a quality problem and the QC engine is not where to look —
go to the provider, the GPU host, or Phase 27's remote execution.

The distinction is deliberate: reporting an unreachable provider as a
quality failure sends an operator to the wrong system.

---

## Turning it off

```bash
INFERENCE_QC_ENABLED=false
```

This restores the pre-Phase-29 path exactly: one provider call, no
measurement, no retry, no trace. It is a real bypass, not a quieter
version of the same loop.

**When this is right:** QC itself is broken, or is rejecting good output
across the board and users are getting nothing. Delivering imperfect
audio beats delivering none.

**When this is wrong:** the retry rate is high. That means QC is working
and the model is not.

Prefer narrowing over switching off:

```bash
INFERENCE_QC_POLICY=CONSERVATIVE   # retries only silence/collapse/transport
INFERENCE_QC_POLICY=STRICT_REPRODUCIBLE  # exactly one call, ever
```

`CONSERVATIVE` keeps the gate — bad audio is still refused — and only
narrows what justifies spending another inference. That is usually the
control you actually want during a cost incident.

---

## Cost control

The hard ceiling is `maximum_total_provider_calls` per profile:
`STANDARD` is 3, `CONSERVATIVE` is 2, `STRICT_REPRODUCIBLE` is 1.
Nothing generates past it whatever the findings.

Every trace records `budget.provider_calls_used` and each attempt
records its `attribution` — `USER_REQUEST` or `QUALITY_RETRY`. That
distinction is recorded at the only time it can be recorded accurately.
Phase 29 implements no billing; the data is there for when something
needs it.

To cut spend immediately, move to `CONSERVATIVE`. To cut it to the floor,
`STRICT_REPRODUCIBLE` — accepting that quality failures then become user-
visible failures rather than retries.

---

## Candidate workspace

Candidate audio accumulates under `candidate_workspace_dir`
(`data/generation-candidates` by default), one directory per generation.
Directories are removed on a terminal outcome — success or failure.

**A directory that is still there means a worker died mid-generation.**
That is by design: the queue's retry of that job will recover the
candidate rather than pay for it again, verifying the SHA-256 first.

If workers are crashing, directories accumulate. Before deleting
anything, check whether the generations are still in flight:

```bash
ls data/generation-candidates | head
psql "$DATABASE_URL" -At -c \
  "select id, status from generations where id in (...)"
```

Directories for generations in a terminal state (`COMPLETED`, `FAILED`)
are safe to remove. Directories for `QUEUED` or `GENERATING` rows are
paid-for audio a resumed job will reuse — deleting them costs a real
inference each.

---

## Verifying a threshold change

Never change a threshold without running the corpus against it:

```bash
uv run python -m luber_inference_qc analyze data/raw-model-output
uv run pytest packages/inference-qc -q
```

The corpus should stay at 97 eligible, 0 rejected. A change that starts
rejecting real corpus tracks is rejecting real songs, and the fixtures
in `packages/inference-qc/tests/qc_fixtures.py` are there to prove the
change still catches what it was meant to catch.

---

## Why there is no console panel

Phase 28's operator console reads the training registries directly from
the filesystem, and says so in its own words: *no database, no mirror, no
cache that can disagree with the files.* Phase 29's data lives somewhere
else entirely — on the `generations` table, one row per user request,
scoped by owner.

Putting a QC panel into that console would mean cutting a cross-owner
database query into a browser-reachable process that was deliberately
built without one. The useful version of that panel is aggregate
reliability metrics; the version that is easy to build by accident is a
browsable list of other people's generations.

The operator questions Phase 29 actually raises — *why did this retry,
which candidate won, what is failing this week* — are all answered by
`explain` and `analyze` today, without a new data path. Phase 30 is
named for observability and quality analytics, and is where an aggregate
surface belongs, designed as one rather than grown out of a debugging
panel.

---

*See also:* [OPERATIONS_RECOVERY.md](OPERATIONS_RECOVERY.md) for worker
and queue recovery, [GENERATION_PIPELINE.md](GENERATION_PIPELINE.md) for
the lifecycle, and [MODEL_PROVIDER.md](MODEL_PROVIDER.md) for provider
configuration.
