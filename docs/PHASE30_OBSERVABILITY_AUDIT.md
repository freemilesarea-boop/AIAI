# Phase 30 audit — what can honestly be observed, and where it may be stored

Written before the code. Three questions decide this phase, and all three
are answerable by reading what Phase 29 actually persists rather than
what an observability system would like to exist.

1. What is the canonical source, and what does it really contain?
2. Where may analytics live, given that the source table holds prompts
   and lyrics?
3. Which of the brief's requested dimensions are measurable, and which
   have to be recorded as UNKNOWN rather than invented?

---

## 1. The source of truth

`generations` rows plus the `inference_qc_trace` column Phase 29 added in
migration 0015. Not logs. Logs are diagnostics that rotate, are sampled
under load, and are not transactional with the row they describe; an
analytics system whose counts depend on log retention cannot answer
"how many generations failed last Tuesday" after the logs age out.

### What the Phase 29 trace really contains

Confirmed by rendering one, not by reading the docstring:

```
qc_schema_version, qc_engine_version, candidate_selection_version,
retry_policy_version, generation_id, request_sha256, base_seed,
policy{name, initial_candidate_count, maximum_candidate_count,
       maximum_retry_rounds, maximum_total_provider_calls,
       maximum_elapsed_seconds, retry_findings[], repeated_failure_limit,
       candidate_selection_enabled, allow_seed_variation},
attempts[{candidate_id, generation_id, attempt_index, request_sha256,
          attribution, seed, provider_request_sha256, status,
          selection_status, raw_sha256, duration_seconds, sample_rate,
          channels, findings[{code, severity, detail, metric, measured,
          threshold, not_measurable, evidence}], score_components,
          technical_selection_score, retry_reason, parent_candidate_id,
          not_selected_reason, provider_error_code, provider_seconds,
          qc_seconds, created_at, schema_version}],
selected_candidate_id, selection{winner_candidate_id, ranking, reasons},
outcome, outcome_detail, finishing_outcome, exhausted,
timings{candidate_phase_seconds, provider_seconds, qc_seconds},
budget{provider_calls_used, candidates_generated, retry_rounds,
       elapsed_seconds, maximum_*}
```

Everything Phase 30 counts comes from this and from the row's own
columns. Nothing is inferred from free text.

### What the row contributes

`status`, `error_code`, `provider`, `model_name`, `model_version`,
`duration_requested`, `language`, `instrumental`, `bpm`, `key_scale`,
`edit_kind`, `reference_audio_id`, `created_at`, `started_at`,
`completed_at`.

## 2. The privacy problem, and why it decides the storage

`generations` holds `prompt`, `lyrics` and `title` as ordinary columns,
and `request_trace` — which looks like a diagnostic blob — contains
`original_prompt` and `original_lyrics` in full:

```
packages/generation-client/src/luber_generation_client/ace_step/provider.py:168
    "original_prompt": compiled.original_prompt,
    "original_lyrics": request.lyrics,
```

So an analytics layer that queried `generations` directly would be one
`SELECT *` away from putting lyrics into a dashboard, a report or a JSON
export. Discipline is not a control here; a boundary is.

**Decision: a projection table.** `inference_observations` holds one row
per generation, with only privacy-safe dimensions and counters. The
analytics engine reads *that* table and has no access path to the
prompt at all. The guarantee becomes structural — there is no column an
analytics query could select a prompt from — which is the same argument
Phase 28 makes about its read model having no field a credential could
occupy.

Three further consequences:

- The observation stores `generation_id` (pseudonymous UUID, already the
  operator's drilldown handle) and `request_sha256`. It stores no
  `user_id`: no metric in this phase is per-user, and a column nobody
  needs is a column that leaks.
- `request_sha256` is drilldown identity, never a grouping dimension
  (Step 74). Grouping by it would produce one bucket per request.
- Aggregation over a projection is indexed column scanning rather than
  parsing a JSON blob per row per query.

## 3. What is measurable, and what is not

| Brief asks for | Available? | How |
|---|---|---|
| provider | yes | `generations.provider` |
| provider/model revision | yes | `model_name` + `model_version`; UNKNOWN when null |
| task type | yes | `edit_kind` + `reference_audio_id` → TEXT_TO_MUSIC / EXTEND / REPLACE_RANGE / COVER / REFERENCE_CONDITIONED |
| requested duration | yes | `duration_requested` |
| language | yes, explicit only | `generations.language`; never inferred from prompt text |
| instrumental | yes | `generations.instrumental` |
| bpm / key requested | yes | `bpm`, `key_scale` — presence only, not value, as a dimension |
| qc policy, engine, retry policy versions | yes | trace version block |
| candidate / provider-call / retry counts | yes | trace `budget` |
| first-candidate accepted | yes | `selected_candidate_id` maps to `attempt_index == 0` |
| retry exhausted | yes | trace `exhausted` |
| findings | yes | attempts[].findings[], split by severity |
| provider latency | yes | `timings.provider_seconds` |
| QC latency | yes | `timings.qc_seconds` |
| total latency | yes | `completed_at − started_at` |
| **finishing latency** | **no** | see below |
| **LUBER git revision** | **partially** | see below |

### Finishing latency is not measured anywhere

Phase 22's `FinishingRecord` carries `outcome`, `finishing_version`,
`source_sha256`, `plan`, `verdict`, `error` — and no timing.
`produce_delivery_assets` does not measure its stages either.

Two options were considered and one rejected:

*Rejected:* add a stopwatch to the finishing record. It would change the
shape of a Phase 22 artefact for a Phase 30 convenience, and Step 106 is
explicit that finishing behaviour and its records stay as they are.

*Taken:* record `delivery_latency_seconds` — the measured span between
the end of the candidate phase and completion, which covers
post-processing, finishing, encoding and upload together — and name it
for exactly that. `finishing_latency_seconds` is not stored, because
storing a number that claimed to be finishing time when it is four
stages would be worse than not having it.

This is a stated limitation, not a silent one.

### The LUBER revision is only honestly knowable at ingest

Nothing records the repository revision on a generation row. A backfill
running today cannot know which commit produced a generation last week,
and writing today's revision onto last week's row would be fabrication.

So: `luber_revision` is written **only by incremental ingestion**, from
configuration, where the ingesting process is the process that produced
the generation. Backfill writes UNKNOWN. Deployment correlation is
served primarily by explicit **markers** (Step 36) carrying a commit and
a timestamp, which is the honest way to put a deployment on a timeline
without claiming every row knows its own build.

## 4. Where the dashboard goes, and the one Phase 28 change it needs

Phase 29 deferred this panel, arguing that a QC panel meant cutting a
cross-owner database query into the training console, which is
deliberately filesystem-only. The projection table answers that: the
analytics API reads `inference_observations`, which is not owner-scoped
because it contains nothing an owner owns.

**Route:** `/ops/inference`, API `/v1/ops/inference/…`. Not
`/ops/training/inference`: inference observability is not training, and
filing it under training would be a lie about the system that a later
phase would have to unpick.

That requires one change to Phase 28. The web proxy currently hard-codes
its upstream:

```
apps/web/src/app/ops/api/[...path]/route.ts
    const target = new URL(`/v1/ops/training/${path.join("/")}`, apiTarget());
```

It becomes a namespace allowlist — `training` and `inference` — so the
operator token still lives in exactly one place and a third namespace
cannot be reached by guessing a path. The training client's paths gain a
`/training` prefix. Behaviour is identical; the gateway stops claiming
to be a training gateway.

**Security is reused, not rebuilt.** `console_available`,
`require_operator` and `enforce_operator_origin` apply to the new router
exactly as they do to the training one, mounted under the same
`create_app` condition. No second auth mechanism, no role, no product
access.

## 5. What the detector must not become

The brief rules out black-box anomaly models, and the reason is worth
stating positively: an operator has to be able to disagree with a
finding. Every regression here is a comparison of two counted rates or
two measured quantiles against a written-down threshold, and the finding
carries both counts, both windows, both thresholds and the deltas. If a
detector cannot be argued with, it will be ignored — and an ignored
detector is worse than none, because it looks like coverage.

Three failure modes the tests exist to prevent:

- **Screaming at small samples.** 1 failure in 2 requests is not a
  regression; it is two requests. Every policy carries a minimum
  current and baseline sample count, and below it the answer is
  INSUFFICIENT_DATA — which is not NORMAL.
- **Screaming at tiny absolute deltas.** 0.1% → 0.2% is a 100% relative
  increase and operationally nothing. Policies require an absolute delta
  as well as a relative one.
- **Flapping.** A metric oscillating around a threshold must not open
  and resolve an incident every run. Incidents deduplicate on a
  fingerprint, and recovery requires a sustained window rather than one
  good request.

## 6. What Phase 30 will not do

Detection only. No provider is disabled, no threshold is changed, no
policy is switched, no service is restarted. The output of a CRITICAL
incident is a recommended operator action and the evidence for it.

Phase 29's behaviour is untouched: no change to candidate selection,
retry policy, QC thresholds or finishing. The one Phase 29-adjacent
write is the ingestion hook, which reads a finished generation and
writes a projection row.

---

*Boundary:* Phase 29 shipped at `460642e`. Generations before it have no
`inference_qc_trace`, and their absence of retries is unknown rather
than zero. The dashboard says so rather than averaging over it.
