# Phase 29 audit — where inference quality control goes, and what it may honestly say

Written before the code, because the two questions that decide this phase
are both answerable by reading what already exists: *where does a
candidate controller belong* and *what can this repository actually
measure*.

The second question turns out to matter more. The brief lists vocal
mismatch, BPM adherence, key adherence, structure and lyric completeness
as things to check. This repository has honest detectors for two of
those, an explicit refusal to build one of them, and nothing at all for
the other two — and Phase 23 already wrote down why.

---

## 1. The generation flow as it stands

```
POST /v1/generations              apps/api/src/luber_api/routes/generations.py
  → row created, request_trace written
  → enqueue(generation_id)        apps/api/src/luber_api/jobs.py
                                    ArqGenerationEnqueuer  (production)
                                    InlineGenerationRunner (tests/dev)
        ↓ Redis queue "generation", max_tries=2
  → arq worker                    services/generation-worker/.../worker.py
      generate(ctx, generation_id)
        ↓
  → GenerationService.execute()   packages/generation-client/.../service.py
      STARTING
      ├─ cover?  _to_cover_request → provider.create_from_audio()
      ├─ edit?   _to_edit_request  → provider.edit()
      └─ else    _to_provider_request → provider.generate()
      GENERATING
        ↓  GenerationResult(audio_path=<raw wav on local disk>, …)
      POST_PROCESSING
      inspect_wav(result.audio_path)          ← the only existing QC
      UPLOADING
      produce_delivery_assets()               packages/.../postprocess.py
        ├─ transcode      → MASTER          (raw, written once)
        ├─ finish_audio() → FINISHED_MASTER (Phase 22, only if it acted)
        └─ encode         → PREVIEW
      record_finishing_trace()
      mark_completed(duration_actual, provider, model, seed)
      COMPLETED
```

Everything the product knows about output quality today is the single
`inspect_wav` call: exists, non-empty, parseable WAV header, sample rate
> 0, channels > 0, frames > 0. A silent file passes it. A file that
collapses to silence after ninety seconds passes it. A 40-second answer
to a 240-second request passes it.

## 2. The insertion point

There is exactly one, and it is narrow:

**`GenerationService.execute`, between the provider call and
`produce_delivery_assets`.**

That line is where the three task branches (generate / edit / cover)
converge on a `GenerationResult` holding a local raw file, and it is
before anything has been transcoded, stored, finished or recorded.
Everything Phase 29 needs is in scope there — the request, the row, the
provider, the raw path — and nothing downstream has happened yet.

So the shape is: `execute()` stops calling the provider once and starts
calling a **`CandidateGenerationController`** once. The controller owns
the loop, the QC, the retries and the selection, and returns the same
`GenerationResult` type `execute()` already handles, plus a trace.

Three places that must **not** grow QC logic:

- **The API route.** It has never executed a provider and must not start
  knowing about candidates.
- **The worker.** It owns job-level concerns — the singleton lock, the
  timeout, `max_tries=2` for interrupted jobs. Candidate retry is a
  different question at a different layer.
- **The provider.** `AceStepProvider` describes and calls one engine.
  A provider that judged its own output would make the judgement
  unavailable to every other provider.

## 3. What already exists, and must be reused rather than rewritten

| Need | Existing code | Verdict |
|---|---|---|
| Decode / structural validity | `luber_audio_utils.inspect_wav` | reuse as the first hard gate |
| Full technical analysis | `luber_audio_finishing.analyze_audio` | reuse; it is the measurement layer |
| Peak, clipping, DC, silence ratio, crest | `analysis.LevelMetrics` | reuse directly |
| Loudness (R128) | `analysis.LoudnessMeasurement` | reuse |
| Spectral slope, bands, air/presence ratios | `analysis.FrequencyMetrics` | reuse |
| Correlation, width, low-band correlation | `analysis.StereoMetrics` | reuse |
| Harshness / sibilance proxies | `analysis.SibilanceMetrics` | reuse, soft only |
| Named risk conditions + thresholds | `risks.RiskFlag`, `evaluate_risks` | reuse for soft findings |
| BPM estimate + confidence | `luber_dataset.factory.musical.estimate_tempo` | reuse |
| Key estimate + confidence | `luber_dataset.factory.musical.estimate_key` | reuse |
| Finishing + its own acceptance gate | `finish_audio`, `acceptance.adjudicate` | unchanged; runs on the winner only |

`analyze_audio` decodes the file once and returns everything above, so
QC costs one decode per candidate — the same decode finishing would do
later. Cached by `(raw_sha256, analysis_version)` so the winner is not
analysed twice.

## 4. What this repository cannot honestly measure

This is the part that constrains the phase.

### Vocal / instrumental presence — NOT MEASURABLE

`packages/dataset/.../classification.py` is explicit, and it is right:

> The repository has no vocal/instrumental classifier and no labelled
> data to validate one against. A spectral heuristic could be written in
> an afternoon and would be wrong often enough to matter — and its
> errors would be invisible, because a wrongly-labelled instrumental
> looks exactly like a correctly-labelled one in a manifest.

Nothing has changed since. So Phase 29 **must not** emit
`CONTROL_VOCAL_MISMATCH` on its own evidence. Building the detector here
would be worse than in Phase 23, because the consequence is not a
mislabelled row — it is throwing away a user's song and spending another
inference to replace it.

What is available is `centre_dominance_db`: mid-versus-side energy in the
200–4000 Hz band, which is *consistent with* a lead vocal. Evidence, not
a verdict.

**Decision.** The check is implemented behind a `VocalPresenceDetector`
protocol with a confidence threshold, and the default implementation
returns UNKNOWN with the reason above. The centre-dominance figure is
recorded alongside as evidence. If a validated detector is ever
introduced it plugs in and the mismatch path — already written and
tested — starts firing. Until then the finding is
`CONTROL_VOCAL_UNKNOWN`, never a rejection.

### Lyric completeness — NOT MEASURABLE

No ASR anywhere in the repository. `LyricLineQA` exists precisely
because the Korean line-omission failure has no automatic detector and
is recorded per line by a human. Phase 29 adds nothing here.

### Structure adherence — NOT MEASURABLE

`MusicalAnalysis.structure_status` is `UNAVAILABLE` throughout the
dataset factory. There is no validated section detector.

### Musical quality, naturalness, catchiness, genre — NOT MEASURABLE

Out of scope by design, and the reason the selection score is named
`technical_selection_score` and never `music_quality_score`.

### BPM and key — MEASURABLE, WITH CONFIDENCE

`estimate_tempo` returns a confidence derived from how far the winning
autocorrelation lag stands above the local baseline; `estimate_key`
returns the margin between the best and runner-up profile. Phase 23
already discards estimates below `MIN_TEMPO_CONFIDENCE = 0.15` and
`MIN_KEY_CONFIDENCE = 0.10` rather than recording a guess. Phase 29
inherits that: below threshold the finding is `NOT_MEASURABLE`, and key
is advisory only in every profile.

### Early collapse — MUST BE BUILT

The brief says "reuse/improve the Phase 22 early-collapse detector".
There is no such detector: the only occurrence of the word "collapse" in
`luber_audio_finishing` is a comment about mono compatibility. Phase 22
measures a whole-file `silence_ratio` but nothing positional, so a track
that ends at 110 s of 240 s and pads the rest with digital silence is
indistinguishable from one with a quiet outro.

This is built here, from the same loaded audio: the last frame whose
short-window RMS exceeds a floor gives an *effective content end*, and
the gap between that and both the file duration and the requested
duration is the finding. It is a new detector and is documented as one.

## 5. Retry ownership

Audited so the two layers cannot both retry the same failure.

| Layer | What it retries | Bound |
|---|---|---|
| ARQ queue (`max_tries = 2`) | A whole **job** that was interrupted — worker killed, timeout, crash | one extra attempt |
| ACE-Step HTTP client | **Nothing.** There is no retry loop; transport errors surface as `GenerationProviderError` | — |
| Phase 29 candidate controller | One **request** whose output failed measurable QC, or whose provider call failed retryably | the policy budget |

There is no existing per-request transport retry, so Phase 29 adds the
only one. The rule that keeps them separate: the queue retries when the
*process* died; the controller retries when the *output* was bad or the
call failed in a way another call could fix. A configuration error
(`MODEL_LOAD_FAILED`, `REFERENCE_AUDIO_UNAVAILABLE`) is non-retryable in
both.

Note the interaction: an ARQ retry re-enters `execute()`, which is why
resume has to be idempotent — §7.

## 6. Task coverage

`execute()` has three branches and they are not equally safe to retry.

| Task | Adaptive retry | Why |
|---|---|---|
| Text-to-music (`generate`) | **Yes** | Seed variation produces a different song from the same request, which is exactly what a retry should do. |
| Reference-conditioned (`generate` with `reference_audio`) | **Yes** | Same, and the reference is carried unchanged on every attempt — a retry that dropped it would deliver a different song than was asked for. |
| Cover (`create_from_audio`) | **No, this phase** | Source-conditioned. Whether a second attempt is "the same cover again" depends on provider semantics not yet audited. |
| Extend / replace-range (`edit`) | **No, this phase** | An edit's value is what it *preserves*. A retry with a new seed may preserve differently, and the product has no semantics for that yet. |

Unsupported tasks still get **QC and the trace** — they simply run one
candidate and never retry. A generation whose single candidate fails
critically fails honestly rather than shipping.

## 7. Crash, resume and concurrency

`execute()` is already re-entrant: it returns early when the row is
COMPLETED, because an ARQ retry can arrive after the work finished. The
controller has to hold the same property one level down.

- **Candidate RAW is hashed.** Every candidate file gets a SHA-256 before
  QC. Resume verifies the hash before reusing a candidate; a corrupt file
  is regenerated rather than trusted.
- **The trace is written as it goes**, not at the end, so a crash between
  the provider returning and QC finishing leaves a record that the call
  was made. Resume reuses that candidate instead of paying for another.
- **Selection is persisted before finishing.** A crash after selection
  resumes with the same winner; it does not re-rank.
- **One controller per generation.** The worker already runs `max_jobs=1`
  behind a singleton lock, and `execute()` refuses a completed row. The
  controller adds a claim on the generation row itself so two executions
  cannot both drive candidates.

## 8. Storage of candidates

Rejected candidates must not become user-visible songs. The existing
`AssetType` vocabulary (`MASTER`, `FINISHED_MASTER`, `PREVIEW`, `STEM`)
is public and every one of those is something a listener can be served,
so a rejected candidate must not be given one.

**Decision.** Candidate RAW files stay in the worker's temporary
directory for the life of the run and are never uploaded unless
selected. What persists is the *trace*: candidate id, seed, digest,
findings, decision. That is enough to explain any delivery, costs
nothing, and cannot leak a broken file into a library.

The consequence is honest and worth stating: a rejected candidate's audio
is **not** retained for later listening. Retaining it would need a
retention policy, an asset role and a cleanup job, and the brief asks not
to store unlimited failed WAVs without one. The trace records the digest,
so a future phase that wants the bytes knows exactly what it is missing.

## 9. What gets built

```
packages/inference-qc/                       new
  versions.py      four independently-bumped version strings
  findings.py      finding codes, severities, retryability
  identity.py      canonical request digest, deterministic seed derivation
  policy.py        CandidatePolicy, profiles, budget
  candidate.py     CandidateGeneration entity and its two state machines
  analysis.py      one decode → QC measurements, cached by (sha256, version)
  collapse.py      the effective-content-end detector (new)
  detectors.py     VocalPresenceDetector protocol + the honest null default
  checks.py        every check, producing findings
  eligibility.py   hard rejection
  scoring.py       technical_selection_score, components visible
  selector.py      deterministic ranking and winner
  planner.py       AdaptiveRetryPlanner
  trace.py         the JSON trace schema
  cli.py           dry-run QC over a file or a directory

packages/generation-client/
  candidates.py    CandidateGenerationController — the one insertion point
  service.py       calls the controller instead of the provider directly

packages/database/
  Generation.inference_qc_trace   one nullable Text column, migration 0015
```

Naming stays honest throughout: `technical_selection_score`, never a
music quality score; `CONTROL_VOCAL_UNKNOWN`, never a mismatch this
repository cannot establish.
