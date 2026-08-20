# Model evaluation and checkpoint qualification

How `luber_evaluation` decides whether a trained checkpoint is better
than what exists, and safe enough to move forward. It implements the
rules in [MODEL_EVALUATION_POLICY.md](MODEL_EVALUATION_POLICY.md); this
document describes the machinery.

The question the package answers is narrow and worth stating exactly:

> Given a checkpoint that finished training, is there evidence it is
> better than an explicit baseline, and is it safe enough to advance?

It does not answer "is this model good". Nothing here can.

---

## 1. What this package refuses to do

Listed first because the refusals are load-bearing, and every one of
them is the kind of thing a well-meaning change would remove.

**It never promotes anything to production.** The furthest it goes is
recording a promotion review that may approve a checkpoint *for
staging*. Serving a model to users is a deployment decision made
elsewhere, deliberately, so that a bug in a qualification gate cannot
change what a user hears.

**It never substitutes zero for a missing measurement.** A metric
carries a status — `MEASURED`, `NOT_MEASURABLE`, `NOT_APPLICABLE`,
`FAILED` — and only a `MEASURED` result may hold a number. The dataclass
raises if that is violated. Recording "no ASR exists" as 0.0 would
manufacture a regression that never happened, and the resulting rejection
would look exactly like a real one.

**It never invents a metric for something it cannot measure.** There is
no automatic score in this project for melody quality, hook strength,
emotion, commercial appeal, instrument realism, vocal naturalness,
musical coherence, or trot-style delivery. The catalogue names those
dimensions and marks them `HUMAN_REQUIRED` with the reason. `luber
evaluation metrics` prints the list, including what cannot be measured,
because the honest answer to "what does this evaluate?" includes the
gaps.

**It never lets a technical pass satisfy a musical claim.** If an
experiment's hypothesis is about a `HUMAN_REQUIRED` dimension, the
outcome is `HUMAN_REVIEW_REQUIRED` even when every automatic gate
passes. That is a real outcome, not a failure to decide.

**It never reads training loss as quality.** Loss is available to
ranking as a final tie-break after outcome, safety, target and the
improvement balance have all tied, and `rank()` raises if asked to order
checkpoints that have no evaluation decision at all.

---

## 2. The shape of an evaluation

```
Phase 25 candidate
        │
        ├─ resolve ──▶ baseline ModelRef + candidate ModelRef + lineage
        │
   run create ──▶ DRAFT      identity frozen: suite, policy, seeds, both sides
        │
   run start  ──▶ VALIDATING digests re-checked, backends built
        │        ──▶ RUNNING     generate + analyse, both sides, same specs
        │        ──▶ COMPLETED   metrics, aggregates, comparison, coverage
        │
   qualify    ──▶ decision   integrity → coverage → safety → regressions → hypothesis
        │
   human-package / human-record   (when a listening claim is at stake)
        │
   promote    ──▶ review     APPROVE_FOR_STAGING | REJECT | HOLD
```

Every step writes to an artifact directory and to the Phase 25 registry,
which this package extends rather than duplicating. Two registries
claiming to know what a checkpoint is would eventually disagree, and the
one people trusted would be whichever they happened to read.

### Identity is frozen at start

Baseline, candidate, suite, policy and seed set become immutable when a
run leaves `DRAFT`. A result citing a suite that has since changed cites
nothing, so `run start` recomputes both digests before generating
anything and `verify` recomputes them again afterwards. Re-running means
creating a new evaluation — there is no flag that reopens a completed
one.

---

## 3. Metrics

### Automatic

Measured by `luber_audio_finishing.analyze_audio` — the same analyser
the finishing engine decides from. Re-implementing loudness or stereo
measurement here would produce a second set of numbers that disagree
with Phase 22's in ways nobody could adjudicate.

Per case and seed: duration error, clipping ratio, silence ratio, peak,
true peak, integrated LUFS, crest factor, spectral centroid, high
frequency energy ratio, stereo width, phase correlation, sample rate,
channels.

Per run, because reliability is a property of the run and not of a case:
generation success / failure / timeout rate, invalid audio rate, silent
output rate, early collapse rate, wrong duration rate. One failure in
three attempts is a 33% failure rate; expressing it per case would let a
median swallow it.

Spectral metrics are recorded and **not** judged as better or worse. A
brighter model is not a better model, and rewarding centroid movement
would train the evaluation to prefer whatever happened to be brighter.
They are `INFORMATIONAL` in the catalogue, which means they appear in
reports and never in a verdict.

### Human-required

`vocal_naturalness`, `korean_pronunciation`, `trot_style_absence`,
`melody_quality`, `instrument_realism`, and the rest. Each carries the
reason no automatic measure exists. They are excluded from
`required_metrics()`, so their absence never depresses coverage, and
they can never be `MEASURED` by any backend — a synthetic profile that
supplies a value for one has that value discarded and the metric
recorded `NOT_MEASURABLE`.

### Noise floors

A metric only counts as moved if it moved further than the suite can
resolve: 5% relative for continuous metrics, 0.02 absolute for rates
(where relative change is misleading — 0.001 to 0.002 is +100% and
irrelevant). A metric may declare its own floor where neither default
fits. `clipping_sample_ratio` does: it is a share of samples within
audio rather than a count over cases, and its hard ceiling (0.01) sits
below the shared rate floor, so under the default no clipping
improvement could ever have registered.

---

## 4. Coverage: BLOCKED versus REJECTED

The distinction decides between "we do not know" and "we know it
failed", and it is drawn precisely because the two are constantly
confused.

- A case the model **tried and failed** is *covered*. The failure is the
  measurement — it is what `generation_failure_rate` counts, and the
  reliability gate is waiting for exactly that number.
- A case with **no recorded outcome at all** — a cancelled run, a crash
  — is *uncovered*.

So a model that fails most of the suite is `REJECTED` on evidence, and a
run that stopped halfway is `BLOCKED` for lack of it. Counting only
successes as covered would have inverted both.

---

## 5. The decision

`decide()` runs in a fixed order, and the order encodes the priorities:

1. **Integrity.** A changed benchmark, a mismatched digest, a missing
   artifact. Blocks before anything else is considered — every number
   downstream of altered inputs is meaningless.
2. **Coverage.** Below the policy floor, `BLOCKED`. A verdict on partial
   evidence is not a verdict.
3. **Hard safety gates.** Absolute ceilings, not comparisons: a
   candidate failing half its generations is unusable whatever the
   baseline did. A gate whose metric was never measured blocks — an
   unmeasured gate is not a passed gate.
4. **Regressions.** Beyond the tolerated severity, or in a
   `never_regress` metric at any severity, `REJECTED`.
5. **Hypothesis.** Only now. Human-required →
   `HUMAN_REVIEW_REQUIRED`; not measurable → `BLOCKED`; measured and not
   supported → `REJECTED`; supported → `QUALIFIED`.

Outcomes: `QUALIFIED`, `REJECTED`, `BLOCKED`, `HUMAN_REVIEW_REQUIRED`,
`PENDING`.

**`QUALIFIED` does not mean production.** It means the checkpoint may
advance to promotion review.

### Policies

`NEUTRAL_CONSERVATIVE` (default) enforces reliability, technical
validity and completeness, and sets no threshold on musical quality — a
policy demanding "melody quality above 0.8" would be demanding a number
nobody can produce. `STRICT` tightens coverage and tolerates no
regression.

Human evidence thresholds are deliberately unset. A preference share
picked before any human has listened would be arbitrary, and would then
be treated as a target.

---

## 6. Ranking several checkpoints

Lexicographic, never weighted: outcome, then worst regression severity,
then the experiment's target metric, then regression count, then
improvement count, then training loss, then id. A weighted score would
let a large target gain buy off a reliability regression, which is
exactly the trade this project must not make silently.

`pareto_front()` is offered where checkpoints genuinely trade off. It
declines to pick a winner rather than inventing one.

---

## 7. Human review

`human-package` builds a blinded A/B package: five questions, each
mapping onto a dimension no automatic metric can measure, plus a Korean
pronunciation question on cases with Korean lyrics. `NO_PREFERENCE` is a
real answer and is excluded from preference denominators — counting it
as half a vote would invent a preference.

The package contains no model identity, no checkpoint id and no metrics.
The mapping from A/B to baseline/candidate is written to a **separate
file**, so that handing someone the package is not the same act as
handing them the answer. Assignment is hash-derived, so it is stable
across rebuilds and is not systematically one-sided.

**Phase 20H currently holds zero human scores.** The benchmark identity
records `human_scores_recorded: 0` and `human_score_store: "absent"`,
and a test asserts it. No baseline human numbers are invented anywhere.

---

## 8. Backends

| Backend | Produces audio | Use |
|---|---|---|
| `synthetic` | **No** | Exercises the whole pipeline without a model. Every value stamped `SIMULATED`; samples carry `synthetic=True` and no audio digest. Has no code path that can write a WAV. |
| `rendered` | Yes | Ingests audio rendered elsewhere, by fixed filename. A missing render is a recorded failure, never a substitution. |
| `ace-step` | Yes | Drives a running ACE-Step server through the product's own provider. Cannot run in CI and does not pretend to — there is no fallback that produces audio when the server is absent. |

Both real backends declare the model they serve and refuse any case for
a different one. An ACE-Step server hosts one model at a time and
nothing in a request says which weights answered it, so a single
misconfigured URL would attribute baseline audio to the candidate and
produce a confident verdict about nothing. The refusal is recorded as a
failed generation rather than raised, so an operator sees it in the
results instead of a traceback halfway through a run.

---

## 9. Provenance

Every sample records evaluation, case, seed, model, checkpoint, mode,
generation-spec digest, and the SHA-256 of the audio on disk. `verify`
recomputes those digests. **No mystery WAVs**: audio a human is about to
judge can always be tied back to the exact model and case that produced
it, and a file swapped after the fact is detected rather than trusted.

Generated audio is not committed. `evaluations/` and `evaluation-runs/`
are gitignored alongside the training registry.

---

## 10. Commands

```
python -m luber_evaluation --registry ./training-registry <command>

suite list                    what can be evaluated, and the benchmark's identity
suite show                    a suite in full, with its digest
metrics                       the catalogue, including what cannot be measured
policies                      qualification policies and their digests

run create   --candidate-id   freeze identity against a Phase 25 candidate
run start    --evaluation-id  generate and measure both sides
run status   --evaluation-id  the record, its verdict and its audit trail
run list                      every evaluation

compare      --evaluation-id  recompute the comparison from recorded aggregates
qualify      --evaluation-id  decide, write the report and the model card

checkpoint rank --run-id      order evaluated checkpoints; list unevaluated ones

human-package --evaluation-id build a blinded listening package
human-record  --evaluation-id record listening responses

verify       --evaluation-id  recompute every claim the evaluation makes
promote      --evaluation-id  record an operator review (never production)
```

Operator-only and local. There is no HTTP surface and no role: an
ordinary LUBER account cannot start an evaluation, read a verdict or
reach evaluation audio, because no path to any of that exists outside
this program.

---

## 11. What is still missing

Recorded so that nobody has to rediscover it.

- **No human baseline exists.** Until P20 is scored, every
  `HUMAN_REVIEW_REQUIRED` outcome is a request for evidence nobody has
  yet gathered.
- **No lyric intelligibility measure.** There is no ASR in this project,
  so `lyric_line_coverage` and `lyric_word_coverage` are attached to
  cases that have lyrics and recorded `NOT_MEASURABLE` with the reason.
  The gap is visible in every report rather than absent from it.
- **No trot detector, no vocal-class detector.** Both would be
  fabrications.
- **Thresholds for human evidence are unset**, and should stay unset
  until there is a baseline to set them against.
