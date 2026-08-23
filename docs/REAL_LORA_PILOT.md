# The bounded real LoRA pilot

**Phase 35.** A training run small enough to be safe and long enough to
show a signal — and the refusals that keep it that way.

---

## 1. The question

Phase 33 proved the trainer starts. Phase 34 measured what it costs.
Neither touched:

> Does real data, through the real stack, produce a coherent training
> signal at all?

A pilot answers that with tens of optimizer steps on rights-cleared
music. It does **not** try to produce a good model, and there is no
vocabulary anywhere in this phase for claiming that it did.

---

## 2. Rights come first, and there is no way round them

A pilot trains on one of exactly two things:

1. **Real material whose every Phase 25 gate passed** — dataset lock,
   curation lock, rights, evaluation leakage, self-generated. There is
   no `--force`, no override parameter and no pilot-specific bypass.
2. **A synthetic fixture**, only when the caller asks for one by name
   with `--allow-synthetic-fixture`.

A synthetic run is stamped `SYNTHETIC_FIXTURE` on its result, in the
API, and in the console. **It validates the mechanism and can never be
evidence about real music.** The two are never rendered alike, and the
runner refuses a fixture that was not explicitly requested.

The current position is recorded in
`docs/PHASE35_REAL_DATA_AUDIT.md`: **zero tracks are eligible for
training today.** Nothing in this phase changes that, and nothing in
this phase may be used to work round it.

---

## 3. The dataset contract

| Property | Requirement |
|---|---|
| Rights | every Phase 25 gate passed |
| Size | at least `PILOT_MIN_TRACKS` = **3** |
| Form | preprocessed `.pt` tensors, from the real ACE-Step preprocessing path |
| Location | beneath the trainer's working directory (ACE-Step refuses `--dataset-dir` anywhere else, *after* loading the model) |
| Immutability | a content digest over every tensor, cited by the pilot identity |

Three is a floor on **evidence quality**, not a claim that three tracks
teach a model anything: below it, a loss series describes which single
recording was sampled rather than training.

Once a pilot identity cites a dataset digest, the data is frozen.
Different bytes are a different pilot, not an edited one.

---

## 4. The step budget — computed, never hoped for

The installed trainer has **no `--max-steps`**. Length is epochs, and
the optimizer-step count follows from the data. From
`trainer_fixed._train_fabric` at the pinned commit:

```python
steps_per_epoch = max(1, ceil(len(train_loader) / gradient_accumulation))
total_steps = steps_per_epoch * max_epochs
```

and `PreprocessedDataModule` builds its loader with `drop_last=False`,
so `len(train_loader) == ceil(samples / micro_batch)`. Therefore:

```
expected_steps = epochs × max(1, ceil(ceil(samples / micro_batch) / accumulation))
```

The `max(1, …)` matters. A dataset smaller than one accumulation window
still takes a step per epoch, because the loop flushes whatever it
accumulated when the epoch ends. A budget that divided and floored would
under-count exactly the small datasets a pilot uses.

Verified against a real observation: Phase 34's profile ran 2 samples at
micro batch 1, accumulation 4, one epoch — and the trainer reported
exactly **1** optimizer step.

`PilotStepBudget.validate()` refuses a plan over the ceiling **before a
process exists**. `PilotStepBudget.for_ceiling()` derives the epoch count
from the ceiling rather than accepting one, so a caller cannot pick a
number that happens to overshoot. A dataset too large for even one epoch
raises: the answer is a smaller dataset, not a larger bound.

---

## 5. The ceilings

| Bound | Value | Constant |
|---|---|---|
| Optimizer steps, whole pilot | **48** | `PILOT_MAX_OPTIMIZER_STEPS` |
| Optimizer steps, one segment | 24 (half the total, exactly) | `PILOT_MAX_SEGMENT_STEPS` |
| Wall clock, one segment | 3600 s | `PILOT_MAX_WALL_CLOCK_SECONDS` |
| Wall-clock ceiling | 5400 s | `PILOT_ABSOLUTE_WALL_CLOCK_SECONDS` |

**Forty-eight** is chosen from what the pilot has to show, not from what
the hardware could stand: enough points that "it did not diverge" is
more than an anecdote, and enough room for a resume with steps on both
sides of it.

The segment ceiling is derived — `PILOT_MAX_OPTIMIZER_STEPS // 2` when a
resume is coming — so the two bounds cannot drift into disagreeing.

**No CLI flag raises any of them.** A caller asking for more gets
`PilotBudgetError`. There is also a last-resort guard inside the trainer
process: if the computed budget and the trainer's actual behaviour ever
disagree, the probe raises `PilotStepCeilingExceeded` and the run stops
rather than continuing quietly.

---

## 6. What must be true before anything starts

In order, and all of them:

1. **Rights** — every Phase 25 gate passed on this material
2. **Dataset kind** — real, or an explicitly requested fixture
3. **Step budget** — computed and inside the ceiling
4. **Capacity** — an *applicable* Phase 34 profile qualifying **this**
   configuration. A qualification for a different rank, batch,
   precision or sequence length is not evidence about this one
5. **Preflight** — the Phase 33 gate is `READY`

Anything else is `BLOCKED`, with a machine-readable reason.

---

## 7. The pilot identity

Deterministic, and covering everything that would make this a different
experiment:

```
plan_digest              dataset_manifest_digest    dataset_id
base_model_id            base_model_upstream_commit ace_step_commit
device  precision  optimizer  lora_rank  lora_alpha
micro_batch_size  gradient_accumulation  epochs  expected_steps
latent_length  encoder_length  seed
```

No timestamp, no hostname, no pid, no free-memory reading. Two runs of
the same pilot produce the same identity.

The **seed** is recorded and reproducible within a machine. It is not a
claim of cross-device reproducibility: a different accelerator will not
reproduce these numbers bit for bit.

---

## 8. What is measured, and from where

From **inside** the trainer process, by pass-through wrappers on the
trainer's own callables — no ACE-Step source is modified:

| Evidence | Seam |
|---|---|
| Loss, learning rate, epoch, per step | the trainer's own yielded step update |
| Gradient norm | the value `lightning.fabric.Fabric.clip_gradients` returns |
| Trainable-parameter fingerprint | `inject_lora_into_dit` (before), end of `train` (after) |
| Base-model integrity | a file digest over the weight files, on the LUBER side |

The pilot forces `log_every_steps = 1`. The trainer only *yields* a step
update when `global_step % log_every == 0`, and the default of 10 would
give a 48-step pilot four points — a loss series with nine tenths of
itself missing, silently. This was found by running one.

### The parameter fingerprint, and a trap in it

The fingerprint is a per-tensor sum and L2 norm — enough to say which
tensors moved and by how much, small enough to travel in a result
record, and far too little to reconstruct a weight from.

Lightning Fabric **wraps** the module it sets up and its wrapper
prefixes every parameter name. A naive before/after comparison therefore
finds no name in common and concludes that nothing changed — reporting
`NO_UPDATE` for a perfectly healthy run. This actually happened on the
first mechanism run, and was caught by comparing two real checkpoints on
disk, which showed all 384 adapter tensors changing.

Two fixes, both kept:

* names are normalised by stripping known wrapper prefixes
  (`_forward_module.`, `_original_module.`, `module.`) before comparing;
* an **empty intersection is reported as unknown, never as zero**. A
  comparison that could not align its two sides has not established that
  nothing moved — it has established nothing.

---

## 9. Reading the loss

Descriptive only:

```
count  finite_count  finite_ratio  first  last  minimum  maximum  mean  median
slope (DERIVED, least squares over the finite points, ≥3 points required)
```

**A monotonic decrease is not required and not implied.** Over tens of
steps the series is noisy, and requiring a downward trend would fail
runs that are working and pass runs that are not. The slope is reported
because an *absurd* sign or magnitude is informative — not because a
negative one means anything good. It is labelled `DERIVED` everywhere.

---

## 10. Classification

```
VALID_SIGNAL           finite loss, finite non-zero gradients, adapter moved
NUMERICALLY_UNSTABLE   non-finite loss or gradients, or the base model changed
NO_UPDATE              finite loss and nothing moved
INSUFFICIENT_EVIDENCE  too few steps, or the evidence was never established
```

There is deliberately no `CONVERGED`, no `IMPROVED` and no `GOOD_MODEL`.
Tens of steps cannot support any of them, and a vocabulary that could
express them would eventually be used to.

Outcomes: `COMPLETED_VALID_SIGNAL`, `COMPLETED_INSUFFICIENT_SIGNAL`,
`BLOCKED`, `FAILED_NUMERIC`, `FAILED_RUNTIME`, `CANCELLED`, `TIMEOUT`.
Never a generic `PASS`.

Failures are a closed vocabulary: `NO_RIGHTS_CLEARED_DATA`,
`DATASET_INVALID`, `MANIFEST_DRIFT`, `STEP_BUDGET_EXCEEDED`,
`CAPACITY_NOT_QUALIFIED`, `PREFLIGHT_BLOCKED`, `PREPROCESSING_FAILED`,
`TRAINER_FAILED`, `LOSS_NONFINITE`, `GRADIENT_NONFINITE`,
`NO_PARAMETER_UPDATE`, `CHECKPOINT_FAILED`, `RESUME_FAILED`,
`STALE_CHECKPOINT`, `TIMEOUT`, `CANCELLED`.

---

## 11. Checkpoint and resume

A pilot runs as **two bounded segments** with a checkpoint between them,
so resume is exercised rather than assumed. One checkpoint is written at
the end of each segment — not one per epoch, which on a 24-epoch segment
would leave 23 adapters nobody will read.

The checkpoint is opened (Phase 33's checkpoint probe): adapter tensors
load, training state deserialises, the step counter is read. The resume
is only a claim if **the step counter advanced**; a segment that
restarted rather than continued is `RESUME_FAILED`.

Every pilot artifact carries `EXPERIMENTAL`, `NON_PRODUCTION`,
`NEVER_AUTO_PROMOTE`.

The base model is verified unchanged by file digest before and after. A
base model that moved is `NUMERICALLY_UNSTABLE` — a LoRA run must never
write to it.

---

## 12. Timeout and cancellation

Each segment has a wall clock. On expiry the **process group** is
killed, the outcome is `TIMEOUT`, and whatever loss points were
collected are preserved. A killed segment is never `COMPLETED`.

---

## 13. Artifacts and cleanup

Written into the pilot workspace beneath the trainer root:

```
pilot_loss.json      the whole result, machine-readable
pilot_loss.md        the operator's version
pilot-A.log, pilot-B.log
output/checkpoints/  one adapter per segment
```

plus a run-scoped `pilot.json` beside the run in the registry.

**None of it is committed.** These are operational artifacts. Remove a
workspace with `--cleanup` on the run, or delete
`<trainer-root>/.luber-canary/pilot-<run-id>` by hand; the registry
record survives either way, which is the point — the evidence outlives
the weights.

Reports carry **counts and digests, never track names, filenames, paths
or lyrics.** A pilot report is read and shared; the material it
describes is not.

---

## 14. Using it

```bash
# What a pilot would do. Starts nothing.
python -m luber_training --registry ./training-registry pilot prepare \
    --run-id run_… --device MPS \
    --trainer-root ~/ace-step-1.5 --python ~/ace-step-1.5/.venv/bin/python \
    --model-dir ~/ace-step-1.5/checkpoints \
    --dataset-dir ~/ace-step-1.5/.luber-pilot/dataset \
    --dataset-build … --curation-build … \
    --latent-length 6000 --encoder-length 256

# Run it: two bounded segments, a checkpoint, a resume.
python -m luber_training --registry ./training-registry pilot run \
    --run-id run_… --device MPS --preflight-status READY … --resume

# Read the record back, and re-check its classification.
python -m luber_training pilot status --run-id run_…
python -m luber_training pilot verify --run-id run_…
```

The console shows it at `/ops/training/runs/<id>` and over
`GET /v1/ops/training/runs/<id>/pilot`. That endpoint is a **read**:
starting a pilot trains on real music and belongs to the operator CLI on
the machine that holds the trainer. Putting a Start button in a
browser-reachable process would move a boundary Phase 28 set
deliberately, for a convenience nobody asked for.

---

## 15. CUDA

The same contract runs unchanged on a CUDA worker: the same gates, the
same budget arithmetic, the same probe, the same classification. The
capacity requirement is what changes — a CUDA pilot needs a **real**
NVIDIA memory profile, and a fixture cannot supply one.

**No NVIDIA hardware has ever been attached to this project. CUDA pilot
status: UNVERIFIED.**

---

## 15b. The first real-data pilot (Phase 35B)

Run against the **real ACE-Step trainer** on **real, operator-authorised
music** — four tracks, 12.4 minutes, all five Phase 25 gates passed. See
`docs/OPERATOR_AUTHORIZED_TRAINING_DATA.md` for what the authorisation
does and does not establish. Apple M4 Pro / 24 GB, MPS, bf16, LoRA rank
16 / alpha 32, micro batch 1, accumulation 4, seed 42, latents up to the
production maximum of 6000 frames:

| | |
|---|---|
| Outcome | `COMPLETED_VALID_SIGNAL` |
| Dataset kind | **`REAL_OPERATOR_AUTHORIZED`** (recorded at the time as `REAL_RIGHTS_CLEARED`; renamed in Phase 36) |
| Preflight / capacity | `READY` / `QUALIFIED` |
| Steps | 48 of 48 (24 per segment, ceiling 48) |
| Loss | first 1.372, last 1.253, min 1.150, max 1.591, **finite ratio 1.0** |
| Slope (DERIVED) | −0.00155 — a line through noise, not a convergence claim |
| Gradients | 24 of 24 finite, 24 non-zero, mean norm 0.475 |
| Trainable tensors changed | **384 of 384**, 11 010 048 trainable parameters |
| Base model | preserved (file digest identical before and after) |
| Resume | epoch 24 → 48, step counter advanced, exit 0 |
| Wall clock | 1465 s across both segments (~30 s/step) |

Two things the run established beyond the signal itself:

- **The 25 frames/s latent rate holds on real audio.** A 178.8 s track
  preprocessed to 4470 latent frames, 134.4 s to 3360, and a 258 s track
  to exactly 6000 — the 240 s production cap. Phase 34 derived that rate
  from the VAE's downsampling ratios; real music confirms it.
- **The conditioning length is longer than the profile default.** Real
  preprocessing produced 769 encoder frames, not the 256 the earlier
  profiles used, so the capacity qualifier correctly refused them and a
  fresh profile was measured at the real shape (9.43 GiB Apple unified,
  sampled peak, REPRESENTATIVE).

**Known gap, carried forward.** The checkpoint integrity check reports
`ok: false` on both segments with one problem: *no provenance record was
written beside this checkpoint*. The checkpoint itself is sound — it
reopened, holds 384 tensors and optimizer state, and the resume ran from
it — but the pilot runner, unlike the canary, writes no provenance
sidecar, so a pilot checkpoint cannot be tied back to its plan by
anything next to it on disk. This predates Phase 35B: the synthetic
mechanism run reported the same problem. It is a metadata gap, not an
integrity one, and it was left as found rather than changed underneath a
completed run.

---

## 15a. Mechanism validation — synthetic, and not real-data evidence

The pilot path has been run end to end against the **real ACE-Step
trainer** on a **synthetic fixture**, to prove the machinery works
before any real material is put through it. Apple M4 Pro / 24 GB, MPS,
bf16, LoRA rank 32, micro batch 1, accumulation 4, 4 synthetic samples
at 512 latent frames, seed 42:

| | |
|---|---|
| Outcome | `COMPLETED_VALID_SIGNAL` |
| Dataset kind | **`SYNTHETIC_FIXTURE`** |
| Steps | 48 of 48 (24 per segment, ceiling 48) |
| Loss | first 3.423, last 3.071, min 2.027, max 6.991, **finite ratio 1.0** |
| Slope (DERIVED) | −0.0147 |
| Gradients | 24 of 24 finite, 24 non-zero |
| Trainable tensors changed | **384 of 384**, max delta 3.79 |
| Base model | preserved (file digest unchanged) |
| Resume | step 24 → 48, advanced |
| Wall clock | ~150 s |

**This is not evidence about real music.** It is stamped
`SYNTHETIC_FIXTURE` in the record, the API and the console for exactly
that reason, and the loss values are the loss of a model fitting noise.
What it establishes is that the pilot's machinery — budget, gates,
probe, loss capture, gradient capture, parameter fingerprint, checkpoint,
resume, classification — works against the real trainer.

Three defects were found by running it, and all three are fixed:

1. **Nine tenths of the loss series was missing.** The trainer yields a
   step update only when `global_step % log_every == 0`, and the default
   of 10 gave a 48-step pilot four points. The pilot now forces
   `log_every_steps = 1`.
2. **A healthy run classified as `NO_UPDATE`.** Fabric's wrapper rewrites
   every parameter name, so the before/after fingerprints shared none and
   the comparison concluded nothing had moved. Caught by comparing two
   real checkpoints on disk, which showed all 384 adapter tensors
   changing. Names are now aligned by suffix, and — the part that
   matters — an unalignable comparison reports **unknown**, never zero.
3. **One checkpoint per epoch.** A 24-epoch segment wrote 24 adapters.
   Now one per segment, which is what the resume needs.

---

## 16. Limits

1. **A pilot is not a training result.** It says the path works. It says
   nothing about convergence, music quality, generalisation, or whether
   the adapter improves anything.
2. **Forty-eight steps is nothing.** Any loss movement over that span is
   noise unless proven otherwise, and this phase does not try.
3. **The seed reproduces within a machine, not across devices.**
4. **A synthetic pilot is a mechanism check.** It is marked as such and
   may never be cited as real-data evidence.
5. **One local pilot at a time.** Two concurrent MPS runs on shared
   unified memory would interfere with each other and with the capacity
   figures that qualified them.
6. **Evaluation is out of scope here.** No held-out evaluation set
   exists that is separated from a training set, because no training set
   exists. Generating audio from a pilot adapter would produce a
   comparison nobody could interpret, so it is not done.

---

## 17. Related documents

* `docs/PHASE35_REAL_DATA_AUDIT.md` — why no real pilot has run
* `docs/DATASET_POLICY.md` — what may and may not be trained on
* `docs/TRAINING_MEMORY_CAPACITY.md` — Phase 34: the capacity requirement
* `docs/TRAINING_PREFLIGHT_AND_CANARY.md` — Phase 33: the readiness gate
