# The first controlled real-data experiment

Phase 35B asked whether the training path works. This asks the next
question and only the next one: does a larger but still bounded run on
real music move the model in a way that shows up on data it never
trained on.

It is not production training, and 24 tracks is not a dataset. What it
is, is the first run in this project with a held-out set that no
optimizer step ever touched.

## The budget, chosen before anything ran

| | |
|---|---|
| Train tracks | 24 |
| Micro batch × accumulation | 1 × 4 |
| Steps per epoch | ceil(24 / 1) / 4 = **6** |
| Epochs per segment | 10 |
| Steps per segment | **60** |
| Segments | 2 (checkpoint and resume between them) |
| **Total optimizer steps** | **120** |
| Module ceiling | 240 steps / 40 epochs — no flag raises it |
| Wall-clock ceiling | 7 200 s per segment, 14 400 s absolute |

Why 120. Fewer than about 100 steps and the validation curve has too few
points to say anything at all; more and this stops being a bounded
experiment and starts being training, which is a decision for a later
phase and a different machine. Six steps per epoch means twenty
validation measurements, which is the smallest number that makes a curve
rather than a pair of readings.

Everything else is the proven Phase 35B Apple-silicon path, unchanged:
MPS, bf16, LoRA rank 16 / alpha 32, AdamW, gradient checkpointing, base
model frozen.

## Validation loss, which the trainer does not have

The installed ACE-Step trainer has no validation loop. It trains and it
reports, and nothing anywhere measures held-out loss — so Phase 36 runs
one from inside the trainer process, at every epoch boundary, using the
**same** objective the trainer optimises, under `no_grad`, with no
optimizer in reach.

Two deliberate departures, because without them the number would be
noise:

- **The generator is reseeded to a fixed value before every sample.**
  The flow-matching loss draws fresh noise and a fresh timestep on every
  call, and two passes over identical weights differ by more than a few
  epochs of training would move them. A fixed draw makes successive
  passes comparable.
- **Classifier-free-guidance dropout is disabled during validation.** It
  is a training regulariser; on four validation tracks it would null the
  conditioning of a quarter of them, and the curve would be measuring
  the unconditional model on a random subset.

So the validation number is a held-out loss under a fixed noise draw. It
is not a sample from the training objective, and that difference matters
when reading it.

## One fixed sequence length, and why

Every training and validation sample is cut to the same **3000 latent
frames** — 120 seconds at the measured 25 frames a second. Truncation
only: 3000 is below the shortest track in either split, so nothing is
padded and no frame of audio is invented.

This was not a design preference. It was diagnosed:

- Phase 35B trained four tracks at up to 6000 latent frames and peaked
  at **9.43 GiB** of Apple unified memory.
- Phase 36's 24 tracks have **24 different latent lengths**. Every
  attempt reached **29 GiB** and died at step 9 — the third step of the
  second epoch, in four separate runs.
- Shortening the validation window from 6000 frames to 2048 moved the
  failure by **zero steps**.
- Stopping an unrelated process that was holding ~10 GB, which took
  system swap from 31 GB to 11 GB, also moved it by **zero steps**.

What was left was the number of distinct shapes. Metal's caching
allocator keeps a working set per tensor shape; one epoch over 24
lengths builds 24 of them, and the second epoch has nowhere to put the
next block. Cutting every sample to one length made the run complete.

Two consequences worth stating plainly. The model sees the first two
minutes of each track, not the whole track. And the Phase 34/35B
capacity profile — measured at a *single* shape — qualified a workload
that then ran at 24 of them; a profile's applicability check compares
the sequence length it measured, and nothing in it counts how many
lengths a dataset contains. That is a real gap in the capacity model,
and it is recorded here rather than papered over.

## What the phase can and cannot conclude

Two verdicts, kept apart:

- **Training signal** — finite losses, finite non-zero gradients, adapter
  tensors that moved, a base model that did not. Says the machinery
  optimised something.
- **Generalization signal** — what the held-out loss did. The only thing
  here that can speak to learning rather than fitting.

The generalization vocabulary is deliberately weak: `HELD_OUT_LOSS_IMPROVED`,
`HELD_OUT_LOSS_DEGRADED`, `NO_MEASURABLE_CHANGE`, `INSUFFICIENT_EVIDENCE`.
There is no `CONVERGED` and no `IMPROVED_QUALITY`, because neither can
be established by a loss number, and a vocabulary that offers the word
invites the claim. A ±2 % band around the first measurement is treated
as noise; the threshold is chosen, not measured, and every verdict that
uses it says so.

Whatever the numbers do, `listening_evaluation_required` is true for any
run that produced a model. Nothing in this phase replaces listening.

## What the run produced

| | |
|---|---|
| Outcome | `COMPLETED` |
| Training signal | **`VALID_SIGNAL`** |
| Generalization signal | **`HELD_OUT_LOSS_IMPROVED`** |
| Steps | 120 of a 120 ceiling, in two segments of 60 |
| Wall clock | 3527 s |
| Train loss | first 1.5895, last 0.9548, min 0.9548, max 1.6851, finite ratio 1.0 |
| Held-out loss | first 1.4724, last 1.1278, min 1.1278, max 1.4728, over 20 measurements |
| Gradients | 120 of 120 finite, 120 non-zero, mean norm 0.3291 |
| Adapter | **384 of 384** trainable tensors changed, 11,010,048 parameters |
| Base model | preserved — file digest identical before and after |
| Resume | epoch 10 to 20, step 60 to 120, exit 0 |
| Checkpoint provenance | segment A verified, segment B verified |

Held-out loss fell **-23.41%** across the run, on four tracks no
optimizer step ever touched. That is the one number here that speaks to
learning rather than fitting, and it is still only that: **no quality
claim, no convergence claim.** `listening_evaluation_required` is
`true`.

Phase 35B's checkpoint gap is closed. Both checkpoints carry a complete
provenance record and both verify against the experiment they came from
— the same check that reported `ok: false` on every pilot checkpoint now
reports no problems.

### A/B listening set

Eight deterministic pairs, drawn from the evaluation split and from
nowhere else: the caption is the operator's own group label, and the
seeds are read out of each track's audio digest, so the set reproduces
without a random number anywhere. Both sides share prompt, seed,
duration (30 s) and step count (8); the adapter is the only difference.
Base and LoRA outputs live in separate directories that are never
written twice, and a manifest maps each pair. Nothing is judged
automatically — a person has to listen.

## What is still not established

- Whether the adapter makes better music. Nobody has listened yet.
- Whether the run converged. 120 steps cannot answer that.
- Whether anything generalises past two-minute windows of 24 tracks.
- Whether the held-out improvement survives a different seed, a
  different split, or a longer run. It was measured once.
