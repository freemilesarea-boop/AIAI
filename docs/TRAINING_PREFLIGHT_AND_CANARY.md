# Training preflight and the bounded canary

**Phase 33.** What LUBER checks in the minutes before a trainer starts,
and the smallest real training run it will perform to prove the answer.

This document states what has been measured, what has been reasoned
about, and what nobody knows. Where a fact came from a fixture rather
than from hardware, it says so.

---

## 1. The question this phase answers

Phase 25 asks whether a dataset **may** be trained on — rights, locks,
evaluation leakage. Phase 27 asks whether the artifacts **arrived** on
the worker intact. Phase 32 asks **where** a workload should run.

None of them asked the question in the middle:

> Given a compiled plan and an eligible machine, will the trainer
> actually start, on that device, at that precision, with that
> optimizer — and can we prove it before the expensive part begins?

Phase 33 answers it, and then goes one step further: it runs a training
run small enough to be safe and real enough to be evidence.

---

## 2. READY, BLOCKED, UNVERIFIED

Three statuses. The third is the one that carries the phase.

| Status | Means | What an operator does |
|---|---|---|
| `READY` | Every mandatory check is **proven** satisfied for the stated intent | Proceed |
| `BLOCKED` | At least one check found a definite incompatibility | Fix the named thing; the reason is machine-readable |
| `UNVERIFIED` | A mandatory check **could not be performed** | Go and measure it, or accept a narrower intent |

**`UNVERIFIED` is not `READY`.** It is not amber-but-fine and no caller
in this repository treats it as a pass. In the console it renders with a
dashed outline and a `?`, never a tick — the same encoding an unmeasured
capability has had since Phase 28.

`BLOCKED` and `UNVERIFIED` both stop a run. They are kept apart because
the next move differs: one is "this machine cannot do it", the other is
"nobody has established whether this machine can do it".

### Intent decides what is mandatory

A preflight is run for a purpose, and the purposes need different
proofs:

* **`CANARY`** — clearing a bounded run. Whether a *production-scale*
  workload fits in memory is irrelevant to it, because it will not run
  one.
* **`FULL_TRAINING`** — clearing a real run. An unmeasured memory
  requirement makes it `UNVERIFIED`.

Nothing in this project has a measured memory requirement, so **every
`FULL_TRAINING` preflight in LUBER today is `UNVERIFIED`**. That is the
honest state, not a bug, and it is why the `CANARY` intent exists: it is
the one that can be answered.

---

## 3. What is checked

Checks are pure functions over evidence. All I/O — filesystem reads,
subprocesses, probes — happens in `luber_training.preflight_collect` and
arrives as frozen dataclasses. That is what makes a preflight
deterministic (identical evidence produces an identical result, digest
included) and what lets the CUDA branches be tested on a machine with no
NVIDIA hardware.

| Group | Checks |
|---|---|
| `plan` | schema supported; digest recomputed; execution device present and resolvable; requirement set does not contradict itself; location legal; precision inside the plan's supported set |
| `dataset` | every Phase 25 gate; manifest present; non-zero eligible samples; no evaluation-only material; **manifest drift** since compilation; referenced files present where they can be seen |
| `trainer` | trainer root, `train.py`, interpreter; torch importable in the *trainer's* interpreter; installed ACE-Step revision matches the audited one; **the compiled argv is accepted by the installed parser**; the optimizer's optional package really imports |
| `hardware` | the selected device exists on the selected target; precision measured working on it; the trainer can actually train in that precision; optimizer legal for the device; GPU count |
| `storage` | dataset readable; output and checkpoint directories writable (by writing); **the dataset resolves inside the trainer's working directory**; free disk against a derived requirement |
| `remote` | worker assigned and registered; worker class verified for a CUDA plan; **capability report fresh enough**; the worker answered |
| `capacity` | device memory; the training memory requirement (always UNKNOWN) |
| `canary` / `checkpoint` | a bounded canary's outcome, its checkpoint integrity and its resume |

### The failure taxonomy

Machine-readable, closed, and deliberately not collapsed into `FAILED`.
Each entry names something an operator does differently.

```
PLAN_INVALID            DATASET_NOT_READY       RIGHTS_BLOCKED
MANIFEST_DRIFT          EVALUATION_LEAKAGE      DEVICE_UNAVAILABLE
PRECISION_UNSUPPORTED   OPTIMIZER_UNSUPPORTED   TRAINER_UNAVAILABLE
DEPENDENCY_MISSING      WORKER_UNAVAILABLE      WORKER_STALE
REMOTE_UNREACHABLE      STORAGE_UNAVAILABLE     INSUFFICIENT_DISK
CAPACITY_UNVERIFIED     CANARY_FAILED           CHECKPOINT_FAILED
RESUME_FAILED
```

### Worker freshness is a policy, and it is named

A capability report older than **12 hours** (`DEFAULT_CAPABILITY_MAX_AGE_SECONDS`)
is `WORKER_STALE`. A rented machine can be returned, resized or have its
driver changed between sessions, and a report from last week describes a
machine that may not exist. A worker with *no readable timestamp* is
`UNKNOWN` rather than stale — we do not know how old it is, and neither
answer is a pass.

---

## 4. Capacity evidence: MEASURED, ESTIMATED, UNKNOWN

Every capacity number carries how it was established, and the three
never mix.

* **MEASURED** — something on the machine reported it, with what took it
  and when.
* **ESTIMATED** — derived by *stated* arithmetic from a measured figure.
  The derivation travels with the number, so a reader can disagree with
  the multiplier rather than having to trust the result. An estimate is
  never promoted to a measurement by being reasonable.
* **UNKNOWN** — nobody looked, or nobody can.

`CapacityReport` has no `fits` field. Whether a real training workload
fits is the question nobody in this project can answer yet, and a
boolean would be the single most misleading field in the codebase.

### Apple unified memory is not VRAM

A 24 GB Mac does not have 24 GB of accelerator memory. It has 24 GB
shared between the GPU, the CPU, the operating system, Postgres, Redis
and whatever else is running. Evidence for MPS carries
`unified_memory: true`, and the console renders it as *"unified memory,
shared with the OS — not VRAM"* rather than putting the figure in a
column beside a GPU's.

### What is UNKNOWN today

* The memory requirement of any LUBER training configuration, on any
  device. Nothing has measured it.
* The disk requirement, until a canary has produced a checkpoint whose
  size can be observed. There is no formula here for "a rank-16 LoRA is
  N megabytes".
* Peak device allocation during a canary. Nothing outside the training
  process can observe MPS allocation, and a resident-set figure taken
  from outside proved not to be defensible, so it is reported as UNKNOWN
  rather than published with a caveat nobody would read.

---

## 5. The bounded canary

Two canaries, and they are not interchangeable.

### `ORCHESTRATION`

Proves LUBER's half: the plan compiles, the bounded config is inside its
envelope, and the command the trainer would receive is produced and
recorded. **It starts no process and trains nothing**, and it says so in
its own result. It is what remains available when the trainer is on a
machine the control plane cannot reach.

### `ACE_STEP`

Proves the trainer: the real ACE-Step DiT is loaded, a real LoRA is
injected, real optimizer steps run, and a real checkpoint is written and
reopened. Only the **data** is synthetic.

A pass is evidence about the mechanism and about nothing else. No music
was involved, the adapter learned noise, and the checkpoint must never
be promoted — which is why LUBER writes a provenance sidecar beside it
saying exactly that.

### The bounds are structural

| Bound | Value | Constant |
|---|---|---|
| Epochs | 1 (2 for a resume canary) | `CANARY_MAX_EPOCHS`, `CANARY_MAX_RESUME_EPOCHS` |
| Samples | ≤ 4 | `CANARY_MAX_SAMPLES` |
| Optimizer steps | ≤ 8 | `CANARY_MAX_OPTIMIZER_STEPS` |
| Wall clock | ≤ 3600 s | `CANARY_ABSOLUTE_WALL_CLOCK_SECONDS` |

`CanaryEnvelope` validates against these **on construction**, so an
envelope that exists is an envelope that is safe and nothing downstream
re-checks. The bounded config is *derived* from the envelope rather than
accepted from a caller: epochs, checkpoint interval, warmup and worker
count are overwritten, and a request that would exceed the step ceiling
raises `CanaryBoundsError` before any process exists.

There is no flag anywhere in `luber_training.canary` that raises a
ceiling. Raising one is a code change with a diff and a review.

The wall clock is enforced by killing the process group. A canary that
could outlive its bound would not be bounded.

### Rights are not relaxed for being small

A canary trains on one of exactly two things:

1. **A synthetic fixture** — upstream's own
   `acestep.training_v2.make_test_fixtures`, which exists for this and
   marks every sample `is_synthetic`. There is no recording in it and
   nothing to have rights in. LUBER never fabricates a training tensor
   itself; using upstream's generator also means the shapes cannot drift
   from what the trainer reads.
2. **Gate-cleared material** — every Phase 25 gate passed, the same
   gates as a full run.

Anything else is refused with `UNAUTHORISED`: *being small is not an
authorisation*. A directory holding more samples than the envelope
allows is refused too.

---

## 6. Checkpoint integrity and provenance

Phase 27 hashes what the trainer wrote and records its size, which
catches a truncated transfer and nothing else. A directory can hold a
correctly-sized, correctly-hashed adapter whose tensors are all zero,
whose training state will not deserialise, or whose step count is
missing — and each of those is a checkpoint that fails the first time
somebody uses it, hours later.

So a canary's checkpoint is **opened**, by the trainer's own
interpreter, and reported on:

* the adapter loads through `safetensors`, with its tensor count
* how many parameters are non-zero (all-zero means nothing was learned)
* `training_state.pt` deserialises, with epoch, global step and
  optimizer state
* a provenance sidecar (`luber_canary_provenance.json`) exists and cites
  the plan this checkpoint was expected from

### Provenance

Written by LUBER beside the checkpoint, because the trainer writes an
adapter and knows nothing about plans, devices or rights. It records the
run, experiment and plan ids, the **plan digest**, the config digest,
the base model and its upstream commit, the ACE-Step revision, the
execution location and device, the resolved precision, the optimizer,
the dataset identity, the envelope, the step count — and a warning that
the checkpoint is a canary's and must never be promoted.

### Resume is only a claim if the step counter moved

Deserialising a file is not resumability. A resume canary runs one
bounded epoch, saves, stops, reloads with `--resume-from`, runs one more
epoch and then **compares the step counts**. If the counter did not
advance, the run restarted rather than resumed, and that is reported as
a failure.

---

## 7. Run identity and provenance chain

```
training plan ──▶ preflight ──▶ worker/target ──▶ canary ──▶ checkpoint
     │                │                             │           │
  plan digest    plan digest                   bounded plan  provenance
                 capability digest               digest        sidecar
```

The **plan digest** stays the immutable training-intent identity.
Volatile runtime measurements — how much memory a machine had today,
which capability digest it reported — live on the preflight and run
records, never on the plan. Phase 32's rule, preserved and tested:
evaluating the same plan against two different machines does not change
its digest.

A canary's bounded plan has a **different** digest from the run's plan,
deliberately. It trains for one epoch on synthetic tensors, and giving
it the run's identity would let its checkpoint be mistaken for the
run's.

---

## 8. Real-machine results

### MPS — measured, on this machine

Apple **M4 Pro**, 24 GB unified memory, macOS. Probed through the
ACE-Step trainer's own interpreter: Python 3.12.11, torch 2.10.0,
`mps_built=True`, `mps_available=True`, `cuda_available=False`.
ACE-Step at the pinned commit `6d467e4b`, base weights already installed
(`acestep-v15-turbo`, 4.5 GB). Nothing was downloaded.

| Precision | Result | Evidence |
|---|---|---|
| `bf16` | **PASS** | model loaded in bfloat16, LoRA injected (2 752 512 trainable of 2 396 625 030), 1 optimizer step, checkpoint written, reopened (384 tensors, 2 751 774 non-zero), resume advanced step 1 → 2 with optimizer state restored |
| `fp32` | **PASS** | same path at `32-true` |
| `fp16` | **FAIL** | `ValueError: Attempting to unscale FP16 gradients.` at the first gradient clip |
| `auto` | **FAIL** | `auto` resolves to `fp16` on MPS, so it fails identically |

The `fp16` failure is a property of the **trainer**, not the hardware.
Phase 32 measured fp16 tensors working on this machine's MPS backend and
they do. `fixed_lora_module` loads the model in fp16 *and* drives
Lightning Fabric at `16-mixed`, and Fabric's `GradScaler` refuses to
unscale gradients that are themselves fp16. The run dies at the first
clip, after the 2.4B model has loaded.

LUBER's `AUTO_BY_DEVICE` still says MPS → fp16, because that is still
what the trainer does. What changed is that `UNTRAINABLE_PRECISION` now
blocks the combination with the measured reason attached, so an operator
is told to request `bf16` or `fp32` explicitly rather than discovering
it three minutes into a load.

### CUDA — UNVERIFIED

**No NVIDIA hardware has ever been attached to this project.** Nothing
in Phase 33 changes that.

* **CUDA real hardware: UNVERIFIED.**
* **CUDA logic: fixture-tested.** Every CUDA case in
  `test_training_preflight.py` runs against a literal `MachineCapability`
  describing a machine nobody owns. The fixture's device is named
  `"NVIDIA (test fixture, not a real card)"` on purpose.
* **CUDA capacity: UNVERIFIED.** No VRAM figure, benchmark or model name
  is claimed anywhere.

**Fixture-tested CUDA logic is not real NVIDIA hardware validation.**

---

## 9. Three defects Phase 33 found in the path to the trainer

All three would have surfaced on the first real dispatch, expensively.

**1. The compiled command could not train.** LUBER emitted
`python train.py fixed …`. The trainer's `--yes` is declared on its
*root* parser, so `fixed --yes` is rejected outright — and LUBER did not
emit it at all. `run_fixed` calls `confirm_start()` before loading
anything; launched detached with stdin closed, the prompt raises
`EOFError`, `confirm_start` returns `False`, and **`run_fixed` returns
0**. From outside that is indistinguishable from success: exit code
zero, no traceback, no checkpoint. Fixed by emitting `--yes --plain`
before the subcommand, and checked by offering the compiled argv to the
installed parser.

**2. `--checkpoint-dir` was pointed at the wrong directory.** To the
trainer it is the root it *reads base model weights from*; the remote
worker was passing the run's own empty output directory. Model loading
would have failed. The worker now passes its configured
`checkpoint_root`, which is what that logical root has always been for.

**3. Checkpoints were looked for where the trainer does not write
them.** The trainer writes per-epoch checkpoints under
`--output-dir/checkpoints/` and the last adapter under
`--output-dir/final/`. Discovery now looks in both, as well as where it
looked before.

A fourth is a constraint rather than a defect: ACE-Step's
`path_safety._SAFE_ROOT` is the working directory at import time, and
`--dataset-dir` is validated against it — **after** the model has
loaded. A dataset outside the trainer's working directory is refused
with "Path escapes safe root". There is no environment variable for it,
so the preflight checks the two paths before anything is loaded and the
canary keeps its workspace beneath the trainer root.

---

## 10. Using it

```bash
# Whether a machine can execute a plan. Starts nothing.
python -m luber_training preflight \
    --run-id run_… --device MPS --location LOCAL --intent CANARY \
    --trainer-root ~/ace-step-1.5 \
    --python ~/ace-step-1.5/.venv/bin/python \
    --dataset-build … --curation-build … --json

# LUBER's half, without a trainer.
python -m luber_training canary orchestration --run-id run_… --device MPS

# The real trainer, bounded: two synthetic samples, one epoch.
python -m luber_training canary ace-step \
    --run-id run_… --device MPS \
    --trainer-root ~/ace-step-1.5 \
    --python ~/ace-step-1.5/.venv/bin/python \
    --model-dir ~/ace-step-1.5/checkpoints \
    --samples 2 --resume --record-preflight --cleanup
```

The console shows the recorded result at
`/ops/training/runs/<id>` and over
`GET /v1/ops/training/runs/<id>/training-preflight` and
`…/canary`. Both are **reads**: collecting evidence means subprocesses
and running a canary starts a trainer, and neither belongs in a process
a browser can reach.

---

## 11. Limits

1. **Passing Phase 33 does not mean a production-scale training workload
   has been proven to fit on a given accelerator.** A canary trains on
   two synthetic tensors of length 64 for one epoch. It says the
   mechanism works. It says nothing about whether a real dataset, at
   real sequence lengths, for thirty epochs, fits in memory or converges.
2. **Fixture-tested CUDA logic is not real NVIDIA hardware validation.**
3. **A canary's checkpoint is worthless as a model.** It learned noise.
   The provenance sidecar says so and nothing may promote it.
4. **No training-time quality signal exists.** A canary reports that a
   loss was computed, not that it was a good one.
5. **Peak device allocation is UNKNOWN.** Nothing outside the training
   process can observe MPS allocation.
6. **The disk requirement is UNKNOWN** until a canary has produced a
   checkpoint on that configuration.
7. **The freshness policy is code.** Changing 12 hours is a deploy.
8. **The MPS results are from an M4 Pro.** A base M4 has a different GPU
   configuration. This machine is a fair *compatibility* proxy for a
   24 GB Mac mini and a poor performance one.

---

## 12. Related documents

* `docs/HARDWARE_EXECUTION_COMPATIBILITY.md` — Phase 32: devices,
  placement, precision, what was measured
* `docs/REMOTE_GPU_EXECUTION.md` — Phase 27: how work reaches a worker
* `docs/TRAINING_ORCHESTRATION.md` — Phase 25: gates, plans, lifecycle
* `docs/TRAINING_CONSOLE.md` — Phase 28: the operator console
