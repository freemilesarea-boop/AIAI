# Training memory and capacity qualification

**Phase 34.** How LUBER measures what a training configuration actually
costs, and how it decides whether that leaves enough room to run
something larger.

Every figure in this document is labelled with how it was obtained.
Where nothing has been measured, it says so.

---

## 1. The question this phase answers

Phase 33 could prove that the trainer starts, takes a step and writes a
checkpoint. It could not answer the question that decides whether a real
run is safe:

> How much memory does this exact configuration consume, how much
> headroom remains, and is that enough evidence to allow a larger run?

Until Phase 34 the answer was `UNKNOWN`, and every `FULL_TRAINING`
preflight was `UNVERIFIED` on that ground alone.

---

## 2. Three domains, never merged

| Domain | What it is | Read from |
|---|---|---|
| `HOST` | The process's resident set, and the machine's available RAM | `psutil` in the trainer process |
| `APPLE_UNIFIED` | Apple's unified memory, as the Metal backend reports it | `torch.mps` |
| `CUDA_DEVICE` | An NVIDIA card's own memory | `torch.cuda` |

They are not comparable and there is no field that could hold either.

**Apple unified memory is not VRAM.** A 24 GB Mac does not have 24 GB of
accelerator memory; it has 24 GB shared between the GPU, the CPU, the
operating system and everything else running. Every Apple figure carries
`unified_memory: true`, and the API, the console and the reports call it
*unified memory, shared with the OS — not VRAM*.

A measured consequence, from the real profile below: on Apple silicon a
9.5 GiB MPS driver allocation showed up beside a **507 MiB** process
resident set. **Host RSS is not the training memory on Apple silicon**,
and anything that treated it as such would under-report by an order of
magnitude. Phase 33 tried exactly that and withdrew the number.

---

## 3. Four evidence sources

| Source | Means |
|---|---|
| `MEASURED` | Something on the machine reported it, with what took it and when |
| `DERIVED` | Arithmetic over measured values, with the arithmetic stated |
| `ESTIMATED` | Derived from a measured figure by a stated multiplier |
| `UNKNOWN` | Nobody looked, or nobody can |

`DERIVED` was added in this phase for peak-minus-baseline and
budget-minus-required: every input is a measurement, and the subtraction
is still somebody's choice of what to subtract. It never renders as
`MEASURED`.

---

## 4. Peaks: runtime versus sampled

```
RUNTIME_PEAK   a high-water mark the runtime itself kept — exact
SAMPLED_PEAK   the largest value a sampler observed — a LOWER BOUND
NOT_AVAILABLE  the runtime exposes no figure for this domain
```

`torch.cuda` keeps `max_memory_allocated` and `max_memory_reserved`.
**The pinned torch 2.10.0 has no `torch.mps.max_memory_allocated`** — it
offers `current_allocated_memory`, `driver_allocated_memory` and
`recommended_max_memory` and nothing else. So every Apple peak in this
project is sampled, and is a lower bound: anything that rose and fell
between two samples was never seen.

That is why sampled evidence gets a larger safety margin than runtime
evidence, and why the two are separate numbers with separate reasons
rather than one number with a footnote.

Apple's figure is the **driver** allocation rather than the current one:
the driver figure is what the process holds from the system, and the
current figure excludes everything the caching allocator has kept back.

---

## 5. Measuring from inside the trainer

The measurements come from **inside the real trainer process**, not from
something watching it. `torch.mps` will only talk to the process that
allocated, and a peak that happens between a forward and a backward is
invisible from outside.

`luber_training._memory_probe` is therefore executed *as* the trainer:
it imports ACE-Step, wraps a handful of its callables with pass-through
recorders, starts a sampler, and runs `train.py` in-process through
`runpy` with the argv LUBER compiled. **No ACE-Step source is
modified.** Each wrapper calls the original, records two numbers around
it, and returns what the original returned; generators stay generators
so `yield from` still works.

### Stages, and where each is observed

| Stage | Seam |
|---|---|
| `BASELINE`, `RUNTIME_INITIALIZED` | before and after `import torch` |
| `MODEL_LOADED` | `cli.train_fixed.load_decoder_for_training` |
| `LORA_ATTACHED` | `fixed_lora_module.inject_lora_into_dit` |
| `OPTIMIZER_CREATED` | `trainer_fixed.build_optimizer` |
| `BATCH_READY`, `FORWARD_COMPLETE` | entry to and return from `FixedLoRAModule.training_step` |
| `BACKWARD_COMPLETE` | `lightning.fabric.Fabric.backward` |
| `OPTIMIZER_STEP_COMPLETE` | the trainer's own yielded step update |
| `CHECKPOINT_BEGIN`, `CHECKPOINT_COMPLETE` | `trainer_fixed.save_checkpoint` |
| `RESUME_LOADED`, `RESUME_STEP_COMPLETE` | `trainer_fixed.resume_checkpoint`, then the next step |
| `FINAL` | after the run, in a `finally` |

A stage that could not be observed is recorded in `not_observed` **with
the reason**. An absent stage and an unobservable one look identical
otherwise.

### The sampler

A daemon thread driven by an `Event`, stopped in a `finally` and joined
with a timeout. It caps its own snapshot count. It is tested to stop
after success, after a raising recorder, after a double stop, and to
leave no thread behind.

### The safety boundary

The sampler can observe that memory has crossed a configured boundary —
the system's available memory falling below a floor, or the MPS driver
allocation passing a share of the runtime's *own*
`recommended_max_memory`. It does **not** predict an OOM and it does not
kill anything from a background thread. It sets a flag, and the next
wrapper to run in the main thread raises. A boundary needs a number to
cross; where the reading is absent, nothing trips.

---

## 6. Profile identity — what a measurement belongs to

A memory measurement is valid only for the configuration that produced
it. The identity digest covers:

```
device            precision           optimizer          strategy
micro_batch_size  gradient_accumulation                  gradient_checkpointing
lora_rank         lora_alpha          target_modules     attention_type
latent_length     encoder_length      model_variant      num_devices
base_model_upstream_commit            ace_step_commit    offload_encoder
```

It deliberately excludes timestamps, hostnames, usernames, pids and any
free-memory reading. Two runs of the same configuration produce the same
identity.

**Micro batch and gradient accumulation are different things.** The
micro batch is what the trainer hands the DataLoader and is what decides
how many samples' activations are held at once; accumulation multiplies
the *effective* batch without holding more. Both are recorded; only the
micro batch is a memory dimension.

### `latent_length` is the field this phase turns on

The VAE downsamples 48 kHz audio by 2·4·4·6·10 = **1920**, which is
**25 latent frames per second**, and preprocessing caps a track at 240
seconds — so a production sample is about **6000 frames**.

Phase 33's canary ran at **64 frames**: about two and a half seconds.
Activation memory scales with that dimension, so the canary's peak is a
real measurement of a workload production does not resemble. Using it to
qualify a real run would be worse than having no measurement — it would
look like evidence.

---

## 7. Applicability

A profile qualifies a request only when every memory-relevant field
matches. Where they differ, the qualification is `UNVERIFIED` and the
differing fields are named.

* batch 1 does not qualify batch 4
* rank 8 does not qualify rank 64
* bf16 does not qualify fp32
* MPS does not qualify CUDA
* two seconds of latents does not qualify four minutes

**One direction is sound and only one.** A profile measured at a
*larger* `latent_length` or `encoder_length` covers a smaller request:
the shorter run allocates less of the same thing. The reverse is the
extrapolation this module exists to refuse.

### Staleness

A profile becomes inapplicable when `ace_step_commit` or `torch_version`
moves — a different trainer may allocate differently, a different torch
may change the allocator. **A later date alone changes nothing** and is
not a staleness signal.

---

## 8. Capacity policy

`capacity-policy-v1`. Every default is a **choice**, not a measurement,
and none is claimed to be optimal for any particular machine.

| Setting | Default | Why |
|---|---|---|
| `host_headroom_fraction` | 0.20 | the operating system and everything else |
| `host_headroom_floor_bytes` | 4 GiB | a percentage of a small machine reserves too little |
| `device_headroom_fraction` | 0.15 | lower than the host's: a CUDA pool has no OS in it, and on Apple the host reserve already protects the same physical memory |
| `device_headroom_floor_bytes` | 2 GiB | — |
| `control_plane_reserve_bytes` | 4 GiB | extra, when the target also serves the API, Postgres, Redis and the orchestrator |
| `sampled_peak_margin` | 1.25 | a sampled peak is a lower bound; this is a margin, not a measurement of how much was missed |
| `runtime_peak_margin` | 1.10 | the runtime missed nothing; this covers fragmentation and run-to-run variation |
| `margin_low_ratio` | 0.85 | above this share of the budget, a pass is reported `MARGIN_LOW` |

**Peak below total is not safe.** A machine that is exactly full is a
machine whose next allocation fails.

On Apple silicon the host and unified-memory domains describe the *same*
physical pool. The two verdicts are evaluated independently and **never
added together**: summing would double-count, and taking the larger
alone would drop the operating system's share. Both must pass on their
own budget.

---

## 9. Qualification

```
QUALIFIED     an applicable, completed, representative profile whose
              measured peak satisfies the policy
MARGIN_LOW    satisfies the policy, with little room
INSUFFICIENT  measured evidence says the policy cannot be met
UNVERIFIED    no applicable evidence
```

`UNKNOWN` never becomes `QUALIFIED`. When several applicable profiles
exist the qualifier uses the one with the **highest** peak: picking the
friendliest would be choosing the evidence to fit the answer.

`luber_training.capacity_policy.qualify()` is the only place this is
decided. Memory-safety logic in a scheduler, a preflight and a console
would eventually disagree, and the copy that mattered would be whichever
one the button used.

---

## 10. What was measured — current machine only

**Apple M4 Pro, 24 GB unified memory, macOS.** Python 3.12.11, torch
2.10.0, ACE-Step `6d467e4b`, MPS built and available, CUDA absent. Base
weights already installed; nothing was downloaded.

Two bounded profiles at **production sequence length** — 6000 latent
frames (240 s at 25 frames/s), encoder length 256, 2 synthetic samples,
1 epoch, bf16, gradient checkpointing on, micro batch 1, gradient
accumulation 4:

| | LoRA rank 4 | LoRA rank 32 |
|---|---|---|
| **Unified-memory peak** (SAMPLED) | **9 751 MiB** | **9 855 MiB** |
| Host RSS peak (SAMPLED) | 507 MiB | 612 MiB |
| Checkpoint-stage peak | 9 209 MiB | 9 393 MiB |
| Resume-leg peak | 5 320 MiB | 5 464 MiB |
| Optimizer steps | 1 | 1 |
| Wall clock | 28.9 s | 28.4 s |
| Samples taken | 113 | ~113 |
| Representativeness | REPRESENTATIVE | REPRESENTATIVE |

Every stage was observed on the rank-32 run, including resume; nothing
was `not_observed`.

**Qualification** for the rank-32 configuration on this machine, with
the control-plane reserve applied:

```
APPLE_UNIFIED  QUALIFIED   peak 9855 MiB → 12 319 MiB required (×1.25 sampled margin)
                           against a 20 889 MiB budget (24 576 total − 3 686 reserved)
HOST           QUALIFIED   peak 612 MiB → 765 MiB required
                           against a 15 564 MiB budget (24 576 total − 9 011 reserved)
```

### What these numbers do **not** say

* Nothing about music quality, convergence, generalisation, or whether
  the resulting model is any good. A memory profile measures memory.
* Nothing about a **longer** sequence, a **larger** micro batch, a
  different precision, or a different model variant. Those are different
  identities and are `UNVERIFIED` until measured.
* Nothing about **wall-clock performance**. The 28-second figure is one
  epoch over two synthetic samples on this machine on this day.
* Nothing about **CUDA**.

---

## 11. Checkpoint and resume memory

Measured separately, because they are not the same allocation.

* The checkpoint write did **not** produce a new high-water mark on
  either profile — the checkpoint-stage figure sits just below the run
  peak. It is measured rather than assumed neutral.
* The resume leg is its own bounded invocation against the checkpoint
  the first leg wrote, and peaked **lower** than a fresh start
  (5 464 MiB against 9 855 MiB). An operator planning a long run that
  may be interrupted needs the larger of the two, which is the fresh
  start.

---

## 12. OOM taxonomy

```
HOST_OOM                 CUDA_OOM                MPS_OOM
UNKNOWN_MEMORY_FAILURE   NOT_A_MEMORY_FAILURE
```

Classified from signatures each runtime actually emits, most specific
first. An ordinary `RuntimeError` stays `NOT_A_MEMORY_FAILURE`: calling
every crash an OOM would make the OOM code meaningless. **A SIGKILL is
on no list** — the kernel OOM killer and `kill -9` are
indistinguishable from outside the process.

---

## 13. Profiles are evidence records

Stored one file per identity under `<registry>/memory_profiles/`, plus a
run-scoped copy at `training_memory_profile.json` beside the run. A bf16
profile never overwrites an fp32 one, and a second machine's measurement
never overwrites the first's.

**They are operational artifacts and are never committed to git.**

A file this build cannot parse is skipped rather than raising: one stale
document must not make every other measurement unreadable. A document
from a different protocol version is refused rather than read with its
unknown fields ignored.

---

## 14. Using it

```bash
# Measure one configuration, inside the real trainer, under a clock.
python -m luber_training --registry ./training-registry \
    profile-memory --run-id run_… --device MPS \
    --trainer-root ~/ace-step-1.5 \
    --python ~/ace-step-1.5/.venv/bin/python \
    --model-dir ~/ace-step-1.5/checkpoints \
    --latent-length 6000 --encoder-length 256 --samples 2 \
    --measure-resume --cleanup

# What the stored profiles permit for a configuration.
python -m luber_training --registry ./training-registry \
    capacity --run-id run_… --device MPS --latent-length 6000

# The preflight consumes the qualification.
python -m luber_training preflight --run-id run_… --intent FULL_TRAINING …
```

The console shows it at `/ops/training/runs/<id>` and over
`GET /v1/ops/training/runs/<id>/capacity`. Both are reads: measuring
memory means running the trainer, and that belongs to the operator CLI
on the machine that has one.

---

## 15. Future hardware

### Planned Apple Silicon Mac mini, 24 GB unified memory

**Status: REQUIRES ON-DEVICE QUALIFICATION.**

The measurements above are from an **M4 Pro**. A base M4 shares the
memory capacity but not the GPU configuration or the memory bandwidth.
Memory-capacity evidence may be *informative* across the two where the
runtime semantics are identical; it is not qualification. The procedure
is in `docs/MAC_MINI_CONTROL_PLANE_RUNBOOK.md`.

### NVIDIA

**Real CUDA profile: NOT RUN. CUDA capacity: UNVERIFIED. CUDA profiler
logic: synthetically verified.**

No NVIDIA hardware has ever been attached to this project. The CUDA
paths use the real `torch.cuda` APIs — `memory_allocated`,
`memory_reserved`, `max_memory_allocated`, `max_memory_reserved`,
`mem_get_info`, `reset_peak_memory_stats` — behind runtime availability
checks, and are exercised against fixtures. A fixture cannot qualify
CUDA, and there is a test that asserts it cannot. The procedure for a
real worker is in `docs/NVIDIA_TRAINING_WORKER_RUNBOOK.md`.

Multi-GPU is `UNVERIFIED`: the schema carries `num_devices` in the
identity, so a single-GPU profile cannot qualify a multi-GPU run.

---

## 16. Limits

1. **No production-scale run has been performed or proven.** A profile
   measures one bounded epoch over two synthetic samples. It qualifies
   memory, not a training outcome.
2. **Apple peaks are lower bounds.** No MPS peak counter exists in the
   pinned torch.
3. **Encoder length is a stated probe parameter, not a production
   measurement.** Nothing here has measured what production lyrics come
   to; a profile does not qualify a longer conditioning sequence.
4. **The policy defaults are choices.** Changing them is a deploy, and
   the version on a stored decision says which ones were in force.
5. **The profiled workload is synthetic.** The tensor *shapes* are
   production-realistic; the values are noise. Memory depends on shapes.
6. **One local profiling workload at a time.** Two concurrent MPS runs
   on shared unified memory would measure each other.
7. **A qualification is not a schedule.** Phase 34 may prove readiness;
   it does not start a run.

---

## 17. Related documents

* `docs/TRAINING_PREFLIGHT_AND_CANARY.md` — Phase 33: the readiness gate
* `docs/HARDWARE_EXECUTION_COMPATIBILITY.md` — Phase 32: devices and placement
* `docs/MAC_MINI_CONTROL_PLANE_RUNBOOK.md` — on-device qualification
* `docs/NVIDIA_TRAINING_WORKER_RUNBOOK.md` — CUDA worker qualification
