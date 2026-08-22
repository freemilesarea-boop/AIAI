# Hardware execution compatibility

Where LUBER's workloads can run, what has been measured, and what is
still unknown.

The short version: **LUBER itself contains no CUDA-only code.** The
training loop belongs to ACE-Step, which accepts `mps` and has done
since the pinned commit. What was missing was a way for LUBER to *say*
so.

---

## 1. Two axes, kept apart

**ExecutionLocation** — `LOCAL` or `REMOTE`.
**ComputeDevice** — `CPU`, `MPS` or `CUDA`.

They are separate fields because they are separate questions. This
repository already has a local backend that uses no accelerator (the dry
run) and a remote backend whose device is whatever the worker reported —
which could be a CPU. `LOCAL` does not mean CPU and `REMOTE` does not
mean CUDA, and one enum would encode both falsehoods permanently.

## 2. What was actually measured

On the development machine — an **Apple M4 Pro, 24 GB, macOS**, probed
through the ACE-Step trainer's own interpreter (Python 3.12.11,
torch 2.10.0):

| Question | Answer | How |
|---|---|---|
| `torch.backends.mps.is_built()` | **True** | probed |
| `torch.backends.mps.is_available()` | **True** | probed |
| `torch.cuda.is_available()` | **False** | probed |
| Precision on MPS | fp32, fp16, **bf16** all work | each dtype allocated on the device and added to itself |
| Precision on CPU | fp32, fp16, bf16 | same |
| Tiny training on CPU | **PASS** | 8 steps: forward, backward, AdamW, cosine scheduler, gradient clip |
| Tiny training on MPS | **PASS** | same |
| Checkpoint MPS → CPU | **loads**, with optimizer state and adapter tensors | `map_location`, then a real optimizer step after loading |
| Checkpoint CPU → MPS | **loads** | same |
| Bounded 64 MiB allocation on MPS | allocated, reported by `torch.mps.current_allocated_memory()`, released | — |

The training smoke uses **a two-layer toy network on synthetic noise**.
It is not ACE-Step, not a DiT, and there is no music in it. It proves
the *mechanism* runs; it proves nothing about whether a real LoRA run
converges.

**No NVIDIA hardware has been measured.** Every CUDA statement in this
project is either logic exercised against a fixture or explicitly not
run.

### Benchmarks, and what they are not

A 512×512 matmul and a tiny forward/backward, on this machine, on this
day: CPU came out *faster* than MPS at that size. That is expected —
the workload is far too small to amortise accelerator dispatch — and it
is exactly why these numbers must not be read as a ranking. They exist
to catch "MPS is not actually being used" and "this path is broken".

**Nothing here may be extrapolated to a Mac mini.** An M4 Pro has a
different core and GPU configuration from a base M4. This machine is a
fair *compatibility* proxy for a 24 GB Mac mini and a poor performance
one.

## 3. What the trainer does with a device

Read from `acestep/training_v2/` at the pinned commit
`6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`.

`--device` accepts `auto | cuda | cuda:N | mps | xpu | cpu`, and `auto`
resolves cuda → mps → xpu → cpu. LUBER's `AUTO` mirrors that order
exactly, so the two cannot disagree about the same run.

Precision under `auto`:

| device | compute dtype | Fabric precision |
|---|---|---|
| `cuda`, `xpu` | `bfloat16` | `bf16-mixed` |
| `mps` | `float16` | `16-mixed` |
| everything else | `float32` | `32-true` |

An **explicit** precision is passed straight through by upstream without
any check that the device can do it. LUBER checks, against the probe.

### fp16 on MPS cannot train — measured in Phase 33

The table above is still what the trainer *does*. Phase 33 measured what
happens next, and on Apple silicon `auto` is not usable:

> `ValueError: Attempting to unscale FP16 gradients.`

`fixed_lora_module` loads the model in `float16` **and** drives Fabric
at `16-mixed`, and Fabric's `GradScaler` refuses to unscale gradients
that are themselves fp16. The run dies at the first gradient clip, after
the 2.4B model has loaded. This is a property of the trainer, not of the
hardware: Phase 32 measured fp16 tensors working on this machine's MPS
backend, and they do.

Measured on Apple M4 Pro / torch 2.10.0 / ACE-Step `6d467e4b`, by a
bounded Phase 33 canary:

| precision on MPS | real training |
|---|---|
| `bf16` | **works** |
| `fp32` | **works** |
| `fp16` | fails at the first clip |
| `auto` | fails — it resolves to `fp16` here |

`AUTO_BY_DEVICE` is unchanged, because it mirrors upstream and upstream
is unchanged. What is new is `luber_training.preflight.UNTRAINABLE_PRECISION`,
which **blocks** the combination with the measured reason attached. See
`docs/TRAINING_PREFLIGHT_AND_CANARY.md` §8.

## 4. Workload classes and where they go

| Workload | Default policy | Why |
|---|---|---|
| `PREPROCESS` | `CPU_ONLY` | `luber_dataset` is numpy and ffmpeg. No tensors |
| `EVALUATION` | `CPU_ONLY` | drives an ACE-Step server over HTTP; the server's device is the server's business |
| `INFERENCE` | `LOCAL_PREFERRED` | the ARQ generation worker |
| `CHECKPOINT_VALIDATION` | `LOCAL_PREFERRED` | needs a device that holds tensors, which is every device |
| `LIGHT_FINE_TUNE` | `MPS_ALLOWED` | small runs whose point is finding out whether the pipeline works |
| `HEAVY_TRAINING` | `REMOTE_CUDA_PREFERRED` | the only thing that wants a rented GPU |

**Heavy training is refused when no CUDA worker is reachable.** It is
not moved to Apple silicon. That would produce a model: the run
completes, the checkpoint looks like every other checkpoint, and the
only record that it trained somewhere nobody chose is a wall-clock time
nobody compares. A caller who genuinely wants a local run passes
`allow_local_fallback`, and the decision records it.

## 5. What must never be silent

- **`adamw8bit` off CUDA is BLOCKED.** ACE-Step's `optim.py` imports
  `bitsandbytes.optim.AdamW8bit` and, on `ImportError`, **logs a warning
  and uses AdamW**. The run would finish having trained with a
  different optimizer than its own plan records.
- **An unsupported precision is BLOCKED**, not downgraded.
- **A requested device that is absent is BLOCKED**, not substituted.
  Asking for CUDA on a Mac gets a refusal, never a CPU.
- **CPU fallback for an MPS gap is off by default** (`MPS_STRICT`), and
  where a non-training workload permits it, every use is recorded on the
  decision. `PYTORCH_ENABLE_MPS_FALLBACK` is **not set anywhere in this
  repository** and must not be set globally — its whole effect is to
  make the substitution invisible.

## 6. Memory

A machine with 24 GB does not have 24 GB for PyTorch. Apple's unified
memory is shared with the GPU, and on the planned topology the same
machine is also the API, Postgres, Redis and the orchestrator.

So a budget reserves a fraction (30% by default, 40% when the machine
also runs the control plane) with a floor of 4 GB, and both are
configurable.

**Feasibility is `UNKNOWN` and says so.** Nothing has measured what the
2B DiT plus optimizer state plus activations needs. Writing "18.4 GB"
would invent the number every scheduling decision depends on. `UNKNOWN`
never reads as a pass — it means the first run on new hardware *is* the
measurement.

Upstream's `vram_8gb.json` … `vram_24gb_plus.json` presets are indexed
by dedicated NVIDIA VRAM. Apple unified memory is a different resource
shared with the operating system, and reading one as the other is the
specific mistake to avoid.

**Local training concurrency is 1.** The Mac is the 24/7 control plane
before it is a trainer.

## 7. Reproducibility across devices

`HardwareRequirements.execution_device` is part of the plan, so it is
part of `TrainingPlan.digest()`. A plan compiled for MPS and one
compiled for CUDA hash differently, which is correct: they are not the
same experiment.

**Same seed on MPS and on CUDA does not mean bit-identical training.**
Different backends use different kernels, different reduction orders and
different default dtypes — `auto` alone is bf16 on CUDA and fp16 on MPS.
This project makes no claim of bitwise reproducibility across hardware
backends and does not test for one. What is reproducible is the *plan*:
same digest means same intent.

Hardware measurements deliberately stay **out** of the plan digest. Free
disk, torch patch version and GPU utilisation are properties of a
machine at a moment; hashing them would make every plan unique.

## 8. Probing an interpreter that is not this one

LUBER's own environment has no `torch` — the control plane never imports
it. The trainer's virtualenv does. A probe restricted to its own process
would report "no accelerator" on the exact machine that can train.

So `luber_hardware/_facts.py` has no LUBER imports and is executed as a
script by whichever interpreter is named:

```
python -m luber_hardware probe --python ~/ace-step-1.5/.venv/bin/python
```

Set `TRAINING_PYTHON_EXECUTABLE` and the operator console's compute
panel does the same thing.

## 9. Privacy

A capability report has **no field** for a hostname, a username, a home
directory, a serial number or a MAC address. Not "we strip them" —
there is nowhere to put one. `cpu_model` is a marketing name
("Apple M4 Pro"), which describes what the machine can do and cannot be
traced to a person.

The capability digest covers operational facts only, so two identically
configured machines produce the same digest. That is deliberate: it
answers "would this run the same way here", not "which box is this".

## 10. For a planned 24 GB Apple Silicon Mac mini

Evidence-based, from this repository. Not a purchasing recommendation —
what the machine would be *able to do*, and what nobody knows yet.

### CONFIRMED — verified on Apple silicon with 24 GB

- **24/7 control plane**: API, PostgreSQL, Redis, the ARQ queue. None of
  it touches an accelerator.
- **Generation worker**: calls ACE-Step over HTTP. No torch in the
  process at all.
- **Dataset preparation**: `luber_dataset` is numpy and ffmpeg.
- **Audio preprocessing and finishing**: numpy, `wave`, ffmpeg.
- **Evaluation**: HTTP plus arithmetic in pure Python.
- **Training orchestration**: compiling plans, staging runs, driving
  remote workers, collecting checkpoints.
- **Checkpoint inspection**: verified — a checkpoint written on CUDA or
  MPS loads here, optimizer state included.
- **The operator console**: non-production deployments only.
- **The training *mechanism* on MPS**: forward, backward, AdamW,
  scheduler, gradient clip, checkpoint save and load, all verified on a
  toy network. fp32, fp16 and bf16 all compute.

### LIKELY — supported by evidence, not yet run end to end

- **Light LoRA fine-tuning on MPS.** The trainer accepts `--device mps`
  and defaults it to fp16; the mechanism is verified. What has not been
  done is a real ACE-Step LoRA run.
- **Local iteration on small datasets**, for finding pipeline problems
  before spending GPU-hours on them.

### UNSUPPORTED here — CUDA or remote only

- **Heavy adapter training.** Policy prefers a remote CUDA worker and
  refuses a local fallback unless explicitly asked.
- **`adamw8bit`.** bitsandbytes has no Metal build, and upstream falls
  back to AdamW with a warning rather than failing.
- **Multi-GPU / DDP.** `num_devices > 1` is a CUDA arrangement.
- **NCCL collectives.** CUDA-only, with no Metal equivalent.
- **Fused AdamW**, which is a CUDA kernel. The unfused path works; it is
  slower, not wrong.

### UNKNOWN until real hardware exists

- **Whether ACE-Step LoRA converges on MPS.** Nobody has run it.
- **Whether the 2B DiT plus optimizer state plus activations fits in
  24 GB of unified memory.** No measurement exists, and the memory model
  reports `UNKNOWN` rather than a number.
- **How long a real epoch takes on Apple silicon**, in absolute terms or
  relative to any GPU.
- **Anything at all about an M4 Mac mini specifically.** Development
  happens on an M4 Pro, which shares the memory size and not the core or
  GPU configuration. It is a compatibility proxy and not a performance
  one.
- **Every NVIDIA figure**, including VRAM sufficiency, because no NVIDIA
  hardware has been attached to this project.

---

## 11. Relationships and limits

**Phase 27 (remote execution).** Untouched. Its `WorkerCapabilities`
wire format, its preflight and its backend are unchanged; Phase 32
*translates* a worker record into a capability rather than replacing it.
Placement chooses `REMOTE + CUDA`; Phase 27 executes it.

**Phase 33 (training preflight and canary).** Builds directly on this
one. Placement chooses a location and a device; the preflight proves the
chosen machine can execute the plan, and the canary demonstrates it by
running a bounded real training step. Phase 33 added one measured fact
back into this document — fp16 on MPS cannot train — and changed no
placement semantics. See `docs/TRAINING_PREFLIGHT_AND_CANARY.md`.

**Phase 34 (memory profiling and capacity qualification).** Builds on
Phase 33. It measures what a configuration costs from inside the real
trainer process and turns that into a capacity qualification the
preflight consumes. It added no placement semantics and changed no
device rule. Apple figures are unified memory throughout — never VRAM.

**Phase 31 (provider resilience).** A different layer answering a
different question. `ProviderRouter` decides which *generation provider*
answers a user's request. Execution placement decides where a
*training or evaluation workload* runs. Nothing flows between them.

**Remote generation does not exist.** Phase 27's remote execution covers
training and checkpoints. Generation still runs through the local ARQ
worker. Nothing in this phase changes that, and nothing here should be
read as implying otherwise.

### Known limits

1. **No NVIDIA hardware has been tested.** CUDA placement logic is
   exercised against fixtures; no GPU benchmark, VRAM figure or model
   name is claimed anywhere. Phase 33 did not change this.
2. **A real ACE-Step LoRA now runs on MPS — bounded.** Phase 33's canary
   loaded the real DiT in bf16, injected a real LoRA, took an optimizer
   step, wrote a checkpoint and resumed from it. It trained on **two
   synthetic tensors for one epoch**. Whether ACE-Step LoRA *converges*
   on Apple silicon is still an open experiment: a canary proves the
   mechanism and nothing about the model.
3. **Memory feasibility was UNKNOWN for every real workload.** Phase 34
   measured it on this M4 Pro, from inside the trainer process, at
   production sequence length: **9 855 MiB of unified memory** for a
   rank-32 LoRA at 6000 latent frames, bf16, micro batch 1. That is a
   `SAMPLED_PEAK` — the pinned torch has no MPS peak counter — and it
   qualifies **only that configuration on that machine**. Everything
   else is still UNKNOWN. See `docs/TRAINING_MEMORY_CAPACITY.md`.
4. **No local training backend.** `LocalDryRunBackend` trains nothing by
   design. Phase 33's canary starts the trainer directly, under hard
   bounds; it is not a backend and cannot run a real job.
5. **Thresholds and policies are code.** Changing the headroom fraction
   or a workload's default policy is a deploy.
