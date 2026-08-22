# Phase 32 audit — what this repository actually needs from hardware

Written before the code, from the repository and from the pinned
ACE-Step clone, not from expectations about what a training project
usually looks like.

Four questions decide this phase:

1. Where does LUBER actually touch a compute device?
2. What does LUBER itself require CUDA for?
3. What does the *trainer* require, and does it know about Apple MPS?
4. What can this Mac actually do, measured rather than assumed?

The third answer is the one that changes the plan.

---

## 1. Where LUBER touches a device

Three files in the whole repository import `torch`, and all three do it
inside a `try` for the purpose of *reporting* hardware:

| File | Purpose |
|---|---|
| `packages/training/src/luber_training/probe.py` | local worker capability probe |
| `packages/training/src/luber_training/remote/capabilities.py` | Phase 27 remote worker probe |
| `packages/training/src/luber_training/plan.py` | records `torch_version` / CUDA version in the environment lock |

**`torch` is not a declared dependency of any package**, and it is not
installed in the monorepo's own environment:

```
$ uv run python -c "import torch"
ModuleNotFoundError: No module named 'torch'
```

That is not an oversight to correct. LUBER is a control plane: it
compiles a plan, stages it, and asks a trainer to execute it. The
trainer has its own environment — `~/ace-step-1.5/.venv`, torch 2.10.0 —
and *that* is the interpreter training runs through. A capability probe
that only ever asked its own interpreter would answer the wrong
question on the control plane, so Phase 32's probe must be able to ask a
**named interpreter**, not just itself.

## 2. What LUBER requires CUDA for

The Step 10 sweep across `packages`, `apps`, `services`, `docs`:

| Pattern | Hits | What they are |
|---|---|---|
| `torch.cuda` | 11 | 8 inside the two guarded probes; 1 in a synthetic test fixture's fake log line; 2 in docs |
| `bitsandbytes` | 3 | all in `docs/PHASE6_ACE_STEP_LORA_AUDIT.md`, describing *upstream's* optional 8-bit Adam |
| `.cuda(`, `device="cuda"`, `autocast`, `GradScaler`, `flash_attn`, `xformers`, `triton`, `nvcc`, `NCCL`, `CUDA_HOME`, `deepspeed`, `accelerate` | **0** | — |
| `PYTORCH_ENABLE_MPS_FALLBACK` | **0** | the project does not depend on it (Step 13) |

**LUBER contains no CUDA-only code.** There is no model forward, no
backward, no optimizer and no autocast anywhere in this repository — the
training loop belongs to ACE-Step.

There is exactly one place where LUBER *decides* a device, and it is a
hard-coded ternary:

```text
# packages/training/src/luber_training/trainer_adapter.py
"--device",
"cuda" if plan.requirements.requires_cuda else "cpu",
```

`HardwareRequirements.requires_cuda` defaults to `True`, and there is no
value of it that produces `mps`. **Today, LUBER cannot express a local
Apple training run at all** — not because the trainer cannot do it, but
because the command compiler has only two branches. That single line is
the reason this phase exists.

### Other packages

| Package | Imports | Needs a GPU? |
|---|---|---|
| `luber_evaluation` | stdlib only | **No.** It drives a running ACE-Step server over HTTP; whichever device *that server* uses is the server's business |
| `luber_dataset` | `numpy`, `subprocess` (ffmpeg) | **No** |
| `luber_audio_utils`, `luber_audio_finishing` | `numpy`, `wave`, ffmpeg | **No** |

So evaluation, dataset preparation and audio preprocessing are already
CUDA-free, and nothing needs to be moved off the GPU because nothing was
ever on it (Steps 34–36).

## 3. What the trainer requires — and it knows about MPS

Read from the pinned clone at `6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`.

**The trainer accepts `mps` explicitly.**
`acestep/training_v2/cli/args.py`:

```
--device   Device: auto, cuda, cuda:0, mps, xpu, cpu (default: auto)
--precision  auto | bf16 | fp16 | fp32
```

**Its `auto` order is cuda → mps → xpu → cpu** (`gpu_utils.resolve_gpu`),
and it names an MPS device `"Apple MPS"` rather than failing.

**It has a per-device precision default** — this is the authoritative
matrix, not something Phase 32 invents
(`training_v2/fixed_lora_module.py`):

| device | `auto` compute dtype | `auto` Fabric precision |
|---|---|---|
| `cuda`, `xpu` | `torch.bfloat16` | `bf16-mixed` |
| `mps` | `torch.float16` | `16-mixed` |
| `cpu` and anything else | `torch.float32` | `32-true` |

An **explicit** `bf16` is passed through unchanged on every device,
including MPS. Upstream does not check whether the device can actually
do it. That is a real trap: a config asking for `bf16` on Apple silicon
is accepted by the trainer and validated by nothing. Phase 32 has to
answer that question from torch before the run starts, because upstream
will not.

**`adamw8bit` is CUDA-shaped and fails soft.** `training_v2/optim.py`
imports `bitsandbytes.optim.AdamW8bit` and, on `ImportError`, **logs a
warning and silently uses AdamW instead**. A run configured for 8-bit
Adam on a Mac would therefore train with a different optimizer than the
one recorded in its own plan. That is precisely the class of silent
substitution this project refuses elsewhere, so it belongs in a
pre-flight refusal rather than in a log line nobody reads.

**Gradient checkpointing is on by default** upstream, and its VRAM
presets (`vram_8gb.json` … `vram_24gb_plus.json`) are indexed by VRAM —
CUDA-shaped language. Apple unified memory is not VRAM, and the presets
should not be read across.

**Full fine-tuning does not exist here.** Phase 20 established that the
only executable strategy is **LoRA/LoKr on DiT attention projections**.
Phase 32 plans around that and nothing larger.

## 4. What this machine is

Measured, not assumed:

| Fact | Value |
|---|---|
| Chip | Apple M4 Pro |
| Cores | 14 (10 performance + 4 efficiency) |
| Unified memory | 24 GiB |
| Architecture | arm64 |
| macOS | 26.3.1 |
| Trainer interpreter | Python 3.12.11, torch 2.10.0 |
| `torch.backends.mps.is_built()` | **True** |
| `torch.backends.mps.is_available()` | **True** |
| `torch.cuda.is_available()` | **False** |

This machine is **not a Mac mini** and must never be labelled as one
(Step 46). It is an Apple Silicon Mac with 24 GB of unified memory,
which makes it a useful *compatibility* proxy for a planned 24 GB Mac
mini and **not** a performance proxy: an M4 Pro has a different core and
GPU configuration from a base M4. Nothing in this phase may extrapolate
a throughput number from here to there.

There is no NVIDIA hardware attached to this project. Every CUDA claim
in this phase is therefore either **synthetically verified** (placement
logic driven by a mocked capability report) or **NOT RUN**. No
benchmark, VRAM figure, or GPU model may be reported for hardware that
does not exist.

## 5. What Phase 27 already built, and must keep

`remote/capabilities.py` is a full CUDA capability probe with a
worker-class classification, a protocol, and preflight arithmetic in
`backends.capability_check`. It is deliberately pessimistic: an
unmeasured capability does **not** satisfy a requirement.

Phase 32 must not rewrite any of it. Two consequences:

1. `WorkerCapabilities` is CUDA/VRAM-shaped and belongs to the remote
   protocol. Phase 32's normalized capability model is a **separate**
   structure that can be *derived from* it, not a replacement — changing
   the wire shape would be a Phase 27 change for no benefit.
2. `capability_check` stays the authority on "can this worker run this
   plan". Placement decides *where a workload goes*; the existing
   preflight still decides whether the chosen worker can hold it.

## 6. What the plan hash already carries

`TrainingPlan.digest()` hashes `canonical_dict()`, which includes
`requirements` (with `requires_cuda` and `supported_precision`) and
`config` (with `precision`). So device *semantics* already move the plan
identity, and a device-class field added to requirements flows into the
digest without new machinery (Step 29).

What must **not** go in: the capability report. Free disk, GPU
utilisation and torch patch versions are properties of a machine at a
moment, not of the training being requested, and hashing them would make
every plan unique.

Note the worker rebuilds the plan from JSON in
`remote/worker.py::_compile`, field by field. Anything added to
`HardwareRequirements` has to be read back there too, or a remote run
would silently train with the default.

## 7. Decisions this audit forces

1. **A separate `luber_hardware` package.** It has no business inside
   `luber_training`: evaluation and preprocessing get placed too, and
   the training package already depends on enough.
2. **The probe must be able to ask another interpreter.** The control
   plane has no torch; the trainer does. A probe restricted to its own
   process would report "torch not installed" on the exact machine that
   can train.
3. **Location and device stay separate axes** (Steps 2–3). This
   repository already has a local backend that is not CPU (dry run,
   which uses nothing) and a remote backend whose device is whatever the
   worker reported. Collapsing them would encode a false equivalence.
4. **`bf16` on MPS must be resolved against torch, not accepted.**
   Upstream passes it straight through.
5. **`adamw8bit` off CUDA must be BLOCKED, not warned.** Upstream's
   silent fallback changes what was trained.
6. **No local training backend exists.** `LocalDryRunBackend` trains
   nothing by design and registers `MOCK` artifacts. Phase 32 adds the
   ability to *express and validate* a local device, and stops short of
   building a second orchestrator.

## 8. Not verified, and deliberately not assumed

- Whether ACE-Step LoRA training **converges** on MPS. Nothing here has
  run it, and the smoke tests in this phase use a toy model, not the
  2B DiT.
- Whether the 2B DiT plus optimizer state plus activations fit in 24 GB
  of unified memory. No measurement exists, so the memory model reports
  `UNKNOWN` rather than a number.
- Any NVIDIA performance figure whatsoever.
- Whether MPS and CUDA produce comparable training results from the same
  seed. They will not be bit-identical; whether they are *equivalent* is
  an experiment nobody has run.
