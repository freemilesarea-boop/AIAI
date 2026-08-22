# NVIDIA training worker runbook

Preparing a rented Linux GPU host as a LUBER training worker.

`docs/FIRST_REMOTE_GPU_CONNECTION.md` is the step-by-step for the *first*
connection, written when Phase 27 was built. This document is the
hardware-facing companion: what the machine has to be, how to prove it
is, and what Phase 32's placement layer does with the answer.

**No cloud provider is assumed.** Nothing here depends on RunPod,
Lambda, Vast or any particular vendor — the worker is a Linux host with
a GPU, an SSH port and a disk.

**No secrets in this document.** No key, no token, no host, no path from
anybody's machine.

---

## 1. What the host has to be

| Requirement | Why |
|---|---|
| Linux, x86_64 | what the trainer and the CUDA wheels target |
| An NVIDIA GPU with a working driver | `nvidia-smi` must answer |
| CUDA-enabled PyTorch | a driver alone proves nothing; see §4 |
| Python 3.11 or 3.12 | ACE-Step's declared range |
| Enough disk for a staged dataset, checkpoints and logs | the run writes all three |
| SSH with key auth | how the control plane reaches it |

**VRAM: no figure is stated here**, because nobody has measured what
LUBER's configuration needs. Upstream's presets range from `vram_8gb`
to `vram_24gb_plus`, which is a hint about the trainer's own defaults
and not a measurement of this project's plans. The preflight reports
VRAM sufficiency as `UNKNOWN` rather than inventing a threshold, and the
first real run is the measurement.

## 2. Driver and CUDA

```bash
nvidia-smi
```

Answers the model, the driver version and the memory. If it does not,
nothing further will work.

`nvidia-smi` describes the *hardware*. It does not tell you whether this
Python can reach it, and a driver can be installed beside a torch built
without CUDA. That is why the worker's own probe asks torch and lets
torch win where the two disagree.

## 3. Python environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <the repository> ~/luber
cd ~/luber
git checkout <the exact commit the control plane will send>
uv sync --all-packages
```

The commit matters. The plan carries a code version, and a worker on a
different commit is running something other than what was approved.

## 4. Prove CUDA rather than assume it

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

`True` and a non-zero count, or this host is not a training worker
whatever `nvidia-smi` said.

Then the normalized probe:

```bash
uv run python -m luber_hardware probe
uv run python -m luber_hardware verify        # a real tiny training step
```

`verify` runs eight steps of a toy network — forward, backward, AdamW,
scheduler, gradient clip — plus a checkpoint round trip and a bounded
64 MiB allocation. It downloads nothing, loads no model weights and
takes seconds. It is the cheapest way to learn that a machine is
mis-provisioned before staging a dataset onto it.

## 5. ACE-Step

Install the trainer at the pinned commit
(`6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`) as
`docs/FIRST_REMOTE_GPU_CONNECTION.md` §6 describes, and record its root
in the worker config. LUBER compiles a command *for* that tree; it does
not vendor it and does not modify it.

If the trainer's environment is separate from LUBER's — it usually is —
probe that one:

```bash
uv run python -m luber_hardware probe --python /path/to/trainer/venv/bin/python
```

## 6. Worker registration

```bash
uv run python -m luber_training.remote init --root ~/luber-worker
uv run python -m luber_training.remote probe
uv run python -m luber_training.remote identity
```

`probe` writes the capability record the control plane reads. A machine
that has never been probed cannot satisfy a CUDA requirement by virtue
of nobody having checked — the preflight treats unmeasured as
unsatisfied, deliberately.

Register it from the control plane with the identity and the capability
record. Then:

```bash
uv run python -m luber_training worker list      # from the control plane
uv run python -m luber_hardware readiness
```

The worker should appear as a `REMOTE + CUDA` target, `READY`.

## 7. What placement does with it

Once the worker is registered and probed:

- `HEAVY_TRAINING` prefers `REMOTE + CUDA` and lands on this machine.
- `auto` precision resolves to **bf16** on CUDA, matching the trainer's
  own default.
- `PREPROCESS` and `EVALUATION` stay on the control plane. Sending them
  here would spend rented GPU-hours on numpy and ffmpeg.

With no CUDA worker registered, heavy training is **BLOCKED** rather
than moved to the Mac. That refusal is the feature.

## 8. Staging, training, collecting

Phase 27 owns all three and this phase changed none of it:

```bash
# from the control plane. `--help` on each for the full argument list —
# staging needs the dataset and curation builds, and dispatch needs the
# connection.
uv run python -m luber_training remote run stage         --run-id <run-id> ...
uv run python -m luber_training remote run dispatch      --run-id <run-id> ...
uv run python -m luber_training remote run remote-status --run-id <run-id> ...
uv run python -m luber_training remote run collect       --run-id <run-id> ...
```

The worker verifies what arrived before running it, and reports state it
observed rather than state it assumed — a trainer process that vanished
without this worker owning it yields an unknown exit code and a state
that says so.

## 9. Checkpoints come back to the Mac

Collected checkpoints load on Apple silicon and on a CPU. That is
measured, not assumed: a checkpoint written on one device and loaded on
another — model, optimizer state and adapter tensors, with a real
optimizer step taken after loading — is part of this phase's test suite.

So the intended topology works: **the GPU host trains, the Mac
evaluates.** Evaluation needs no GPU at all; it drives an ACE-Step
server over HTTP and computes in pure Python.

## 10. Shutdown

Rented hardware costs money while it exists.

```bash
uv run python -m luber_training remote run collect --run-id <run-id>   # first
uv run python -m luber_training.remote cleanup --run-id <run-id>       # on the worker
```

Collect before destroying. A checkpoint left on a machine that has been
returned is gone, and the run that produced it cost more than the
transfer would have.

Nothing in LUBER provisions or de-provisions hardware. Renting and
releasing a machine is a human action taken with the provider, on
purpose.

## 11. What has not been done

**No NVIDIA hardware has ever been attached to this project.** Every
CUDA path in this repository is exercised against fixtures. This runbook
is written from the code that will run and from the trainer's own
source; the first person to follow it end to end will be the first
person to test it, and should expect to correct it.
