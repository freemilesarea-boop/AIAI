# GPU Provider Checklist

Provider-neutral. This deliberately names no vendor and quotes no price:
both change faster than documentation, and a stale recommendation is
worse than none.

Work through it *before* renting. Most of these are cheap to check up
front and expensive to discover mid-run.

---

## Hardware

- [ ] **NVIDIA GPU.** The trainer's CUDA path is the only one this
      project will use. `--device` also accepts `mps` and `xpu`; neither
      has been verified here.
- [ ] **VRAM.** No figure is recommended, because none has been
      measured. The SMOKE run exists to produce the first real number —
      until then treat VRAM as an open question rather than a spec.
- [ ] **GPU count.** One unless a plan sets `num_devices > 1`. Multiple
      GPUs means one bigger run, not several concurrent models.
- [ ] **BF16 support.** The probe reports it. A plan requesting `bf16`
      on hardware without it is refused rather than silently downgraded.
- [ ] **CPU and RAM** sufficient for the data loader. `num_workers`,
      `prefetch_factor` and `pin_memory` all cost host memory.

## Storage

- [ ] **Capacity** for the dataset, the preprocessed tensors *and* the
      checkpoints. Preprocessing writes `.pt` tensors that can exceed
      the source audio.
- [ ] **A persistent volume**, or an explicit plan to move checkpoints
      off before termination. A terminated instance takes its disk.
- [ ] **Throughput.** A slow volume starves the GPU, and you pay for the
      GPU either way.

## Network

- [ ] **Inbound transfer** of the dataset — how long, and is it billed.
- [ ] **Egress cost** for pulling checkpoints back. Frequently the
      surprise line on the invoice.
- [ ] **Bandwidth limits or caps.**

## Access

- [ ] **SSH access**, or a documented alternative. The remote backend
      contract assumes a transport it can resolve from a
      `credential_ref`.
- [ ] **Credential handling.** Store the *name*; never put a key into a
      plan, a run record, a log or the repository.
- [ ] **Session persistence** — does a dropped connection kill the
      process. If so, run under something that survives it.

## Software

- [ ] **CUDA runtime compatible with the pinned torch.** Verify with the
      probe, not with the provider's marketing page.
- [ ] **Driver version** the probe can read.
- [ ] **Python version** matching the pinned environment.
- [ ] **Root or container permissions** sufficient to install
      dependencies.

## Billing and lifecycle

- [ ] **Hourly rate and billing granularity** — per second, per minute,
      per hour. Recorded against the run afterwards; nothing fetches it
      automatically.
- [ ] **Idle billing.** Does a stopped-but-not-terminated instance cost
      money.
- [ ] **Shutdown behaviour.** Does terminating destroy the volume.
- [ ] **Spot or preemptible?** If so:
      - the run must tolerate interruption
      - checkpoints must be frequent enough to lose little
      - a preempted run becomes `LOST`, not `FAILED` — we know we lost
        contact, not that training stopped
- [ ] **Quota and availability** for the instance type, in the region
      you want.

## Before the first real run

- [ ] Probe reports `GPU_TRAINING_READY`.
- [ ] Dataset and curation digests verified **on the host**.
- [ ] `SMOKE` completed and its checkpoint reached `READY`.
- [ ] Peak VRAM, checkpoint size and step time recorded — turning three
      `UNKNOWN_REQUIREMENT` entries into measurements.
- [ ] Checkpoint backhaul tested. Discovering egress is broken after a
      long run is the expensive way to learn it.
