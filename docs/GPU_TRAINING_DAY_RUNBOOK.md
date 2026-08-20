# GPU Training Day Runbook

What the operator does when the first NVIDIA server becomes available.

The point of Phase 25 is that this list contains no architecture work.
Provision, register, verify, smoke, then run.

No provider is named and no credentials appear anywhere in this
document.

---

## Before you rent anything

Two things must already exist, and neither is created on GPU day:

1. **An approved, frozen dataset** — a Phase 23 build with
   `dataset_lock.json`, and a Phase 24 curation with
   `curation_lock.json`. If the curation was computed against a
   different dataset lock, the gate will refuse it, and that is easier
   to fix at home than on a metered machine.
2. **An evaluation-only list** — the track ids that must never enter
   training, P20 benchmark material above all. Without this file the
   leakage gate has nothing to protect.

Run `python -m luber_training run validate` locally against the dry-run
backend first. Every gate except the worker check can pass before a
server exists, and each one that fails costs nothing to fix now.

---

## 1. Provision the host

Choose from the checklist in
[`GPU_PROVIDER_CHECKLIST.md`](GPU_PROVIDER_CHECKLIST.md).

Note the hourly rate and the shutdown behaviour **before** starting.
A host that bills while idle and one that terminates on disconnect fail
in opposite directions.

---

## 2. Clone the exact LUBER commit

```bash
git clone <repo> luber && cd luber
git checkout <the exact commit you intend to train from>
git status --porcelain    # must be empty
```

A dirty tree fails preflight. "That commit plus whatever was in the
editor" is not a revision anyone can reproduce, and the run record would
name a revision that never existed.

---

## 3. Install the pinned environment

```bash
uv sync --all-packages
```

---

## 4. Install and verify the pinned ACE-Step

```bash
git clone <ace-step> ~/ace-step-1.5 && cd ~/ace-step-1.5
git checkout 6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0
git rev-parse HEAD        # must match exactly
```

The training config records this commit and refuses to validate against
a different one. If upstream has moved, **re-run the capability audit**
before training rather than adjusting the constant — the flags this
project compiles were read from that tree, and a changed parser means
silently-ignored parameters.

---

## 5. Probe the worker

```bash
python -m luber_training worker probe --output worker_capabilities.json
cat worker_capabilities.json
```

Confirm `worker_class` is `GPU_TRAINING_READY` and that
`cuda_available` is `true`. If it is `null`, torch cannot see CUDA —
fix that before going further. A `null` is not a small problem to work
around; the scheduler treats an unmeasured capability as unsatisfied,
and it is right to.

---

## 6. Register the worker

```bash
python -m luber_training --registry ./training-registry worker register \
    --name gpu-host-1 --backend remote-gpu \
    --host-identity <stable host identifier> \
    --capabilities worker_capabilities.json \
    --credential-ref <the NAME of a credential, never its value>
```

The probe's own classification is used. A worker does not become
GPU-ready by being registered with an optimistic flag.

---

## 7. Transfer the approved locked dataset

Move the dataset build, the curation build and the approved audio.

Transfer only what curation selected. Everything else is material the
gates already refused, and it does not belong on a rented machine.

---

## 8. Verify the digests on the host

```bash
python -m luber_dataset.factory verify --output ./dataset-build
python -m luber_dataset.factory verify-curation \
    --output ./curated-build --manifest ./dataset-build/dataset_manifest.jsonl
```

A digest that changed in transit is a corrupted transfer, and training on
it would produce a run nobody can reproduce.

---

## 9. Run the preflight

```bash
python -m luber_training --registry ./training-registry run create \
    --experiment-id <exp_…> --preset SMOKE --backend remote-gpu \
    --worker-id <wrk_…> \
    --dataset-build ./dataset-build --curation-build ./curated-build

python -m luber_training --registry ./training-registry run validate \
    --run-id <run_…> --worker-id <wrk_…> \
    --dataset-build ./dataset-build --curation-build ./curated-build \
    --evaluation-only ./evaluation_only.txt
```

Read the preflight report. **`UNKNOWN` is not `PASS`.** Two entries are
expected to be unknown on the first ever run:

- `minimum_vram_mb` — no VRAM figure has been measured for any LUBER
  configuration
- `disk_capacity` — no checkpoint size has been measured

That is the honest state today. The SMOKE run is what turns both into
measurements.

---

## 10. Execute SMOKE first, and only SMOKE

SMOKE is one epoch at rank 4. **It teaches the model nothing, and its
checkpoint is worthless.** That is the point: it proves the data loads,
the adapter injects, a step runs and a checkpoint writes.

Watch for:

- the adapter injecting onto `q_proj k_proj v_proj o_proj`
- a step completing without OOM
- `adapter_config.json` and `adapter_model.safetensors` appearing
- the checkpoint reaching `READY` — meaning it validated and hashed

**Record the numbers this run finally makes real**: peak VRAM,
checkpoint size on disk, seconds per step. Put them into the training
config's hardware requirements so the next preflight can check rather
than shrug.

If SMOKE fails, stop. Every failure it can produce is cheaper here than
in a long run.

---

## 11. Verify the checkpoint lifecycle

```bash
python -m luber_training --registry ./training-registry checkpoint list --run-id <run_…>
python -m luber_training --registry ./training-registry run bundle --run-id <run_…>
```

Confirm the checkpoint is `READY`, its digest is recorded, and the
bundle names the dataset lock, curation lock, config, plan, environment
and repository commit. If the bundle cannot explain the run today, it
will not explain it in six months.

---

## 12. Only then, the first controlled experiment

```bash
python -m luber_training --registry ./training-registry run create \
    --experiment-id <exp_…> --preset LORA_SMALL --backend remote-gpu ...
```

Then `run validate`, then `run start`.

`LORA_SMALL` before `LORA_STANDARD`. The question a first real run
answers is whether the dataset moves the model at all, and a short run
answers it for a fraction of the cost.

---

## 13. Afterwards

- A `COMPLETED` run does **not** mean the model improved. Create an
  `EvaluationCandidate` and stop.
- **Nothing is promoted here.** Promotion needs evaluation evidence,
  which is the next phase.
- Back up checkpoints, metrics and the run bundle **before** destroying
  the host. A terminated instance takes its disk with it.
- Record the actual cost against the run.

---

## If something goes wrong

| Symptom | Likely cause | Action |
|---|---|---|
| `RIGHTS_GATE_FAILED` | a selected track lacks explicit permission | fix the sidecar and re-curate. There is no override, by design |
| `EVALUATION_LEAKAGE` | benchmark material in the selection | check the evaluation-only list; the gate matches on digest too |
| `DATASET_LOCK_INVALID` | manifest changed after freezing, or a bad transfer | re-verify locally, re-transfer |
| `CODE_VERSION_DIRTY` | uncommitted changes on the host | commit or stash; do not use `--allow-dirty` for a real run |
| `INSUFFICIENT_HARDWARE` | probe reports less than the plan needs | re-probe; do not edit the requirement to fit |
| run stuck `RUNNING`, worker silent | lost contact | it becomes `LOST`, not `FAILED`. **Check the host before assuming training stopped** — it may still be billing |
| OOM | rank, batch size or gradient checkpointing | reduce rank first; a new run, not an edited one |

---

## What this runbook does not do

- It does not recommend a provider or quote a price.
- It does not claim any VRAM or throughput figure. Nothing here has been
  measured on NVIDIA hardware by this project.
- It does not improve model quality. Phase 25 improves reproducibility,
  safety and operability; the first run may well produce a worse model,
  and the evaluation phase exists to find out.
