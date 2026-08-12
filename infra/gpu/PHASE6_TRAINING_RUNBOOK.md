# Phase 6 — NVIDIA LoRA Training Runbook

For running `LUBER_LORA_PILOT_V1` on rented NVIDIA hardware.

**Do not train on the development Mac.** Phase 5 measured swap at
16.05 GB of 17.41 GB (92%) while running inference alone; Phase 2
measured the LM alone driving swap to 18.7 GB and free disk to 5.7 GB.
The Mac stays an inference and benchmarking machine.

This runbook is written but **not yet executed** — see "Preconditions",
which is currently blocked on authorized training data.

---

## 1. Hardware target

Upstream states ~17 GB VRAM during training and recommends 20 GB+
(`docs/en/LoRA_Training_Tutorial.md`). A 48 GB card gives comfortable
headroom for longer songs and batch > 1.

| Preference | GPU | VRAM | Why |
|---|---|---|---|
| **First choice** | L40S | 48 GB | Ada generation, widely available, cheap per hour |
| Equivalent | RTX 6000 Ada | 48 GB | Same class |
| Acceptable | A100 | 80 GB | Overkill for the pilot but fine |
| **Not required** | H100 | 80 GB | No pilot justification; do not pay for it |

Also provision: **≥ 200 GB disk** (10 GB weights + dataset + tensor
cache + checkpoints), and CUDA 12.x.

---

## 2. Environment setup

```bash
nvidia-smi                      # confirm GPU, driver, CUDA
python3 --version               # must be 3.11 or 3.12
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

## 3. Repository checkout at the pinned commit

The pin is not optional — training against a moving `main` makes the
run unreproducible and incomparable with the Phase 5 baseline.

```bash
git clone --filter=blob:none https://github.com/ace-step/ace-step-1.5.git
cd ace-step-1.5
git checkout 6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0
git rev-parse HEAD    # must print 6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0

uv sync
# 8-bit Adam saves VRAM; upstream falls back to plain AdamW with a
# warning when this is absent (observed on the Mac).
uv pip install bitsandbytes
```

## 4. Model download

Official mechanism only (`acestep/model_downloader.py`, HuggingFace with
ModelScope fallback). ~10.1 GB.

```bash
export ACESTEP_CHECKPOINTS_DIR=/workspace/checkpoints
uv run python -c "
from acestep.model_downloader import download_main_model
ok, msg = download_main_model()
print(ok, msg)
"
du -sh "$ACESTEP_CHECKPOINTS_DIR"/*
```

## 5. Dataset upload

Only a manifest that passed the rights and quality gates
(`packages/dataset`) may be uploaded. Verify the content hash on the
GPU host so the run cites the dataset it actually trained on.

```bash
rsync -avP ./trainset_v1/ user@gpu-host:/workspace/dataset/
```

Layout must match upstream's expectation (basename-matched):

```
/workspace/dataset/
├── track001.wav
├── track001.lyrics.txt        # exact lyrics, section tags, line breaks
├── track001.json              # caption, bpm, keyscale, timesignature, language
└── …
```

```bash
uv run python -c "
import json, hashlib, pathlib
m = json.load(open('/workspace/dataset/manifest.json'))
h = hashlib.sha256()
for t in sorted(m['tracks'], key=lambda t: t['track_id']):
    h.update(t['track_id'].encode()); h.update(t['audio_sha256'].encode())
print('recomputed:', h.hexdigest()); print('manifest  :', m['content_hash'])
assert h.hexdigest() == m['content_hash'], 'dataset hash mismatch'
"
```

## 6. Preprocessing

Training consumes pre-computed tensors, not raw audio.

```bash
export ACESTEP_CHECKPOINTS_DIR=/workspace/checkpoints
uv run python -m acestep.training.dataset_builder \
    --input /workspace/dataset \
    --output /workspace/tensors
```

Restart between preprocessing and training to release VRAM (upstream
notes this explicitly).

## 7. Training

Values follow the LoRA audit. Note two deliberate departures from the
code defaults, both justified in `docs/PHASE6_ACE_STEP_LORA_AUDIT.md`:

- **epochs 800, not the 100 default** — upstream's own guidance for a
  10–20 track set (finding L3).
- **`val_split` non-zero** — the default of 0.0 makes overtraining
  undetectable (finding L5).

```bash
uv run python train.py \
    --dataset /workspace/tensors \
    --output /workspace/runs/LUBER_LORA_PILOT_V1 \
    --lora-rank 8 \
    --lora-alpha 16 \
    --learning-rate 1e-4 \
    --batch-size 1 \
    --gradient-accumulation 4 \
    --max-epochs 800 \
    --save-every-n-epochs 50 \
    --val-split 0.1 \
    --seed 42 \
    --shift 3.0 \
    --mixed-precision bf16 \
    2>&1 | tee /workspace/runs/train.log
```

> Confirm exact flag names against `train.py --help` at the pinned
> commit before the first run. The values above are audited; the CLI
> spelling is not yet verified against a live checkout.

Resume is supported (`resume_from` in the trainer):

```bash
uv run python train.py … --resume-from /workspace/runs/LUBER_LORA_PILOT_V1/epoch_400
```

## 8. Checkpoint policy (Step 16 — do not assume later is better)

`save_every_n_epochs 50` over 800 epochs gives 16 checkpoints. Keep and
evaluate at minimum:

| Label | Epoch |
|---|---|
| early | ~100 |
| middle | ~400 |
| late | ~800 |

Watch the validation curve and the samples for: mode collapse, genre
overfitting, melodic repetition, loss of prompt adherence, degraded
instrumental quality, and vocal over-specialisation. A rising
validation loss with falling training loss means stop and take an
earlier checkpoint.

```bash
# Hash every checkpoint for the run manifest.
find /workspace/runs/LUBER_LORA_PILOT_V1 -name "*.safetensors" \
    -exec sha256sum {} \;
```

## 9. Benchmark inference on the GPU host

Run the same benchmark prompts, same seeds, so results are comparable
with `LUBER_BASELINE_P5_V1`.

```bash
export ACESTEP_LM_BACKEND=vllm         # CUDA; the Mac used mlx
uv run acestep-api --host 127.0.0.1 --port 8001 &

# From the LUBER repo, pointing at the GPU host's engine:
uv run python scripts/benchmark/run_ab_experiment.py \
    --config benchmarks/music_quality/configs/phase6_ab.json \
    --ace-step http://gpu-host:8001
```

The GPU host also unblocks the two experiments the Mac could not run:

- `GPU_REQUIRED_FOR_LM_BENCHMARK` — LM-enabled configuration
- `GPU_REQUIRED_FOR_XL_BENCHMARK` — `acestep-v15-xl-turbo`

Run both before concluding anything about base-model ceilings.

## 10. Persist artifacts before shutdown

The instance is ephemeral; everything worth keeping must leave it.

```bash
rsync -avP /workspace/runs/LUBER_LORA_PILOT_V1/ ./runs/LUBER_LORA_PILOT_V1/
rsync -avP /workspace/runs/train.log ./runs/
# Benchmark audio stays out of git; metadata and manifests come back.
```

## 11. Cost logging

Record in the run manifest (`TrainingRunManifest`): GPU model, hourly
rate, start and end time, GPU hours, and total cost. A pilot whose cost
is unknown cannot inform the decision about a larger run.

```bash
date -u +%Y-%m-%dT%H:%M:%SZ    # at start and at end
nvidia-smi --query-gpu=name,memory.total --format=csv
```

## 12. Shutdown

```bash
pkill -f acestep-api
# Verify artifacts are off the box, then destroy the instance.
```

**Stop the instance explicitly.** A 48 GB GPU left running overnight
costs more than the pilot it was rented for.

---

## Preconditions — current status

| Requirement | Status |
|---|---|
| Training pipeline audited | ✅ `docs/PHASE6_ACE_STEP_LORA_AUDIT.md` |
| Dataset rights gate implemented | ✅ `packages/dataset` |
| Dataset quality gate implemented | ✅ `packages/dataset` |
| Run-manifest reproducibility enforced | ✅ validated in code |
| GPU runbook | ✅ this document |
| **Authorized training audio supplied** | ❌ **BLOCKED** |

**No rights-cleared audio has been supplied to this project.** The
dataset pipeline is complete and tested, and it currently has nothing
lawful to ingest. Training does not start until the operator provides
audio with confirmed commercial ML training rights — and the gate will
reject anything that arrives without them.

To unblock, supply per track: the audio file, exact lyrics with section
tags, annotations (caption, bpm, keyscale, timesignature, language), and
a rights record naming the holder, a document reference, and a
confirmation date.
