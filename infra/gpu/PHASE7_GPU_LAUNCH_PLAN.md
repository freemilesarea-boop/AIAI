# Phase 7 — GPU Launch Plan (pre-purchase)

Everything up to the moment money is spent. **No instance has been
provisioned and no payment has been made.**

Dataset ready to ship: `LUBER_TRAINSET_PILOT_V1`
Content hash: `d14b6911ff4217c33dc36c2804e905038febd31071d0cb1c7e742742a09aee49`
10 tracks · 27.84 min · 48 kHz stereo · ~1.6 GB

Every training argument below was **read from the actual CLI** at the
pinned commit, not inferred. Verbatim `--help` output is committed at
`docs/upstream_train_cli_6d467e4.txt`.

---

## 1. Upstream CLI — verified, and different from the earlier draft

An earlier revision of this runbook guessed the interface and was
wrong in almost every argument. The real entry point takes a
**subcommand**:

```
train.py [-h] [--plain] [--yes] {vanilla,fixed,estimate} ...
```

| Subcommand | Purpose |
|---|---|
| `vanilla` | Reproduces the existing (upstream's word: *bugged*) training, for backward compatibility |
| **`fixed`** | **Corrected training: continuous timesteps + CFG dropout** ← use this |
| `estimate` | Gradient sensitivity analysis, no training |

Corrections against the earlier draft:

| Earlier guess | Actual |
|---|---|
| `train.py --dataset` | `train.py fixed --dataset-dir` |
| `--output` | `--output-dir` (required) |
| `--lora-rank 8` | `--rank` — **default 64** |
| `--lora-alpha 16` | `--alpha` — **default 128** |
| `--learning-rate` | `--lr` |
| `--max-epochs` | `--epochs` |
| `--save-every-n-epochs` | `--save-every` |
| `--mixed-precision bf16` | `--precision {auto,bf16,fp16,fp32}` |
| `--val-split 0.2` | **does not exist on the CLI** |
| `python -m acestep.training.dataset_builder` | `train.py fixed --preprocess --audio-dir … --dataset-json … --tensor-output …` |
| — | `--base-model {turbo,base,sft,xl_turbo,xl_base,xl_sft}` |
| — | `--optimizer-type {adamw,adamw8bit,adafactor,prodigy}` |
| — | `--attention-type {self,cross,both}` (default `both`) |
| — | `--gradient-checkpointing` **on by default** |
| — | `--offload-encoder` (saves ~2–4 GB VRAM) |
| — | `--cfg-ratio` (default 0.15) |

### The validation-split problem

`val_split` exists in `acestep/training/configs.py` (default `0.0`) and
is honoured by `data_module.py`, but **the CLI never exposes it**. A
CLI-driven run therefore trains with no validation set, and LUBER's own
run-manifest validator refuses a zero split precisely because
overtraining becomes undetectable.

Resolution used here: **hold out at the dataset level.** Preprocess 8
tracks into the training tensor directory and keep 2 aside, untouched by
training. Overtraining is then assessed by comparing checkpoints on the
held-out prompts plus `--sample-every-n-epochs`, rather than by a loss
curve the CLI cannot produce. This is weaker than a real validation
loss and is recorded as a known limitation, not hidden.

---

## 2. GPU sizing

### Derivation at our actual settings

`acestep-v15-turbo` (2B DiT), bf16, batch 1, gradient accumulation 4,
LoRA rank 64 / alpha 128, `--attention-type both`, gradient
checkpointing on, longest track 199.8 s (under the 240 s default).

| Component | VRAM |
|---|---|
| 2B DiT weights (bf16) | ~4.7 GB |
| Text encoder Qwen3-Embedding-0.6B | ~1.2 GB (removable via `--offload-encoder`) |
| VAE | ~0.34 GB |
| LoRA trainable params (fp32; upstream forces fp32 for trainables) | ~0.2–0.3 GB |
| Gradients (fp32) | ~0.2–0.3 GB |
| AdamW states ×2 | ~0.5–0.6 GB (`adamw8bit`: ~0.15 GB) |
| Activations, batch 1, checkpointing on | ~3–6 GB |
| Fragmentation / allocator headroom | ~1–2 GB |
| **Total** | **≈ 11–15 GB** |

Upstream independently states ~17 GB typical, 16 GB minimum, 20 GB+
recommended. The derivation lands just under that, which is the
expected relationship — treat upstream's figure as the anchor.

| | VRAM | Examples | Verdict |
|---|---|---|---|
| **Minimum** | **16 GB** | RTX 4080, A4000 | Works only with `--gradient-checkpointing` and `--offload-encoder`. No margin; an OOM restart costs more than the card saves. |
| Comfortable | 24 GB | RTX 4090, L4, A10G | Fits our settings with headroom. Viable if cost matters. |
| **Recommended** | **48 GB** | **L40S 48 GB**, RTX 6000 Ada 48 GB | Runs without offload, leaves room to try rank 128 or batch 2–4 without re-renting. |
| Not required | 80 GB | A100 / H100 | 3–5× the price for no pilot benefit. |

**Recommendation: L40S 48 GB.** The delta over a 24 GB card is roughly
USD 0.20/hr — about USD 1 across this pilot — which is less than one
OOM-forced restart.

---

## 3. Transfer bundle

| Item | Source | Size |
|---|---|---|
| 10 audio files | operator's `AI 음원`, **copied, never moved** | ~1.6 GB |
| `pilot_manifest.json` | `~/.luber/pilot_manifest.json` | 8 KB |
| `dataset.json` | generated below, for `--dataset-json` | small |

```bash
STAGE=~/luber-gpu-bundle
mkdir -p "$STAGE/audio_train" "$STAGE/audio_holdout"
uv run python - <<'PY'
import json, shutil
from pathlib import Path
stage = Path.home() / "luber-gpu-bundle"
manifest = json.loads((Path.home() / ".luber" / "pilot_manifest.json").read_text())
catalog = {e["sha256"]: e["absolute_path"]
           for e in json.loads((Path.home() / ".luber" / "discovery_catalog.json").read_text())}
tracks = sorted(manifest["tracks"], key=lambda t: t["track_id"])
# 8 train / 2 held out — the CLI has no --val-split, so the split is physical.
for index, track in enumerate(tracks):
    target = "audio_holdout" if index >= 8 else "audio_train"
    shutil.copy2(Path(catalog[track["audio_sha256"]]),
                 stage / target / f"{track['track_id']}.wav")   # copy, never move
shutil.copy2(Path.home() / ".luber" / "pilot_manifest.json", stage / "pilot_manifest.json")
# Minimal labelled dataset for --dataset-json. Fields we do not know are
# left empty rather than guessed: no language, no bpm, no key, no lyrics.
json.dump(
    [{"audio_path": f"audio_train/{t['track_id']}.wav", "caption": "", "lyrics": ""}
     for t in tracks[:8]],
    (stage / "dataset.json").open("w"), indent=2, ensure_ascii=False)
print(f"staged 8 train + {len(tracks) - 8} holdout to {stage}")
PY
```

Verify the copies before upload — a corrupted transfer must not become
a silently different dataset:

```bash
uv run python - <<'PY'
import hashlib, json
from pathlib import Path
stage = Path.home() / "luber-gpu-bundle"
manifest = json.loads((stage / "pilot_manifest.json").read_text())
missing = []
for track in manifest["tracks"]:
    for folder in ("audio_train", "audio_holdout"):
        path = stage / folder / f"{track['track_id']}.wav"
        if path.is_file():
            if hashlib.sha256(path.read_bytes()).hexdigest() != track["audio_sha256"]:
                missing.append(track["track_id"])
            break
    else:
        missing.append(track["track_id"])
print("TRANSFER CORRUPTED: " + ", ".join(missing) if missing else "all 10 hashes match")
PY
```

---

## 4. Exact commands for the GPU host

```bash
# ── 0. verify the machine ────────────────────────────────────────────
nvidia-smi                              # confirm 48 GB and CUDA 12.x
python3 --version                       # 3.11 or 3.12
df -h /workspace                        # >= 200 GB free

# ── 1. toolchain ─────────────────────────────────────────────────────
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# ── 2. upstream at the pinned commit (never main) ────────────────────
cd /workspace
git clone --filter=blob:none https://github.com/ace-step/ace-step-1.5.git
cd ace-step-1.5
git checkout 6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0
git rev-parse HEAD    # must equal 6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0
uv sync
uv pip install bitsandbytes             # enables --optimizer-type adamw8bit

# ── 3. confirm the CLI matches this runbook before spending time ─────
uv run python train.py --help
uv run python train.py fixed --help     # diff against docs/upstream_train_cli_6d467e4.txt

# ── 4. model weights, official downloader only ───────────────────────
export ACESTEP_CHECKPOINTS_DIR=/workspace/checkpoints
uv run python -c "
from acestep.model_downloader import download_main_model
ok, msg = download_main_model(); print(ok, msg)"

# ── 5. dataset (run from your Mac) ───────────────────────────────────
# rsync -avP ~/luber-gpu-bundle/ user@GPU_HOST:/workspace/dataset/

# ── 6. preprocess the 8 training tracks to tensors ───────────────────
cd /workspace/ace-step-1.5
uv run python train.py fixed \
    --preprocess \
    --audio-dir /workspace/dataset/audio_train \
    --dataset-json /workspace/dataset/dataset.json \
    --tensor-output /workspace/tensors \
    --max-duration 240 \
    --base-model turbo \
    --dataset-dir /workspace/tensors \
    --output-dir /workspace/runs/preprocess_only \
    --epochs 0 --yes

# ── 7. OVERFIT TEST FIRST — cheap, and it decides everything ─────────
uv run python train.py fixed \
    --base-model turbo \
    --dataset-dir /workspace/tensors \
    --output-dir /workspace/runs/overfit \
    --precision bf16 \
    --adapter-type lora --rank 64 --alpha 128 --attention-type both \
    --lr 1e-4 --batch-size 1 --gradient-accumulation 4 \
    --epochs 60 --save-every 20 \
    --optimizer-type adamw8bit --scheduler-type cosine \
    --gradient-checkpointing --offload-encoder \
    --seed 42 --shift 3.0 --num-inference-steps 8 \
    --sample-every-n-epochs 20 \
    --yes 2>&1 | tee /workspace/runs/overfit.log

# Gate before spending more: loss moves, a checkpoint is written, it
# reloads, LoRA ON vs OFF differ on the same prompt+seed, no NaN/Inf.
# If ON and OFF are indistinguishable: STOP and destroy the instance.
# ~USD 1 spent instead of ~USD 6 on a pilot that could not have worked.

# ── 8. pilot LoRA — only after the overfit test passes ───────────────
uv run python train.py fixed \
    --base-model turbo \
    --dataset-dir /workspace/tensors \
    --output-dir /workspace/runs/LUBER_LORA_PILOT_V1 \
    --precision bf16 \
    --adapter-type lora --rank 64 --alpha 128 --dropout 0.1 \
    --attention-type both --bias none \
    --lr 1e-4 --batch-size 1 --gradient-accumulation 4 \
    --epochs 600 --save-every 50 --warmup-steps 100 \
    --optimizer-type adamw8bit --scheduler-type cosine \
    --gradient-checkpointing --offload-encoder \
    --seed 42 --shift 3.0 --num-inference-steps 8 --cfg-ratio 0.15 \
    --sample-every-n-epochs 100 \
    --log-dir /workspace/runs/LUBER_LORA_PILOT_V1/runs \
    --yes 2>&1 | tee /workspace/runs/pilot.log

# ── 9. hash every checkpoint for the run manifest ────────────────────
find /workspace/runs -name "*.safetensors" -exec sha256sum {} \;

# ── 10. retrieve artifacts, confirm locally, then destroy ────────────
# rsync -avP user@GPU_HOST:/workspace/runs/ ./runs/
```

`--rank 64 / --alpha 128` are the CLI defaults, kept deliberately.
Upstream's `configs.py` dataclass says 8/16, but the CLI overrides it,
and a first pilot should run the interface's own defaults rather than a
value from a different layer.

Epochs 600 with 8 tracks ≈ 1,200 optimizer steps at batch 1 /
accumulation 4. Upstream's tutorial suggests 800 epochs for 10–20
tracks; 600 on 8 tracks sits in that range while keeping the first run
short.

---

## 5. Checkpoints

Written to `--output-dir` as PEFT adapter directories
(`lora_checkpoint.py` prefers `decoder.save_pretrained`).

`--save-every 50` over 600 epochs → 12 checkpoints. Compare at minimum
early (~100), middle (~300), late (~600). Later is not automatically
better: with no validation loss available, judge on the held-out tracks
and the `--sample-every-n-epochs` audio.

Expected artifacts:

```
/workspace/runs/LUBER_LORA_PILOT_V1/
├── checkpoint-epoch-50/   … -600/     PEFT adapter dirs (~50-200 MB each)
├── runs/                              TensorBoard logs
└── pilot.log
```

---

## 6. Time and cost

| Stage | Estimate |
|---|---|
| Setup, `uv sync`, bitsandbytes | 15–25 min |
| Model download (~10.1 GB) | 5–15 min |
| Dataset upload (1.6 GB) | 3–10 min |
| Preprocessing (8 tracks → tensors) | 10–20 min |
| Overfit test (60 epochs) | 15–25 min |
| Pilot LoRA (600 epochs, 8 tracks) | 1.5–3 h |
| Benchmark inference (A/B/C) | 20–40 min |
| Download + shutdown | 10–15 min |
| **Total** | **≈ 3–5.5 h** |

| Provider class | GPU | ~USD/hr | Pilot total |
|---|---|---|---|
| Community (Vast.ai, RunPod Community) | L40S 48 GB | 0.40–0.80 | **USD 1.5–4.5** |
| Managed (RunPod Secure, Lambda) | L40S 48 GB | 0.86–1.10 | **USD 3–6** |
| Managed | RTX 6000 Ada 48 GB | 0.75–1.10 | USD 2.5–6 |

**Budget USD 10** to absorb one failed run and a retry. Rates move —
re-check at rental time.

---

## 7. What this pilot can and cannot prove

**Can prove:** whether a LUBER LoRA measurably moves ACE-Step's output
at all. That is the Phase 7 question.

**Cannot prove:** anything about Korean lyric completion. The 10 tracks
have **no lyrics** and unlabelled vocal language, so `dataset.json`
ships empty captions and empty lyrics. `KOREAN_LINE_OMISSION` and
`LYRIC_LINE_SKIP` receive no training signal whatsoever.

Expect movement in timbre, production character, and arrangement
tendency. A null result on the Korean findings is not evidence the
approach fails — they were never in the data.

---

## 8. Stop condition

**Stop here.** The next step rents hardware and spends money, which is
the operator's decision. Nothing above has been executed.
