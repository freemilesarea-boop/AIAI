# Phase 7 — GPU Launch Plan (pre-purchase)

Everything up to the moment money is spent. **No instance has been
provisioned and no payment has been made.** Nothing below runs until
the operator rents hardware.

Dataset ready to ship: `LUBER_TRAINSET_PILOT_V1`
Content hash: `d14b6911ff4217c33dc36c2804e905038febd31071d0cb1c7e742742a09aee49`
10 tracks · 27.8 minutes · 48 kHz stereo · ~1.6 GB

---

## 1. GPU specification

Upstream states ~17 GB VRAM during LoRA training and recommends 20 GB+
(`docs/en/LoRA_Training_Tutorial.md`). A 48 GB card gives headroom for
full-length songs and batch > 1 without offload.

| Requirement | Value |
|---|---|
| GPU | **L40S 48 GB** (first choice) or **RTX 6000 Ada 48 GB** |
| Acceptable alternative | A100 40/80 GB |
| **Not required** | H100 — no pilot justification, roughly 3–5× the cost |
| VRAM | ≥ 24 GB strictly; 48 GB recommended |
| Disk | ≥ 200 GB (10 GB weights + 1.6 GB dataset + tensor cache + checkpoints) |
| RAM | ≥ 32 GB |
| CUDA | 12.x |
| Python | 3.11 or 3.12 |

## 2. Cost estimate

Rates are typical on-demand prices at the time of writing and **must be
re-checked before renting** — they move, and per-provider billing
granularity differs.

| Provider class | GPU | ~USD/hr |
|---|---|---|
| Community marketplace (Vast.ai, RunPod Community) | L40S 48 GB | 0.40 – 0.80 |
| Managed (RunPod Secure, Lambda) | L40S 48 GB | 0.86 – 1.10 |
| Managed | RTX 6000 Ada 48 GB | 0.75 – 1.10 |
| Managed | A100 80 GB | 1.50 – 2.50 |

Estimated pilot time on a 48 GB card:

| Stage | Estimate |
|---|---|
| Setup, `uv sync`, bitsandbytes | 15 – 25 min |
| Model download (~10.1 GB) | 5 – 15 min |
| Dataset upload (1.6 GB) | 3 – 10 min |
| Preprocessing → tensors (10 tracks) | 10 – 20 min |
| **Overfit test** (1–2 tracks, short) | 10 – 20 min |
| **Pilot LoRA** (10 tracks, ~600 epochs) | 1.5 – 3 h |
| Benchmark inference (A/B/C sets) | 20 – 40 min |
| Artifact download + shutdown | 10 – 15 min |
| **Total** | **≈ 3 – 5.5 h** |

**Expected cost: USD 3 – 6** on a community L40S, **USD 4 – 8** managed.
Budget **USD 10** to absorb one failed run and a retry.

Cost control: the overfit test comes first and is cheap. If LoRA ON and
OFF produce identical audio, stop and destroy the instance — roughly
USD 1 spent instead of 6 on a pilot that could not have worked.

## 3. Transfer bundle

Ship exactly these. Nothing else needs to leave this machine.

| Item | Source | Size |
|---|---|---|
| 10 audio files | the operator's `AI 음원` folder, **copied, not moved** | ~1.6 GB |
| `pilot_manifest.json` | `~/.luber/pilot_manifest.json` | 8 KB |
| Training runbook | `infra/gpu/PHASE6_TRAINING_RUNBOOK.md` | — |

Staging command — copies out of the source folder and leaves the
originals untouched:

```bash
STAGE=~/luber-gpu-bundle
mkdir -p "$STAGE/audio"
uv run python - <<'PY'
import json, shutil
from pathlib import Path
stage = Path.home() / "luber-gpu-bundle"
manifest = json.loads((Path.home() / ".luber" / "pilot_manifest.json").read_text())
catalog = {e["sha256"]: e["absolute_path"]
           for e in json.loads((Path.home() / ".luber" / "discovery_catalog.json").read_text())}
for track in manifest["tracks"]:
    src = Path(catalog[track["audio_sha256"]])
    shutil.copy2(src, stage / "audio" / f"{track['track_id']}.wav")   # copy, never move
shutil.copy2(Path.home() / ".luber" / "pilot_manifest.json", stage / "pilot_manifest.json")
print(f"staged {len(manifest['tracks'])} tracks to {stage}")
PY
```

Then verify the copies before uploading — a corrupted transfer must not
become a silently different dataset:

```bash
uv run python - <<'PY'
import hashlib, json
from pathlib import Path
stage = Path.home() / "luber-gpu-bundle"
manifest = json.loads((stage / "pilot_manifest.json").read_text())
ok = True
for track in manifest["tracks"]:
    path = stage / "audio" / f"{track['track_id']}.wav"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != track["audio_sha256"]:
        print(f"MISMATCH {track['track_id']}"); ok = False
print("all hashes match" if ok else "TRANSFER CORRUPTED — do not upload")
PY
```

## 4. Exact commands for the GPU host

Run in order. Stop at the first failure rather than continuing.

```bash
# ── 0. verify the machine ────────────────────────────────────────────
nvidia-smi
python3 --version                       # 3.11 or 3.12
df -h /workspace                        # >= 200 GB free

# ── 1. toolchain ─────────────────────────────────────────────────────
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# ── 2. upstream at the pinned commit (not main) ──────────────────────
cd /workspace
git clone --filter=blob:none https://github.com/ace-step/ace-step-1.5.git
cd ace-step-1.5
git checkout 6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0
git rev-parse HEAD          # must equal 6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0
uv sync
uv pip install bitsandbytes # 8-bit Adam; upstream warns and falls back without it

# ── 3. model weights via the official downloader only ────────────────
export ACESTEP_CHECKPOINTS_DIR=/workspace/checkpoints
uv run python -c "
from acestep.model_downloader import download_main_model
ok, msg = download_main_model(); print(ok, msg)"

# ── 4. dataset (from your machine) ───────────────────────────────────
# rsync -avP ~/luber-gpu-bundle/ user@GPU_HOST:/workspace/dataset/

# ── 5. verify the dataset arrived intact ─────────────────────────────
cd /workspace/dataset && uv run python - <<'PY'
import hashlib, json
from pathlib import Path
manifest = json.loads(Path("pilot_manifest.json").read_text())
bad = [t["track_id"] for t in manifest["tracks"]
       if hashlib.sha256((Path("audio") / f"{t['track_id']}.wav").read_bytes()).hexdigest()
       != t["audio_sha256"]]
assert not bad, f"corrupted: {bad}"
print(f"{len(manifest['tracks'])} tracks verified, dataset hash {manifest['content_hash']}")
PY

# ── 6. preprocess to tensors ─────────────────────────────────────────
cd /workspace/ace-step-1.5
uv run python train.py --help          # confirm flag spellings at this commit
uv run python -m acestep.training.dataset_builder \
    --input /workspace/dataset/audio --output /workspace/tensors

# ── 7. OVERFIT TEST FIRST — 1-2 tracks, cheap, decides everything ────
uv run python train.py \
    --dataset /workspace/tensors --output /workspace/runs/overfit \
    --lora-rank 8 --lora-alpha 16 --learning-rate 1e-4 \
    --batch-size 1 --gradient-accumulation 4 \
    --max-epochs 60 --save-every-n-epochs 20 --val-split 0.2 \
    --seed 42 --shift 3.0 --mixed-precision bf16 \
    2>&1 | tee /workspace/runs/overfit.log

# Required before spending more:
#   loss moves, checkpoint written, checkpoint reloads,
#   LoRA ON vs OFF differ on the same prompt+seed, no NaN/Inf.
# If ON and OFF are indistinguishable: STOP, destroy the instance,
# investigate. Do not run the pilot.

# ── 8. pilot LoRA (only after the overfit test passes) ───────────────
# 10 tracks is a small set: upstream's guidance is ~800 epochs for
# 10-20 tracks, not the code default of 100.
uv run python train.py \
    --dataset /workspace/tensors --output /workspace/runs/LUBER_LORA_PILOT_V1 \
    --lora-rank 8 --lora-alpha 16 --learning-rate 1e-4 \
    --batch-size 1 --gradient-accumulation 4 \
    --max-epochs 600 --save-every-n-epochs 50 --val-split 0.2 \
    --seed 42 --shift 3.0 --mixed-precision bf16 \
    2>&1 | tee /workspace/runs/pilot.log

# ── 9. hash every checkpoint for the run manifest ────────────────────
find /workspace/runs -name "*.safetensors" -exec sha256sum {} \;

# ── 10. retrieve artifacts, then destroy the instance ────────────────
# rsync -avP user@GPU_HOST:/workspace/runs/ ./runs/
# Confirm artifacts landed locally BEFORE terminating.
```

## 5. Checkpoint policy

`--save-every-n-epochs 50` over 600 epochs gives 12 checkpoints. Compare
at least early (~100), middle (~300), late (~600). A rising validation
loss against a falling training loss means take the earlier checkpoint —
later is not automatically better.

`--val-split 0.2` is mandatory here: with 10 tracks that is 2 held out,
and the run manifest refuses a zero split precisely because
overtraining is otherwise undetectable.

## 6. What this pilot can and cannot prove

**Can prove:** whether a LUBER LoRA measurably moves ACE-Step's output
at all — the actual Phase 7 question.

**Cannot prove:** anything about Korean lyric completion. The 10 tracks
carry **no lyrics** and the vocal language is unlabelled, so there is no
lyric supervision in this dataset. The Phase 5 failures
`KOREAN_LINE_OMISSION` and `LYRIC_LINE_SKIP` are untouched by this run.

Expect movement in timbre, production character, and arrangement
tendency. Do not expect the trot-vocal or Korean-pronunciation findings
to shift, and do not read a null result on those as evidence the
approach fails — they were never in the training signal.

## 7. Stop condition

**Stop here.** The next action requires renting hardware and spending
money, which is the operator's decision. Nothing above has been
executed.
