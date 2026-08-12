# Phase 6 — ACE-Step 1.5 LoRA Training Audit

Audited 2026-08-12 against the **pinned** upstream checkout only
(`~/ace-step-1.5` @ `6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`).
Every value below is cited from upstream code or upstream docs at that
commit. No community tutorials, blogs, or forum posts were used.

Where the tutorial and the code disagree, **the code is authoritative**
and the discrepancy is recorded.

---

## 1. Hardware requirements (upstream `docs/en/LoRA_Training_Tutorial.md`)

| VRAM | Upstream statement |
|---|---|
| 16 GB | "Generally sufficient, but longer songs may cause out-of-memory errors" |
| **20 GB+** | **Recommended.** "VRAM usage typically stays around 17 GB during training" |

This is modest — a 48 GB L40S or RTX 6000 Ada has ample headroom, and
even a 24 GB card would work for the pilot. It also confirms the Phase 6
decision not to train on the Mac: this machine's unified memory is
already ~92% swap-saturated running *inference* alone.

Upstream also warns that Gradio must be restarted several times during
preprocessing to free VRAM — relevant only to the UI path, which the
runbook avoids in favour of the API/CLI path.

---

## 2. Dataset format (`docs/en/LoRA_Training_Tutorial.md` §Data Preparation)

Flat directory, one set of files per track, matched by basename:

```
dataset/
├── song1.mp3            # audio
├── song1.lyrics.txt     # lyrics (exact, with section tags)
├── song1.json           # annotations (optional)
├── song1.caption.txt    # caption (optional; may live in the JSON instead)
└── …
```

Annotation JSON — **all fields optional**:

```json
{
    "caption": "A high-energy J-pop track with synthesizer leads and fast tempo",
    "bpm": 190,
    "keyscale": "D major",
    "timesignature": "4",
    "language": "ja"
}
```

**Finding L1 — the training format wants exactly the metadata LUBER
never sends at inference.** `bpm`, `keyscale`, `timesignature`, and
`language` are first-class training annotations, and Phase 5 found
LUBER omits the first three at inference time. Training on conditioned
data while inferring unconditioned is a train/serve mismatch. The
LUBER dataset schema therefore captures all four as required fields.

`keyscale` must use upstream's vocabulary (7 notes × 5 accidentals × 2
modes = 70 valid strings, e.g. `"F# minor"`), per
`acestep/constants.py::VALID_KEYSCALES`.

`language` must be in `VALID_LANGUAGES`, which includes `ko`.

---

## 3. Preprocessing pipeline

Modules under `acestep/training/dataset_builder_modules/`:

| Module | Role |
|---|---|
| `preprocess_audio.py` | Audio loading and conditioning |
| `preprocess_vae.py` | VAE encoding to latents |
| `preprocess.py` / `preprocess_context.py` | Orchestration |
| `metadata.py` | Annotation handling |
| `label_utils.py` | Auto-labelling helpers |
| `update_sample.py` | Per-sample mutation |

The documented flow is: load models → load data → review dataset →
auto-label → review/edit → save dataset JSON → **generate tensor
files**. Training consumes pre-computed tensors, not raw audio, so
preprocessing is a separate, cacheable stage.

Auto-annotation helpers exist (`scripts/lora_data_prepare/`, including
a Gemini captioner and a local Key-BPM finder that exports CSV). LUBER
will supply its own curated annotations rather than rely on
auto-captioning, because caption quality directly shapes what the LoRA
learns.

---

## 4. LoRA configuration (`acestep/training/configs.py`)

Read from the dataclass defaults — these are the code's values:

| Parameter | Default | Notes |
|---|---|---|
| `r` (rank) | **8** | Dimension of the low-rank matrices |
| `alpha` | **16** | Scaling is `alpha / r` = 2.0 |
| `dropout` | **0.1** | |
| `target_modules` | **`["q_proj", "k_proj", "v_proj", "o_proj"]`** | Attention projections only — no MLP, no cross-attention-specific targets |
| `bias` | `"none"` | Bias parameters not trained |

**Finding L2 — the default adapter is small and attention-only.** Rank
8 over four attention projections is a light-touch adapter. That is
appropriate for style/timbre adaptation, which is exactly the Phase 6
target (vocal character), and less likely to fix structural or lyric
scheduling behaviour. It sets a realistic expectation for what the
pilot can achieve: the LoRA is a plausible lever on H8/H9 (vocal
style), not on H6/H7 (lyric omission).

---

## 5. Training configuration (`acestep/training/configs.py`)

| Parameter | Code default | Tutorial guidance |
|---|---|---|
| `shift` | **3.0** (fixed) | "Fixed: turbo uses shift=3.0" |
| `learning_rate` | **1e-4** | 1e-4 for LoRA |
| `batch_size` | **1** | Increase to 2–4 if VRAM allows |
| `gradient_accumulation_steps` | **4** | Effective batch = batch × accumulation |
| `max_epochs` | **100** | ~100 songs → 500 epochs; **10–20 songs → 800 epochs** |
| `save_every_n_epochs` | **10** | Smaller for short runs |
| `weight_decay` | **0.01** | |
| `mixed_precision` | **bf16** | bf16 on CUDA/XPU, fp16 on MPS, fp32 on CPU |
| `gradient_checkpointing` | false | Enable if VRAM-constrained |

Timesteps are described as "discrete timesteps from turbo shift=3.0
schedule (8 steps)" — training matches the turbo inference schedule.

**Finding L3 — the code default (100 epochs) and the tutorial guidance
(800 epochs for a 10–20 track set) differ by 8×.** The code default is
not tuned for a small dataset. A LUBER pilot of 20–50 tracks should
follow the tutorial's small-dataset guidance rather than the dataclass
default, and must checkpoint often enough to compare early/middle/late
(Step 16) rather than trusting the final epoch.

**Finding L4 — the demonstration dataset in the tutorial is a
commercial album.** Upstream trains on *ナユタン星からの物体Y* by
NayutalieN (13 tracks) and states the tutorial "is intended solely for
educational purposes… Please use your own original works to train your
LoRA." LUBER follows the disclaimer, not the demonstration: only
rights-cleared audio enters `LUBER_TRAINSET_V1`, enforced in code.

---

## 6. Optimizer and schedule (`acestep/training/trainer.py`)

- `AdamW` from `torch.optim`, with **8-bit Adam via bitsandbytes when
  available** ("OPTIMIZATION: Use 8-bit Adam to save some VRAM"),
  falling back to standard AdamW with a warning when bitsandbytes is
  absent.
- Schedulers imported: `CosineAnnealingWarmRestarts`, `LinearLR`,
  `SequentialLR` — i.e. warmup followed by cosine restarts.
- Only LoRA parameters are passed to the optimizer ("Setup optimizer -
  only LoRA parameters").
- Trainable parameters are forced to fp32 before optimizer/Fabric setup
  (`_ensure_optimizer_params_fp32`), while compute runs in bf16.

**Note for the runbook:** bitsandbytes was *not* installed in the local
macOS environment (the server logs a warning at startup). On the CUDA
training host it should be installed to get the 8-bit optimizer.

---

## 7. Checkpointing (`acestep/training/lora_checkpoint.py`)

`save_lora_weights()` has three paths, in priority order:

1. **PEFT adapter** — if the model exposes `decoder.save_pretrained`,
   saves a standard adapter directory. This is the normal path.
2. **Full state dict** — only when `save_full_model=True`.
3. **Manual LoRA state dict** — collects LoRA-named parameters as a
   fallback.

Adapter directories are the portable artifact to persist and hash.

---

## 8. Resume and validation (`acestep/training/trainer.py`)

- **Resume:** supported — `resume_from: Optional[str]` names a
  checkpoint directory, threaded through the training entrypoint.
- **Validation:** supported — `val_split` is read from the training
  config, a `val_dataloader` is constructed when available, and the
  trainer tracks `plot_val_steps` / `plot_val_loss` and `best_val_loss`.

**Finding L5 — validation is opt-in via `val_split`, which defaults to
0.0.** A pilot that leaves it at the default trains with no validation
signal at all, which makes Step 16 (detecting overtraining) impossible.
The runbook sets a non-zero `val_split`.

---

## 9. LoKr — a supported alternative

`acestep/training/lokr_utils.py` plus tutorial §LoKr. Kronecker-product
decomposition instead of low-rank factorization.

| | LoRA | LoKr |
|---|---|---|
| Default learning rate | 1e-4 | **0.03** |
| Default epochs | 10 | 500 |
| `linear_alpha` | — | 128 (typically 2× dim) |
| `weight_decompose` (DoRA) | — | true |
| Speed | baseline | "what previously took an hour can often complete in around 5 minutes" |

Upstream markets LoKr as better suited to consumer GPUs. The Phase 6
pilot uses **LoRA**, because it is the better-documented path and the
speed advantage is irrelevant on rented 48 GB hardware — but LoKr is a
legitimate fallback if iteration speed becomes the constraint.

---

## 10. Inference integration

Tutorial §"Using LoRA" documents loading a trained adapter for
generation, and the Gradio UI and API both expose LoRA selection
(`train_api_lora_start_route.py`, `train_api_lokr_start_route.py`
handle training; the generation path accepts an adapter).

**Implication for LUBER.** Serving a LoRA means the ACE-Step server
must load the adapter, and LUBER's `AceStepProviderConfig` would need
an adapter identifier to record which weights produced a track. That is
a small provider change, deferred until a LoRA actually exists and is
worth serving — recording it here so it is not discovered late.

---

## 11. What this audit changes about the Phase 6 plan

1. **Hardware is not the obstacle.** ~17 GB during training means a
   48 GB GPU is comfortable; the pilot does not need an H100.
2. **The default adapter targets attention only at rank 8.** Expect
   vocal-character and timbre movement, not structural or lyric-
   scheduling fixes. This aligns the LoRA pilot with root-cause
   hypotheses H8/H9 and explicitly *not* with H6/H7.
3. **Epoch guidance must come from the tutorial, not the code
   default** — 800 epochs for a 10–20 track set versus a default of
   100.
4. **`val_split` must be set explicitly** or overtraining cannot be
   detected.
5. **Training annotations require bpm/keyscale/timesignature/language**,
   which the LUBER dataset schema now enforces — and which also exposes
   the train/serve mismatch at inference.
6. **Upstream's own disclaimer requires original or licensed works.**
   The rights gate is not LUBER being cautious; it is what upstream
   instructs.
