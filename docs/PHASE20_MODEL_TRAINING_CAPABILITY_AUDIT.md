# Model and training capability — audit

What the installed ACE-Step can actually be trained to do, read from the
checkout at `6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0` rather than from
its README. Nothing in this document was inferred from a filename.

Three confidence levels are used throughout and are not mixed:
**VERIFIED** (read in the source, or executed), **POSSIBLE** (the code
suggests it but it was not confirmed), **NOT SUPPORTED** (looked for and
absent).

---

## 1. Identity

| | |
|---|---|
| Engine | ACE-Step, commit `6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0` |
| Model | `acestep-v15-turbo` |
| Inference steps | 8 (turbo configuration) |
| Runtime here | Apple Silicon, MLX; `torch 2.10.0`, MPS available |
| Serving | `acestep-api` on loopback :8001 |

## 2. Training capability

### LoRA / adapter fine-tuning — **VERIFIED**

Present and real, not aspirational:

- `acestep/training/lora_injection.py` builds a `peft.LoraConfig` and
  injects adapters into the DiT.
- `acestep/training_v2/` holds the trainer used by it —
  `trainer_fixed.py` (`FixedLoRATrainer`), `fixed_lora_module.py`,
  optimiser, timestep sampling, TensorBoard hooks, and a preprocessing
  pipeline (`preprocess_vae.py`, `preprocess_prompt.py`).
- `peft 0.18.1` is installed in the engine's own virtualenv.
- Seven VRAM presets ship with the repository (`presets/vram_8gb.json`
  … `vram_24gb_plus.json`, plus `quick_test`, `recommended`,
  `high_quality`), each with concrete rank, batch size, gradient
  accumulation and optimiser settings.

**Trainable surface.** `get_dit_target_modules` selects, from the DiT
decoder only, every `nn.Linear` whose name contains `q_proj`, `k_proj`,
`v_proj` or `o_proj` — the attention projections. Confirmed by the
presets, which all carry `"target_modules_str": "q_proj k_proj v_proj
o_proj"`.

That is the whole trainable surface. The VAE, the text encoder and the
vocoder are not touched by this path.

### Full-parameter fine-tuning — **NOT SUPPORTED** by the installed tooling

No full fine-tune entry point exists in `training_v2`. Searching the
configuration and settings modules for a train-mode switch returns
nothing, and the trainer's own docstring describes itself as "adapter
fine-tuning". A full fine-tune would mean writing a trainer, not
configuring one.

`train.py` exists at the repository root and a `docs/en/
Large_Scale_SFT_Training_Guide.md` is present. Neither was traced to a
working full-parameter path here, so both are **POSSIBLE, not verified**,
and this phase does not plan around them.

### Preprocessing contract — **VERIFIED**

`acestep/training_v2/preprocess_vae.py` sets `TARGET_SR = 48000`. Any
dataset standardisation this project defines must resample to 48 kHz or
it will be resampled anyway, by code the project does not control. This
single constant is why the dataset specification names 48 kHz rather
than choosing a comfortable-sounding number.

### Dataset contract — **VERIFIED**

The engine's own dataset API (`train_api_dataset_models.py`) records, per
item: `caption`, `genre`, `lyrics` (defaulting to `[Instrumental]`),
`keyscale`, `timesignature`, `language`, plus the audio. LUBER's dataset
manifest is a superset of these fields, so an export is a projection
rather than a redesign.

## 3. What this means for the observed problems

The reported failures do not all live in the same place, and the LoRA
surface reaches only some of them.

| Reported problem | Plausible locus | Reachable by DiT-attention LoRA? |
|---|---|---|
| Trot-like vocal delivery | Learned style prior in the DiT | **Likely yes** — this is style, and style is what adapters move |
| Melody weakness | DiT | Possibly, with enough good data |
| Instrument/vocal timbre resolution | DiT *and* the VAE/decoder | **Partly at best.** If the ceiling is the audio decoder, no attention adapter will lift it |
| Korean lyric omission | Lyric conditioning / alignment | **Unclear — and this is the important one** |
| Korean pronunciation | Text encoder / lyric representation | Not reachable if the text encoder is the cause |
| Mix balance, stereo width | DiT output distribution | Possibly |

The Korean row is the one that changes what should be done first. If
lines are dropped because the lyric conditioning does not attend to them,
that is a conditioning failure, and training attention adapters on more
Korean audio may improve it — or may not, if the text encoder never
represented the missing text. This is a **hypothesis**, and the first
experiment is designed to test it cheaply rather than assume it.

## 4. Not verified, and deliberately not assumed

- Whether the reference-timbre encoder used at inference (Phase 15R) is
  trainable through this path. Not investigated; not needed yet.
- Whether MLX training works at all on this Mac. The trainer is
  PyTorch-based and the presets are VRAM-indexed, which is CUDA-shaped
  language. **No training has been run here**, so any claim about Apple
  Silicon training throughput would be invented.
- Whether `train.py` supports full fine-tuning end to end.

## 5. Consequence

The one training strategy this project can actually execute today is
**LoRA on DiT attention projections**. That is what
`TRAINING_ARCHITECTURE.md` plans around, and it is why the first
experiment targets a *style* problem — the class of problem adapters are
known to move — rather than audio resolution, which may be bounded by a
component this method cannot touch.

---

*Read-only audit. No ACE-Step file was modified, and its working tree is
unchanged at the pinned commit.*
