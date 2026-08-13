# Phase 10 — Inference Parameter Audit

Every parameter below was read from the pinned source at
**`6d467e4b`** (`acestep/api/http/release_task_models.py`,
`acestep/inference.py`, `acestep/core/generation/handler/*`). Nothing
here is inferred from documentation or guessed. Where a parameter is
declared but has no effect on this deployment, that is stated with the
line that makes it inert.

**Deployment context that decides most of this table:** LUBER runs
`acestep-v15-turbo` with `thinking=false` and `use_cot_*=false` — a
**DiT-only, turbo-checkpoint** configuration. A large share of the
engine's parameter surface is either LM-only or base-model-only and is
therefore dead on arrival here.

## 1. Parameters that can plausibly affect quality on this deployment

| Parameter | Upstream default | LUBER value | Range / values | Role per source | Compute impact |
|---|---|---|---|---|---|
| `inference_steps` | `8` (`release_task_models.py:41`) | `8` (`ACE_STEP_INFERENCE_STEPS`) | int, **no clamp anywhere in the REST path** — only `> 0` is checked (`lyric_timestamp.py:66`) | Number of diffusion denoising steps | **Linear.** The one real quality/compute lever available |
| `infer_method` | `"ode"` (`:91`) | `"ode"` (not sent; default applies) | `"ode"` \| `"sde"` | Diffusion sampler selection | ~neutral |
| `custom_timesteps` | `""` (`:98`) | not sent | comma-separated floats | "Overrides `inference_steps` and `shift`" — full manual control of the noise schedule | equals implied step count |
| `instruction` | `DEFAULT_DIT_INSTRUCTION` | not sent | free text | DiT-side instruction prefix | none |
| `batch_size` | server default 2 | `1` | int | Samples per request | linear |

`inference_steps` is the headline finding of this audit: **the REST path
applies no upper bound.** The Phase 2 audit's note that turbo is
"1–20 (8 recommended)" is guidance, not enforcement. Values above 20 are
accepted and will run.

## 2. Declared but inert on this deployment

| Parameter | Upstream default | Why it does nothing here |
|---|---|---|
| `guidance_scale` | `7.0` (`:42`) | **Turbo does not support CFG.** `inference.py:69`: "Only support for non-turbo model". `generate_music.py:286` forces it to 1.0 to avoid double-application. Sending 7.0 changes nothing. |
| `shift` | `3.0` (`:101`) | Field's own description: "**Only effective for base models, not turbo models.**" |
| `use_adg` | `False` (`:88`) | Base-model only. |
| `cfg_interval_start` / `cfg_interval_end` | `0.0` / `1.0` (`:89-90`) | CFG scheduling — meaningless when CFG is off. |
| `thinking` | `False` | Requires the 5 Hz LM, disabled on this host. |
| `use_cot_caption` / `use_cot_language` | `True` | LM CoT enhancement; LUBER explicitly disables both. |
| `lm_negative_prompt` | — | **The only negative-prompt field upstream, and it is LM-only.** With the LM off there is no negative prompt on this path at all. |
| `sample_mode` / `use_format` / `sample_query` | `False` / `False` / `""` | Sample-formatting helpers, unrelated to generation quality. |
| `audio_code_string` | `""` | Supplies LM-generated semantic codes; no LM here. |

## 3. Task-type parameters LUBER does not use

`reference_audio_path`, `src_audio_path`, `audio_cover_strength`,
`cover_noise_strength`, `repainting_start` / `repainting_end`,
`repaint_mode`, `repaint_strength`, `repaint_latent_crossfade_frames`,
`repaint_wav_crossfade_sec`, `chunk_mask_mode`.

All belong to `cover` / `repaint` / `lego` / `extract` task types. LUBER
uses `text2music` only. Note `inference.py:815-819` silently forces
`audio_duration = None` for those task types — a trap for any future
phase that wires them.

## 4. Analysis-only switches

`analysis_only`, `full_analysis_only`, `extract_codes_only` — return
analysis instead of audio. Not quality parameters, but potentially
useful diagnostics later.

## 5. Consequence for Phase 10

The honest conclusion of this audit is **narrow**: on a turbo,
DiT-only deployment, essentially one meaningful inference lever exists.

- `guidance_scale`, `shift`, `use_adg`, CFG intervals — **all inert.**
  Any experiment varying them would be measuring noise, and reporting a
  result from one would be reporting a fiction.
- `infer_method` ode/sde is available, but Phase 6 already A/B'd it and
  found no user-visible win plus worse end-level drift. Re-running it
  without new evidence would be repeating a settled experiment.
- **`inference_steps` is the experiment worth running**, and it is
  unbounded, so 8 → 16 → 32 is a legitimate progression.
- `custom_timesteps` is a more advanced version of the same lever and is
  left for a later phase if the step count proves to matter.

This is why the Phase 10 DOE is small: the parameter surface that can
actually move quality on this deployment is small. A large sweep would
have produced a large table of parameters that provably do nothing.
