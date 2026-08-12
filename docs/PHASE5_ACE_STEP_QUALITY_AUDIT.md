# Phase 5 — ACE-Step 1.5 Quality Surface Audit

Audited 2026-08-12 against the **pinned** upstream checkout only
(`~/ace-step-1.5` at `6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`). Every
statement below cites upstream code or upstream docs at that commit. No
blogs, no third-party summaries, no inference from ACE-Step 1.0.

Where upstream's own documents disagree with upstream's code, **the code
wins** and the discrepancy is recorded.

---

## 1. Model variants actually available at the pin

From `acestep/model_downloader.py` (`MAIN_MODEL_REPO`, `SUBMODEL_REGISTRY`)
and the README Model Zoo.

### Bundled in the unified repo `ACE-Step/Ace-Step1.5` (~10.1 GB)

| Component | Size | Notes |
|---|---|---|
| `acestep-v15-turbo` (2B DiT) | 4.79 GB | Default DiT. **This is LUBER's production model.** |
| `acestep-5Hz-lm-1.7B` | 3.76 GB | Default LM (optional planner) |
| `Qwen3-Embedding-0.6B` | 1.21 GB | Text encoder |
| `vae` | 0.34 GB | Audio codec |

### Separately downloadable (`SUBMODEL_REGISTRY`)

LM: `acestep-5Hz-lm-0.6B`, `acestep-5Hz-lm-4B`
2B DiT: `acestep-v15-base`, `acestep-v15-sft`, `acestep-v15-turbo-shift1`,
`acestep-v15-turbo-shift3`, `acestep-v15-turbo-continuous`
XL (4B) DiT: `acestep-v15-xl-base`, `acestep-v15-xl-sft`, `acestep-v15-xl-turbo`

### Upstream's own quality ratings (README Model Zoo)

| Model | Steps | CFG | Quality | Diversity |
|---|---|---|---|---|
| `acestep-v15-base` | 50 | yes | Medium | High |
| `acestep-v15-sft` | 50 | yes | High | Medium |
| **`acestep-v15-turbo`** | **8** | **no** | **Very High** | Medium |
| `acestep-v15-xl-base` | 50 | yes | High | High |
| `acestep-v15-xl-sft` | 50 | yes | Very High | Medium |
| `acestep-v15-xl-turbo` | 8 | no | Very High | Medium |

**Finding Q1.** Upstream rates the 2B turbo we already run as "Very High"
quality — the *same* rating it gives XL turbo and XL sft. By upstream's
own table, moving to XL is not an obvious quality upgrade; XL's stated
advantage is "higher audio quality" from a 4B decoder (README line 140),
which is a fidelity claim, not a musicality claim. This weakens the
assumption that XL is the answer to composition/vocal weaknesses.

---

## 2. Marketing claim — recorded, NOT adopted as a gate

README line 41 and line 60 state ACE-Step v1.5 achieves *"quality beyond
most commercial music models"* and specifically **"between Suno v4.5 and
Suno v5"**.

**This is an upstream marketing claim and is explicitly not used as a
Phase 5 pass criterion.** Per the Phase 5 mandate, parity with Suno 4.5
may only be asserted against a blinded comparison with legitimately
obtained Suno reference audio. No such reference audio exists in this
project. See the baseline report's parity section.

---

## 3. Full generation parameter surface

From `acestep/api/http/release_task_models.py` (`GenerateMusicRequest`)
and `release_task_request_builder.py`.

### Core musical conditioning

| Parameter | API default | LUBER sends? | Notes |
|---|---|---|---|
| `prompt` (alias `caption`) | `""` | ✅ compiled | Music description |
| `lyrics` | `""` | ✅ verbatim | Empty / `[inst]` ⇒ instrumental |
| `vocal_language` | `"en"` | ✅ | Must be in `VALID_LANGUAGES` |
| `audio_duration` | `None` | ✅ | Aliases: `duration`, `target_duration` |
| **`bpm`** | `None` | ❌ **not sent** | Integer tempo conditioning |
| **`key_scale`** | `""` | ❌ **not sent** | e.g. `"F# minor"`; 70 valid values |
| **`time_signature`** | `""` | ❌ **not sent** | Metre conditioning |
| `instruction` | `DEFAULT_DIT_INSTRUCTION` | ❌ (default used) | `"Fill the audio semantic mask based on the given conditions:"` |

### Sampling / diffusion

| Parameter | API default | LUBER sends? | Notes |
|---|---|---|---|
| `inference_steps` | `8` | ✅ `8` | Turbo 1–20 (8 recommended); base 1–200 (32–64) |
| `guidance_scale` | `7.0` | ❌ | **Irrelevant for turbo** — see Finding Q2 |
| `shift` | `3.0` (REST) | ❌ (default used) | **Already correct** — see Finding Q3 |
| `timesteps` | `None` | ❌ | Custom schedule; overrides steps+shift |
| `infer_method` | `"ode"` | ❌ | `"ode"` or `"sde"` — **unexplored** |
| `use_adg` | `False` | ❌ | Adaptive Dual Guidance, **base model only** |
| `cfg_interval_start` / `_end` | `0.0` / `1.0` | ❌ | CFG window; moot for turbo |
| `seed` + `use_random_seed` | `-1` / `True` | ✅ when set | |
| `batch_size` | `None` (server 2) | ✅ `1` | |
| `use_tiled_decode` | `True` | ❌ (default) | Memory management |

### LM (planner) parameters — inert while LM is disabled

`thinking` (`False`), `lm_model_path`, `lm_backend` (`vllm`|`pt`|`mlx`),
`use_cot_caption` (`True`), `use_cot_language` (`True`),
`constrained_decoding` (`True`), `lm_temperature` (`0.85`),
`lm_cfg_scale` (`2.5`), `lm_top_k`, `lm_top_p` (`0.9`),
`lm_repetition_penalty` (`1.0`), `lm_negative_prompt`
(`"NO USER INPUT"`), `sample_mode`, `sample_query`, `use_format`,
`allow_lm_batch`.

LUBER sends `thinking=False`, `use_cot_caption=False`,
`use_cot_language=False`.

### Not used by LUBER (other task types)

`global_caption`, `reference_audio_path`, `src_audio_path`,
`audio_cover_strength`, `cover_noise_strength`, `audio_code_string`,
`task_type` (`text2music`), repainting/repaint_* family,
`chunk_mask_mode`, `analysis_only`, `full_analysis_only`,
`extract_codes_only`, `track_name`, `track_classes`,
`is_format_caption`.

---

## 4. Findings that matter for quality

### Finding Q2 — `guidance_scale` is auto-corrected for turbo (no gap)

`acestep/core/generation/handler/generate_music.py:288`:

```
if self.is_turbo_model() and guidance_scale != 1.0:
    ... "guidance_scale {:.1f} -> 1.0 (turbo does not use CFG)."
```

Turbo bakes guidance into distillation. **LUBER omitting
`guidance_scale` costs nothing.** Do not "fix" this.

### Finding Q3 — `shift` is already 3.0 on the REST path (no gap)

Upstream's own docs contradict each other:

- `docs/en/INFERENCE.md:376` — Python API default `1.0`, *"Recommended
  3.0 for turbo models"*, and line 361: *"the only timestep/guidance
  parameter you need to set manually for turbo is `shift=3.0`"*.
- `release_task_models.py:92-94` — REST default **`3.0`**, described as
  *"Only effective for base models, not turbo models."*

Code resolves it: `release_task_request_builder.py:83` uses
`parser.float("shift", 3.0)` and `job_generation_setup.py:175` passes
`shift=req.shift` into the pipeline (whose own signature defaults to
`1.0` at `generate_music.py:207`).

**Conclusion: LUBER, by not sending `shift`, receives the REST default
of 3.0 — the value INFERENCE.md recommends for turbo.** This is correct
today, but it is *correct by accident*: it depends on an upstream
default, and the two docs disagree about whether it even applies to
turbo. Recommend pinning it explicitly so an upstream default change
cannot silently alter output.

### Finding Q4 — musical metadata conditioning is entirely unused

`bpm`, `key_scale`, and `time_signature` are first-class conditioning
inputs that LUBER never sends. The request model notes: *"Regardless of
thinking, if some metas are missing, server may use LM to fill them."*
**LUBER runs with the LM disabled**, so missing metadata is not filled by
a planner either. Tempo, key, and metre are therefore left entirely to
the DiT's own priors on every generation.

This is a plausible root cause for weak structural/rhythmic consistency
and is cheap to test — it needs no training, only request changes.
Classified `INFERENCE_CONFIGURATION_PROBLEM` pending measurement.

### Finding Q5 — `infer_method` (ode/sde) never explored

Default `"ode"`. `"sde"` is a supported alternative sampler at the pin.
Unexplored, zero-cost to A/B.

### Finding Q6 — Korean is a supported vocal language

`acestep/constants.py:13` `VALID_LANGUAGES` includes `'ko'` (and `'ja'`,
`'zh'`, `'yue'`). LUBER sends `ko`, which is valid. Korean is supported
but this says nothing about *pronunciation quality* — that is what the
Phase 5 listening benchmark measures.

### Finding Q7 — `key_scale` has a strict vocabulary

`VALID_KEYSCALES` = 7 notes × 5 accidentals × 2 modes = 70 combinations
(e.g. `"C major"`, `"F# minor"`, `"B♭ major"`). Any future key
conditioning must emit exactly these strings.

---

## 5. Memory requirements (upstream `docs/en/GPU_COMPATIBILITY.md`)

| VRAM | Tier | XL (4B) DiT | LM models | Max duration (LM / no LM) |
|---|---|---|---|---|
| ≤4 GB | 1 | ❌ | none | 4 / 6 min |
| 6–8 GB | 3 | ❌ | 0.6B | 8 / 10 min |
| 12–16 GB | 5 | ⚠️ marginal | 0.6B, 1.7B | 8 / 10 min |
| **16–20 GB** | **6a** | ✅ with offload | 0.6B, 1.7B | 8 / 10 min |
| 20–24 GB | 6b | ✅ | 0.6B, 1.7B, 4B | 8 / 8 min |
| ≥24 GB | unlimited | ✅ | all | 10 / 10 min |

XL weights ≈ **9 GB** bf16 vs ≈ **4.7 GB** for 2B (README line 140).

**This machine.** ACE-Step self-reports `tier6a` (17.76 GB unified) on
MPS. That table is written for discrete CUDA VRAM; on Apple Silicon the
"VRAM" is unified memory shared with the OS, so tier6a's "✅ (offload)"
for XL does **not** imply XL is safe here. Measured evidence from Phase 2
on this exact machine: enabling the 1.7B LM alongside the 2B DiT drove
swap from ~8 GB to **18.7 GB** and free disk from 17 GB to **5.7 GB**.
That is why LUBER runs `ACESTEP_INIT_LLM=false`.

---

## 6. Official benchmark methodology

`docs/en/BENCHMARK.md` documents `profile_inference.py`, which measures
**performance, not musical quality**: wall time, LM planning time, DiT
diffusion time, VAE decode time, across `profile` / `benchmark` /
`tier-test` / `understand` / `create_sample` / `format_sample` modes.

**Finding Q8. Upstream ships no musical-quality benchmark.** There is no
official listening protocol, rubric, or quality dataset at this pin. The
README's Suno comparison is asserted without a reproducible method in the
repository. Phase 5's benchmark therefore cannot reuse an upstream
quality harness — it has to define its own, which is what
`benchmarks/music_quality/` does.

---

## 7. Configuration matrix — what is actually runnable here

Only combinations upstream supports:

| ID | Configuration | Supported upstream | Local feasibility |
|---|---|---|---|
| **A** | `acestep-v15-turbo`, DiT-only (LM off, thinking off) | ✅ | ✅ **baseline, in production** |
| B | `acestep-v15-turbo` + LM 1.7B (thinking off, metas via LM) | ✅ | ⚠️ weights present; high swap risk |
| C | `acestep-v15-turbo` + LM + `thinking=true` | ✅ | ⚠️ same risk, higher |
| D | `acestep-v15-xl-turbo`, DiT-only | ✅ | ❌ ~9 GB download; 17 GB disk free |
| E | `acestep-v15-xl-turbo` + LM | ✅ | ❌ |

Configurations that do **not** exist and must not be invented: turbo +
`guidance_scale` (overridden), turbo + `use_adg` (base only), and any
`shift` sweep expectation on turbo beyond what §4 Q3 establishes.

---

## 8. Zero-training levers this audit surfaces

Ordered by cost, all testable without any training:

1. `bpm` / `key_scale` / `time_signature` conditioning (Q4)
2. `infer_method=sde` vs `ode` (Q5)
3. `inference_steps` 8 → 12–20 within turbo's supported range (Q3 table)
4. Explicit `shift=3.0` pin instead of relying on the default (Q3)
5. LM-enabled metadata completion (config B) — hardware permitting
6. Prompt-compiler changes (audited separately in the baseline report)

Whether any of these actually improves musical quality is an empirical
question. Phase 5 measures the baseline; it does not tune.
