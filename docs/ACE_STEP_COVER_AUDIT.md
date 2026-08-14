# ACE-Step 1.5 — the real cover / audio-to-audio contract

Read from the pinned source at `6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`
(`~/ace-step-1.5`), not from documentation, blogs, or ACE-Step 1.0. Where
a claim below is behavioural rather than structural it is marked as
*unverified until calibration*.

Loaded model on this deployment: `acestep-v15-turbo` (`is_turbo: true`),
8 inference steps, LM disabled.

---

## Endpoint and transport

| | |
|---|---|
| Endpoint | `POST /release_task` (same as every other task) |
| `task_type` | `"cover"` — also `"cover-nofsq"`, see below |
| Source audio | `src_audio` multipart file field, or `src_audio_path` inside the system temp dir |
| Result | `POST /query_result`, then `GET /v1/audio?path=…` |

`cover` is in `TASK_TYPES_TURBO` (`constants.py:82`), so the loaded turbo
checkpoint supports it. `extract` / `lego` / `complete` are base-only and
remain unavailable.

Transport is identical to Phase 13B/13C: `validate_audio_path` rejects
absolute paths outside the system temp directory, so LUBER uploads the
bytes.

## Request fields that matter

From `GenerateMusicRequest` (`api/http/release_task_models.py`), forwarded
into `GenerationParams` by `api/job_generation_setup.py:154-190`:

| Field | Default | Notes |
|---|---|---|
| `task_type` | `"text2music"` | must be `"cover"` |
| `src_audio_path` / `src_audio` upload | — | the source |
| `audio_cover_strength` | `1.0` | **the real strength dial** — see below |
| `cover_noise_strength` | `0.0` | **inert on this deployment** — see below |
| `prompt`, `lyrics`, `vocal_language` | — | ordinary conditioning |
| `bpm`, `key_scale`, `time_signature` | unset | ordinary metadata conditioning |
| `seed`, `use_random_seed` | random | as for text2music |
| `audio_duration` | — | **ignored when a source is supplied**, see Duration |
| `reference_audio_path` | — | a *separate* mechanism; not used by cover |

`task_type` is **not validated** against the loaded checkpoint anywhere on
the `/release_task` path. An unsupported task is forwarded and produces
undefined output rather than an error, so LUBER must gate on the model
itself.

## How cover differs from repaint — architecturally

This is the important finding, and it is structural rather than a matter
of degree.

**Repaint** (Phases 13B/13C) masks a time range. Outside the mask the
sampler re-imposes the VAE-encoded source at every step
(`_repaint_step_injection`), so the preserved audio *is* the original
recording — measured at correlation 0.9997.

**Cover** masks nothing. `conditioning_masks.py:68-71` gives the whole
canvas `chunk_masks = ones` and sets `is_cover = True`; there is no
`repaint_mask` and no step injection. Everything is regenerated.

What the source contributes is a *semantic summary*, not audio
(`modeling_acestep_v15_turbo.py:1683-1696`):

```python
lm_hints_5Hz, indices, llm_mask = self.tokenize(hidden_states, ...)
lm_hints_25Hz = self.detokenize(lm_hints_5Hz)
src_latents = torch.where(is_covers > 0, lm_hints_25Hz, src_latents)
context_latents = torch.cat([src_latents, chunk_masks], dim=-1)
```

The source is VAE-encoded, quantised down to **5 Hz semantic tokens**
(~200 ms per token), detokenized back to 25 Hz, and handed to the model as
*context*. So cover conditions on a lossy musical/semantic sketch of the
source, never on its waveform.

Consequence to expect, and to test rather than assert: composition,
structure and broad contour have a channel to survive; the specific
recording and the singer's voice do not. Whether that is enough to call
the product "Remix" is what calibration decides.

`cover-nofsq` shares cover's instruction text and differs in bypassing the
FSQ quantiser path. Not exercised in this phase — one primitive at a time.

## Source-preservation strength

There are two candidate dials and **only one of them works here.**

### `audio_cover_strength` — live, and the one to calibrate

Implemented identically in both samplers
(`models/turbo/modeling_acestep_v15_turbo.py:2091`,
`models/mlx/dit_generate.py:324`):

```python
cover_steps = int(num_steps * audio_cover_strength)
...
if step_idx >= cover_steps and not _switched_to_non_cover:
    encoder_hidden_states = encoder_hidden_states_non_cover
```

It sets **for how many diffusion steps the source-derived conditioning is
used** before the model switches to plain text conditioning. Higher keeps
the source in play longer.

Range 0.0–1.0, default 1.0. With turbo's 8 steps the value quantises to
eighths, so the distinguishable settings are:

| `audio_cover_strength` | cover steps (of 8) |
|---|---|
| 1.00 | 8 |
| 0.75 | 6 |
| 0.50 | 4 |
| 0.25 | 2 |
| 0.00 | 0 — equivalent to text2music |

A sweep of 1.00 / 0.75 / 0.50 / 0.25 therefore covers the whole usable
space at this step count; finer values would be rounding noise.

### `cover_noise_strength` — INERT on this deployment

The request model documents it as "0.0=pure noise, 1.0=closest to source
audio", and the PyTorch turbo model implements it as an img2img-style
start (`modeling_acestep_v15_turbo.py:2050-2064`): begin the schedule from
a partially-noised source latent instead of pure noise.

It never reaches the MLX path. `cover_noise_strength` appears **zero
times** in `core/generation/handler/diffusion.py` (the MLX bridge) and
**zero times** in `models/mlx/dit_generate.py` (the MLX sampler).

This deployment runs MLX: `server.log` records **97 of 97** generations as
`DiT diffusion complete via MLX`, and `gpu_config` resolves tier6a with
the mlx backend on this machine's 17.8 GB of unified memory.

So on this runtime `cover_noise_strength` is a parameter that is accepted,
logged, and does nothing. It must not be exposed as a product control, and
it must not be used to explain any observed behaviour.

## Duration semantics

Different from text2music, and not to be assumed from it.

`padding_utils.py:36-38`:

```python
if is_cover_task:
    # Cover task: Use src_audio directly without padding
    batch_target_wavs = processed_src_audio
```

The canvas is the source audio itself. Requested `audio_duration` is not
used to size it — only the no-source branch calls `create_target_wavs`.
Expectation: output duration ≈ source duration. Verified in calibration
below rather than assumed.

## Lyrics, prompt, seed

Ordinary conditioning, carried exactly as in text2music: `prompt` and
`lyrics` are compiled and embedded, `seed` honours `use_random_seed=false`.
There is no lyric-to-time alignment anywhere in the engine, so nothing
here lets LUBER claim a lyric lands at a particular moment.

## Known incompatibilities and traps

1. `cover_noise_strength` is inert on MLX (above).
2. `task_type` is unvalidated server-side; sending a base-only task to
   turbo yields undefined audio, not an error.
3. `reference_audio_path` is a *separate* conditioning path from
   `src_audio` and is not what cover uses. Mixing them up would produce a
   different operation.
4. `audio_cover_strength` quantises to 1/8 at turbo's step count.
5. Cover shares no code with repaint's preservation guarantee. Nothing
   about Phases 13B/13C transfers to it.

## Product outcome (Phase 13D-2)

Classified **COVER_ONLY** and shipped as **Create Cover** after the
product owner's listening pass confirmed the results are musically usable.
That verdict is subjective and is recorded, with its limits, in
`benchmarks/remix_cover/README.md` — it is not a benchmark against any
other system.

The product exposes two strengths mapped onto the two calibrated engine
values (1.00 and 0.75) with the direction inverted, since more
transformation means less adherence. `cover_noise_strength` is never sent.

## Does the engine expose explicit source preservation?

Not in the sense repaint does. There is no mask, no step injection, and no
parameter that preserves source *samples*. `audio_cover_strength` controls
how long a semantic sketch of the source steers generation — influence,
not preservation. Any product built on this must be named accordingly.
