# ACE-Step 1.5 — reference audio conditioning

Read from the pinned source at `6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`
and then tested with six real generations. Nothing here rests on parameter
names alone.

Runtime: `acestep-v15-turbo`, 8 inference steps, MLX, LM disabled.

**Status: calibration complete, awaiting human listening.** No product
feature was built. The measured classification is provisional
REFERENCE_STYLE_READY — see the classification section for exactly what
that does and does not cover.

---

## The capability, and where it actually goes

`reference_audio_path` (JSON) or a `ref_audio` multipart upload. It is a
**different mechanism from `src_audio`**, and confusing the two would
produce a different operation entirely.

The path, traced end to end:

```
ref_audio upload
  → process_reference_audio            normalise to stereo 48k
  → infer_refer_latent                 VAE tiled_encode → acoustic latents
      conditioning_embed.py:18-69
  → timbre encoder                     embed_tokens: timbre_hidden_dim → hidden_size
      modeling_acestep_v15_turbo.py:1076-1100
  → model.prepare_condition(...)       merged with text + lyric embeddings
      service_generate_execute.py:164-170
  → encoder_hidden_states
  → mlx_generate_diffusion(encoder_hidden_states_np=...)
```

### Active on MLX: yes — and grepping the sampler would have said no

`refer_audio_acoustic_hidden_states_packed` appears **zero times** in
`core/generation/handler/diffusion.py` (the MLX bridge) and **zero times**
in `models/mlx/dit_generate.py` (the MLX sampler).

That is not evidence of inertness. The reference is consumed *before* the
sampler: `prepare_condition` runs in PyTorch and folds the timbre
embeddings into `encoder_hidden_states`, and it is that already-conditioned
tensor which crosses to MLX. The sampler never needs to know a reference
existed.

This is the opposite conclusion to `cover_noise_strength` (Phase 13D),
which really is MLX-inert. The two look identical to a grep and differ
completely in fact, which is why each parameter has to be traced rather
than pattern-matched. The behavioural evidence below confirms the trace.

## What is extracted from the reference

Not a waveform, not codec tokens, not a fingerprint. The reference is
VAE-encoded to acoustic latents and passed through a dedicated **timbre
encoder** whose projection is literally named
`timbre_hidden_dim → hidden_size`. The result joins text and lyric
embeddings as a third conditioning stream in cross-attention.

When no reference is supplied the stream is not absent — it is filled with
30 s of silence (`generate_music_request.py:127`), which
`infer_refer_latent` maps to the model's `silence_latent`. So every
generation already carries a reference channel; supplying audio replaces
silence with real timbre embeddings.

## Controls

**There are none.** No scale, weight, dropout, guidance or strength
parameter touches the reference stream anywhere in the pinned source. The
conditioning is binary: a real reference, or the silence latent.

This is why the experiment below has no strength sweep. Any "influence:
low / medium / high" control would be invented, not calibrated.

## Experiment

Six generations, 30 s each, one variable at a time. Lyrics, language,
vocal gender, duration and seed held constant except where a run's name
says otherwise. Full parameters and hashes in
`benchmarks/reference_audio/results.jsonl`.

References are LUBER-generated instrumentals, chosen to be far apart so
that transfer would be detectable at all:

| Reference | Centroid | Rolloff 85% |
|---|---|---|
| A — hard electronic, saw synths | 2794 Hz | 5772 Hz |
| B — sparse acoustic guitar | 665 Hz | 821 Hz |

## Results

Spectral centroid, the clearest and most defensible signal:

| Run | Centroid | vs control |
|---|---|---|
| 00 prompt only | 1391 Hz | — |
| 01 + reference A | **2180 Hz** | **+789 toward A** |
| 02 + reference B | **1262 Hz** | **−129 toward B** |
| 05 reference A, different seed | 2219 Hz | +39 vs run 01 |
| 04 contradictory prompt, no reference | 803 Hz | −588 |
| 03 contradictory prompt + reference A | 1135 Hz | +332 vs run 04 |

Rolloff and spectral flatness move in the same directions (control 2990 Hz
→ 4685 with A, 2361 with B).

### What this supports

1. **Reference audio causally changes generation.** Same prompt, same
   seed, only the reference differs, and the output moves toward the
   reference in every case.
2. **The direction is correct both ways.** A brightens, B darkens. A
   single-direction effect could have been an artefact; two opposite ones
   from opposite references cannot easily be.
3. **The effect is far above seed noise.** Two runs differing only by seed
   land 39 Hz apart; reference A moves the result 789 Hz. Roughly 20×.
4. **It survives a contradictory prompt.** With a prompt explicitly asking
   for sparse acoustic folk, reference A still pulled the output +332 Hz
   brighter than the same prompt alone.
5. **Prompt conditioning is stronger.** The prompt moved the result
   ~1045 Hz where the reference moved it ~789 Hz. The reference modulates
   within the territory the prompt sets; it does not override it.
6. **Nothing is copied.** SI-SDR between reference and output is −51 dB.
   No sample-level relationship exists, which is the safety property that
   matters most for a feature that accepts user uploads.
7. **Lyrics are fully independent.** Every output sings the same supplied
   Korean lyrics while both references are instrumental.

### What this does not support

- **Magnitude is asymmetric.** Reference A closed 56% of the brightness
  gap to itself; reference B closed only 18%. A plausible explanation is
  that the base prompt ("warm indie pop, electric piano, soft drums")
  already sat near B, leaving less room to move — but that is an
  explanation, not a measurement, and it is untested.
- **Only spectral/production character is demonstrated.** Centroid,
  rolloff and flatness are three views of one dimension.
- **Harmony, rhythm, melody and structure transfer are NOT demonstrated.**
  Chroma-sequence similarity sat at 0.76–0.86 for every pair including
  unrelated ones, and onset correlation was 0.01–0.37 with no pattern.
  Neither metric discriminated, so neither supports a claim in either
  direction.
- **MFCC cosine was useless here.** Every pair scored 0.95–0.999,
  including cross-controls. Reported for completeness and used for
  nothing — the same saturation failure that time-averaged chroma showed
  in Phase 13D.
- **Tempo is unreliable.** The autocorrelation estimator returned obvious
  octave errors (161 BPM for a slow ballad). Run 01 landing at 63 BPM
  against reference A's 62 is suggestive and is *not* claimed.
- **Vocal character transfer is unmeasured.** No speaker-embedding model
  was installed to invent a number for it.

## Classification

**REFERENCE_STYLE_READY — provisional, pending human listening.**

Justified on the definition's own terms: the reference reliably transfers
production character (brightness, spectral density) in the correct
direction, reproducibly across seed, without pretending to preserve or
copy the recording.

Two honest qualifications:

1. It is reliable in *direction*, variable in *magnitude* (56% vs 18%).
2. Only production character is demonstrated. It is **not**
   REFERENCE_MUSICAL_CONDITIONING, which would require multiple musical
   dimensions, and nothing here establishes melody, harmony, rhythm or
   structure transfer.

The classification is not final. Whether a 789 Hz brightness shift is
*perceptible as style transfer* is a listening judgement, and the same
discipline as Phase 13D applies: the measurements say the effect is real
and causal; only a listener can say it is useful. The package is at
`~/Desktop/LUBER_PHASE13E_REFERENCE_LISTENING/`.

## Proposed product design — not implemented

Nothing was built. This is the shape a future phase should implement *if*
listening confirms the effect.

**Name:** Reference Track. Never "sounds like", never an artist name.

**Contract sketch** (engine-neutral, no ACE-Step vocabulary):

```python
class ReferenceAudioCondition(BaseModel):
    reference_audio: Path  # uploaded or an existing LUBER master
    reference_duration_seconds: float
```

It attaches to an ordinary text-to-music request rather than replacing it,
because that is what the engine does: the prompt still drives the song.

**No influence control.** The engine exposes no dial, so the product must
not either. A three-level slider would be a fabricated control — the
failure Phase 13D avoided with `cover_noise_strength`.

**Acceptable copy**, supported by the measurements:

- "Use this track as a musical reference for the new song."
- "Guides the new song's sound and production character."
- "Your description still leads; the reference shapes the texture."

**Copy that must never appear**: copy this song, sound exactly like this,
clone this artist or singer, preserve this vocal, reproduce this
recording. The first is contradicted by −51 dB SI-SDR; the rest were never
measured at all.

**Open questions for the productization phase**: whether user-uploaded
audio (as opposed to LUBER-generated) behaves the same; what rights and
content checks an upload path requires; and whether the asymmetric
magnitude leaves the feature feeling inert for some references.
