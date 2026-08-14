# Reference audio calibration (Phase 13E)

Does supplying a reference track actually change a new song, and if so,
what does it change?

Engine contract and full reasoning: `docs/ACE_STEP_REFERENCE_AUDIO_AUDIT.md`.
Engine: ACE-Step `6d467e4b`, `acestep-v15-turbo`, 8 steps, MLX, LM disabled.

**Status: calibration complete, awaiting human listening. No product
feature was built.**

## Contents

Metadata and code only. No audio is committed; the WAVs live in the
scratchpad and in `~/Desktop/LUBER_PHASE13E_REFERENCE_LISTENING/`.

| File | What it is |
|---|---|
| `results.jsonl` | one record per run: every engine parameter, both seeds, reference and output SHA-256 |
| `analysis.json` | descriptors for every track and every pairwise comparison |
| `references.json` | provenance of the two reference tracks |
| `scripts/reference_calibrate.py` | runs the six generations against the engine |
| `scripts/reference_analyse.py` | descriptor and comparison pass |

## Method

Six 30 s generations, one variable at a time, same Korean lyrics and same
seed throughout except where noted:

| Run | Prompt | Reference | Seed |
|---|---|---|---|
| 00 | base | — | 777777 |
| 01 | base | A (electronic) | 777777 |
| 02 | base | B (acoustic) | 777777 |
| 03 | contradictory | A | 777777 |
| 04 | contradictory | — | 777777 |
| 05 | base | A | 131313 |

Run 04 exists so run 03 is interpretable: without it, a difference in 03
could be the new prompt rather than the reference. Run 05 establishes the
seed noise floor, without which no difference between the others would
mean anything.

No strength sweep: source inspection found no scale, weight or dropout on
the reference stream. It is binary — a real reference, or a silence
latent — so any "influence level" would have been invented.

References are LUBER-generated instrumentals made for this experiment and
chosen to be far apart (centroid 2794 Hz vs 665 Hz). No commercial
recordings were used.

## Results

Spectral centroid, the clearest signal:

| Run | Centroid | Change |
|---|---|---|
| 00 prompt only | 1391 Hz | — |
| 01 + reference A | **2180 Hz** | **+789 toward A** |
| 02 + reference B | **1262 Hz** | **−129 toward B** |
| 05 reference A, other seed | 2219 Hz | +39 vs run 01 |
| 04 contradictory, no reference | 803 Hz | −588 |
| 03 contradictory + reference A | 1135 Hz | +332 vs run 04 |

Rolloff and flatness agree in direction. SI-SDR between any reference and
any output is around −51 dB.

## What the numbers support

- The reference **causally changes** the output, in the **correct
  direction for both references**.
- The effect is **~20× the seed noise floor** (789 Hz vs 39 Hz).
- It **survives a contradictory prompt** (+332 Hz).
- The **prompt is stronger** (~1045 Hz vs ~789 Hz): the reference
  modulates within the prompt's territory rather than overriding it.
- **Nothing is copied.** −51 dB SI-SDR, no sample-level relationship.
- **Lyrics are independent** — instrumental references, sung outputs.

## What they do not support

- **Magnitude is asymmetric**: A closed 56% of the gap, B only 18%.
- Only **production character** is demonstrated. Centroid, rolloff and
  flatness are three views of one dimension.
- **No evidence either way** on harmony, rhythm, melody or structure:
  chroma-sequence similarity was 0.76–0.86 for every pair including
  unrelated ones, and onset correlation 0.01–0.37 with no pattern.
- **MFCC cosine was useless**: 0.95–0.999 for everything, cross-controls
  included. Reported, used for nothing.
- **Tempo estimates are unreliable** (octave errors) and support no claim.
- **Vocal character is unmeasured.**

## Classification

**REFERENCE_STYLE_READY — provisional, pending listening.** Reliable in
direction, variable in magnitude, demonstrated for production character
only. Whether a 789 Hz brightness shift reads as style transfer is a
listening judgement, and the phase stops here until it is made.

## Reproducing

```
uv run python scripts/reference_calibrate.py <outdir> <reference_a.wav> <reference_b.wav>
uv run python scripts/reference_analyse.py <outdir>
```

`<outdir>/references/` must hold the two reference WAVs. Six generations,
roughly 45–90 s each on this hardware.
