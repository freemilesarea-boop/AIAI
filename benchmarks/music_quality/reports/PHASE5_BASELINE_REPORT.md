# Phase 5 Baseline Report — LUBER_BASELINE_P5_V1

Measurement only. No training, no prompt tuning, no configuration changes were made in response to these results.

## Executive Summary

- Generations attempted: **26**
- Completed: **26** (100%)
- Technical failure rate: **0.0%** (gate: <2%)
- Human evaluations recorded: **0**

> **Status of this report.** Every number below comes from 26 real
> ACE-Step 1.5 generations executed through the production Phase 4
> pipeline. No mock provider was used, no prompt was edited mid-run, and
> no configuration was tuned in response to a result.
>
> **This report contains no human listening scores.** The listening
> instrument, rubric, and blind A/B system are built and ready, but the
> evaluation itself requires a human ear and has not been performed.
> Everything reported under "objective analysis" is a *measurement*, not
> a judgement of whether the music is good. See
> "What this baseline does and does not establish".


## Benchmark Configuration

- Baseline id: `LUBER_BASELINE_P5_V1`
- Benchmark version: `v1`
- ACE-Step version: `1.5.0` @ `6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`
- Configurations: `A_turbo_dit_only`
- Models: `acestep-v15-turbo`

## Hardware

- CPU: Apple M4 Pro
- Cores: 14
- Memory: 24 GB
- Backend: Apple Silicon MPS + MLX (ACE-Step macOS path)

## Generation Counts

- Korean vocal: **17**
- English vocal: **5**
- Instrumental: **4**
- Long-form (>=180s): **4**

## Generation Speed

- Wall-clock per generation: median **44.8s**, min 25.2s, max 105.2s
- Real-time factor: median **0.94x** (wall-clock seconds per second of audio)

## Technical Failure Flags

_No data._

## Genre Breakdown (overall musical quality)

_No data._

## Language Breakdown

_No data._

_No data._

## Duration Breakdown

_No data._

## Human Scores vs Internal Quality Gate

| Dimension | Measured / Target |
|---|---|
| overall_musical_quality | n/a / 8.0 → **MISS** |
| commercial_release_readiness | n/a / 8.0 → **MISS** |
| vocal_naturalness | n/a / 8.0 → **MISS** |
| lyrics_pronunciation | n/a / 8.0 → **MISS** |
| prompt_adherence | n/a / 8.0 → **MISS** |
| song_structure | n/a / 8.0 → **MISS** |

### All rubric dimensions

_No data._

## Artifact Frequency

_No data._

_Artifact-rate gate: <10% of tracks with an obvious artifact._

## Seed Variance

_No data._

---

# Objective Musical-Structure Analysis

Measured with `bench/analysis.py` on all 26 completed masters. Raw
per-track values are committed in
`results/pilot_baseline_p5_v1_structure.jsonl`.

These are **measurements, not verdicts**. None of them can tell you
whether a melody is good or a vocal sounds human. What they can do is
localise *where* to listen, and they surfaced one strong pattern the
pipeline metrics completely missed.

## Finding A — 30-second tracks systematically fade out (highest impact)

| Requested duration | n | Median end-level drift | Median silence ratio | Median sections | Median repetition |
|---|---|---|---|---|---|
| **30 s** | 15 | **−9.68 dB** | 0.073 | 3 | 0.978 |
| 60 s | 7 | **−0.19 dB** | 0.032 | 8 | 0.954 |
| 180 s | 4 | **−1.95 dB** | 0.012 | 21 | 0.949 |

"End-level drift" is the loudness of the final third minus the first
third. At 30 s the track is typically **~10 dB quieter at the end than
at the start**; at 60 s and 180 s it is essentially flat.

Worst individual cases, all at 30 s:

| Track | End-level drift | Silence ratio |
|---|---|---|
| ROCK-04 | **−30.91 dB** | 0.183 |
| KPOP-06 | −22.99 dB | 0.153 |
| RNB-04 | −20.66 dB | 0.127 |
| BALLAD-01 | −17.68 dB | 0.167 |
| KPOP-01 (seed 1001) | −16.85 dB | 0.000 |
| JAZZ-01 | −15.04 dB | 0.117 |

**Why this matters.** 30 seconds is the *default preset in the LUBER
UI* and the shorter of only two options users are offered. The
configuration most users will hit is the one that most reliably
produces a track that trails off into near-silence. A user asking for a
30-second track is asking for 30 seconds of music, not ~22 seconds
followed by a fade.

**What this is not.** These tracks passed every technical gate — none
were flagged `EXCESSIVE_SILENCE` (the threshold is 35% and the worst
here is 18%). This is exactly the class of problem technical metrics
miss and structural analysis catches.

## Finding B — long-form is the *strongest* tier, not the weakest

The usual assumption is that long-form degrades. Measured here, 180 s
tracks have the **lowest** end-level drift (−1.95 dB), the **lowest**
silence ratio (0.012), the **lowest** repetition (0.949), the **most**
detected section changes (median 21), and the **lowest** spectral drift
between first and last third (10.6–19.4, vs 20–72 for 30 s).

All four 180 s generations completed with no technical flags, including
three Korean vocal songs with full `[Intro]/[Verse]/[Pre-Chorus]/
[Chorus]/[Bridge]/[Final Chorus]/[Outro]` structures.

Long-form musical *quality* is still unassessed — drift being low does
not mean the bridge is any good — but there is no objective evidence of
long-form collapse in this baseline, and the resource cost is
surprisingly low (see Finding D).

## Finding C — seeds change structural quality substantially

| Prompt | Spectral divergence | Energy variation across seeds | Repetition across seeds |
|---|---|---|---|
| BALLAD-01 | 0.403 | 13.8 / 11.8 / **5.0** (spread 8.8) | 1.000 / 1.000 / 0.864 |
| KPOP-01 | 0.325 | 12.6 / 12.2 / **6.3** (spread 6.4) | 0.935 / 0.995 / 0.896 |

Same prompt, same configuration, three seeds: one take has more than
**twice** the dynamic variation of another. The model is not producing
a consistent quality level — it is sampling across a wide band, and a
single lucky generation would badly misrepresent it. This is the
concrete justification for never judging the engine on one output.

## Finding D — speed is not the bottleneck

- Wall-clock per generation: median **44.8 s** (min 25.2 s, max 105.2 s)
- Real-time factor: median **0.94×** (min 0.26×, max 2.86×)
- 180 s tracks ran at RTF **0.26–0.31×** — three minutes of audio in
  under a minute of compute.

Longer requests are *more* compute-efficient per second of audio. There
is no performance argument for keeping users on the 30 s preset.

## Finding E — near-total self-similarity clusters at 30 s

Six tracks scored max repetition ≥ 0.999, meaning frames far apart in
the track are almost spectrally identical: ACOUSTIC-01 (0.9999),
LOFI-01 (0.9999), BALLAD-01 (0.9999 / 0.9995 / 0.9992), ROCK-04
(0.9996). **Every one is a 30 s track.** For LOFI a static loop is
stylistically fine; for a ballad or a rock song it suggests one texture
repeated rather than a section that develops.

## Finding F — upstream normalizes peak level to exactly −1.0 dBFS

Every one of the 26 masters peaks at 0.8912–0.8913 (−1.00 dBFS) with no
clipping anywhere. ACE-Step is peak-normalizing its output. Two
consequences: peak level carries no diagnostic information in this
benchmark, and LUBER's Phase 4 decision to add no normalization of its
own is correct — the level is already controlled upstream.

---

# Prompt Compiler Audit

`AceStepPromptCompiler` appends conditioning to every user prompt. Both
the original and compiled prompt are recorded per generation so a
quality problem can be attributed rather than guessed at.

| Input | Compiled output |
|---|---|
| `bright K-pop with female vocal` | `bright K-pop with female vocal, female lead vocal, natural female singing voice` |
| `Instrumental K-pop backing track, …, no vocals` | `Instrumental K-pop backing track, …, no vocals, instrumental, no vocals` |
| `lo-fi chill beat to study to` | `lo-fi chill beat to study to, instrumental, no vocals` |

**Finding G — the compiler duplicates conditioning it does not check for.**
When a user already writes "female vocal", the compiler appends "female
lead vocal, natural female singing voice" anyway — the concept appears
three times. For instrumentals the duplication is literal: a prompt
ending "no vocals" becomes "…no vocals, instrumental, no vocals".

This is unconditional string concatenation with no awareness of what
the prompt already says. It spends prompt budget restating conditioning
instead of describing music, and it is the cheapest possible thing to
fix — no training, no configuration, one function.

Whether it actually degrades output is **not established by this
baseline**; it is a hypothesis with an obvious A/B test attached.

---

# Configuration Experiments — resource outcomes

## LM-enabled configuration: `GPU_REQUIRED_FOR_LM_BENCHMARK`

Not run. Not because the weights are missing — `acestep-5Hz-lm-1.7B`
(3.5 GB) is already downloaded locally — but because the machine has no
headroom:

| Measurement immediately before the planned experiment | Value |
|---|---|
| Swap in use | **16.05 GB of 17.41 GB (92%)** |
| Free swap | 1.35 GB |
| Free disk | 18 GB |

Loading a further 3.5 GB model into that state would force macOS to
grow its swapfile against limited disk. This is not speculative: during
Phase 2 on this same machine, enabling the LM drove swap to **18.7 GB**
and free disk to **5.7 GB**, which is why `ACESTEP_INIT_LLM=false` is
the production setting. Classified `BLOCKED_LOCAL_RESOURCE` /
`GPU_REQUIRED_FOR_LM_BENCHMARK` rather than risk destabilising the
development machine.

## XL Turbo configuration: `GPU_REQUIRED_FOR_XL_BENCHMARK`

Not run and **not downloaded**. `acestep-v15-xl-turbo` is a ~9 GB
download (weights ~9 GB resident vs ~4.7 GB for the 2B) against 18 GB
free disk and the swap position above. Upstream's compatibility table
rates XL as supported at this tier only *with CPU offload*, and that
table describes discrete CUDA VRAM, not Apple unified memory shared
with the OS.

Worth noting from the audit: upstream's own Model Zoo rates
`acestep-v15-turbo` — the model we already run — as **"Very High"**
quality, the same rating it gives `acestep-v15-xl-turbo`. XL's claimed
advantage is audio fidelity from a larger decoder, not better
composition. XL is therefore a weaker Phase 6 candidate than its size
suggests.

---

# What this baseline does and does not establish

## Established

- The production pipeline generates real ACE-Step music reliably:
  **26/26 completed, 0% technical failure rate** (gate: <2%).
- No silent, corrupted, clipped, or wrong-duration output occurred.
- Generation speed is comfortable and improves with length.
- Structural behaviour differs sharply by duration, and the **30 s
  default is the weakest tier** by every structural measure taken.
- Seed choice materially changes structural quality for the same prompt.
- The prompt compiler emits duplicated conditioning.

## NOT established

- **Whether the music is any good.** No human has listened to these
  tracks. There are zero rubric scores in `listening/scores.jsonl`.
- Vocal naturalness, vocal tone, and whether the voice is convincing.
- **Korean pronunciation quality** — 받침, 연음, mixed Latin/numerals.
  This was a primary Phase 5 concern and it is entirely unmeasured; no
  automated proxy for it exists in this toolkit and none was invented.
- Lyrics alignment: whether the supplied lyrics are actually sung, in
  order, in the right sections.
- Melodic and harmonic quality; genre authenticity; mix balance.
- Commercial release readiness.

## Suno 4.5 parity

**INTERNAL QUALITY BASELINE ONLY.**
**SUNO 4.5 PARITY NOT YET VERIFIED.**

No Suno reference audio exists in this project. None was obtained, and
no scraping, automation, or account access was attempted — the
reference importer is local-file-only by construction and has no
network capability (enforced by test). Upstream's README claims quality
"between Suno v4.5 and Suno v5"; that is an upstream marketing claim,
it is recorded in the audit, and it is **not** treated as evidence.

Parity can only be claimed after a blinded A/B against reference audio
the user legitimately holds. The blind A/B system is built and waiting
for that input.

---

# Quality Gap Classification

| # | Finding | Classification | Confidence |
|---|---|---|---|
| A | 30 s tracks fade out ~10 dB | `INFERENCE_CONFIGURATION_PROBLEM` | High — measured, n=15 |
| E | Near-total repetition at 30 s | `MODEL_CAPABILITY_PROBLEM` (short-form) | Medium |
| C | Large seed-to-seed quality spread | `MODEL_CAPABILITY_PROBLEM` | High — measured |
| G | Compiler duplicates conditioning | `PROMPT_COMPILER_PROBLEM` | High — confirmed in code |
| Q4 | `bpm` / `key_scale` / `time_signature` never sent | `INFERENCE_CONFIGURATION_PROBLEM` | High — confirmed in code |
| Q5 | `infer_method=sde` never tried | `INFERENCE_CONFIGURATION_PROBLEM` | Unknown |
| — | Korean pronunciation | `UNKNOWN` — unmeasured | n/a |
| — | Vocal naturalness | `UNKNOWN` — unmeasured | n/a |
| — | Melody / harmony quality | `UNKNOWN` — unmeasured | n/a |
| — | LM contribution | `UNKNOWN` — `GPU_REQUIRED` | n/a |
| — | XL contribution | `UNKNOWN` — `GPU_REQUIRED` | n/a |

Note how much of the table is `UNKNOWN`. That is the honest state after
a measurement-only phase with no listening: the objective work found
real problems, but the questions that actually decide whether this
product competes — vocals, Korean, melody — are still open.

---

# Recommended Next Experiments (Phase 6 candidates)

Ordered by evidence strength and cost. None require training.

1. **Human listening pass on the 26 existing tracks.** Zero generation
   cost — the audio, tool, and rubric already exist. This converts most
   of the `UNKNOWN` rows above into measured findings and is the
   prerequisite for every quality decision after it.
2. **Duration A/B: 30 s vs 60 s on identical prompts.** Directly tests
   Finding A. If 60 s is consistently better, the cheapest quality win
   available is changing the default preset.
3. **Metadata conditioning A/B.** Send `bpm` / `key_scale` /
   `time_signature` (Finding Q4) on the same prompts and compare.
4. **Prompt compiler de-duplication A/B** (Finding G).
5. **Seed-count strategy.** Given Finding C, generating 2–3 candidates
   and selecting may raise delivered quality more than any parameter
   change.
6. **LM and XL benchmarks on an NVIDIA GPU** — blocked locally.

