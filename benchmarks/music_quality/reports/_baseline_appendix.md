---

# Objective Musical-Structure Analysis

Measured with `bench/analysis.py` on all 26 completed masters. Raw
per-track values are in `results/pilot_baseline_p5_v1_structure.jsonl`.

These are **measurements, not verdicts**. The human verdict at the top
of this report is the authority on quality; this section only localises
*where* the problems live.

## Finding A — 30-second tracks systematically fade out

| Requested duration | n | Median end-level drift | Median silence ratio | Median sections | Median repetition |
|---|---|---|---|---|---|
| **30 s** | 15 | **−9.68 dB** | 0.073 | 3 | 0.978 |
| 60 s | 7 | **−0.19 dB** | 0.032 | 8 | 0.954 |
| 180 s | 4 | **−1.95 dB** | 0.012 | 21 | 0.949 |

"End-level drift" is the loudness of the final third minus the first
third. At 30 s a track is typically **~10 dB quieter at the end than at
the start**; at 60 s and 180 s it is essentially flat.

Worst cases, all 30 s: ROCK-04 (−30.91 dB, 18.3% silence), KPOP-06
(−22.99 dB), RNB-04 (−20.66 dB), BALLAD-01 (−17.68 dB), KPOP-01
(−16.85 dB), JAZZ-01 (−15.04 dB).

30 s was the default preset in the UI and the shorter of only two
options offered. **Phase 6 retires it as the primary quality tier.**

## Finding B — long-form is the strongest tier, not the weakest

180 s tracks have the lowest end-level drift (−1.95 dB), lowest silence
(0.012), lowest repetition (0.949), most section changes (median 21),
and lowest spectral drift. All four completed with no technical flags,
including three Korean vocal songs with full
`[Intro]/[Verse]/[Pre-Chorus]/[Chorus]/[Bridge]/[Final Chorus]/[Outro]`
structures. There is no objective evidence of long-form collapse.

## Finding C — seeds change structural quality substantially

| Prompt | Spectral divergence | Energy variation across seeds | Repetition across seeds |
|---|---|---|---|
| BALLAD-01 | 0.403 | 13.8 / 11.8 / **5.0** (spread 8.8) | 1.000 / 1.000 / 0.864 |
| KPOP-01 | 0.325 | 12.6 / 12.2 / **6.3** (spread 6.4) | 0.935 / 0.995 / 0.896 |

Same prompt, same configuration, three seeds: one take has more than
**twice** the dynamic variation of another. No single generation
represents the engine.

## Finding D — speed is not the bottleneck

Median wall-clock 44.8 s (min 25.2, max 105.2); median real-time factor
**0.94×**; 180 s tracks at RTF **0.26–0.31×**. Longer requests are more
compute-efficient per second of audio.

## Finding E — near-total self-similarity clusters at 30 s

Six tracks scored max repetition ≥ 0.999 — ACOUSTIC-01, LOFI-01,
BALLAD-01 (×3), ROCK-04 — and **every one is 30 s**.

## Finding F — upstream normalizes peak to exactly −1.0 dBFS

All 26 masters peak at 0.8912–0.8913 with no clipping. ACE-Step
peak-normalizes its output, so peak level carries no diagnostic
information here, and LUBER's Phase 4 decision to add no normalization
of its own is correct.

Note this does **not** contradict the human "excessive high frequency"
finding: peak normalization controls level, not spectral balance.

---

# Prompt Compiler Audit

`AceStepPromptCompiler` appended conditioning unconditionally, with no
check of what the prompt already said:

| Input | Compiled output |
|---|---|
| `bright K-pop with female vocal` | `bright K-pop with female vocal, female lead vocal, natural female singing voice` |
| `Instrumental K-pop backing track, …, no vocals` | `Instrumental K-pop backing track, …, no vocals, instrumental, no vocals` |

**Finding G.** The concept appears up to three times, and instrumentals
duplicate "no vocals" literally. This spends prompt budget restating
conditioning instead of describing music.

Fixed in Phase 6 (see `docs/PHASE6_QUALITY_ROOT_CAUSE_MAP.md` and the
compiler A/B results).

---

# Configuration Experiments — resource outcomes

## LM-enabled: `GPU_REQUIRED_FOR_LM_BENCHMARK`

Not run. `acestep-5Hz-lm-1.7B` (3.5 GB) is present locally, but swap was
at **16.05 GB of 17.41 GB (92%)** with 18 GB free disk immediately
before the planned experiment. Phase 2 measured the LM driving swap to
**18.7 GB** and free disk to **5.7 GB** on this same machine.

## XL Turbo: `GPU_REQUIRED_FOR_XL_BENCHMARK`

Not run, not downloaded (~9 GB against limited disk). Upstream's own
Model Zoo rates `acestep-v15-turbo` — the model already in use — as
**"Very High"**, the same rating it gives `acestep-v15-xl-turbo`. XL's
claimed advantage is decoder fidelity, not composition.

---

# What this baseline established

## Objectively

- 26/26 completed, **0% technical failure rate** (gate <2%).
- No silent, corrupted, clipped, or wrong-duration output.
- Generation speed is comfortable and improves with length.
- The 30 s tier is the weakest by every structural measure.
- Seed choice materially changes structural quality.
- The prompt compiler emitted duplicated conditioning.

## By human listening

The evaluator reviewed all 26 tracks and **rejected every one**,
scoring the baseline **2/10** overall. The failures are musical and
sonic, not pipeline failures:

- frequency balance, instrument fidelity, and overall texture are poor
- excessive high-frequency energy and sibilance
- Korean lyric sentences frequently omitted; lines skipped
- unwanted trot-like (뽕끼) vocal character
- vocal style not contemporary for the K-pop / pop / R&B target

**The technical pipeline passed and the music failed.** That divergence
is the single most important result of Phase 5: 0% technical failure
alongside 100% human rejection means every remaining quality problem
lives in the model and its conditioning, not in LUBER's plumbing.

## Suno 4.5 parity

**INTERNAL QUALITY BASELINE ONLY. SUNO 4.5 PARITY NOT ACHIEVED.**

No Suno reference audio exists in this project; none was obtained, and
no scraping or automation was attempted. Upstream's README claim of
quality "between Suno v4.5 and Suno v5" is recorded in the audit as a
marketing claim and is contradicted, for this product's Korean-pop
target, by the evaluator's 2/10.
