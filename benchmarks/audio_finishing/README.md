# Audio finishing baseline (Phase 14A)

Does the reported perceptual weakness — dull, flat, narrow, sometimes
harsh — show up in measurements of LUBER's own output? And if it does,
what should an engine be allowed to do about it?

Engine and full reasoning: `docs/AUDIO_FINISHING_ARCHITECTURE_AUDIT.md`.
Engine version `p14-v1`. No production behaviour changed.

**Status: engine built and measured, awaiting human listening. It is not
integrated into generation.**

## Contents

Metadata and code only. No audio is committed; the A/B pairs live in
`~/Desktop/LUBER_PHASE14_FINISHING_LISTENING/`.

| File | What it is |
|---|---|
| `baseline_results.jsonl` | full analysis of every corpus track, one per line |
| `baseline_summary.json` | corpus distributions for every metric and band |
| `ab_results.json` | the five A/B pairs: plan, filter graph, before/after, safety |
| `scripts/finishing_baseline.py` | analyses a corpus and summarises it |
| `scripts/finishing_ab.py` | builds the listening package and measures both sides |

## Corpus

40 existing LUBER masters — all 32 in local storage (20 x 30 s, 8 x 60 s,
4 x 180 s) plus 8 outputs from the Phase 13D cover and 13E reference
calibrations. Nothing was generated for this phase. No commercial music.
No directory scanning: paths are arguments.

## What the numbers say about the listening report

| Claim | Verdict |
|---|---|
| Tonal balance is inconsistent | **Supported, strongly.** Air-to-midrange spans -39.0 to -13.4 dB across one model's own output; sd 6.0 dB |
| Upper frequencies often roll off | **Supported for a minority.** 9/40 dark in both slope and air; 11/40 in the darkest quarter |
| Excessive highs, harshness, sibilance | **Also present.** Sibilance peak excess to 20.0 dB, harshness to 22.2 dB |
| Both at once | **8/40 tracks.** This is why no fixed shelf could be correct |
| Instruments feel flat | **Not visible here.** 50 ms crest runs 7.0-9.3 dB against a 6.5 dB flatness threshold |
| Stereo can feel narrow | **A tail, not a rule.** 4/40 below the narrow threshold |

One more measured fact shapes everything downstream: every generated
master arrives at **exactly -1.0 dBFS**, sd 0.00. There is 1 dB of
headroom in the whole catalogue and every correction is paid out of it.

## Thresholds

Absolute, not corpus percentiles. Percentile thresholds would define
"correct" as "average for this model", guarantee a fixed fraction is
always flagged, and make the engine chase its own output.

Ratios are measured against the 400 Hz-2 kHz midrange, which overlaps
none of the bands compared to it. An earlier 300 Hz-3 kHz reference
shared 300-400 Hz with the low-mid band and 2.5-3 kHz with the harshness
band; a ratio whose numerator sits in its denominator saturates, and both
measurements did.

| Flag | Threshold | Corpus hits |
|---|---|---|
| HIGH_FREQUENCY_DEFICIT | air < -27 dB and slope < -6.5 dB/oct | 9/40 |
| AIR_DEFICIT | air < -30 dB | 11/40 |
| PRESENCE_DEFICIT | presence < -21 dB | 7/40 |
| LOW_MID_MUD | low-mid > +5.5 dB | 9/40 |
| LOW_END_EXCESS | sub+bass share > 0.70 | 3/40 |
| SIBILANCE_RISK | peak excess > 17 dB and p90 > -17 dB | 5/40 |
| HARSHNESS_RISK | peak excess > 14 dB and p90 > -11 dB | 8/40 |
| STEREO_TOO_NARROW | width < 0.11 | 4/40 |
| STEREO_TOO_WIDE | width > 0.45 | 0/40 |
| TRANSIENT_FLATNESS | 50 ms crest < 6.5 dB | 0/40 |

**15 of 32 storage masters produce NO_ACTION.** Doing nothing is the
engine's most common answer, and a flag that fired on most of a corpus
would be describing the model rather than detecting a defect.

Width is measured above 120 Hz. Side energy below that is not image
width — it is bass that partially cancels in mono, and the engine's own
repair is to remove it. Counting it would make a track measure *narrower*
after being fixed, which happened before the metric was corrected.

## Reproducing

```
uv run python scripts/finishing_baseline.py <outdir> <master.wav> [...]
uv run python scripts/finishing_ab.py <listening-dir> <outdir> <master.wav> [...]
```

Sequential by design; the whole 32-track corpus finishes in about 24 s on
an M-series laptop.
