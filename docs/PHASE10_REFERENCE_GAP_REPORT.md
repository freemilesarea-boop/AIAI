# Phase 10 — Reference Gap Report

> **Headline: the "excessive high-frequency energy" finding from Phase 9
> does not survive measurement against real commercial music, and I am
> retracting it.** LUBER's broadband brightness sits *inside* the
> commercial distribution on every measure. The perceived harshness is
> real, but it is a **dynamics** problem in a narrow band, not a tonal
> tilt — and that changes what Phase 10 should try to fix.

## 1. What the reference band is

`benchmarks/music_quality/reference/commercial_reference_profile.json`

- **n = 100 unique commercially released tracks**, FLAC, from a Billboard
  Hot 100 chart snapshot (2026-03-28) on the user's machine.
- A second on-disk folder held **byte-identical copies** of the same 100
  tracks; the content-hash dedup caught it, so the cohort is 100, not
  200. Worth stating plainly, because reporting "n=200" would have
  doubled the apparent evidence without adding a single track.
- Loudness and level from ffmpeg `ebur128`/`astats` over the whole file;
  spectral descriptors in numpy over the middle 60 s at 48 kHz.
- Aggregate statistics only. No filenames, paths, per-track values, or
  fingerprints are stored. Nothing was copied into the repository and
  nothing was trained on.

### Limitations of this reference set

1. **One chart snapshot of international pop.** It is not a Korean
   reference set, which matters for a Korean-first product.
2. **These are finished commercial masters.** LUBER output is
   deliberately unmastered. Some of the gap below is that difference and
   nothing more.
3. n=100 is enough for a distribution, not for genre-level claims.

## 2. The comparison

LUBER side: the three Phase 9 long-form gates plus four 30 s Phase 9
experiment tracks, measured through the **identical** descriptor
pipeline. "In band" means inside reference p10–p90.

| Descriptor | ref p10 | ref p50 | ref p90 | LUBER range | In band |
|---|---|---|---|---|---|
| integrated LUFS | −10.11 | −8.15 | −6.80 | −14.60 … −12.60 | **0/7** |
| RMS dBFS | −12.49 | −10.30 | −8.45 | −18.11 … −15.22 | **0/7** |
| crest factor dB | 8.45 | 10.14 | 12.40 | 14.22 … 17.11 | **0/7** |
| spectral centroid Hz | 2465 | 3138 | 3908 | 2396 … 3283 | 6/7 |
| energy > 8 kHz | 0.088 | 0.137 | 0.203 | 0.125 … 0.177 | **7/7** |
| energy > 10 kHz | 0.052 | 0.084 | 0.136 | 0.078 … 0.116 | **7/7** |
| band low (20–250) | 0.162 | 0.219 | 0.314 | 0.234 … 0.336 | 6/7 |
| band mid (250–4k) | 0.403 | 0.493 | 0.574 | 0.422 … 0.482 | 7/7 |
| band high (4k–20k) | 0.201 | 0.284 | 0.377 | 0.213 … 0.304 | 7/7 |
| stereo correlation | 0.644 | 0.815 | 0.941 | 0.562 … 0.791 | 5/7 |
| loudness range LU | 2.99 | 5.55 | 10.11 | 5.10 … 14.70 | 3/7 |
| dynamic range proxy dB | 2.24 | 4.01 | 7.56 | 4.37 … 51.51 | 2/7 |

## 3. Retraction of the Phase 9 brightness claim

Phase 9's results document states that a spectral centroid of ~3.0–3.1
kHz and ~15 % of energy above 8 kHz are "both very high for produced
music, where a centroid of 1.5–2.5 kHz and low single-digit
high-frequency percentages are typical", and concludes that the
"excessive high-frequency energy" complaint is "objectively real".

**That was wrong.** It was reasoning from remembered general norms
rather than from measurement. Measured against 100 actual commercial
releases:

- commercial median centroid is **3138 Hz** — LUBER's 3033–3137 Hz is
  essentially exactly the median, not an outlier;
- commercial median energy above 8 kHz is **0.137** — LUBER's
  0.125–0.177 straddles the median and stays inside p10–p90;
- LUBER is in band on `band_high`, `energy_above_10k` and `band_mid` too.

On broadband tonal balance, **LUBER is a normal-sounding modern record.**
Any Phase 10 work premised on "reduce the highs" would have been
chasing a defect that the reference data does not support.

The human perception of harshness is not thereby dismissed. It is
relocated — see §5.

## 4. The gaps that are real: loudness and dynamics

Three descriptors miss the band on **every single track**, all in the
same direction:

- **~5 dB quieter** than commercial (−14.6…−12.6 LUFS vs −10.1…−6.8).
- **~6 dB lower RMS**.
- **~4–5 dB more crest factor** (14.2–17.1 dB vs 8.4–12.4 dB), i.e.
  markedly *less* compressed.

This is a **mastering-stage difference, not a generation defect**, and
it is expected by design: LUBER's audio pipeline transcodes and does
nothing else, while every reference track has been through a commercial
mastering chain. It is the single largest measurable gap and also the
least interesting one, because it says nothing about whether the music
is any good.

It does have one real consequence: **A/B listening between LUBER output
and commercial music is invalid without loudness matching.** A 5 dB
level difference will dominate any subjective comparison. Any listening
test that pits LUBER against reference material must level-match first.

One LUBER track shows a dynamic-range proxy of **51.5 dB**, far outside
the reference p90 of 7.6. That is a near-silent window somewhere in the
track (almost certainly an intro or outro), not a mastering property.
Flagged for the listening pass rather than corrected.

## 5. Where the harshness actually lives

Broadband energy cannot distinguish "bright" from "harsh". A narrow-band
plus temporal analysis can, and it finds a clear, consistent signature.
Reference: 30 commercial tracks; LUBER: 5 tracks; medians of the
per-track band energy share.

| Band | ref p50 | ref p90 | LUBER median | |
|---|---|---|---|---|
| 2–3 kHz | 0.0921 | 0.1130 | **0.0541** | far below |
| 3–4 kHz | 0.0522 | 0.0646 | 0.0401 | below |
| 4–5 kHz | 0.0453 | 0.0584 | 0.0423 | below |
| 5–6 kHz | 0.0438 | 0.0589 | 0.0422 | below |
| 6–8 kHz | 0.0469 | 0.0627 | 0.0423 | below |
| 8–10 kHz | 0.0515 | 0.0617 | 0.0550 | above median |
| **10–12 kHz** | 0.0408 | 0.0587 | **0.0549** | above median, near p90 |
| 12–16 kHz | 0.0288 | 0.0454 | 0.0314 | above median |
| 16–20 kHz | 0.0098 | 0.0208 | **0.0062** | far below |

**Temporal character of the 5–8 kHz band** — the measurement that
separates a bright mix from a harsh one:

| | reference | LUBER | ratio |
|---|---|---|---|
| mean 5–8 kHz share | 0.0837 | 0.0621 | **0.74×** |
| 99th-percentile share | 0.2415 | 0.2795 | **1.16×** |
| burstiness (p99 / mean) | 2.93 | 4.47 | **1.52×** |

### What this says

LUBER has **less** average energy in the sibilance region than
commercial music, but **higher peaks** there, and its high-mid energy is
**1.5× burstier**. That is the acoustic signature of **un-de-essed,
uncompressed sibilants**: the average is fine, individual consonants
spike. Commercial masters have de-essing and compression that hold those
peaks down; LUBER has neither.

Two further shape differences worth recording:

- **A scoop at 2–4 kHz** (0.054 vs 0.092 at 2–3 kHz — roughly *half* the
  commercial share). This is the presence/intelligibility region. A
  scooped 2–4 kHz with intact 8–12 kHz is a recognisable "thin and
  glassy" character, and is a plausible contributor to both the
  "poor instrument texture" and the Korean-consonant-clarity complaints.
- **A rolled-off top octave** above 16 kHz (0.0062 vs 0.0098), which is
  consistent with a codec or VAE band limit rather than a mixing choice.

## 6. What this redirects

| Planned Phase 10 step | Verdict from this data |
|---|---|
| §14 gentle tonal EQ toward the reference band | **Largely unjustified as originally framed.** LUBER is already inside the band on every broadband descriptor. Pulling the highs down would move it *away* from commercial norms. The only defensible tonal move this data supports is a small **2–4 kHz presence lift**, which is the opposite of what "too bright" implied. |
| §13 conservative dynamic de-essing | **Justified, and now targeted.** The evidence is the 1.52× burstiness and the 1.16× peak share in 5–8 kHz — a dynamics problem with a specific band and a specific mechanism. |
| Loudness/dynamics gap | Real but expected; a mastering question, not a model question. Must be neutralised (level-matched) before any human A/B against commercial material. |

## 7. What this cannot tell you

- Whether any of it *sounds* better. Every number here is a descriptor,
  not a judgement.
- Whether the 2–4 kHz scoop is a model property or a property of these
  particular prompts.
- Whether the burstiness is vocal sibilance specifically, or cymbals and
  transients. The 5–8 kHz band contains both. Establishing that it is
  vocal requires listening to the isolated moments, or comparing
  instrumental against vocal generations — see the raw-output
  root-cause work.
