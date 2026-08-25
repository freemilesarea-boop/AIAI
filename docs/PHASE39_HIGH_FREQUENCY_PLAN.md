# Phase 39 — the high band, and what the evidence says to change

Phase 38 fixed the two things it set out to fix and did not fix the
third. The operator heard rhythm and arrangement move, vocal melody
hold, and the high end stay closed — with a metallic character that is
worse than merely dull.

This document is the diagnosis and a proposed experiment. Nothing has
been trained for Phase 39.

## The measurement that decides it

Band energies are stated against the same file's 200–2000 Hz band, so a
level difference cannot masquerade as a tilt. Flatness is
geometric-over-arithmetic mean: 1.0 is noise, low is tonal. Narrow peaks
are dB above a median-smoothed floor between 4 and 16 kHz.

| stage | 8–12 kHz | flatness 6–12 kHz | narrow peaks |
|---|---|---|---|
| source library | −12.50 dB | 0.925 | 0 |
| training windows | −13.05 dB | 0.924 | 0 |
| VAE encode → decode | −14.81 dB | 0.903 | 2 |
| generated, base, **no adapter** | −18.12 dB | 0.830 | 16 |
| generated, Phase 38 | −17.61 dB | 0.793 | 12 |

Windowing costs nothing. The VAE costs about a decibel. The diffusion
stage costs three more, collapses the flatness that makes air sound like
air, and adds ten to fifteen resonances that no reference file has.

**The base model with no adapter attached shows the same defect.** Phase
38 sits *above* it in 4–12 kHz. Whatever this is, the adapter did not
introduce it.

## What it is not

**Not the dataset.** The Phase 38 selection is the brightest subset in
the library: HF energy share 0.0205, against 0.0096 for the library and
0.0077 for the POP tracks it declined. The tiering did exactly what it
was designed to do. The model then failed to reproduce the brightness it
was given.

**Not a filter.** 48 kHz end to end. The only resample in the code is
the MP3 export branch, which FLAC output never takes. The only
post-processing is a peak normalisation to −1 dBFS — which is why every
generated file peaks at exactly 0.891251. Gain, not EQ.

**Not an inference setting.** Steps were swept 8 → 16 → 32 → 60 and
shift 1.0 → 3.0 → 5.0 → 7.0. No setting closed the gap and every one of
them kept 10–24 narrow peaks against the VAE round trip's 2. More steps
did not help; the current shift of 3.0 was among the better ones.

## What it is

Not missing energy. 16–20 kHz is already comparable to reference; the
deficit sits in 6–16 kHz *and* arrives with tonal structure. So it is a
distribution problem (C), narrow-band resonance (D) and a loss of
broadband noise texture (E), all arising in the generation stage.

**This is why a boost is the wrong instrument.** There is no smooth tilt
to tilt back. Lifting 6–16 kHz raises the resonances along with the air
and makes the metallic character louder, not smaller. The operator's
instinct not to assume a boost was correct.

## The proposal, and its honest odds

The tiering scored HIGH_END from high-band *energy*: energy ratio,
high-band RMS, spectral centroid. Every one of those is a level measure.
The defect the operator described is a *texture* — flatness 0.79 where
the material sits at 0.92. **The axis we selected on does not measure
the thing that is wrong.**

So the smallest change the evidence actually supports is to add one
measurement, not to change the training:

- add high-band spectral flatness (6–16 kHz) to `audio_features`
- re-tier with it carried in the HIGH_END axis alongside the energy terms
- keep rank, alpha, optimizer, learning rate, precision, batch geometry,
  epochs, step ceiling and dataset size exactly where Phase 38 has them
- keep beat-aware window starts and arrangement weighting untouched, so
  the two mechanisms credited with the rhythm and arrangement gains are
  preserved by composition rather than rebuilt

Regression protection is structural: the rhythm mechanism (onset-snapped
starts) and the arrangement mechanism (per-window weighting) are not
touched, and RHYTHM, ARRANGEMENT and VOCAL stay on the A/B card as
blocking axes, not as notes.

**The odds should be stated plainly.** The base model shows this defect
with no adapter, and roughly 2.8 dB of the 4.5 dB gap is available
before the VAE becomes the limit. Phase 37 reached −16.07 dB, so a LoRA
can move this band. But a data-selection change may well not fix a
defect that lives in the base model's sampler, and the experiment must
be allowed to return "no effect" without being retried into a result.

If it does return no effect, the next lever is not another LoRA. It is
the decode path or the base checkpoint, and that is a different kind of
change than this phase should make.

## A known bias that bears on all of this

Phase 38's own document records it: `PreprocessedDataModule` takes no
per-sample weight, so the arrangement weights were computed, recorded as
evidence, and **never enforced by the loader**. The same would be true of
any flatness weighting. Enforcing them needs a weighted sampler, which
is a trainer change and would alter rhythm and arrangement exposure at
the same time — so it is a separate experiment with its own regression
risk, not a rider on this one.

## Status

Diagnosis complete. No Phase 39 training has been run and no long run is
scheduled. The A/B card below is proposed, not executed.

| axis | Phase 38 control | Phase 39 candidate |
|---|---|---|
| air / openness | | |
| metallic artifact | | |
| high-frequency naturalness | | |
| vocal sibilance | | |
| cymbal / percussion texture | | |
| rhythm continuity — **must not regress** | | |
| arrangement richness — **must not regress** | | |
| vocal melody — **must not regress** | | |

Identical prompts, lyrics, seeds, duration and generation settings on
both sides. The operator decides; nothing here selects a winner.
