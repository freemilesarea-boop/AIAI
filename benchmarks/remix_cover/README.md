# Cover / audio-to-audio calibration (Phase 13D)

Does ACE-Step's `cover` task condition strongly enough on a source
recording to justify a LUBER product feature — and if so, what should
that feature honestly be called?

Engine contract: `docs/ACE_STEP_COVER_AUDIT.md`.
Engine: ACE-Step `6d467e4b`, model `acestep-v15-turbo`, 8 steps, MLX,
LM disabled.

**Outcome: COVER_ONLY. Shipped as "Create Cover" (Phase 13D-2).**

The measurements alone stopped short of a product decision — they showed
real but modest source influence and could not establish that the song is
*recognisable*, which is a listening judgement. The product owner has
since listened and supplied it; see "Human listening verdict" below. The
classification moved from WEAK_AUDIO_CONDITIONING to COVER_ONLY on that
evidence, and the feature ships under the name the engine's behaviour
supports.

## Human listening verdict

Recorded 2026-08-14 from the product owner, after listening to the package
in `~/Desktop/LUBER_PHASE13D_LISTENING/`:

- The source-conditioned results are musically usable enough to justify
  exposing the capability.
- Subjectively, and given that no project-specific fine-tuning or training
  has been done, overall generation quality feels roughly comparable to an
  early-generation Suno v2-era system.
- The operation behaves more truthfully as a **cover** — source-conditioned
  regeneration — than as a preservation-based remix.

Two limits on how that verdict may be used:

1. It is a **subjective product-owner assessment, not a measured
   benchmark**. LUBER has run no comparison against Suno, and nothing in
   this repository may state or imply that LUBER objectively matches Suno
   v2. The remark is recorded because it is the reason the gate opened,
   not as a performance claim.
2. It concerns usability, not preservation. The measurements below still
   stand: the recording is not retained, and no product copy may say it is.

The engine findings are unchanged: the source demonstrably influences the
output, and that influence collapses below `audio_cover_strength = 0.75`.

## Contents

Metadata and code only. Audio is not committed — the WAVs live in the
scratchpad and in `~/Desktop/LUBER_PHASE13D_LISTENING/`.

| File | What it is |
|---|---|
| `results.jsonl` | one record per run: every engine parameter, both seeds, output SHA-256 |
| `analysis_signal_and_timbre.json` | waveform/SI-SDR/spectral/duration measurements |
| `analysis_musical_timevarying.json` | chroma-sequence, pitch-contour and structure measurements |
| `calibrate.py` | runs the seven generations against the engine |
| `analyse_signal.py` | signal + timbre pass |
| `analyse_musical.py` | time-varying musical pass |
| `wav_metrics.py` | shared WAV reader and SI-SDR |

## Method

One source: LUBER generation `a9ae6249-0d22-49d9-99c0-afeb64f88575`,
30.000 s, 48 kHz stereo pcm_s24le, SHA-256 `3cb9ff97…`, Korean female
vocal, known prompt and lyrics.

Seven runs, one variable at a time, `seed=424242` fixed throughout:

- **Baseline** — source's own prompt, `audio_cover_strength = 1.00`
- **Strength sweep** — same prompt, strength 0.75 / 0.50 / 0.25
- **Style transfer** — strength 1.00, three target prompts (polished
  K-pop, contemporary R&B / neo-soul, indie pop). No artist names.

Strength values were chosen from the source, not invented: turbo runs 8
diffusion steps and `cover_steps = int(8 × strength)`, so the dial
quantises to eighths and 1.00 / 0.75 / 0.50 / 0.25 spans the usable range.

`cover_noise_strength` was **not** swept. It is implemented only in the
PyTorch sampler and never reaches the MLX path this deployment runs
(`server.log`: 97/97 generations `via MLX`), so sweeping it would have
produced four identical files and a false finding.

## Two different questions, kept apart

**Signal preservation** — "is this the same recording?" A transformation
*should* score low. High would mean nothing changed.

**Musical preservation** — "is this the same song?" This survives
transformation, so it is what decides whether the output derives from the
source at all.

Every number is read against controls, because a bare correlation means
nothing:

- an **independent generation from the same prompt and lyrics** — isolates
  what the *source audio* contributed beyond the prompt;
- a **block-shuffled copy of the source** — same audio content, wrong
  order, so it isolates genuine time alignment.

## Results

Signal, and time-averaged timbre:

| run | dur | wav corr | SI-SDR dB | centroid Hz |
|---|---|---|---|---|
| baseline 1.00 | 30.00 | 0.0944 | −20.5 | 1860 |
| strength 0.75 | 30.00 | 0.0773 | −22.2 | 1918 |
| strength 0.50 | 30.00 | 0.0170 | −35.4 | 1800 |
| strength 0.25 | 30.00 | 0.0160 | −35.9 | 1679 |
| style K-pop | 30.00 | 0.0830 | −21.6 | 1723 |
| style R&B | 30.00 | 0.0589 | −24.6 | 1754 |
| style indie pop | 30.00 | 0.0673 | −23.4 | 1512 |
| **control: unrelated, same prompt** | 30.00 | **0.0039** | **−48.2** | 2143 |

Time-varying musical structure:

| run | chroma seq | pitch contour | structure |
|---|---|---|---|
| baseline 1.00 | 0.781 | 0.344 | **0.507** |
| strength 0.75 | 0.787 | 0.321 | **0.502** |
| strength 0.50 | 0.748 | 0.330 | 0.325 |
| strength 0.25 | 0.734 | 0.209 | 0.258 |
| style K-pop | 0.748 | 0.490 | 0.438 |
| style R&B | 0.747 | 0.408 | 0.387 |
| style indie pop | 0.748 | 0.285 | 0.403 |
| control: unrelated A | 0.743 | 0.041 | 0.270 |
| control: unrelated B | 0.724 | 0.141 | 0.310 |
| control: source shuffled | 0.707 | 0.237 | 0.052 |

## What the numbers support

1. **Duration is exactly preserved.** All seven outputs are 30.00 s, as
   the source-is-the-canvas code path predicted. No truncation, no drift.

2. **Source influence is real.** At strength 1.00 the output sits 28 dB
   above an unrelated same-prompt generation in SI-SDR (−20.5 vs −48.2),
   with pitch-contour correlation 0.34 against a 0.04–0.14 floor and
   structure agreement 0.51 against 0.27–0.31. Three independent measures
   agree, so this is not one metric's artefact.

3. **The recording is not preserved.** −20.5 dB SI-SDR is nowhere near
   the 26–78 dB that repaint achieves on genuinely preserved audio in
   Phases 13B/13C. Cover regenerates everything; it does not retain the
   performance.

4. **There is a cliff at 0.75.** Structure agreement is 0.50 at strength
   0.75–1.00 and 0.26–0.33 at 0.25–0.50 — indistinguishable from an
   unrelated song. **The usable engine range is 0.75–1.00**, and below it
   the feature silently becomes text-to-music.

5. **Style prompts change the output** without destroying derivation: the
   spectral centroid moves 1512–1860 Hz across targets while pitch-contour
   correlation *rises* (0.49 for K-pop). Whether the requested style is
   perceptually present needs a listener.

## What the numbers do not support

- **Melody recognisability.** Pitch-contour 0.34 is well above the floor
  but far from strong, and the shuffled-source control reaching 0.24
  shows part of any such score is content overlap rather than
  time-alignment. "The tune is recognisable" is a perceptual claim and is
  not made here.
- **Vocal identity.** Nothing here measures speaker identity, and no
  speaker-embedding model was installed to invent one.
- **Lyric preservation.** Korean lyrics were supplied to every run, but
  whether they are sung, and intelligibly, cannot be measured without ASR
  ground truth this project does not have.
- **Tempo.** The autocorrelation estimator returned 79–201 BPM across
  outputs with obvious octave errors (79 ≈ 161/2). It is reported in the
  JSON and is not used for any conclusion.
- **Time-averaged chroma is useless here.** It scored 0.98 for an
  unrelated song. The first analysis pass used it, could not discriminate,
  and was replaced by the time-varying pass rather than reported as a
  finding.

## How the classification was reached

The gate for shipping was REMIX_READY or COVER_ONLY.

- **Not REMIX_READY.** A remix keeps the recording. This does not — the
  performance and voice are fully regenerated, and −20.5 dB SI-SDR is
  nowhere near repaint's 26–78 dB on genuinely preserved audio.
- **COVER_ONLY**, on the combination of the measurements above and the
  listening verdict. The measurements establish that the output is
  derived from the source rather than from the prompt alone; the listener
  established that the derivation is musically usable. Neither alone was
  sufficient, which is why the calibration phase deliberately stopped and
  waited.

## What shipped, and the constraints it inherited

Phase 13D-2 exposes this as **Create Cover**. Never "Remix": the name has
to match what the engine does to the audio.

| Product preset | Engine `audio_cover_strength` | Measured |
|---|---|---|
| Closer to the original (`subtle`) | 1.00 | structure 0.507 |
| More transformed (`strong`) | 0.75 | structure 0.502 |

Two presets, not three. Only two engine values sit inside the calibrated
band, so a third would be a setting nobody measured. The mapping is
**inverted** — more transformation means *lower* adherence — and that
inversion is asserted by a test, because getting it backwards would leave
both labels lying while everything still appeared to work.

Nothing below 0.75 is reachable from the product, and the provider refuses
an out-of-band value rather than clamping it.

The UI copy is bounded by the same evidence: it says a cover is "a new
performance inspired by this song's musical structure", and states that
the original recording and vocal are not kept. It does not claim melody,
performance or voice preservation, because none of those were measured or
observed.

## Reproducing

```
uv run python calibrate.py <outdir> <source.wav>
uv run python analyse_signal.py <outdir> <source.wav> <control.wav>
uv run python analyse_musical.py <outdir> <source.wav> <controls...>
```

Seven generations, roughly 45 s each on this hardware.
