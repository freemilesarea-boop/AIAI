# Measuring the library, and training on the parts that hold up

Phase 37 gave the model more of each song. It did not ask whether the
material was worth giving more of. Phase 38 measures the library, sorts
it, and trains on the part that has live high end, a steady pulse and a
busy arrangement — with windows that start on a beat instead of wherever
even spacing happened to land.

Only the data representation changes. Rank, alpha, optimizer, learning
rate, precision, batch geometry and the fixed 3000-frame shape are all
exactly where Phase 37 left them.

## What was measured

The library carried two folder names and nothing else, so nine measures
were computed from the audio itself with numpy and the standard
library's `wave` module. No scipy, no librosa: the source is 16-bit PCM
at 48 kHz and a short-time Fourier transform gives all of it.

| median | POP | Lofi |
|---|---|---|
| high-frequency energy share (>8 kHz) | **0.0187** | 0.0003 |
| spectral centroid | **1073 Hz** | 308 Hz |
| high-band RMS | **−34.1 dB** | −51.0 dB |
| transient density | 2.52 /s | 3.59 /s |
| onset density | 5.46 /s | 6.46 /s |
| beat stability | 0.412 | 0.395 |
| tempo consistency | 0.867 | **0.991** |
| drum/bass alignment | **0.198** | 0.094 |
| layer density | **0.417** | 0.281 |

POP carries **62 times** the high-frequency energy of Lofi, three and a
half times the centroid, and 17 dB more high-band level. Lofi wins on
tempo consistency, which is what a loop-based production does.

Stated limits, because these names are more confident than the
measurements: onset density is spectral-flux peaks per second, not a
transcription. Drum/bass alignment is a correlation between two
band-limited onset envelopes — a proxy, named one. Layer density is
spectral entropy; it counts no instruments and identifies none.

## Four axes, one of them deliberately empty

`HIGH_END`, `RHYTHM` and `ARRANGEMENT` are scored from the measures
above, as percentile ranks *within the library being classified*.

`VOCAL` is **never scored.** Nothing in this repository distinguishes a
sung note from a lead synth, and a number invented for that axis would
be the most misleading value in the file. It exists in the vocabulary
because the listening evaluation is organised around it, and the tiering
records that it is unmeasured rather than guessing.

Tiers are ranks, not verdicts. The enum says `TIER_A`, not `GOOD`, and
every assignment carries the thresholds it was decided under.

| | TIER_A | TIER_B | TIER_C |
|---|---|---|---|
| POP | **38** | 28 | 9 |
| Lofi | 1 | 10 | 42 |

## What Phase 38 trains on

Tier A and B, POP only, 50 train / 8 validation / 8 evaluation tracks,
93 training windows. **62 authorised tracks are deliberately unseen.**

POP-only is a decision worth stating. Restricting by tier alone left a
training split of 58 POP against 2 Lofi while the held-out sets stayed
balanced 4/4 — the model would have been trained on one kind of material
and measured on another. Since the request was a POP/R&B quality
experiment, every split is POP and the held-out sets measure what the
training split actually contains.

## Beat-aware window starts

An arbitrary crop can open halfway through a snare hit, so the model's
first frames are the tail of an event whose attack it never saw. Each
window start is now nudged onto the nearest onset within 50 latent
frames — two seconds, enough to reach a downbeat at any tempo in range,
small enough that the even spread still decides *which parts of the
song* are covered.

Of 123 windows placed, **47 already began on an onset and 76 were
snapped onto one. None fell back to an arbitrary position.** Two windows
can never collapse onto the same frame: a snap that would collide keeps
its even-spaced position and records that it did.

## Arrangement-weighted exposure

Per-window weights now tilt toward the busier windows of each track,
while every *track* still contributes the same total — a four-window
song does not outvote a one-window song, and within a song its fuller
sections carry more of that song's share.

**Known bias, carried forward from Phase 37 and still true.** The
installed `PreprocessedDataModule` takes no per-sample weight; it
iterates a directory. These weights are recorded as evidence and are not
enforced by the loader. Enforcing them needs a weighted sampler, which
is a trainer change and not this phase's variable.

## What the run produced

**This section was corrected on 2026-08-25.** It previously claimed "336
optimizer steps across two resumed segments"; commit `cbcf865` carries
that wording, and its message repeats it.

The plan was 336 optimizer steps as one continuation: segment A running
1–168, segment B resuming A and running 169–336. Segment A did that.
**Segment B did not resume segment A's weights — it trained 168 steps
from a fresh LoRA.** The preserved adapter, the one the operator
evaluated, therefore carries **168 optimizer steps of training, not
336.** There is no 336-step cumulative adapter; the two segments never
composed.

Within segment B, the numbers themselves stand: 168 finite steps, 384 of
384 trainable tensors changed, the base model digest identical before
and after, and held-out loss 1.2298 → 0.9362 across 14 measurements on
19 tracks no optimizer step touched. Loss "1.5103 → 1.0962" spans both
segments and so describes two runs rather than one curve.

That is a training-path result and a generalization signal. It is not a
quality claim, and the checkpoint stays EXPERIMENTAL, NON_PRODUCTION,
NEVER_AUTO_PROMOTE.

### Why it looked like a resume

PEFT writes adapter tensors as `...lora_A.weight` and holds them in the
model as `...lora_A.<adapter>.weight`. The trainer resumes with
`decoder.load_state_dict(state_dict, strict=False)`, so every key missed,
nothing was restored, and — because `strict=False` — nothing failed.

Meanwhile `epoch`, `global_step`, the optimizer and the scheduler *did*
load, from `training_state.pt`. So the run reported a successful resume,
continued the counter from 168, and wrote a checkpoint labelled step 336.
Everything downstream believed it.

Three pieces of evidence settle it. Segment A and segment B recorded the
same first training loss to fifteen decimal places — 1.510331004858017 —
and the same first gradient norm, 0.26953125. And the mechanism
reproduces: a PEFT adapter saved with `lora_B = 0.5`, resumed the way the
trainer does it, comes back with `lora_B` still 0.0.

What there is *no* evidence of is a contemporaneous warning. `strict=False`
produced none, which is the whole problem.

The loader is now guarded: a continuation re-loads the adapter with the
name component restored, compares every tensor against the file it came
from, and refuses to train on a mismatch rather than starting fresh in
silence.

**None of this changes the audio.** The adapter bytes are untouched and
the operator's listening verdict below stands exactly as recorded, for
this 168-step adapter.

## What the operator heard

Four sides — base, Phase 36, Phase 37, Phase 38 — were generated from
identical prompts, seeds, durations and step counts, 8 axis prompts at
30 s and 3 songs at 195 s, and the operator listened.

| axis | verdict | what was reported |
|---|---|---|
| RHYTHM | **pass, significantly improved** | the interrupted, tangled, stumbling timing is no longer meaningfully present |
| ARRANGEMENT | **pass, significantly improved** | instrumentation noticeably richer, arrangement more complete |
| VOCAL | **pass** | the strong melodic behaviour remains acceptable |
| HIGH_END | **fail** | still flat and closed, and worse than dull: a metallic, steely, brittle character |

The two things Phase 38 set out to change are the two that moved. The
axis it deliberately did not score — VOCAL — did not regress. High end
was tiered *for* and did not improve, which is the finding that matters:
selecting brighter material did not produce brighter output.

No numeric score exists. No MOS, no rating, no blind preference rate was
collected, and none may be inferred from the table above.

## Where the high end actually goes

Phase 39's question is why, so the chain was measured before anything
was proposed. Band energies below are stated against the 200–2000 Hz
band of the same file, so a level change cannot masquerade as a tilt.

| | 8–12 kHz | flatness 6–12 kHz | narrow peaks >6 dB, 4–16 kHz |
|---|---|---|---|
| source library | −12.50 dB | 0.925 | 0 |
| training windows | −13.05 dB | 0.924 | 0 |
| VAE encode→decode, no diffusion | −14.81 dB | 0.903 | 2 |
| generated, base model, no adapter | −18.12 dB | 0.830 | 16 |
| generated, Phase 38 | −17.61 dB | 0.793 | 12 |

Reading down that column is the whole answer. Windowing costs nothing.
The VAE costs about 1 dB and adds a couple of narrow peaks. The
diffusion stage costs a further 3 dB, drops the flatness that makes air
sound like air, and adds ten to fifteen narrow-band resonances that no
reference file has. **The base model with no adapter attached shows the
same deficit**, and Phase 38 sits slightly *above* it in 4–12 kHz.

So the metallic character is not something the adapter introduced, and
it is not the dataset: the Phase 38 training selection is the brightest
subset in the library — HF energy share 0.0205 against 0.0096 for the
library and 0.0077 for the POP tracks it did not select.

It is also not a filter anywhere in the chain. Everything runs at 48 kHz
end to end, the only resample is on the MP3 path this never takes, and
the sole post-processing stage is a peak normalisation to −1 dBFS —
which is why every generated file peaks at exactly 0.891251. Gain, not
EQ.

A boost would therefore be the wrong instrument. The deficit is not a
smooth tilt to be tilted back; it is tonal structure standing where
broadband noise should be, and lifting that band lifts the resonances
along with the air.
