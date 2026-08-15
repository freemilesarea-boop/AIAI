# LUBER audio finishing — architecture audit and Phase 14A engine

Read from the repository as it stands, then tested against 40 existing
LUBER masters. Phase 14A built and proved the engine. **Phase 14B wired
it into the delivery pipeline** — see §8 for what changed; everything
above it describes the engine, which 14B did not alter.

This work belongs to `luber-music-ai` alone. No code, model,
configuration or binary from any other mastering project was read,
copied, referenced or depended on.

---

## 1. The pipeline as it exists today

```
ACE-Step writes a WAV
  → GenerationService._run                       generation-client/service.py
  → inspect_wav(result.audio_path)               structural validation, stdlib
  → produce_delivery_assets(...)                 generation-client/postprocess.py
      → transcode_master_wav_async               ffmpeg → 48 kHz / 24-bit / stereo
      → encode_preview_mp3_async                 ffmpeg → 320 kbps CBR, from the master
      → storage.put(master_key), storage.put(preview_key)
  → repo.create_audio_asset(MASTER), (PREVIEW)   audio_assets rows
  → repo.mark_completed(...)
  → GET /generations/{id}/download                api/routes/generations.py
  → AudioStorage.download_target                  signed URL, or streamed locally
  → web player / download
```

Three properties of that path matter here:

**Nothing in it changes the sound.** `luber_audio_utils.transcode` says so
explicitly and means it: no loudness normalisation, no limiter, no EQ, no
dithering choice. Container, sample rate, channel count and sample width
only. So the stored MASTER *is* the model output, resampled — which is
what makes it usable as a raw reference.

**The preview is derived from a stored master**, not from the provider's
file, so what is streamed and what is downloaded describe the same audio.
Since 14B that is the *delivery* master — see §8.

**Storage keys are deterministic** — `audio/<generation-id>/master.wav`.
A retry overwrites its own object rather than accumulating new ones.

### The insertion point

Inside `produce_delivery_assets`, between the master transcode and the
preview encode:

```
transcode_master_wav_async   → raw master  (unchanged, stored)
finish_audio                 → finished master  ← the new stage
encode_preview_mp3_async     → preview, from whichever master ships
```

That position is the right one for four reasons. The audio is already
normalised to one format, so the engine never has to handle whatever the
model emitted. The preview is generated downstream, so it follows the
shipping master automatically. The raw master is produced before
finishing runs, so no ordering exists in which a finishing failure leaves
a generation without a master. And storage is addressed by key, so a
second asset costs a key, not a redesign.

**Implemented in 14B, at exactly this point.** The insertion point was
re-verified against the code before wiring, and the surrounding call
graph was unchanged.

## 2. Raw preservation

Non-negotiable, and enforced in code rather than by convention:

- `finish_audio` refuses `destination == source`.
- It refuses any input already carrying a finishing stamp.
- It writes only to the destination; a test asserts the source file is
  byte-identical afterwards.
- A NO_ACTION plan writes **nothing**. The raw master is already the
  deliverable, and producing an identical copy of it would only give the
  two a chance to drift apart.

### Does this need a migration?

**Not for the asset type — and that is a measurement, not an assumption.**
`AudioAsset.asset_type` is `String(20)` (`models/generation.py`). There is
no `sa.Enum` and no `CheckConstraint` anywhere in
`packages/database/alembic/versions/`, so the database does not constrain
the value. Adding `FINISHED_MASTER` to the `AssetType` StrEnum is a
Python-side change only, and `MASTER` keeps its existing meaning so no
stored row is reinterpreted.

Durable finishing metadata is a different question, and 14B answered it
with migration `0011`: one nullable Text column, `generations.finishing_trace`,
matching the `request_trace` pattern from 0004. See §8.

The unique constraint is `(generation_id, asset_type)`, so a raw and a
finished master coexist as two rows without touching it.

## 3. The engine

`packages/audio-finishing`, `luber_audio_finishing`. numpy plus ffmpeg;
no ML models, no GPU, no plugins. Version `p14-v1`.

```
RAW MASTER
  → analyze_audio          measurement only, no judgement
  → evaluate_risks         named flags, each carrying its own evidence
  → FinishingDecisionEngine → FinishingPlan (data, renders nothing)
  → finish_audio           two ffmpeg passes, structured argv
  → verify                 clipping, ceiling, duration, rate, channels, balance
FINISHED MASTER
```

Analysis is deliberately separate from judgement: one is what is true
about the file, the other is what counts as a defect. Keeping them apart
is what makes a threshold arguable without re-measuring anything.

### What is measured

| Group | Fields |
|---|---|
| Technical | duration, sample rate, channels, bit depth, frames |
| Level | peak, RMS, crest factor, DC offset, clipped and near-clipped counts, silence ratio, 50 ms crest distribution |
| Loudness | integrated LUFS, LRA, true peak, short-term P10/P50/P90 (ffmpeg `ebur128`) |
| Frequency | 8 bands, centroid, rolloff85, bandwidth, flatness, slope in dB/octave, air / low-mid / presence ratios, adaptive low-mid peak |
| Sibilance | 6-9 kHz and 2.5-5 kHz ratios plus their peak excess over their own median |
| Transient | spectral flux distribution, onset rate, transient density |
| Stereo | L/R balance, mid and side energy, width, correlation, low-band correlation and width, high-band width |
| Spatial | stereo decorrelation, high-band decorrelation, envelope decay — all proxies, all decision-inert |

Bands: sub 20-60, bass 60-150, low-mid 150-400, mid 400 Hz-2 kHz,
presence 2-5 kHz, brilliance 5-10 kHz, air 10-16 kHz, ultra-high
16-20 kHz.

Four decisions inside the analyser are worth stating because a plausible
implementation would get each of them wrong and still look fine:

**Bands above Nyquist are absent, not empty.** A 22.05 kHz file has no
16-20 kHz band. Reporting zero energy there reads as "no air" and
provokes a correction the file cannot benefit from.

**Mono files have no stereo metrics.** Not 1.0 correlation — `None`.
Reporting perfect correlation would read as "perfectly mono-compatible
stereo", which is a different claim about a different file.

**Percentiles come from a gated frame set.** Frames more than 40 dB below
the loudest are excluded. Without the gate, fade-outs and silence
dominate the low percentiles and every track with an intro appears to
lose its high frequencies in places.

**The body reference band is 400 Hz-2 kHz, and overlaps nothing.** The
first version used 300 Hz-3 kHz, which shared 300-400 Hz with the low-mid
band and 2.5-3 kHz with the harshness band. A ratio whose numerator also
sits in its denominator saturates: thick 300-400 Hz content raised both
sides and read as balanced, and a harshness burst stopped registering
past about 13 dB however loud it got. Both were found by tests, both are
now impossible by construction, and a test asserts the non-overlap.

### Cost

Frames are reduced in blocks and the full spectrogram is never held.
Analysis stays in single-digit megabytes and runs at roughly 0.2 s per
30 s stereo track; a full finish is about 1.4 s per track including two
ffmpeg passes and three `ebur128` runs. The 32-track corpus renders
sequentially in 24 s on this laptop.

## 4. The baseline corpus

40 existing LUBER masters: all 32 in local storage (20 x 30 s, 8 x 60 s,
4 x 180 s) plus 8 outputs from the Phase 13D cover and 13E reference
calibrations. Nothing was generated for this phase, no commercial music
was read, and no directory was scanned — paths are arguments.

Full numbers in `benchmarks/audio_finishing/`.

### Does the listening report survive measurement?

**Partly, and the part that fails is worth more than the part that passes.**

*Supported without qualification: tonal balance is inconsistent.* The air
band relative to the midrange spans **-39.0 to -13.4 dB** across one
model's own output — a 25 dB spread, standard deviation 6.0 dB. Presence
spans 17.8 dB, low-mid 19.6 dB. This is the case for an adaptive engine
rather than a fixed curve, and it is not a close call.

*Supported with qualification: the high-frequency deficit is real but not
universal.* Nine of forty tracks are both dark in slope and low in air.
Eleven of forty fall in the darkest quarter for air. So "upper frequencies
often feel reduced" describes a substantial minority — not the catalogue.
Reporting it as universal would have justified a fixed shelf on every
track, which the same corpus shows would be wrong.

*The opposite failure mode is present in the same corpus.* Sibilance peak
excess reaches 20.0 dB and harshness 22.2 dB. **Eight tracks are dark and
spiky at once.** A fixed high-shelf boost would have made those eight
worse, which is exactly why the engine has a suppression rule instead.

*Also measured, and load-bearing:* every generated master arrives at
**exactly -1.0 dBFS**, standard deviation 0.00. There is 1 dB of
headroom in the entire catalogue, and every correction has to be paid for
out of it.

*Not supported:* nothing in the corpus indicates flattened transients.
50 ms crest factor runs 7.0-9.3 dB against a flatness threshold of
6.5 dB — no track comes close. The reported "instruments feel flat" is
not visible in any transient measurement taken here, which is one reason
transient processing is deferred rather than shipped small.

## 5. Decisions

Thresholds are absolute, not corpus percentiles. Setting them at
percentiles would define "correct" as "average for this model", guarantee
a fixed fraction of tracks is always flagged, and make the engine chase
its own output. **Fifteen of thirty-two storage masters produce
NO_ACTION** — the engine's most common answer is to do nothing.

No commercial recording informed any number here.

| Risk | Response | Ceiling |
|---|---|---|
| HIGH_FREQUENCY_DEFICIT / AIR_DEFICIT | high shelf, corner 7 kHz or 10 kHz by which flag fired | +3.0 dB |
| PRESENCE_DEFICIT | peaking lift at 3.2 kHz, Q 0.8 | +2.0 dB |
| LOW_MID_MUD | peaking cut at the track's own low-mid peak, Q 0.9 | -2.5 dB |
| LOW_END_EXCESS | low shelf at 90 Hz | -1.5 dB |
| STEREO_TOO_NARROW / TOO_WIDE | side-channel gain | ±1.5 dB |
| STEREO_IMBALANCE | per-channel gain, re-measured after the chain | ±1.5 dB |
| LOW_END_PHASE_RISK | side channel high-passed at 120 Hz | — |
| HARSHNESS_RISK / SIBILANCE_RISK | **lower the shelf ceiling** to 1.5 / 1.0 dB | — |
| TRANSIENT_FLATNESS | recorded, no action | — |

Every rule corrects a *fraction* of what it measures — 40%, or 50% for
mud — and never all of it. A generation 8 dB dark becomes one 5 dB dark,
not one that is 0 dB dark. Full correction would impose a single spectral
opinion on every song.

Chain order: balance → subtractive EQ → additive EQ → mid/side → level.
Cuts before boosts, so a boost lifts audio that has already lost its
excess.

### Contradictions resolve toward doing less

Harshness and sibilance do not cancel the brightness rules; they lower
their ceiling, so a dark *and* sibilant track still gets the small lift
its darkness justifies. On the corpus this caps a requested +3.04 dB at
+1.00 dB. Presence is different — the harshness band (2.5-5 kHz) and the
presence band (2-5 kHz) are the same region, so lifting one is lifting
the other, and the rule does not fire at all.

The engine must never solve "dull" by creating "painful".

### What is deferred, and why

**Dynamic harshness control.** Measured, not assumed. ffmpeg's
`adynamicequalizer` was run over a sibilant baseline track at detection
thresholds from 0.0005 to 0.1 and in adaptive mode. Wherever it acted, it
moved the 6-9 kHz median and 90th percentile by the *same* amount,
leaving peak excess at 21.05-21.23 dB against an untreated 21.21 dB. That
is a static cut wearing a dynamic label, and a static scoop is precisely
what a de-esser must not be. Deferred with the numbers recorded.

**Transient shaping.** No measured need (see §4) and no verified bounded
implementation. A shaper applied without a need attacks sustained pads
and reverb tails as readily as drums.

**Spatial and reverb.** The spatial metrics are proxies with known
confounds — a held pad and a reverb tail decay alike; a wide synth and a
room read alike. No threshold among them separates a dry mix from a
deliberately dry one. Adding ambience to every master would be an
aesthetic decision presented as a correction. Better to preserve a dry
generation than wash the mix in fake room.

**A limiter.** Not needed at p14-v1 correction sizes, and it would change
the dynamics of the comparison the listening test is about.

## 6. Output safety

Two passes. The corrective filters change the peak by an amount no
formula predicts, and the masters arrive with 1 dB of headroom, so the
level is measured after filtering rather than guessed before it. Pass one
renders the chain at 32-bit float where nothing can clip; pass two
applies one measured gain and writes the deliverable.

That gain is the smaller of two numbers: the one matching finished
loudness to source loudness, and the one keeping true peak under
-1.0 dBTP. **Peak safety wins when they disagree**, which can leave the
finished file slightly quieter — up to 1.4 LU on one corpus track. That
is deliberate. Loudness matching exists so an A/B comparison is about
tone rather than about which file is louder.

Balance is corrected from the *filtered* audio, not the source. The
mid/side stage shifts it — scaling or high-passing the side changes L and
R asymmetrically wherever mid and side correlate — and on one track that
was a 1.19 dB shift, more than the 0.8 dB the engine treats as a defect
worth fixing. Uncorrected, the engine would introduce the fault it flags.
Verification asserts the output is centred.

### The crossover that was replaced

Bass-mono was first built as a Linkwitz-Riley crossover: mono-ise the low
band, sum it back. Its halves sum flat in magnitude, but they rotate phase
through 360 degrees, and that reshapes transients. Sample peak rose by up
to 3 dB with no change in loudness, which the level stage then had to give
back as 3 dB of gain reduction; crest factor on one track went 14.3 →
17.4 dB for an effect nobody can hear.

It was replaced by a mid/side section that filters the side channel
alone. The mid — the mono sum, and most of the energy — passes through
untouched, and a test asserts the M/S round trip is bit-exact. Peak
inflation went from 3.1 dB to 1.4 dB worst case and crest drift from
+3.1 dB to +0.7 dB.

### Verified on every render

No clipping. True peak and sample peak under the ceiling. Duration within
10 ms. Sample rate and channel count unchanged. Bit depth never reduced.
Output balance centred. A render failing any of these raises and the
output file is deleted rather than shipped.

### Idempotency

Finished files carry `comment=luber_finishing=p14-v1` in their WAV INFO
chunk. `finish_audio` refuses any input carrying that stamp: a second
pass would measure audio the first pass already changed, and the result
would depend on how many times it ran. Same source plus same version
produces the same plan and byte-identical output.

## 7. Phase 14A status

**Engine built and measured.** The objective results showed the
processing did what each plan said, within its stated ceilings, without
clipping and without changing duration, rate, channel count or dynamics.

They did not show it sounds better; nothing measurable can. Five
RAW/FINISHED pairs went to `~/Desktop/LUBER_PHASE14_FINISHING_LISTENING/`
and the phase stopped there.

## 8. Phase 14B — integration

Listening review completed, so the engine was wired into delivery. The
engine itself was not changed.

### Assets

Three roles now, from two masters:

| Role | Meaning |
|---|---|
| `MASTER` | the raw generation master — model output, format-normalised, written once |
| `FINISHED_MASTER` | the finishing result, present only when the engine acted |
| `PREVIEW` | derived from whichever master is being delivered |

`MASTER` deliberately keeps its pre-14B meaning. Renaming it to
`RAW_MASTER` would have read better but would have reinterpreted every
stored row and broken any client mid-deploy; adding a value alongside it
breaks nothing and leaves old rows correct as they stand.

The cost of that choice is that "MASTER" no longer means "the master to
serve", so every consumer that filtered for it by hand is now a place the
wrong one can be picked silently. All five were routed through two
selectors in `luber_schemas.assets`:

* `select_delivery_master` — finished if present, else raw. Downloads,
  playback, the preview encode.
* `select_raw_master` — always raw. Extend, replace-section, cover.

Edits read the **raw** master on purpose. Feeding a finished master back
into the model and finishing the result would stack corrections across
generations — a track extended five times would carry five high-shelf
lifts — and the child gets its own finishing pass regardless.

### Failure policy

Two policies, deliberately different. The transcode, the preview and
their uploads still fail the whole generation. Finishing does not: the
raw master is a complete, shippable product and was the entire product
before 14B, so a finishing failure is logged, recorded as `FAILED` in the
trace, and the raw ships.

That fallback cannot publish a bad file. The engine verifies its own
output and deletes it rather than returning it, so the pipeline is only
ever deciding whether to ship the enhancement, never whether it is safe.
Only `FinishingError` is treated this way; an unexpected exception is a
defect in the wiring and still fails the generation.

### Provenance

`generations.finishing_trace` (migration `0011`, nullable Text holding
JSON) records the outcome, the engine version, the digest of the raw
master the decision was made from, and the whole plan. It exists because
an absent `FINISHED_MASTER` cannot distinguish three real states: the
engine declined, the engine failed, or the generation predates 14B. NULL
is the third.

### Retry

Storage keys are deterministic and `create_audio_asset` upserts, so a
retry overwrites rather than accumulates. The engine is deterministic, so
a retry reaches the same decision. The one case that needs handling is a
future engine version declining where an earlier one acted: the stale
`FINISHED_MASTER` row is retracted, row first and object second, so a
half-cleaned state is unreferenced bytes rather than a broken download.
