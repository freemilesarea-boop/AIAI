# Phase 9 — Full-Song Generation: Results

> **What Phase 9 established:** this deployment can technically produce
> 120/180/240-second tracks, cheaply and reliably, and the objective
> measurements now corroborate several of the human evaluator's
> complaints with numbers.
>
> **What Phase 9 did not establish:** that those tracks are *good*. No
> automated measurement in this repository can answer that, and nothing
> here claims to. The human listening pass is the deciding step and it
> has not happened yet.

Engine pin: ACE-Step 1.5 @ `6d467e4b`. Source audit:
[PHASE9_LONG_FORM_ENGINE_AUDIT.md](PHASE9_LONG_FORM_ENGINE_AUDIT.md).

---

## 1. Long-form gates — all three passed technically

Each is one real generation through the production path: HTTP → Postgres
→ Redis → ARQ worker → real ACE-Step → post-processing → MASTER WAV +
MP3 preview. Fixed seed `20260813`, Korean vocal, structured lyrics,
BPM 84, A minor, 4/4.

| Gate | Requested | Actual | Wall clock | RTF | Master | Preview |
|---|---|---|---|---|---|---|
| 120 s | 120 s | 120.0 s | **96.3 s** | 0.80× | 34.6 MB WAV | 4.8 MB MP3 |
| 180 s | 180 s | 180.0 s | **89.5 s** | 0.50× | 51.8 MB WAV | 7.2 MB MP3 |
| 240 s | 240 s | 240.0 s | **76.4 s** | 0.32× | 69.1 MB WAV | 9.6 MB MP3 |

**The headline technical finding: long-form is not expensive here.**
Wall clock is effectively flat across 120–240 s, and the 240 s run was
*faster* than the 180 s run. Cost is dominated by fixed per-request
overhead plus 8 fixed inference steps; sequence length is not the
bottleneck at these sizes on MPS. Run-to-run variance is dominated by
machine contention, not by requested duration.

This corrects the pre-gate extrapolation from 30 s samples (RTF
1.54–2.30×), which predicted ~550 s for a 240 s track. That estimate was
wrong by roughly 7×.

### Resource envelope (24 GB M-series, also the user's daily machine)

| Gate | Peak swap used | Min swap free | Min free disk | Peak load |
|---|---|---|---|---|
| 120 s | 19.7 GB | 729 MB | 11 GB | 5.6 |
| 180 s | 21.1 GB | 604 MB | 11 GB | 15.0 |
| 240 s | 23.1 GB | **206 MB** | 11 GB | 6.6 |

The machine started this phase with only 888 MB of free swap and 13 GB
of free disk. All three gates completed and memory was released
afterwards, but the 240 s run **transiently crossed the 400 MB
free-swap caution threshold set before the run** (206 MB). No crash, no
thrash that affected the run, and disk recovered to 14–18 GB.

The honest read: 240 s is achievable on this hardware but is close to the
edge, and this is not a machine to run long-form batches on.
Concurrency stayed at 1 throughout, which the engine enforces anyway
(`ACESTEP_QUEUE_WORKERS=1`).

---

## 2. Audio analysis — the numbers agree with the listener

Measured on the MASTER WAV of each gate, in four equal windows.

| | 120 s | 180 s | 240 s |
|---|---|---|---|
| Peak | −2.67 dBFS | −1.69 dBFS | −1.32 dBFS |
| RMS | −17.35 dBFS | −15.80 dBFS | −16.30 dBFS |
| Crest factor | 14.7 dB | 14.1 dB | 15.0 dB |
| Clipping | 0.0 | 0.0 | 0.0 |
| Silence ratio | 0.022 | 0.040 | 0.025 |
| **Spectral centroid** | **3137 Hz** | **3096 Hz** | **3033 Hz** |
| **Energy above 8 kHz** | **15.6 %** | **15.2 %** | **14.8 %** |
| **SIBILANCE_RISK_PROXY** | **0.147** | **0.142** | **0.141** |
| Level drift across windows | 1.6 dB | 4.3 dB | 2.2 dB |
| Sibilance growth (last/first) | 1.26 | 1.29 | 1.10 |
| Flags | none | none | none |

### What this says

**The "excessive high-frequency energy" complaint is objectively real.**
A spectral centroid around 3.0–3.1 kHz and ~15 % of spectral energy
above 8 kHz are both very high for produced music, where a centroid of
1.5–2.5 kHz and low single-digit high-frequency percentages are typical.
This is the model's own balance: LUBER's post-processing transcodes and
does nothing else. Phase 5 found frequency problems at 30 s; they are
still here at 240 s, essentially unchanged.

**Long form does not obviously make it worse.** Sibilance growth of
1.10–1.29 from first window to last is below the 1.5 flag threshold, and
the centroid is *slightly lower* on longer tracks. So the high end is
bad from the start rather than degrading over the song. That is a
useful negative result: it points at the model's overall spectral
balance rather than at long-form drift.

**Levels are technically sound.** No clipping, no excessive silence,
crest factors of 14–15 dB, and level drift of 1.6–4.3 dB across windows.
Nothing here needs fixing at the delivery layer.

### `SIBILANCE_RISK_PROXY` — what it is not

It is the share of spectral energy in 5–10 kHz. Cymbals, synths, and
noise live there too. It is meaningful for comparing windows within a
track and tracks against each other. **It is not a vocal-sibilance
detector**, and a high value is not proof that a vocal is sibilant. The
name carries the caveat deliberately.

---

## 3. Control verification

### BPM — verified, and it works

| Gate | Requested | Estimated | Difference | Confidence |
|---|---|---|---|---|
| 120 s | 84 | **83.96** | 0.04 | 0.386 |
| 180 s | 84 | **83.96** | 0.04 | 0.364 |
| 240 s | 84 | **83.96** | 0.04 | 0.460 |

This is the strongest positive result in Phase 9. Phase 8 proved the BPM
parameter *reached* the engine; this proves the **rendered audio
actually follows it**, to within 0.05 BPM on all three long-form tracks.
The estimator is onset-envelope autocorrelation, validated against
synthetic click tracks at 80/100/120/140 BPM.

### Key — partial, low confidence, mode unreliable

| Gate | Requested | Estimated | Tonic match | Verdict |
|---|---|---|---|---|
| 120 s | A minor | C# minor | ✗ | LOW_CONFIDENCE |
| 180 s | A minor | A major | ✓ (tonic) | LOW_CONFIDENCE |
| 240 s | A minor | A major | ✓ (tonic) | ESTIMATED |

Two of three recover the **tonic**; none agrees on **mode**. The
estimator is Krumhansl-Schmuckler correlation over an FFT chroma with no
harmonic whitening or tuning correction, and its major/minor decision is
known-unreliable on dense produced music — which is why it emits a
`verdict` field rather than a bare answer.

**Status: `HUMAN_OR_EXTERNAL_ANALYSIS_REQUIRED`.** The evidence is
suggestive that the tonic is honoured and says nothing trustworthy about
mode. Do not report key control as verified.

### Time signature — not verified, by design

No validated automatic method exists here, so the code returns the
constant `HUMAN_OR_EXTERNAL_ANALYSIS_REQUIRED` rather than a number.
The requested value is preserved in metadata and in the request trace,
and there is a human QA field for it.

---

## 4. Conditioning experiments

Both experiments were bounded exactly as specified and use fixed seeds,
so the manipulated variable is the only difference within each pair.

### 4a. Contemporary vocal conditioning (3 prompts × 2 variants)

Variant B appends: *contemporary Korean pop vocal, restrained vibrato,
clean modern phrasing, natural conversational pronunciation, modern R&B
phrasing*. No living artist is referenced.

| Prompt | Δ centroid | Δ sibilance proxy | Δ HF ratio | Audio differs |
|---|---|---|---|---|
| ballad | **−196 Hz** | −0.0150 | −0.0019 | yes |
| R&B | +107 Hz | +0.0051 | +0.0093 | yes |
| city pop | **−268 Hz** | −0.0164 | −0.0114 | yes |

**Objective conclusion: prompt conditioning does move the output.** All
three pairs differ despite identical seeds, and two of three moved
*darker and less sibilant* — the direction we want. The R&B pair moved
the other way.

**What this does not establish:** whether the vocal sounds less
trot-like or more contemporary. A spectral shift is not a style
judgement. With n=3 and an inconsistent direction, the correct summary
is "prompt conditioning is not inert, and is worth a listening test" —
not "prompt conditioning fixes the vocal". Listening decides.

### 4b. Korean lyric line formatting (3 sets × 2 variants)

Variant B holds the **words constant** and only re-breaks lines and
tightens section segmentation.

| Set | Δ centroid | Δ sibilance proxy | Audio differs |
|---|---|---|---|
| long_lines | −54 Hz | −0.0013 | yes |
| dense_verse | +88 Hz | +0.0049 | yes |
| untagged | −248 Hz | −0.0184 | yes |

Line shape changes the output too — unsurprising, since lyrics are
conditioning input. **Whether it changes line-omission behaviour cannot
be measured here at all.** Omission is exactly the failure with no
automatic detector, which is why Phase 9 built the line-level QA record
instead of pretending to measure it. Listening decides.

---

## 5. What Phase 9 built for the omission problem

The most damaging Korean failure is whole lines not being sung. Phase 9
does not claim to detect it. It builds the record:

- `expected_lyric_lines()` derives, from the submitted sheet, exactly
  what the model was asked to sing — indexed, with its section, tags and
  blank lines excluded.
- `lyric_line_qa` stores one verdict per line: `COMPLETE`, `PARTIAL`,
  `SKIPPED`, `DUPLICATED`, `UNKNOWN`. The submitted text is snapshotted
  alongside, so the record survives later edits.
- `UNKNOWN` is a first-class answer. On a dense mix a listener often
  genuinely cannot tell, and forcing a guess would poison the dataset
  this exists to build.
- `GET/PUT /v1/generations/{id}/qa` reads and writes it; re-reviewing
  corrects the record rather than appending a conflicting opinion.
- The listening package pre-populates the line list from the real
  submitted lyrics.

Once enough reviews exist, "which line positions get skipped, in which
sections, at which densities" becomes a query rather than an anecdote.

---

## 6. Full-song product surface

- **Durations offered: 30 / 60 / 120 / 180 / 240 s.** Each is a
  validated point. The engine accepts 600 s and the API schema 360 s;
  neither is offered, because neither has been validated end to end.
- **Presets:** Short Demo, Full Pop Song, Ballad, R&B, Band Song,
  Instrumental. A preset sets duration, instrumental flag, and a
  structure skeleton. It never writes lyrics.
- **Structure templates:** Pop, Ballad, R&B, Band, Verse/Chorus.
- **Applying either never silently destroys writing.** With words
  already in the sheet the UI asks, and "add after my lyrics" is the
  first option; replacing needs a second explicit click. A bare skeleton
  is swapped without ceremony because nothing is lost.
- **Templates are conditioning aids, not controls.** ACE-Step reads
  section tags as part of the lyric text. A template makes a
  recognisable arrangement more likely and enforces nothing. The UI copy
  says so.

### Lyric budget engine

`TOO_MANY_LYRICS`, `TOO_FEW_LYRICS`, `SECTION_OVERLOAD`,
`VERSE_OVERLOAD`, `CHORUS_OVERLOAD`, plus Hangul-block counting. Total
density has exactly one owner at any duration: `analyze_density` below
120 s, `analyze_lyric_budget` at or above it, so the editor never shows
the same advice twice.

### Two real bugs the gates exposed and Phase 9 fixed

1. **`EMPTY_SECTION` fired for `[Intro]` and `[Outro]`** — which are
   routinely instrumental in a sung song. Found on the very first real
   full-song request. Fixed via `OPTIONALLY_WORDLESS_SECTIONS`; an empty
   *verse* still warns.
2. **`[Final Chorus]` was not a recognised tag** — so the Phase 9
   templates would have warned about their own tags. Found on the first
   180 s run. Added as a chorus alias.

Both were only findable by running the real thing.

---

## 7. Duration-aware timeout

`timeout_for(duration) = max(generation_timeout, base + multiplier × duration)`,
defaults `600 s` floor, `300 s` base, `4.0 ×`.

| Requested audio | Budget | Measured wall clock | Margin |
|---|---|---|---|
| 30 s | 600 s | 46–69 s | ~9× |
| 120 s | 780 s | 96 s | 8× |
| 180 s | 1020 s | 90 s | 11× |
| 240 s | 1260 s | 76 s | 16× |

Honest framing: given the measured numbers, this is **defence in depth,
not a fix for an imminent breach**. The pre-gate extrapolation suggested
the 600 s default was nearly breached at 240 s; measurement showed
otherwise. It is still the right shape — a timeout must tell "provider
dead" apart from "long request", and one flat number cannot — and it
can only ever be *more* generous than Phase 8, so nothing that worked
before can start failing. An operator who deliberately tightens the
timeout still gets exactly what they asked for.

---

## 8. What still requires a human

Everything that decides whether Phase 9 mattered:

- Whether the 180 s and 240 s tracks hold together musically — opening,
  verse identity, chorus recurrence, bridge, ending.
- Whether Korean lyric lines were skipped, and which ones.
- Whether contemporary vocal conditioning moved the vocal style.
- Whether shorter lyric lines reduce omission.
- Whether the vocal still sounds trot-like or dated.
- Key mode; time signature.

The listening package at
`benchmarks/music_quality/phase9/listening/index.html` covers all 15
real tracks: the three long-form gates and the twelve experiment tracks.
It asks for an overall 1–10 first, shows detailed dimensions only at 5
or above, and offers the twelve required failure tags plus per-line
completeness with `UNKNOWN` as the default.

---

## 9. Known limitations

1. **Phase 9 did not improve musical quality.** It made full songs
   reachable and made defects measurable. The 2/10 baseline stands until
   a listener says otherwise.
2. **`SIBILANCE_RISK_PROXY` is a band-energy proxy**, not a sibilance
   detector.
3. **Key mode estimation is unreliable**; tonic is suggestive only.
4. **Time signature is unverified.**
5. **Tempo estimation is validated on synthetic click tracks only** —
   the 0.04 BPM agreement on real tracks is strong evidence, but the
   estimator has no real-music ground truth.
6. **n=3 per experiment arm.** Enough to show conditioning is not inert;
   not enough to establish a direction.
7. **240 s is close to this machine's resource edge** (206 MB free swap
   at the trough). Not a batching platform.
8. **No progress percentage.** Upstream exposes none, and
   `/v1/stats avg_job_seconds` averages over all durations, so it would
   be a fabricated number. Status remains stage-based and factual.
9. **Long-form output has not been listened to.**

## 10. What Phase 9 does NOT solve

- Musical quality, arrangement coherence, vocal realism, instrument
  fidelity.
- The trot-like vocal character and dated vocal style.
- Korean lyric-line omission — Phase 9 measures and records it; it does
  not reduce it.
- Excessive high-frequency energy. It is now *quantified*
  (centroid ~3 kHz, ~15 % above 8 kHz) and deliberately **not**
  cosmetically corrected: de-essing or EQ-ing the output would hide a
  model defect behind post-processing, which this phase explicitly
  refuses to do.
- Suno 4.5 parity. Not approached, not claimed.
