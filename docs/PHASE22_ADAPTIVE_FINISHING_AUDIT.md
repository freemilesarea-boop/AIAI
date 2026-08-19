# Phase 22 — Adaptive Audio Finishing Engine

Audit of the Phase 14 finishing engine, the gaps found against the
Phase 22 mission, and what was built to close them.

Scope note: this concerns the LUBER generative music engine only. The
separate Louver Mastering AI project was not imported, referenced or
consulted, and nothing here is derived from it.

---

## 1. What Phase 14 already provides

The audit began by reading `packages/audio-finishing` in full and then
running it over real RAW masters, because what an engine does and what
its source appears to say are different questions.

The existing engine is strong, and Phase 22 builds on it rather than
replacing it:

| Mission requirement | Phase 14 status |
|---|---|
| RAW immutable | **Complete.** Never writes to its input, refuses a destination equal to its source, refuses an already-stamped file. |
| Analyze | **Complete.** 742 lines of objective measurement: bands, slope, ratios, sibilance, stereo, transients, loudness. |
| Conservative adaptive correction | **Complete.** Every rule is proportional, partial, ceilinged, and records what it asked for versus what it got. |
| No fixed global preset | **Complete.** Thresholds are absolute and within-track; 14 of 40 baseline masters tripped nothing. |
| `NO_ACTION` on good input | **Complete.** An empty plan writes no file at all. |
| Analyze finished output | **Partial.** Measured, but only for technical safety. |
| Accept only if safer/better | **Absent.** |
| RAW fallback | **Partial.** Existed for *failures*, not for *bad results*. |

The design principles already in place — proportional correction,
contradictions resolving toward doing less, deferrals recorded with
reasons — are the right ones and were extended, not altered.

---

## 2. Gaps found

### 2.1 Safety was standing in for quality

`processor._verify` checked clipping, peak ceiling, duration, sample
rate, channel count and output balance. Every one of those passes on a
render that did the exact opposite of what it intended — **a mis-signed
shelf is exactly as peak-safe as a correct one.** The engine could apply
a lift, measurably darken the track, and ship it.

Nothing compared the finished audio to the raw master on the dimensions
the plan claimed to be correcting.

### 2.2 Corrections were not watched for collateral damage

Running the Phase 14 engine over five real masters showed the cost of a
correction going unmeasured. On `00c8f77a` the high-shelf lift did what
it intended — air ratio `-35.67 → -32.45` — and also moved:

- sibilance peak excess `15.68 → 16.03 dB`
- harshness peak excess `13.03 → 13.52 dB`

Neither crossed a threshold, so neither was flagged, and nothing in the
engine was watching the trade. The shelf that adds air adds sibilance
with it; that is not a defect in the shelf, it is a fact about shelves,
and it has to be measured rather than hoped about.

### 2.3 Only half of each condition axis was supported

The mission requires opposite conditions handled **independently**.
Phase 14 handled one side of each:

| Axis | Handled | Missing |
|---|---|---|
| dark / bright | dark (`HIGH_FREQUENCY_DEFICIT`, `AIR_DEFICIT`) | **bright** — a tilted-up master got no response |
| narrow / phase-unsafe | narrow, wide, low-band phase | **broadband phase** — no detection, no floor under widening |
| muddy / bass-intentional | muddy (`LOW_MID_MUD`) | **bass-intentional** — a deliberate heavy low end was treated as excess |

The bass case was the sharpest. `LOW_END_EXCESS` fires on 7 of the 57
real masters, and 5 of those have clean, mono-compatible low ends — the
weight *is* the track. Phase 14 would have cut all 7. It only avoided
doing so by accident: the requested cut fell below `MIN_ACTION_DB` in
most cases, so the rule was very nearly inert rather than correct.

### 2.4 A declined correction was indistinguishable from an absent one

`_presence_lift` returned `[]` when harshness was present. That is the
right decision, and it left no trace. A rule that silently returns
nothing looks identical to a rule that never fired, and the two mean
opposite things about the audio.

---

## 3. What was built

### 3.1 `acceptance.py` — the adjudicator

The core addition. Every render is measured and put to three classes of
check, and **any** failure rejects:

- **Safety** — everything `_verify` did, now recorded per check with its
  numbers rather than raised as a joined string.
- **Efficacy** — each action names a metric and a direction; the finished
  audio must have moved it that way by a worthwhile share of what was
  requested.
- **Regression** — sibilance, harshness, broadband and low-band
  correlation, crest factor, loudness, and low-mid thickness are checked
  *whether or not the plan targeted them*.

Every check runs even after one fails, so a verdict carries the full
picture rather than the first objection.

### 3.2 Rejection as a first-class outcome

`finish_audio` no longer raises on a bad result. It returns a
`FinishingResult` with `output_path=None`, the full verdict, and the
rejected file deleted from disk. `FinishingOutcome.REJECTED` joins
`FINISHED`, `NO_ACTION` and `FAILED`.

Four ways to deliver the raw master, kept apart because they call for
different responses:

| Outcome | Meaning | Response |
|---|---|---|
| absent record | the engine never ran | nothing |
| `NO_ACTION` | nothing needed correcting | nothing |
| `REJECTED` | corrected, measured, judged worse | tune the rules |
| `FAILED` | the engine could not run | fix the bug |

### 3.3 The missing halves of each axis

- **`EXCESSIVE_BRIGHTNESS`** → `HIGH_SHELF_CUT`, corner at 10 kHz, held
  to a tighter ceiling than the lift (2.0 dB vs 3.0 dB) because cutting
  discards information a later stage cannot restore. Requires a high air
  ratio **and** a shallow slope, so a *spiky* track is not mistaken for a
  *tilted* one — a shelf is the wrong tool for a band spike.
- **`low_end_is_intentional`** → suppresses `LOW_SHELF_CUT` when the low
  end does not smear into 150–400 Hz and survives a mono fold-down.
- **`BROADBAND_PHASE_RISK`** and `SAFE_TO_WIDEN_CORRELATION` → a floor
  under the widening rule.

### 3.4 `SuppressedAction`

Per-track record of a correction the rules called for and the engine
declined, with the evidence that overrode it. Distinct from
`DeferredDecision`, which is a standing choice about a whole area.

---

## 4. Threshold calibration

Anchored on all 57 real RAW masters, following the existing doctrine:
absolute rather than corpus-relative, and rare by design.

| Constant | Value | Corpus behaviour |
|---|---|---|
| `EXCESSIVE_BRIGHTNESS_AIR_DB` | −16.0 | median −25.7, p90 −18.3, max −12.9 |
| `BRIGHT_SLOPE_DB_PER_OCTAVE` | −5.0 | median −6.2, max −3.5 |
| both together | | **3 of 57** |
| `INTENTIONAL_LOW_END_CORRELATION` | 0.90 | separates 5 intentional from 2 muddy, of 7 flagged |
| `BROADBAND_PHASE_CORRELATION` | 0.20 | **0 of 57** — a tripwire, not a description |
| `SAFE_TO_WIDEN_CORRELATION` | 0.50 | 0 of the 6 narrow masters |

`BROADBAND_PHASE_CORRELATION` firing on nothing is intentional and
matches the precedent set by `TRANSIENT_FLAT_CREST_DB`: it exists to
catch a regression, not to describe current output.

---

## 5. Two findings that changed the work

### 5.1 Narrow and out-of-phase are the same measurement

Width is `side/(mid+side)`, so anti-correlated channels put nearly all
their energy in the side and measure as **wide**. Inverting one channel
of a narrow mix produces `width 1.000, correlation −1.000` — not a
narrower one.

A track therefore cannot be both narrow and broadband out of phase, and
`SAFE_TO_WIDEN_CORRELATION` is structurally unreachable from real audio.
It was kept as a floor under the widening rule and is tested against a
constructed analysis, with the unreachability documented rather than
quietly presented as a feature.

The genuine phase-unsafe case — wide, anti-correlated — is handled: the
image is narrowed and the bass summed to mono.

### 5.2 The synthetic "healthy" fixture was not healthy

The test fixtures' baseline was shaped noise at −4 dB/octave, chosen to
sit "comfortably clear of every deficit threshold" — clear on one side
only, which was sufficient while the engine could only detect darkness.

It measures an air ratio of **−9.2 dB**, brighter than any of the 57 real
masters, whose maximum is −12.9. Once brightness became detectable, every
fixture built on that baseline carried a defect it was never meant to
have.

Fixed with an 8 dB trim above 10 kHz applied to the shared stereo
helper, landing the baseline at slope −6.04 and air −17.2 — clearing the
dark rule's slope condition and the bright rule's air condition
*separately*, so the fixture does not depend on one half of an `AND` to
stay healthy. Steepening the slope instead was rejected: it drags in
`LOW_END_EXCESS`, `TRANSIENT_FLATNESS` and `STEREO_TOO_NARROW`, the same
confound the existing `dull_stereo` docstring already warns about.

---

## 6. Verification

Objective analysis, synthetic DSP fixtures, and 5 real RAW smoke tracks.
No human listening was required or requested.

Smoke results (`data/audio`, first 5 masters, RAW SHA-256 verified
identical before and after in every case):

| Master | Outcome | Detail |
|---|---|---|
| `00c8f77a` | FINISHED | 16/16 checks passed |
| `03da4daa` | FINISHED | 15/15 |
| `06db2d47` | FINISHED | 15/15 |
| `07d6cf93` | FINISHED | 15/15; presence lift suppressed by harshness |
| `0bb96b5e` | FINISHED | **brightness trim** applied; **bass cut suppressed** as intentional |

`0bb96b5e` exercises both new conditions in one track, and both correctly:
it is trimmed for brightness and its heavy low end is left alone.

Widened to 22 masters to see the outcome mix rather than a slice:

| Outcome | Count |
|---|---|
| `NO_ACTION` | 7 |
| `FINISHED` | 14 |
| `REJECTED → RAW` | 1 |

RAW SHA-256 verified identical before and after on all 22, and no
rejected render was left on disk. Good input producing `NO_ACTION` on
roughly a third of the corpus is the property that matters most here:
the engine is not looking for work.

The single rejection is `0c758827`, whose low-mid cut moved the band by
0.07 dB against 1.11 dB requested. It is the case that produced the
phase's most useful finding — see §7.

---

## 7. Efficacy floors are per filter shape

The first `MIN_EFFICACY_FRACTION` was a single global 0.5. Running it on
real audio rejected `06db2d47` because its low-mid cut delivered 0.98 dB
against 1.99 dB requested — 49%, missing the floor by one percent.

That is not a bad render. It is a **bell judged against a band average**.
A shelf moves an entire band, so it delivers essentially all of its gain
to the metric; a bell only cuts near its centre, so a band-average metric
necessarily moves less than the bell's gain. Measured across the corpus:

- shelves deliver 92–111% of the requested move
- bells deliver substantially less, by construction

Measured across the corpus:

| Action | Delivered / requested | n |
|---|---|---|
| `HIGH_SHELF_LIFT` | 0.90 – 1.34 | 9 |
| `PRESENCE_LIFT` | 0.86 | 1 |
| `HIGH_SHELF_CUT` | 0.74 | 1 |
| `LOW_MID_CUT` | 0.06 – 0.72, median 0.49 | 8 |

The reason for the spread is geometric, and the two worst cases prove
it: the bells that delivered 0.06 and 0.26 are centred at **387 Hz and
363 Hz**, against a band that ends at 400. Most of the filter acts
outside the window the metric measures. Those renders are not wrong —
the metric is a poor witness to what they did.

So the floors are set per action kind:

- shelves and the presence bell: **0.5**, against measured 0.74–1.34
- the low-mid bell: **0.20**, catching a cut that achieved essentially
  nothing while accepting an edge-centred one that genuinely acted

This is knowingly the weakest check in the adjudicator, and it is
documented as such in the source: for a bell judged against a fixed
band, direction is verifiable and magnitude is not. Under the original
single floor, `06db2d47` was rejected by one percentage point and the
raw master shipped in place of a correction that had worked.

---

## 8. What was deliberately not done

- **No model training.** Out of scope and explicitly excluded.
- **No human listening.** Phase 20H stays deferred; acceptance here is
  entirely objective.
- **No dynamics processing.** The Phase 14 deferrals for transient
  shaping, spatial processing and dynamic EQ were re-read and stand;
  each was measured rather than assumed, and nothing in Phase 22
  changes the evidence.
- **No loudness normalisation target.** Finished audio is matched to the
  source's loudness and may end up quieter for peak safety, never louder.
- **Nothing from the Louver Mastering AI project.**
