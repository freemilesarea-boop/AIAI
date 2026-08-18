# Phase 20 listening rubric

Extends the v1 rubric rather than replacing it. The anchors are
unchanged, because moving anchors between baselines destroys the only
thing a baseline is for. What Phase 20 adds is the set of dimensions a
human listener actually complained about and v1 had no place to record:
trot-like delivery, Korean lyric completeness, stereo depth, instrument
resolution.

**Everything here scores the RAW model master.** Not the Phase 14
finished master. Finishing exists to make delivery better and it does;
scoring it here would credit the equaliser for the model's work and hide
exactly the deficiencies this baseline is meant to expose.

## Anchors

Fixed before any listening. They are not adjusted after seeing results.

| Score | Meaning |
|---|---|
| 10 | Competitive with a professional commercial release. |
| 9 | Professional quality; a listener would not know it was generated. |
| 8 | Genuinely releasable with minor work. |
| 7 | Usable with editing — a competent producer could fix it. |
| 6 | Demo quality, the better end. |
| 5 | Usable demo, obvious shortcomings. |
| 4 | Bad. Recognisably music, but a listener would stop. |
| 3 | Major quality problems throughout. |
| 2 | Barely music. |
| 1 | Not usable as music output. |

Score what you hear, not what the prompt promised. **Omit** a dimension
that does not apply — vocal dimensions on an instrumental are left blank,
never scored 0 or 5. An omitted score is excluded from averages; a filled
one distorts them.

---

## COMPOSITION

| Dimension | Question |
|---|---|
| `melody_quality` | Is the melody worth hearing again? |
| `harmonic_coherence` | Do the chords make sense and resolve? |
| `phrasing` | Does the musical line breathe like a written one? |
| `hook_strength` | Is there something you would remember an hour later? |
| `commercial_plausibility` | Could this appear on a playlist without explanation? |

## ARRANGEMENT

`section_structure`, `transitions`, `energy_progression`,
`repetition_control`, `long_form_coherence`

Long-form coherence is scored only on the 120 s and 180 s cases.

## INSTRUMENT QUALITY

`timbral_realism`, `instrument_definition`, `transient_quality`,
`separation`, `production_resolution`

`production_resolution` is the "does this sound like a real recording or
like a compressed rendering of one" axis — the low-resolution character
listeners reported.

## VOCAL QUALITY *(vocal cases only)*

`vocal_naturalness`, `vocal_timbre`, `pitch_stability`, `vocal_phrasing`,
`emotional_appropriateness`

## KOREAN VOCAL *(Korean vocal cases only)*

| Dimension | Question |
|---|---|
| `pronunciation` | Would a Korean listener hear correct Korean? |
| `lyric_completeness` | Was every supplied line actually sung? |
| `syllable_timing` | Do syllables land musically, not rushed or stretched? |
| `phrase_omission` | Scored inversely: 10 = nothing dropped. |
| `segmentation_naturalness` | Are breaks in sensible places? |

`lyric_completeness` and `phrase_omission` are scored against the
**expected lyrics shown alongside the player**, which are stored verbatim
in the benchmark file for exactly this purpose.

## VOCAL STYLE

| Dimension | Question |
|---|---|
| `trot_absence` | 10 = no trot-like delivery at all. 1 = pervasive. |
| `vibrato_control` | 10 = appropriate. 1 = exaggerated, wobbling. |
| `ornament_appropriateness` | Are melodic ornaments suited to the genre? |
| `genre_appropriateness` | Does the delivery match what was asked for? |

`trot_absence` on the four `TROT-*` cases is the load-bearing
measurement of this phase. Those prompts explicitly ask for restrained
contemporary delivery. If the score is low **there**, the bias is in the
model and prompting will not fix it — which is what decides whether the
first fine-tuning experiment targets vocal style at all.

## MIX / SONICS

`frequency_balance`, `low_mid_clarity`, `presence`,
`high_frequency_detail`, `harshness` (10 = none), `sibilance` (10 = none),
`stereo_image`, `depth`, `ambience`, `dynamics`

## OVERALL

`listenability`, `commercial_readiness`, `would_release` (yes/no, not a
score), `overall_preference`

`overall_preference` is the single number used for baseline-versus-
candidate comparison. Everything else explains it.

---

## Procedure

1. The tool shows a benchmark ID, the prompt, and expected lyrics. It
   does **not** show the model, the checkpoint, or the generation id.
2. Play the whole track before scoring anything.
3. Score the dimensions that apply; skip the rest.
4. Add artifact tags from `TAXONOMY.md`. Tags are observations, not
   scores, and any number may apply.
5. Free-text note for anything the rubric has no dimension for. These
   notes are where the *next* rubric version comes from.

Sessions can be resumed. A partially scored baseline is normal and is
reported as partial — never averaged as if complete.

## What must not happen

- No score may be entered without listening to the track.
- No dimension may be added, removed or re-anchored mid-baseline.
- No model identity may be revealed before scoring.
- A missing score stays missing. Estimating one to fill a table is
  fabricating data, and a fabricated baseline makes every future
  comparison meaningless.
