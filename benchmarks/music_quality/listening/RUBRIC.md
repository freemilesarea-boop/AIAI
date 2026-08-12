# LUBER Music Quality Listening Rubric

Every dimension is scored **1–10 integer**. Score what you actually
hear, not what the prompt promised. If a dimension does not apply
(vocal dimensions on an instrumental), **omit it** — do not score it 0
or 5, because an omitted score is excluded from averages while a filled
one distorts them.

## Anchors used throughout

| Score | Meaning |
|---|---|
| 1–2 | Broken. Unlistenable or fundamentally wrong. |
| 3–4 | Bad. Recognisably music, but a listener would stop. |
| 5–6 | Demo quality. Obviously machine-made; usable as a sketch. |
| 7 | Usable with editing. A competent producer could fix it. |
| 8 | Commercially usable as-is. |
| 9 | Professional commercial quality. |
| 10 | Top-tier release quality. |

Be strict. An 8 means you would genuinely ship it.

---

## 1. Overall Musical Quality
Your holistic judgement as a listener. Would you keep listening? This
is not the average of the other dimensions — a track can be technically
clean and still be lifeless.

## 2. Composition / Melody
Is there a real melodic idea? Is it memorable, shaped, and developed —
or does it wander, loop a two-bar cell, or never resolve?

## 3. Harmony
Chord choices and voice leading. Do progressions make sense for the
genre? Penalise random modulations, unresolved dissonance that reads as
error rather than intent, and static one-chord drift.

## 4. Rhythm / Groove
Does it feel good? Timing stability, pocket, and whether the drums and
bass agree. Penalise drift, smeared transients, and grooves that fight
the tempo.

## 5. Arrangement
Instrument choice, layering, and whether parts leave space for each
other. Does the track change over time, or is it one loop repeated?

## 6. Song Structure
Are sections distinguishable, and do they do their job? Verse ≠ chorus,
chorus recurs recognisably, bridge contrasts, intro and outro exist and
resolve. For prompts with an explicit `[Verse]/[Chorus]/[Bridge]`
structure, score how faithfully that structure is realised.

## 7. Vocal Naturalness *(vocal only)*
Does it sound like a human sang it? Breath, phrasing, micro-timing, and
vibrato. Penalise robotic delivery, formant wobble, and pitch that snaps
between notes.

## 8. Vocal Tone *(vocal only)*
Timbre quality and suitability. Is the voice pleasant, consistent in
identity across the track, and appropriate for the genre and the
requested vocal gender?

## 9. Lyrics Pronunciation *(vocal only)*
Are the supplied words intelligible and correctly pronounced? For
Korean specifically, listen for:
- **받침** (final consonants) — dropped or wrong
- **연음** (liaison across syllables) — unnatural breaks
- mixed Latin words and numbers inside Korean lines
- very short syllables swallowed at speed
- long sustained syllables in slow ballads losing vowel identity

## 10. Lyrics Alignment *(vocal only)*
Are the supplied lyrics actually sung, in order, in the right sections?
Penalise omission, invented words, repeated lines that were not
repeated in the source, and lyrics landing in the wrong section.

## 11. Prompt Adherence
Did it deliver what the prompt asked — instrumentation, tempo feel,
mood, and named elements? Judge against the prompt text, not against
what would have sounded nicer.

## 12. Genre Authenticity
Would someone who listens to this genre accept it as that genre?
Penalise generic "AI pop" that ignores genre conventions.

## 13. Mix Balance
Relative levels, frequency balance, stereo image. Penalise buried
vocals, muddy low-mids, harsh highs, and a mix that collapses in mono.

## 14. Artifact / AI Weirdness
**Higher is better** (10 = no artifacts). Score the *absence* of
digital artifacts: warbling, metallic ringing, smeared transients,
phantom voices, sudden dropouts, nonsense syllables.

## 15. Commercial Release Readiness
Answer one question:

> If this track were placed in a Spotify or YouTube Music playlist
> alongside human-made songs, would it stand out as noticeably worse
> *because* it is AI-generated?

| Score | Meaning |
|---|---|
| 1 | Unusable |
| 5 | Obvious AI / demo |
| 7 | Usable with editing |
| 8 | Commercially usable |
| 9 | Professional commercial quality |
| 10 | Top-tier commercial release |

---

## Artifact tags

Attach every tag that applies. Tags are the raw material for gap
classification, so tag generously.

`VOCAL_ROBOTIC` `VOCAL_WOBBLE` `BAD_PRONUNCIATION` `LYRIC_OMISSION`
`LYRIC_REPETITION` `MELODY_REPETITIVE` `STRUCTURE_COLLAPSE`
`BAD_TRANSITION` `RHYTHM_DRIFT` `HARMONY_WEIRD` `INSTRUMENT_ARTIFACT`
`MIX_MUDDY` `MIX_HARSH` `HIGH_FREQ_ARTIFACT` `LOW_END_PROBLEM`
`UNNATURAL_REVERB` `UNWANTED_NOISE` `GENERIC_COMPOSITION` `PROMPT_MISS`
`GENRE_MISS` `OTHER`

---

## Listening procedure

1. **Blind by default.** Do not look at the configuration before
   scoring. The listening tool hides it until you save.
2. **Listen to the whole track.** For long-form (≥180 s) you must not
   score from the first 30 seconds — drift, structure collapse, and
   vocal identity loss appear late by definition.
3. **One pass, then score.** Re-listen only for dimensions you are
   unsure about.
4. **Do not compare against your memory of a previous run.** Use the
   A/B tool when you need a comparison.

## Internal quality gate

These are the Phase 5 targets. They are aspirations, not results, and
are never lowered to make a run pass.

| Dimension | Target |
|---|---|
| Overall Musical Quality | ≥ 8.0 |
| Commercial Release Readiness | ≥ 8.0 |
| Vocal Naturalness | ≥ 8.0 |
| Lyrics Pronunciation (Korean) | ≥ 8.0 |
| Prompt Adherence | ≥ 8.0 |
| Song Structure | ≥ 8.0 |
| Fatal technical failure rate | < 2% |
| Obvious AI artifact rate | < 10% |
