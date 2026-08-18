# Failure taxonomy

Structured tags for what went wrong in a generation. A tag is an
observation, not a score: any number may apply to one track, and their
absence is not a claim of quality.

These exist to survive past this baseline. When a fine-tuned candidate is
compared against the frozen baseline, "it sounds better" is not an
analysable statement; "`VOCAL_TROT_STYLE` dropped from 14 tracks to 3 and
`KOREAN_LYRIC_OMISSION` was unchanged" is. The tag vocabulary is
therefore fixed in the same way the rubric anchors are — a new tag means
a new taxonomy version.

## Vocal

| Tag | Meaning |
|---|---|
| `VOCAL_SYNTHETIC` | Timbre is recognisably machine-made. |
| `VOCAL_TROT_STYLE` | Trot-like (뽕끼) delivery or phrasing. |
| `VOCAL_EXCESSIVE_VIBRATO` | Wobble beyond what the genre supports. |
| `VOCAL_PITCH_INSTABILITY` | Drifting or unstable pitch. |

## Korean

| Tag | Meaning |
|---|---|
| `KOREAN_PRONUNCIATION` | Words are wrong or unintelligible as Korean. |
| `KOREAN_LYRIC_OMISSION` | Supplied lyrics not sung. Pair with the line. |
| `KOREAN_SYLLABLE_TIMING` | Syllables rushed, stretched or misaligned. |
| `LYRIC_PHRASE_SKIPPED` | A whole phrase absent. The severe form of omission. |

`KOREAN_LYRIC_OMISSION` and `LYRIC_PHRASE_SKIPPED` are the tags a future
ASR alignment pass will be able to confirm mechanically. Until then they
are human judgements and are recorded as such — no word-error rate is
computed or quoted, because none has been measured.

## Instruments and structure

| Tag | Meaning |
|---|---|
| `INSTRUMENT_SYNTHETIC` | Instruments do not sound like instruments. |
| `INSTRUMENT_BLUR` | Sources smeared together, poorly defined. |
| `TRANSIENT_WEAK` | Attacks soft; drums and plucks lack definition. |
| `ARRANGEMENT_COLLAPSE` | Arrangement falls apart partway. |
| `LONG_FORM_DRIFT` | Piece wanders from what it started as. |
| `REPETITION_EXCESS` | Repeats past the point of interest. |

## Frequency

| Tag | Meaning |
|---|---|
| `LOW_END_EXCESS` | Boomy, dominating low end. |
| `LOW_MID_MUD` | Congested 200–500 Hz. |
| `MID_HOLLOW` | Scooped mids; thin body. |
| `PRESENCE_EXCESS` | Forward and fatiguing 2–5 kHz. |
| `PRESENCE_DEFICIT` | Dull, recessed, lacking articulation. |
| `HIGH_FREQUENCY_DEFICIT` | Rolled off; no air. |
| `HIGH_FREQUENCY_EXCESS` | Over-bright. |
| `HARSHNESS` | Abrasive upper-mid edge. |
| `SIBILANCE` | Excessive "s" energy on vocals. |

`HIGH_FREQUENCY_DEFICIT` and `HIGH_FREQUENCY_EXCESS` are both in the
reported symptoms and both are expected to appear — on different tracks.
Whether the corpus leans one way is a question for the measurements, not
for a single listener's impression.

## Stereo and space

| Tag | Meaning |
|---|---|
| `STEREO_NARROW` | Close to mono; no width. |
| `STEREO_UNSTABLE` | Image wanders or phases. |
| `DEPTH_FLAT` | Everything at one distance; no front-to-back. |
| `REVERB_UNNATURAL` | Space sounds artificial or mismatched. |

## Composition

| Tag | Meaning |
|---|---|
| `MELODY_WEAK` | Melody uninteresting or aimless. |
| `MELODY_TROT_LIKE` | Melodic shape itself is trot-like, apart from delivery. |
| `GENRE_MISMATCH` | Not the genre that was asked for. |

`MELODY_TROT_LIKE` is kept separate from `VOCAL_TROT_STYLE` deliberately.
A trot-shaped melody sung plainly and a modern melody sung with trot
inflection are different failures with different fixes, and collapsing
them into one tag would hide which one the model actually has.

## Endings and integrity

| Tag | Meaning |
|---|---|
| `EARLY_FADE` | Fades out before the intended duration. |
| `ABRUPT_END` | Stops without an ending. |

`EARLY_FADE` is one of the few tags the objective analysis can
corroborate — a high trailing silence ratio is measurable. Where the
measurement and the listener disagree, the listener is right about the
music and the measurement is right about the file; record both.
