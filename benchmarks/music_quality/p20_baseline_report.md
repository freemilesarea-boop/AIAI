# Phase 20 baseline report — `luber-baseline-p20-v1`

Frozen state of LUBER's generative music quality, measured on **RAW model
masters only**. The Phase 14 finished master is a delivery artefact; using
it here would credit the equaliser for the model's work.

The document has two halves and they must not be confused. The objective
half is measured. The human half **has not happened**.

| | |
|---|---|
| Baseline id | `luber-baseline-p20-v1` |
| LUBER commit | `5a1d0674851a097c37612fb8b63e65946e2a02f2` |
| Engine | ACE-Step `6d467e4b…`, `acestep-v15-turbo`, 8 steps |
| Benchmark | `BENCHMARK_P20.json` — 28 cases (12 GEN, 4 TROT, 8 KO, 4 LONG) |
| Rubric / taxonomy | `RUBRIC_P20.md`, `TAXONOMY.md` |
| Manifest | `p20_baseline_manifest.json` |

---

## Objective findings

Measured with `luber_audio_finishing.analyze_audio` over **45 RAW
masters** — the project's full generated corpus, including nine cases
generated for this benchmark. Medians with full range:

| Metric | Median | Range |
|---|---|---|
| Integrated loudness (LUFS) | −14.1 | −17.3 … −12.5 |
| Crest factor (dB) | 16.2 | 14.1 … 19.6 |
| Spectral centroid (Hz) | 1762 | 442 … 3972 |
| Spectral slope (dB/oct) | −6.2 | −11.2 … −3.5 |
| Air ratio (dB) | −26.1 | −39.6 … −12.9 |
| Low-mid ratio (dB) | 3.2 | −4.7 … 19.0 |
| Presence ratio (dB) | −15.6 | −26.5 … −3.0 |
| Sibilance ratio (dB) | −26.2 | −38.1 … −9.1 |
| Harshness ratio (dB) | −17.3 | −28.6 … −4.8 |
| Stereo width | 0.179 | 0.073 … 0.418 |
| Stereo correlation | 0.762 | 0.369 … 0.939 |
| Silence ratio | 0.075 | 0.016 … 0.592 |

### What the numbers actually support

**Narrow stereo — corroborated.** Median width 0.179, median correlation
0.762. Twelve of 45 tracks fall below 0.15 width and three exceed 0.9
correlation, which is close to mono. The human report of weak stereo
depth is not just an impression; it is the corpus's central tendency.

**Inconsistent top end — corroborated, and it resolves an apparent
contradiction.** Listeners reported *both* rolled-off highs and excessive
high-frequency energy. Air-band energy spans 26.7 dB (−39.6 to −12.9) and
spectral centroid spans nearly an order of magnitude (442 to 3972 Hz).
Both reports are right, about different tracks. The defect is variance,
not a fixed tilt — which matters, because a global EQ fix would make half
the corpus worse.

**At least one structural failure is real and measurable.** One track is
59.2% silence, with two more above 18%. That is an early-fade or collapse
case that needs no listener to confirm.

**Dynamics are a current strength.** Median crest factor 16.2 dB with
loudness around −14 LUFS: the model is not producing over-compressed
output. This is worth protecting — a future candidate that "sounds
louder" by flattening this is a regression, and the promotion policy
treats it as one.

**Low-mid mud — not confirmed either way.** Median low-mid ratio 3.2 dB
with a 23.7 dB spread. There is no corpus-wide tendency here; the
listener reports may be about specific tracks. Human scoring will settle
it.

### What these numbers cannot do

They cannot hear a weak melody, a trot-like phrase, a dropped Korean
line, or a chorus that fails to lift. Every one of the primary reported
failures is invisible to them. The measurements narrow where to look;
they do not substitute for listening, and no quality claim is made from
them.

---

## Human scores — **PENDING**

**No human listening has been performed against this baseline.** Every
rubric dimension is UNKNOWN.

| Dimension group | Status |
|---|---|
| Composition | PENDING |
| Arrangement | PENDING |
| Instrument quality | PENDING |
| Vocal quality | PENDING |
| Korean vocal | PENDING |
| Vocal style (incl. `trot_absence`) | PENDING |
| Mix / sonics | PENDING |
| Overall | PENDING |

**Current overall quality: UNKNOWN.**

The figure of roughly 2/10 from earlier sessions is a recollection of
listening to *different* generations under no fixed rubric. It is not a
measurement against this baseline and is not recorded as one. Carrying it
forward as if it were would corrupt the first real comparison.

The known human reports — synthetic timbre, Korean phrase omission,
trot-like delivery, weak melody — are treated here as **hypotheses to
score**, not as established properties of every track.

---

## Generated for this baseline

Nine cases, through the production path (queue, worker, storage), one at
a time:

| Case | Duration | Seed | Wall time |
|---|---|---|---|
| GEN-01 | 60 s | 3710749089 | 92.2 s |
| GEN-06 | 60 s | 456103037 | 99.3 s |
| GEN-10 | 60 s | 2862688926 | 107.7 s |
| TROT-01 | 60 s | 2806572120 | 97.9 s |
| TROT-02 | 60 s | 329408103 | 72.7 s |
| KO-01 | 30 s | 2874670796 | 68.4 s |
| KO-02 | 30 s | 1563700957 | 47.0 s |
| KO-03 | 30 s | 2243652556 | 36.3 s |
| KO-05 | 30 s | 229337410 | 45.1 s |

9/9 completed. The remaining 19 benchmark cases are defined but not yet
generated — the suite is frozen, the corpus against it is partial, and
that is stated rather than papered over.

## Next

The blind listening pass. Until it exists there is no baseline quality
score, and no candidate model can be compared to anything.

The single most decisive measurement in it is `trot_absence` on the four
`TROT-*` cases, whose prompts explicitly demand restrained contemporary
delivery. If the trot character survives an explicit instruction against
it, the bias is in the model rather than the prompt — and that one result
decides whether the first fine-tuning experiment is worth running at all.
