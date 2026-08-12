# Phase 6 — Quality Root Cause Map

Maps each Phase 5 human finding to the engineering layers that could
produce it.

**These are hypotheses.** Nothing here is a claim of causality. A
hypothesis is only promoted to a cause when an experiment isolates it,
and the experiment column records what would settle it.

The starting position is unusual and worth stating plainly: the
pipeline scored **0% technical failure** while the music scored
**2/10 with 26/26 rejected**. Every failure below is therefore
downstream of a correctly-functioning pipeline — the audio decodes,
hits its duration, never clips, and is byte-verified end to end. The
problems are in what the model generates and how it is conditioned.

## Layer vocabulary

| Layer | Meaning |
|---|---|
| `MODEL_BASE_LIMITATION` | The 2B turbo DiT cannot do this at all |
| `VAE_OR_AUDIO_DECODER_LIMITATION` | The codec/VAE reconstruction is the ceiling |
| `TRAINING_DATA_DOMAIN_GAP` | The base model never saw enough of this domain |
| `VOCAL_STYLE_BIAS` | The base model's vocal prior is wrong for the target |
| `KOREAN_LANGUAGE_ALIGNMENT_GAP` | Korean text→phoneme→audio alignment is weak |
| `LYRIC_ALIGNMENT_GAP` | Lyric-to-timeline scheduling drops or reorders content |
| `PROMPT_COMPILER_PROBLEM` | LUBER's own prompt construction |
| `INFERENCE_CONFIGURATION_PROBLEM` | Parameters we send (or fail to send) |
| `POST_PROCESSING_PROBLEM` | LUBER's Phase 4 delivery stage |
| `UNKNOWN` | Not yet attributable |

---

## H1 — Frequency balance is poor

**Human finding:** overall frequency balance is wrong.

| Hypothesis | Layer | Confidence | How to settle it |
|---|---|---|---|
| Base model's spectral prior is genuinely off-target | `MODEL_BASE_LIMITATION` | Medium | Compare against XL turbo (4B decoder) on a GPU |
| VAE reconstruction colours the spectrum | `VAE_OR_AUDIO_DECODER_LIMITATION` | Medium | Encode→decode a *reference* commercial track through the VAE alone and measure the spectral delta; that isolates the codec from the generator |
| LUBER post-processing altered it | `POST_PROCESSING_PROBLEM` | **Ruled out** | Phase 4 applies format conversion only — no EQ, no normalization. Verified in `transcode.py` and its tests |

**Measured context.** Median spectral centroid across the pilot was
3255 Hz (Korean vocal) and 3482 Hz (English). For reference, dense
commercial pop masters typically sit lower. This is consistent with the
human "too bright" report but is not proof on its own.

**Explicitly not a fix:** applying corrective EQ to the master. See
Step 2 policy below.

---

## H2 — Instrument fidelity and instrument sound quality are poor

| Hypothesis | Layer | Confidence | How to settle it |
|---|---|---|---|
| 2B DiT lacks capacity for convincing instrument timbre | `MODEL_BASE_LIMITATION` | **High** | XL turbo (4B) A/B on GPU — upstream's stated XL advantage is exactly "higher audio quality" |
| VAE is the reconstruction ceiling | `VAE_OR_AUDIO_DECODER_LIMITATION` | Medium | VAE round-trip test on reference audio |
| Base training data under-represents well-recorded acoustic instruments | `TRAINING_DATA_DOMAIN_GAP` | Medium | Only addressable by training |

This is the finding most likely to be a genuine base-model ceiling
rather than something configuration can reach.

---

## H3 — Overall audio texture / perceived quality is poor

| Hypothesis | Layer | Confidence | How to settle it |
|---|---|---|---|
| Codec artifacts from the 5 Hz latent representation | `VAE_OR_AUDIO_DECODER_LIMITATION` | Medium | VAE round-trip test |
| 8 inference steps is too few | `INFERENCE_CONFIGURATION_PROBLEM` | **Low** | Turbo's supported range is 1–20 with 8 recommended; try 12–20. Cheap to test, unlikely to be transformative |
| `infer_method=ode` vs `sde` changes texture | `INFERENCE_CONFIGURATION_PROBLEM` | Unknown | Direct A/B — never tried |

---

## H4 — Excessive high-frequency energy · H5 — Excessive sibilance

Grouped: both are "too much energy above ~5 kHz", but they have
different likely origins.

| Hypothesis | Layer | Confidence | How to settle it |
|---|---|---|---|
| Vocal-specific sibilance is a vocal-model artifact | `VOCAL_STYLE_BIAS` | Medium | Compare instrumental vs vocal tracks' high-band energy — if only vocals are harsh, it is the vocal path |
| Broadband brightness is a decoder characteristic | `VAE_OR_AUDIO_DECODER_LIMITATION` | Medium | VAE round-trip |
| Upstream peak normalization to −1.0 dBFS emphasises peaks | `INFERENCE_CONFIGURATION_PROBLEM` | **Low** | Peak normalization changes level, not spectral tilt. Measured: every track peaks at exactly −1.0 dBFS |
| A de-esser would fix it | `POST_PROCESSING_PROBLEM` | **Deferred by policy** | Would mask, not fix. See Step 2 |

**Available measurement.** The pilot's band-energy profile is already
recorded per track in
`results/pilot_baseline_p5_v1_structure.jsonl` (5 bands). Comparing the
6 kHz–20 kHz band between instrumental and vocal tracks is a zero-cost
first discriminator.

---

## H6 — Korean lyric sentences frequently omitted
## H7 — Model skips to the following lyric line

Grouped: both are lyric-scheduling failures.

| Hypothesis | Layer | Confidence | How to settle it |
|---|---|---|---|
| Lyric-to-timeline scheduling cannot fit the supplied text into the requested duration | `LYRIC_ALIGNMENT_GAP` | **High** | Duration sweep with fixed lyrics: if omissions fall as duration rises, it is a fit problem, not a language problem |
| 30 s tracks force compression of a full lyric set | `INFERENCE_CONFIGURATION_PROBLEM` | **High** | 15 of the 26 pilot tracks were 30 s and showed a −9.68 dB median fade — consistent with the model running out of room. Phase 6 moves the default to 60 s |
| Korean tokenisation/alignment is weaker than English | `KOREAN_LANGUAGE_ALIGNMENT_GAP` | Medium | Same lyric content and duration in ko vs en; compare omission rates |
| The 5 Hz LM (disabled here) normally plans lyric placement | `INFERENCE_CONFIGURATION_PROBLEM` | **Medium-High** | LUBER runs DiT-only. Upstream notes the LM fills missing metadata and plans codes. This is the strongest untested configuration hypothesis and needs a GPU |

**This is the most actionable cluster.** Two of the four hypotheses are
configuration, one is testable at zero training cost, and the duration
evidence from Phase 5 already points the same way.

---

## H8 — Unwanted trot-like (뽕끼) vocal character
## H9 — Vocal style not contemporary

Grouped: both describe the vocal prior being wrong for the target.

| Hypothesis | Layer | Confidence | How to settle it |
|---|---|---|---|
| Base model's Korean vocal training data skews older/traditional | `TRAINING_DATA_DOMAIN_GAP` + `VOCAL_STYLE_BIAS` | **High** | Only a LoRA on modern Korean vocal data can settle this |
| Prompt conditioning is too weak to steer vocal style | `PROMPT_COMPILER_PROBLEM` | Medium | The compiler emitted "female lead vocal, natural female singing voice" — generic, and it says nothing about era or genre-vocal identity. Test explicit contemporary descriptors |
| Compiler duplication diluted the useful prompt tokens | `PROMPT_COMPILER_PROBLEM` | Medium | Fixed in Phase 6; A/B measures it |

**Why this matters most commercially.** A wrong vocal character cannot
be post-processed away and is the difference between "AI demo" and
"releasable K-pop". This is the primary justification for the LoRA
pilot targeting modern Korean pop/R&B vocals.

---

## Summary — where the effort should go

| Cluster | Dominant layer | Reachable without training? |
|---|---|---|
| H6/H7 lyric omission & skipping | `LYRIC_ALIGNMENT_GAP` + configuration | **Partly** — duration, LM |
| H8/H9 vocal style | `TRAINING_DATA_DOMAIN_GAP` / `VOCAL_STYLE_BIAS` | **No** — needs LoRA |
| H2 instrument fidelity | `MODEL_BASE_LIMITATION` | **Probably not** — XL is the lever |
| H1/H3/H4/H5 spectral & texture | `VAE_OR_AUDIO_DECODER_LIMITATION` | **Unclear** — VAE round-trip decides |
| Compiler duplication | `PROMPT_COMPILER_PROBLEM` | **Yes** — fixed in Phase 6 |

Two conclusions follow, and they are uncomfortable in different ways:

1. **The cheap fixes are unlikely to move a 2/10 to a 5/10.** Compiler
   de-duplication and parameter changes are worth doing — they are
   nearly free and they remove confounds — but the dominant findings
   (vocal character, instrument fidelity) map to model capability and
   training data, not to configuration.

2. **The lyric-omission cluster is the exception.** It has real
   configuration hypotheses with evidence already pointing at them, and
   it is the one place where a no-training change could plausibly
   produce a visible improvement.

---

## Step 2 policy — no mastering to hide model failures

The following must **not** be "fixed" by processing the master:

- excessive sibilance
- harsh high frequencies
- poor instrument realism
- bad vocal character
- lyric omissions

A de-esser or corrective EQ may later be legitimate production polish,
but adding one now would raise benchmark scores while the model got no
better — and would destroy the ability to measure whether training
actually worked. Phase 4's delivery stage stays format-conversion-only.

Generation failure and post-processing opportunity are tracked
separately, and Phase 6 changes nothing in the delivery stage.
