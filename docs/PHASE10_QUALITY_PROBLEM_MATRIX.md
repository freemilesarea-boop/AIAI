# Phase 10 — Quality Problem Matrix and Model Ceiling

Every defect the product owner reported, classified against the evidence
Phase 10 could actually gather. Where evidence is absent the row says
so; no row claims a cause it cannot support.

**Two Phase 10 findings reshape this whole table**, so they are stated
before it:

1. **Inference compute is capped.** `inference_steps` is live below 8 and
   **saturates at 8**: values of 16 and 32 produce *byte-identical*
   audio to 8 (verified by SHA256 on two prompts). Steps 1 and 4 differ
   and are measurably worse. So "throw more compute at it" is not
   available on `acestep-v15-turbo` — the engine refuses. Combined with
   `guidance_scale`, `shift`, `use_adg` and CFG intervals all being inert
   on a turbo/DiT-only deployment, **there is essentially no inference
   configuration lever left.**
2. **LUBER is not too bright.** Against 100 commercial masters its
   broadband high-frequency energy is *inside* the p10–p90 band on every
   descriptor. The harshness is a **1.5× burstiness** in 5–8 kHz — a
   dynamics defect, not a tonal one — alongside a **2–4 kHz scoop**.

---

## Matrix

### Poor frequency balance

- **Human:** poor frequency balance, excessive high-frequency energy.
- **Evidence:** LUBER in band on centroid (6/7), >8 kHz (7/7), >10 kHz
  (7/7), band_high (7/7) vs commercial reference. **But** narrow-band:
  2–3 kHz share 0.054 vs commercial 0.092 (a ~40 % deficit), 10–12 kHz
  0.055 vs 0.041, 16–20 kHz 0.0062 vs 0.0098.
- **Likely causes:** `MIX_BALANCE`, `BASE_MODEL`. Not
  `POST_PROCESSING` — LUBER only transcodes.
- **Testable now:** a broad 2–4 kHz presence lift (post-processing).
- **Needs training:** whether the model can *generate* proper presence.
- **Classification:** `IMPROVED_WITH_POST_PROCESSING` (partially) /
  `TRAINING_REQUIRED` (root cause).

### Excessive sibilance

- **Human:** excessive sibilance.
- **Evidence, and this is the strongest causal finding in Phase 10:**
  mean 5–8 kHz share is *below* commercial (0.062 vs 0.084), but the
  99th percentile is *above* (0.280 vs 0.242) and burstiness is
  **1.52×** commercial. Peaky, uncontrolled high-mid transients.
- **Likely causes:** `POST_PROCESSING` (absence of de-essing —
  commercial masters have it, LUBER has none), plus `BASE_MODEL` for
  whatever produces the peaks.
- **Testable now:** conservative dynamic de-essing on 5–8 kHz. The
  evidence names the band and the mechanism.
- **Classification:** `IMPROVED_WITH_POST_PROCESSING` — **highest
  confidence remediation in this phase**, and not yet built.

### Poor instrument fidelity / texture

- **Human:** synthetic, smeared, poor texture.
- **Evidence:** none gathered. The instrument diagnostic was not run.
  The 2–4 kHz scoop is *consistent* with "thin/glassy" but does not
  establish it.
- **Likely causes:** `BASE_MODEL`, `VAE_DECODER`, `TRAINING_DATA_BIAS`.
- **Testable now:** the per-instrument diagnostic (not run).
- **Classification:** `UNKNOWN`, leaning
  `BASE_MODEL_REPLACEMENT_MAY_BE_REQUIRED`. Now that inference compute is
  proven capped, a *configuration* fix for fidelity is close to ruled
  out — the remaining levers are prompt conditioning, post-processing,
  or a different model.

### Korean lyric-line omission / skipping

- **Human:** whole lines not sung. The most damaging defect.
- **Evidence:** none — **no automated detector exists and none was
  faked.** Phase 9 built the per-line QA record; it holds only
  `UNKNOWN` placeholders because the listening pass has not happened.
  The Phase 9 lyric-formatting A/B produced audio but the deciding
  metric is human.
- **Likely causes:** `LYRIC_CONDITIONING`, `BASE_MODEL`,
  `TRAINING_DATA_BIAS`.
- **Testable now:** the A/B tracks exist and are in the listening
  package; only listening is missing.
- **Classification:** `UNKNOWN` — **blocked on human listening, not on
  engineering.**

### Trot-like / outdated vocal delivery

- **Human:** trot-like, not contemporary K-pop/R&B.
- **Evidence:** prompt conditioning demonstrably changes the audio (all
  3 Phase 9 pairs differ under identical seeds; 2 of 3 moved darker and
  less sibilant). Whether it moved *style* is unmeasured.
- **Likely causes:** `TRAINING_DATA_BIAS` (primary — a model sings the
  style it was trained on), `VOCAL_CONDITIONING` (partial lever).
- **Important negative result:** there is **no usable negative prompt**.
  `lm_negative_prompt` is the only one upstream and it is LM-only; the
  LM is disabled here. Trot bias cannot be suppressed by negation, only
  displaced by positive conditioning.
- **Classification:** `TRAINING_REQUIRED` for the bias itself;
  `IMPROVED_WITH_CONFIG` possible via conditioning, unproven.

### Poor overall audio quality / commercial usability

- **Human:** 2/10, FAIL.
- **Evidence:** the measurable gaps are loudness (~5 dB quieter) and
  crest factor (~4–5 dB less compressed) — both **mastering-stage**
  differences, expected by design, and not evidence about the music.
- **Classification:** `TRAINING_REQUIRED` for the musical component;
  the technical delivery gap is `IMPROVED_WITH_POST_PROCESSING` but
  would be cosmetic.

---

## Model ceiling classification

| Defect | Classification | Confidence |
|---|---|---|
| Excessive sibilance | `IMPROVED_WITH_POST_PROCESSING` | High — band and mechanism identified |
| Frequency balance (2–4 kHz scoop) | `IMPROVED_WITH_POST_PROCESSING` (partial) | Medium |
| Broadband "too bright" | **Not a defect** — retracted, inside commercial band | High |
| Loudness / dynamics vs commercial | `IMPROVED_WITH_POST_PROCESSING` (mastering) | High, but cosmetic |
| Trot-like vocal | `TRAINING_REQUIRED` | Medium-high |
| Outdated vocal style | `TRAINING_REQUIRED` | Medium-high |
| Korean line omission | `UNKNOWN` — blocked on listening | — |
| Instrument fidelity | `BASE_MODEL_REPLACEMENT_MAY_BE_REQUIRED` | Medium |
| Instrument texture | `BASE_MODEL_REPLACEMENT_MAY_BE_REQUIRED` | Medium |
| Overall musical quality | `TRAINING_REQUIRED` | High |

### The honest summary

**`SOLVED_WITH_CONFIG`: nothing.** Not one defect was solved by
configuration, and Phase 10 established *why* rather than merely failing
to find one: on a turbo, DiT-only deployment the inference surface is
inert (`guidance_scale`, `shift`, `use_adg`, CFG intervals) or saturated
(`inference_steps` caps at 8). There is no knob left to turn.

That is a genuinely useful negative result. It means further
no-training effort should go to **post-processing** (de-essing, presence,
loudness) and **prompt conditioning**, and that the musical defects —
instrument fidelity, texture, vocal style, and whatever drives line
omission — are training or model-replacement questions.

**The 2/10 baseline should be expected to stand.** Nothing in Phase 10
plausibly moves it, because nothing in Phase 10 changed what the model
generates.
