# Phase 6 — No-Training A/B Results

24 real generations (6 prompts × 4 variants), 60 s each, **fixed seed
per prompt** so exactly one variable changes per cell. Driven directly
against the ACE-Step engine at the pinned commit; no mock inference.

Composition: 2 Korean K-pop vocal, 1 Korean ballad, 1 Korean R&B,
1 English vocal, 1 instrumental.

Duration is 60 s, not 30 s: Phase 5 established the 30 s tier as
structurally weakest, so benchmarking model quality there would measure
the known-bad mode.

**Result set:** 24/24 completed, **zero technical flags**.

| Variant | What changed | Median gen | Median centroid | Median end-drift | Median repetition |
|---|---|---|---|---|---|
| **V1** legacy compiler | Phase 5 control | 49.2 s | 3840 Hz | −6.17 dB | 0.9534 |
| **V2** dedup compiler | prompt text only | 35.1 s | 3749 Hz | −6.28 dB | 0.9642 |
| **V3** dedup + metadata | `bpm`/`key_scale`/`time_signature` | 39.2 s | **3533 Hz** | **−4.88 dB** | 0.9454 |
| **V4** dedup + `sde` | sampler | 40.0 s | 3550 Hz | **−9.57 dB** | 0.9629 |

Paired per-prompt deltas against V1 (the honest read for a fixed-seed
design — medians across variants can hide per-prompt effects):

| Variant | Centroid Hz (lower better) | End-level drift dB (toward 0 better) | Repetition (lower better) |
|---|---|---|---|
| V2 dedup | −8.6 median, **3/6** improved | +0.25 median, **3/6** improved | −0.001, 4/6 |
| **V3 metadata** | **−141 median, 4/6** improved | **+1.31 median, 5/6** improved | +0.01, 2/6 |
| V4 sde | −260 median, 4/6 improved | **−6.46 median, 1/6** improved | +0.00, 2/6 |

---

## Finding 1 — metadata conditioning is the best no-training change

`V3_dedup_metadata` is the only variant that improves both measured
proxies for human complaints:

- **Spectral centroid down 141 Hz** (4/6 prompts), which points the
  right way on the "excessive high-frequency energy" finding.
- **End-level drift improved on 5 of 6 prompts** (+1.31 dB median),
  the best result on the fade-out problem that Phase 5 identified.

This costs nothing: `bpm`, `key_scale`, and `time_signature` are
first-class upstream request fields that LUBER has simply never sent
(Phase 5 finding Q4). Sending them is a request change, not a model
change.

**Selected as BEST NO-TRAINING CONFIG.**

## Finding 2 — the sampler change is a bad trade

`infer_method=sde` produces the largest centroid reduction (−260 Hz)
but wrecks the ending: end-level drift worsens by a median 6.46 dB and
improves on only **1 of 6** prompts. JAZZ-01 degraded by −19.9 dB.

Reporting only the centroid would have made `sde` look like the
winner. It is not — it buys darkness by fading out, which is the exact
failure mode Phase 5 flagged. **Not recommended.**

## Finding 3 — the compiler fix is correct but not a quality win

`V2_dedup_compiler` is structurally near-neutral: −8.6 Hz centroid and
+0.25 dB drift, each improving only 3 of 6 prompts — indistinguishable
from noise at n=6.

This is worth stating plainly because it would be easy to present the
compiler fix as a quality improvement. It is not, on this evidence.
It is a **correctness** fix: it removes literal duplication
(`"…no vocals, instrumental, no vocals"`), stops conditioning from
fighting user descriptors, and frees prompt budget. Those are good
reasons to keep it. "It made the music better" is not a claim this
experiment supports.

The one incidental observation: V2 had the fastest median generation
(35.1 s vs 49.2 s), consistent with shorter prompts, though n=6 makes
that suggestive rather than established.

## Finding 4 — none of this closes a 2/10 gap

Effect sizes are small. A 141 Hz centroid shift and a 1.3 dB drift
improvement are real and measurable, and they are nowhere near enough
to move a human verdict of "2/10, all 26 rejected, not commercially
usable".

That matches the root-cause map's prediction: the dominant findings
(trot-like vocal character, instrument fidelity, Korean lyric
omission) map to model capability and training data, not to request
parameters. The configuration work removes confounds and buys small
gains; it does not substitute for training.

---

## Caveats

- **These are objective proxies, not listening results.** Centroid and
  level drift correlate with "too bright" and "fades out", but no human
  has heard the V3 tracks. The A/B audio is available to the listening
  tool for a blind pass.
- **n = 6 per variant.** Enough to rank candidates, not enough for
  confident per-genre claims.
- Only upstream-supported parameters were tested. `guidance_scale` is
  auto-corrected to 1.0 for turbo and `use_adg` is base-model only, so
  both are inert here — see the Phase 5 audit rather than re-testing.
- `shift` was not swept: the REST default is already 3.0, which is what
  upstream recommends for turbo.

## Recommended production change

Send `bpm`, `key_scale`, and `time_signature` when known, and keep the
de-duplicated compiler. Both are cheap, neither risks a regression, and
V3 is the only variant that improved both measured dimensions.

This has **not** been applied to the production provider in Phase 6 —
the experiment establishes the candidate; wiring per-request metadata
through the API contract is a product change that deserves its own
scope rather than being smuggled into a measurement phase.
