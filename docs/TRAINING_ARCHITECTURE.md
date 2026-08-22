# Training architecture

How LUBER would actually train a model, given the engine it actually
has. Every strategy here was checked against the installed ACE-Step
(`PHASE20_MODEL_TRAINING_CAPABILITY_AUDIT.md`); nothing is included
because it is common practice elsewhere.

**No training has been run.** Numbers below that come from the engine's
own configuration are marked as such; numbers that are estimates are
marked as estimates. There are no measured throughput figures for this
project because there is nothing to measure yet.

---

## 1. Available strategies

| Strategy | Status | Why |
|---|---|---|
| **LoRA on DiT attention** | **Available** | `inject_lora_into_dit`, `FixedLoRATrainer`, `peft 0.18.1` installed, seven VRAM presets shipped |
| Adapter on other components | Not available | The injector targets the DiT decoder only |
| Partial-component fine-tune | Not available | No entry point |
| Full-parameter fine-tune | Not available in the installed tooling | No train-mode switch in `training_v2`; the trainer describes itself as adapter fine-tuning. A root `train.py` and an SFT guide exist but were not traced to a working path |

So there is one strategy, and the plan is built around it rather than
around what would be ideal.

## 2. LoRA on DiT attention

**Trainable surface.** Every `nn.Linear` in the DiT decoder named
`q_proj`, `k_proj`, `v_proj` or `o_proj`. Not the VAE, not the text
encoder, not the vocoder.

**Trainable parameters.** For rank *r* on a *d*×*d* projection, each
adapter adds 2·*r*·*d*. The engine's presets span rank 16 to 128; the
exact count depends on the checkpoint's hidden size and layer count,
which this audit did not enumerate, so the figure is deliberately not
quoted. It is available by loading the model and summing
`requires_grad` parameters — a one-line measurement to take before the
first run rather than a guess to record now.

**Hyperparameters** — from the engine's own presets, not invented:

| Preset | Rank | Alpha | Batch | Grad accum | LR | Optimiser |
|---|---|---|---|---|---|---|
| `vram_8gb` | 16 | 32 | 1 | 8 | 1e-4 | adamw8bit |
| `vram_24gb_plus` | 128 | 256 | 2 | 2 | 1e-4 | — |

**Rollback.** An adapter is a separate artefact. The base checkpoint is
never written to, so rollback is "do not load the adapter". This is the
main reason LoRA is a comfortable first step: the worst outcome is a file
nobody loads.

## 3. Which component the problems live in

The distinction between evidence and hypothesis matters here, because
choosing wrong wastes the first experiment.

**Evidence** (measured on 45 RAW masters):

- Stereo width median 0.179, correlation 0.762 — the corpus really is
  narrow. This corroborates the human report objectively.
- Air-band energy spans −39.6 to −12.9 dB and spectral centroid spans
  442 to 3972 Hz — the corpus is *inconsistent* rather than uniformly
  dull or uniformly bright. Both human reports were right, about
  different tracks.
- Silence ratio reaches 0.59 on at least one track — a real early-fade
  or collapse case, not an impression.
- Crest factor median 16.2 dB — dynamics are a current strength.

**Hypothesis** (not measured, and labelled so):

- Trot-like delivery is a learned style prior in the DiT. Adapters move
  style, so this is the most likely thing LoRA can fix.
- Korean line omission is a lyric-conditioning failure rather than an
  audio-quality one. If the text encoder never represents the missing
  line, DiT-attention LoRA cannot recover it.
- Synthetic timbre may be bounded by the VAE/decoder. If so, **no
  strategy available here fixes it**, and continuing to try would be the
  expensive mistake this document exists to prevent.

## 4. First experiment

One hypothesis, one subset, one method, one checkpoint, one criterion.

> **Hypothesis.** The trot-like vocal delivery is a style prior in the
> DiT, and it can be reduced by LoRA on DiT attention trained on
> contemporary Korean vocals.

Chosen over the Korean-omission and timbre problems for one reason: it is
the hypothesis this method can actually test. A negative result is as
useful as a positive one — if explicit anti-trot prompting fails *and* a
style adapter fails, the problem is deeper than adapters reach, and that
is worth knowing before spending on a GPU host.

| | |
|---|---|
| **Dataset** | Smallest useful GOLD subset of contemporary Korean vocal tracks, `OWNED` or `LICENSED_FOR_TRAINING`. Minimal synthetic share — training on current output would reinforce the very delivery being removed |
| **Method** | LoRA, rank 16–32, engine preset hyperparameters unchanged |
| **Baseline** | `luber-baseline-p20-v1`, frozen |
| **Evaluation** | The four `TROT-*` cases plus the `GEN` Korean vocal cases, scored blind on `trot_absence` |
| **Acceptance** | `trot_absence` improves on the TROT cases **and** `lyric_completeness`, `pitch_stability` and technical safety do not materially regress |

**Prerequisite, and it is a hard one.** The experiment cannot start until
a GOLD Korean vocal subset with established rights exists. Today the
project has none: every audio file it owns is its own generated output,
which is the one thing that must not dominate this particular run. The
first blocking task is data acquisition with real provenance, not
training.

**Not run.** No training has been started and none is authorised.

## 5. Compute

Derived from the engine's VRAM presets, which are the most reliable
statement available about what this model needs. LUBER has measured
none of it.

| Tier | VRAM | Notes |
|---|---|---|
| **Inference only** | ~8 GB, or Apple Silicon unified memory | Observed: ACE-Step peaks ~1.2 GB RSS serving 180 s on this Mac |
| **Small LoRA** (rank 16, batch 1, grad accum 8) | ~8 GB | The engine ships a preset for exactly this |
| **Serious LoRA** (rank 128, batch 2) | 24 GB+ | Preset names RTX 3090 / 4090 / A100 / H100 |
| **Full fine-tune** | Not applicable | No supported path; estimating VRAM for something that cannot be run would be fiction |

System RAM: 32 GB+ for preprocessing (VAE latents for a corpus are large
and are written before training). Storage: the corpus at 48 kHz stereo
WAV plus latents — budget several times the raw audio size.

Expected bottleneck for a small run is data, not compute. There is no
corpus yet.

**Phase 33 update.** A bounded canary has now loaded the real ACE-Step
DiT (2 396 625 030 parameters) on Apple silicon in bf16, injected a
rank-4 LoRA (2 752 512 trainable), taken an optimizer step, written a
checkpoint and resumed from it — on **two synthetic tensors for one
epoch**. That establishes the mechanism and nothing about the table
above: no memory requirement for a real workload has been measured on
any device, and the preflight reports it as UNKNOWN rather than reading
a figure off the presets. See `docs/TRAINING_PREFLIGHT_AND_CANARY.md`.

No cloud vendor is named and no prices are quoted; neither was asked for
and both would date immediately.

## 6. What this Mac is for

Good for, and demonstrated: development, benchmark inference (a 180 s
track in ~77 s), objective analysis, dataset validation and manifest
generation, preprocessing, listening.

**Not demonstrated:** training of any kind. The trainer is PyTorch with
CUDA-shaped VRAM presets; whether it trains usefully under MPS is
untested here. Any claim that this machine can fine-tune would be
invented, and the honest position is that a tiny adapter run is worth
*attempting* to find out, not worth assuming.

## 7. Run reproducibility

Every training run must record, before it starts:

base model identity and checkpoint hash · LUBER git commit · dataset
manifest path **and hash** · split · every hyperparameter · seed ·
hardware · library versions (torch, peft, ACE-Step commit) · output
checkpoint identity · start and end time · loss curve · benchmark
version used for evaluation.

A checkpoint that cannot name its dataset manifest hash cannot be
reasoned about later and should be deleted rather than kept.

## 8. Model registry

A JSON file, not a platform.

| State | Meaning |
|---|---|
| `BASELINE` | The current production reference. Exactly one |
| `EXPERIMENT` | Trained, not yet evaluated |
| `CANDIDATE` | Evaluated objectively, awaiting blind human listening |
| `ACCEPTED` | Beat the baseline under the promotion policy; may become the new baseline |
| `REJECTED` | Did not. Kept, with the reason recorded |

Each entry: parent/base model, training run id, dataset manifest hash,
benchmark results, human listening status. Rejected checkpoints are kept
— a record of what did not work is why the second experiment is cheaper
than the first.

Promotion rules live in `MODEL_EVALUATION_POLICY.md`.
