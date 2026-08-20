# ACE-Step Training Capability Audit

Re-audited against the **installed** tree at
`~/ace-step-1.5`, commit `6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`,
by reading the argument parser, the preprocessing loader and the
checkpoint helpers directly. Not from prior documentation.

The purpose is narrow: establish what the trainer genuinely accepts, so
that the orchestration layer exposes exactly that and nothing else. A
config field LUBER offers and the trainer ignores is worse than a
missing one — it looks like a setting and does nothing.

---

## 1. Entry point

```
python train.py <subcommand> [args]
```

| Subcommand | Purpose |
|---|---|
| `fixed` | corrected training: continuous timesteps + CFG dropout |
| `vanilla` | reproduces the older (upstream-acknowledged buggy) path |
| `estimate` | gradient sensitivity analysis, no training |

`fixed` is the only subcommand LUBER compiles to. `vanilla` exists for
backward compatibility with a path upstream itself describes as bugged,
and `estimate` does not train.

---

## 2. Training strategy support

| Strategy | Status | Evidence |
|---|---|---|
| **LoRA (PEFT)** | **SUPPORTED** | `--adapter-type lora`, with `--rank`, `--alpha`, `--dropout`, `--target-modules`, `--bias`, `--attention-type` |
| **LoKR (LyCORIS)** | SUPPORTED | `--adapter-type lokr` plus seven `--lokr-*` flags |
| **Full fine-tune** | **NOT SUPPORTED** | No subcommand, no flag, no code path. `trainer_fixed.py` collects `parameters() if p.requires_grad`, which under adapter injection is the adapter only. |

Full fine-tuning is therefore **not** offered by the orchestration
layer. It is not a missing feature to be added later by configuration —
the installed trainer has no entry point for it, and exposing a
`training_strategy: FULL` that silently trained an adapter would be a
lie in the run record.

---

## 3. Configuration surface (exact)

Every flag below was read from `acestep/training_v2/cli/args.py`.

### Model and paths
`--checkpoint-dir`, `--model-variant`, `--dataset-dir`, `--output-dir`

### Device and platform
`--device` (`auto`, `cuda`, `cuda:N`, `mps`, `xpu`, `cpu`),
`--precision` (`auto`, `bf16`, `fp16`, `fp32`),
`--num-devices`, `--strategy` (`auto`, `ddp`)

### Training
`--lr/--learning-rate`, `--batch-size`, `--gradient-accumulation`,
`--epochs`, `--warmup-steps`, `--weight-decay`, `--max-grad-norm`,
`--seed`, `--shift`, `--num-inference-steps`,
`--optimizer-type` (`adamw`, `adamw8bit`, `adafactor`, `prodigy`),
`--scheduler-type` (`cosine`, `cosine_restarts`, `linear`, `constant`,
`constant_with_warmup`),
`--gradient-checkpointing` / `--no-gradient-checkpointing`,
`--offload-encoder`

### LoRA
`--rank` (default 64), `--alpha` (128), `--dropout` (0.1),
`--target-modules` (default `q_proj k_proj v_proj o_proj`),
`--bias` (`none`/`all`/`lora_only`),
`--attention-type` (`self`/`cross`/`both`)

### Checkpointing and logging
`--save-every` (epochs), `--resume-from`, `--log-dir`, `--log-every`,
`--log-heavy-every`, `--sample-every-n-epochs`

### Data loading
`--num-workers`, `--pin-memory`, `--prefetch-factor`,
`--persistent-workers`

---

## 4. Fields that do NOT exist

Checked explicitly, because they are conventional enough to be assumed:

| Field | Status |
|---|---|
| `--max-steps` | **ABSENT** — training length is epochs only |
| `--validation-interval` | **ABSENT** — the nearest thing is `--sample-every-n-epochs`, which generates audio samples, not a validation loss |
| `--checkpoint-interval` (steps) | **ABSENT** — `--save-every` counts **epochs** |

Consequences for the orchestration layer:

- `TrainingConfig` has **no** `max_steps` field.
- `TrainingConfig` has **no** `validation_interval`, and no validation
  loss metric is promised.
- `checkpoint_every_epochs` is named for what it measures. Calling it
  `checkpoint_interval` would invite the reader to assume steps.

This is the substance of Step 54: any LUBER field the trainer does not
accept is removed, not silently dropped at compile time.

---

## 5. Dataset contract

Training consumes **preprocessed tensors**, not audio:

```
--dataset-dir <directory containing preprocessed .pt files>
```

Those tensors are produced by a separate preprocessing pass
(`--preprocess --audio-dir ... --dataset-json ... --tensor-output ...`)
which reads a JSON file of the form:

```json
{
  "custom_tag": "optional dataset-level tag",
  "samples": [
    {
      "filename": "track.wav",
      "caption": "...",
      "lyrics": "[Verse]\n…"  or  "[Instrumental]",
      "genre": "",
      "bpm": null,
      "keyscale": "",
      "timesignature": "",
      "duration": 0,
      "custom_tag": ""
    }
  ]
}
```

Indexed by `filename` (basename fallback), or by the basename of
`audio_path`. Missing entries receive defaults, including a caption
derived from the filename and `"[Instrumental]"` lyrics — which is
exactly why LUBER supplies the metadata explicitly rather than letting
the loader guess.

`preprocess_vae.TARGET_SR` fixes the working sample rate.

**LUBER's canonical manifest is not changed to match this.** The
adapter converts Phase 23/24 export records into this shape; the
canonical manifest keeps its own semantics.

---

## 6. Checkpoint format

`trainer_helpers.save_checkpoint` / `save_adapter_flat` write:

- `adapter_config.json`
- `adapter_model.safetensors`
- `training_state.safetensors` (epoch, global step)
- a torch-pickled training state file

So a checkpoint is an **adapter**, not a full model. The checkpoint
registry records `checkpoint_kind: ADAPTER` accordingly, and a "full
model" kind exists in the vocabulary only so a future trainer could use
it — nothing produces one today.

Resume: `--resume-from <checkpoint directory>`.

---

## 7. Presets shipped upstream

`acestep/training_v2/presets/`: `quick_test.json`, `recommended.json`,
`high_quality.json`, `vram_8gb.json`, `vram_12gb.json`, `vram_16gb.json`,
`vram_24gb_plus.json`.

The VRAM-named presets are upstream's, and LUBER does **not** reuse their
names or repeat their implied hardware claims. LUBER presets describe
*intent* (`SMOKE`, `LORA_SMALL`, `LORA_STANDARD`, `LORA_HIGH_QUALITY`)
and state their VRAM requirement as `UNKNOWN_REQUIREMENT` until a real
worker measures it.

---

## 8. Local (Apple Silicon) training feasibility

`--device` accepts `mps`, so the trainer will *attempt* Apple Silicon.
That is not evidence it works.

Measured on this machine: the LUBER virtualenv has **no torch installed
at all**, so nothing about local training has been demonstrated —
neither MPS nor CPU.

**Status: `LOCAL_TRAINING_SMOKE_UNSUPPORTED`.**

Not "unsupported by the trainer" — unverified by us. No local smoke run
was executed, and none is claimed. The local Mac therefore registers as
a `DEVELOPMENT_ONLY` worker and cannot satisfy a plan requiring CUDA.

Should a local smoke ever be attempted, it would need torch installed in
an environment with the trainer's dependencies, and would still be a
*plumbing* check rather than training: a few epochs on a handful of
tensors teaches nothing.

---

## 9. Distributed training

`--num-devices N` with `--strategy ddp`. Single-process by default. Not
exercised, and the worker registry records GPU count from a worker
probe rather than assuming it.

---

## 10. What this audit does not establish

- **No performance numbers.** No throughput, no step time, no VRAM
  figure has been measured on any NVIDIA hardware by this project.
  Every such field is `UNKNOWN_REQUIREMENT` until a probe reports it.
- **No quality claims.** Nothing here says LoRA training will improve
  vocal quality, Korean pronunciation or trot bias. It says the trainer
  accepts LoRA parameters.
- **No checkpoint-size estimate.** Adapter size depends on rank and
  target modules; nothing has been produced to measure.
