# Frozen Baseline — `LUBER_BASELINE_P5_V1`

The control for Phase 6. Any future quality comparison must state which
baseline it is measured against, and only runs on the *same*
`benchmark_version` may be compared directly.

Frozen 2026-08-12.

## Engine

| Item | Value |
|---|---|
| ACE-Step upstream | `https://github.com/ace-step/ace-step-1.5` |
| Pinned commit | `6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0` |
| Package version | `1.5.0` |
| DiT model | `acestep-v15-turbo` (2B) |
| LM | **disabled** (`ACESTEP_INIT_LLM=false`) |
| `thinking` | `false` |
| `use_cot_caption` / `use_cot_language` | `false` |
| `inference_steps` | `8` |
| `guidance_scale` | not sent — auto-corrected to `1.0` for turbo upstream |
| `shift` | not sent — REST default `3.0` applies |
| `bpm` / `key_scale` / `time_signature` | not sent |
| `batch_size` | `1` |
| `audio_format` | `wav` |
| Runtime backend | Apple Silicon MPS + MLX |

## Pipeline

| Item | Value |
|---|---|
| Prompt compiler | `AceStepPromptCompiler` @ commit `26c72d9` |
| Post-processing | Phase 4 delivery normalization @ commit `26c72d9` |
| Production master | WAV / 48 kHz / stereo / 24-bit PCM |
| Preview | MP3 / 48 kHz / stereo / 320 kbps CBR |
| Loudness processing | none (peak already normalized upstream to −1.0 dBFS) |

## Benchmark

| Item | Value |
|---|---|
| Benchmark version | `v1` |
| Prompt dataset | `prompts/BENCHMARK_V1.json` — 60 prompts, 10 genres, 45% Korean vocal |
| Pilot manifest | `manifests/pilot_baseline.json` — 26 units |
| Results | `results/pilot_baseline_p5_v1.jsonl` |
| Structure analysis | `results/pilot_baseline_p5_v1_structure.jsonl` |
| Report | `reports/PHASE5_BASELINE_REPORT.md` |

## Measured baseline (objective only)

| Metric | Value |
|---|---|
| Generations | 26 |
| Success rate | 100% |
| Technical failure rate | 0% |
| Median generation time | 44.8 s |
| Median real-time factor | 0.94× |
| Median end-level drift @ 30 s | −9.68 dB |
| Median end-level drift @ 60 s | −0.19 dB |
| Median end-level drift @ 180 s | −1.95 dB |
| Seed spectral divergence | 0.32–0.40 |

**Human listening scores: none.** This baseline is objective-only. See
the report's "What this baseline does and does not establish".

## Hardware the baseline was measured on

Apple M4 Pro, 14 cores, 24 GB unified memory, macOS 26.3.1, MPS + MLX.
Numbers here are not portable to CUDA hardware; re-baseline before
comparing across machines.
