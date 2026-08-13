# Phase 9 — Long-Form Duration Engine Audit

Audited directly against the pinned upstream source at
**`6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`** (`~/ace-step-1.5`), not
against assumptions carried over from Phase 2 or Phase 8. Every claim
below cites the file and line it came from. Where the source does not
answer a question, that is stated rather than guessed.

The question this audit exists to answer: **can this deployment produce
coherent 180–240 second songs, and what breaks if we ask it to?**

---

## 1. True supported duration range

| Bound | Value | Source |
|---|---|---|
| `DURATION_MIN` | 10 s | `acestep/constants.py:58` |
| `DURATION_MAX` | 600 s | `acestep/constants.py:59` |

The upstream comment is explicit about intent:
`10s: Short loops, musical excerpts` / `600s: Full songs, extended
compositions (10 minutes)`.

**Where these bounds are actually enforced matters more than their
values.** `DURATION_MIN`/`DURATION_MAX` are referenced in exactly two
places:

- `acestep/constrained_logits_processor.py:130,205` — constrains the
  **LM's** generated `duration` metadata token to the range.
- `acestep/llm_inference.py:270-290` — clamps the LM's effective
  duration and derives a token budget.

Both are **LM paths**. LUBER runs DiT-only (`thinking=false`,
`use_cot_*=false`), so **neither clamp applies to our requests**. The
REST `release_task` path accepts `audio_duration` as a plain
`Optional[float]` and passes it through; `generate_music_request.py:86`
only nulls it when `<= 0`.

**Consequence:** upstream will not stop LUBER from requesting an absurd
duration on the DiT path. LUBER's own `DURATION_MAX = 360` in
`luber_schemas.songcraft` is the only real guard, and the Phase 9
product ceiling of 240 s is a LUBER policy decision, not an engine
limit. That is the correct place for it.

---

## 2. Duration parameter flow

```
LUBER GenerationRequest.duration_seconds
  → payload["audio_duration"] (float)            provider.py:_build_payload
  → release_task_models.audio_duration           (aliases: duration, target_duration)
  → generate_music_request.py:61,86-97           nulled if <= 0
  → batch_prep.py:101-102                        calculated_duration = float(audio_duration)
  → conditioning_target.py                       target latent allocation
  → dit_generate_kwargs["audio_duration"]        inference.py:837
```

For `task_type="text2music"` (LUBER's only path) the requested duration
is authoritative. Note `inference.py:815-819`: for `cover`, `repaint`,
`lego` and `extract`, `audio_duration` is **silently forced to `None`**
and taken from the source audio instead. LUBER wires none of those task
types, so this does not affect us — but it is a trap for any future
phase that does.

### Latent length

`conditioning_target.py:64,90`: `expected_latent_length =
current_wav.shape[-1] // 1920`.

At 48 kHz that is **25 latent frames per second of audio**, with a floor
of 128 frames (`conditioning_target.py:121`).

| Requested audio | Latent frames |
|---|---|
| 30 s | 750 |
| 60 s | 1 500 |
| 120 s | 3 000 |
| 180 s | 4 500 |
| 240 s | 6 000 |
| 600 s | 15 000 |

A 240 s request is an **8× longer sequence** than the 30 s requests
Phase 8 validated. This is the single most important number in this
audit: everything downstream scales off it.

---

## 3. Chunking behaviour — the critical distinction

**Generation is not chunked. Only VAE decode is.**

- **DiT denoising**: no windowing, tiling, or segmentation. The DiT
  denoises the entire latent sequence in one pass, for every inference
  step. Searching `acestep/core/generation` for chunk/window/segment
  logic returns only VAE encode/decode helpers and unrelated matches.
- **VAE decode**: `vae_decode_chunks.py::_tiled_decode_inner` implements
  genuine tiled decode with `chunk_size`/`overlap`, an adaptive-overlap
  reduction loop, a `min_overlap = 4` floor to avoid boundary artifacts,
  batch-sequential decode when `bsz > 1`, and an OOM fallback to CPU
  decode (`vae_decode_chunks.py:52-54`).
- **VAE encode**: `vae_encode.py:52` picks `chunk_size = 48000 * 15` if
  `gpu_memory <= 8` else `48000 * 30`, `overlap = 48000 * 2`. Encode only
  runs for audio-conditioned tasks, which LUBER does not use.

**Implication:** tiled decode bounds *decode* memory at long durations,
but nothing bounds *DiT* memory. If long-form fails on memory, it will
fail during denoising, not during decode.

---

## 4. Memory implications

### There is no memory guard on this machine

`memory_utils.py::_vram_guard_reduce_batch` is the only automatic
protection, and it opens with:

```python
if batch_size <= 1:
    return batch_size
...
if device == "cpu" or device == "mps":
    return batch_size
```

Two independent reasons it never fires for us: LUBER always sends
`batch_size = 1`, and this host is MPS. **Long-form generation on this
Mac runs with no upstream memory backstop.** Resource safety is entirely
LUBER's and the operator's responsibility.

### The engine's own scaling estimate

For CUDA, `memory_utils.py:168-169`:

```python
duration_sec = float(audio_duration) if ... else 60.0
per_sample_gb = 0.5 + max(0.0, 0.15 * (duration_sec - 60.0) / 60.0)
```

That is **linear in duration**, not quadratic: 0.5 GB at ≤60 s rising to
0.95 GB at 240 s, ×1.7 for XL, ×2.0 for CFG base models. Upstream
therefore does not expect attention memory to blow up quadratically with
sequence length — consistent with a memory-efficient attention kernel.

This is an upstream heuristic for a different backend and is treated
here as **indicative only**. It is a reason to attempt 120 s, not
evidence that 240 s is safe. Empirical measurement on MPS is the only
thing that settles it, which is why Phase 9 escalates 120 → 180 → 240
with monitoring instead of jumping to 240.

### MPS decode offload

`memory_utils.py:120-127`: on MPS, decoded audio is offloaded to CPU
unless effective MPS memory ≥ 32 GB, overridable with
`ACESTEP_MPS_DECODE_OFFLOAD`. This host has 24 GB, so **offload is on** —
helpful for long-form, since the decoded waveform for 240 s stereo
float32 is ~92 MB before conversion and does not sit in accelerator
memory.

---

## 5. Duration-dependent conditioning

`audio_duration` is not only a length: it is **conditioning metadata**.

`metadata_utils.py::_dict_to_meta_string` renders the metadata block the
model is conditioned on as:

```
- bpm: {bpm}
- timesignature: {timesignature}
- keyscale: {keyscale}
- duration: {duration} seconds
```

`_build_metadata_dict` writes `duration` as `f"{int(duration)} seconds"`.
So the requested duration reaches the model twice — once as the latent
allocation and once as a text token in the conditioning metadata. A
request for 240 s tells the model both "make this many frames" and
"this is a 240-second piece", which is the mechanism by which duration
could plausibly influence arrangement rather than just length.

Whether it *does* influence arrangement is an empirical question this
audit cannot answer from source.

---

## 6. Timeout implications

Three independent timeouts sit on the path. Only one is currently
dangerous.

| Timeout | Value | Where | Risk at 240 s |
|---|---|---|---|
| LUBER provider `generation_timeout` | **600 s default** | `ace_step/provider.py:43` | **REAL** |
| Upstream `TASK_TIMEOUT_SECONDS` | 3600 s, hardcoded | `api_server.py:111` | none |
| LUBER `request_timeout` (per HTTP call) | 60 s | `ace_step/provider.py:41` | none — polls are fast |

### Correction: measured long-form timing (added after the gates ran)

The extrapolation below was written **before** the long-form gates and
turned out to be pessimistic. The measured results supersede it and are
stated first so nobody acts on the superseded estimate.

| Audio | Wall clock | RTF | Notes |
|---|---|---|---|
| 30 s ×5 | 46–69 s | 1.54–2.30× | Phase 8 E2E |
| **120 s** | **96.3 s** | **0.80×** | Phase 9 gate |
| **180 s** | **89.5 s** | **0.50×** | Phase 9 gate |
| **240 s** | **76.4 s** | **0.32×** | Phase 9 gate |

**Wall clock is effectively flat with duration between 120 s and 240 s**,
and the 240 s run was *faster* than the 180 s run. Two things follow:

1. Cost is dominated by a fixed per-request overhead (model warm-up,
   text encode, scheduler setup) plus 8 fixed inference steps. Sequence
   length is not the bottleneck at these sizes on MPS — consistent with
   §4's linear-not-quadratic memory heuristic.
2. Run-to-run variance is dominated by **machine contention**, not by
   requested duration. The 30 s samples spanning 46–69 s on an idle-ish
   machine are the same effect.

So the 600 s default is **not** in immediate danger at 240 s: measured
worst case is 96 s, a ~6× margin. Duration-aware timeout is still
implemented in Phase 9, but the honest justification is defence in depth
against cold starts and contention, **not** an imminent breach. The
original reasoning is kept below because the correction is the
interesting part.

### The 600 s extrapolation (superseded — kept for the record)

Five real ACE-Step generations on this host during the Phase 8 E2E:

| Audio | Wall clock | RTF |
|---|---|---|
| 30 s | 46.3 s | 1.54× |
| 30 s | 59.0 s | 1.97× |
| 30 s | 65.5 s | 2.18× |
| 30 s | 68.4 s | 2.28× |
| 30 s | 69.0 s | 2.30× |

Mean RTF ≈ 2.05×, range 1.54–2.30×. Naively extrapolated at the observed
worst case, 240 s of audio would take ≈ 550 s — **inside the 600 s
default timeout by under a minute**, on a contended daily-driver
machine. A single slow run would be killed by LUBER as a "provider
timeout" while the engine was still working perfectly.

That is the concrete, evidenced justification for duration-aware
timeouts required by Phase 9 §8. It is not "multiply the timeout by
something"; it is "the default is empirically too small for the
durations we are about to expose, and the failure mode is
misdiagnosis — a healthy engine reported as a dead one."

Note the RTF is not a clean constant: a fixed startup cost (text encode,
scheduler setup) is amortised over longer runs, so long-form RTF may be
*better* than 2.05×. The 120 s gate measures this directly.

### Upstream's own timeout is not a concern

`TASK_TIMEOUT_SECONDS = 3600` is hardcoded (not env-configurable) and is
applied in `query_result_service.py:53` — a task still reporting
`status == 0` an hour after `create_time` is reported as failed. At any
plausible RTF, a 240 s request finishes far inside an hour.

---

## 7. Queue implications

`api_server.py:194-195`:

```python
QUEUE_MAXSIZE = int(os.getenv("ACESTEP_QUEUE_MAXSIZE", "200"))
WORKER_COUNT = int(os.getenv("ACESTEP_QUEUE_WORKERS", "1"))  # Single GPU recommended
```

- **The engine processes one job at a time by default.** A 240 s request
  occupies the engine for its entire duration; every other request
  queues behind it. HTTP 429 is returned only when 200 jobs are pending.
- LUBER adds its own serialisation: one ARQ generation worker.
- `ACESTEP_AVG_JOB_SECONDS` (default 5.0) and `ACESTEP_AVG_WINDOW` (50)
  feed `/v1/stats`'s `avg_job_seconds`. That average is computed over a
  rolling window of *all* jobs, so mixing 30 s and 240 s requests makes
  it a poor ETA source. **Not a basis for a progress estimate.**

For local long-form validation this means concurrency is naturally 1,
which matches the Phase 9 resource-safety requirement.

---

## 8. Transient audio lifecycle

- Output lands under `ACESTEP_TMPDIR` (or `<cache_root>/tmp`), in
  `api_audio` (`lifespan_runtime.py:61-78,135`).
- `JOB_STORE_CLEANUP_INTERVAL = 300` (`api_server.py:112`) — a cleanup
  worker prunes completed jobs every 5 minutes
  (`worker_runtime.py:33-38`).
- `GET /v1/audio?path=...` serves server-local files that are
  **transient by design**.

LUBER already downloads promptly inside `AceStepProvider.generate` and
persists to its own storage before returning, so the cleanup window is
not a risk. It becomes one only if a future phase defers download.

**Long-form disk note:** a 240 s master at 48 kHz/24-bit stereo is
≈ 69 MB, plus the MP3 preview and the engine's own transient copy —
roughly 150 MB of churn per long generation across both processes.

---

## 9. Findings that change Phase 9's plan

1. **Duration-aware timeout is defence in depth, not an emergency.**
   Measured 240 s wall clock is 76 s against a 600 s default — a ~8×
   margin. Implemented anyway for cold starts and contention. (§6)
2. **No memory backstop exists on MPS.** Escalate 120 → 180 → 240 with
   live monitoring and hard abort thresholds; never jump straight to
   240. (§4)
3. **Failure, if it comes, will be in DiT denoising, not VAE decode.**
   Decode is tiled with a CPU fallback; denoising is one unchunked pass
   over an 8×-longer sequence. (§3)
4. **Engine-side concurrency is 1.** Long-form validation cannot be
   parallelised, which happens to match the safety requirement. (§7)
5. **Duration is conditioning, not just length.** Asking for 240 s tells
   the model it is writing a 240-second piece. (§5)
6. **LUBER's 360 s cap is the only real duration guard** on the DiT
   path; the 240 s product ceiling is correctly a LUBER policy. (§1)

## 10. Questions this audit cannot answer from source

- Actual MPS peak memory at 120/180/240 s. **Measured empirically.**
- Whether long-form output stays *musically* coherent — structure,
  vocal identity, melodic development. **Human listening only.**
- Whether the model honours the duration metadata as an arrangement cue.
- Whether lyric-line omission worsens with more sections. **Human
  listening only.**
