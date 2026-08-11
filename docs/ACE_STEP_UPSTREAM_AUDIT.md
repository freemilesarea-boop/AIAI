# ACE-Step 1.5 Upstream Audit

Audited: 2026-08-11, directly from a fresh clone of the official
upstream repository (no blogs, no ACE-Step 1.0 material).

## Pinned version

| Item | Value |
|---|---|
| Upstream repository | `https://github.com/ace-step/ace-step-1.5` |
| Audited commit (`ACE_STEP_UPSTREAM_COMMIT`) | `6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0` (main, 2026-06-26) |
| Package version (`ACE_STEP_VERSION`, upstream `pyproject.toml`) | `1.5.0` |
| Nearest release tag at audit time | `v0.1.8` (`dce621408bee8c31b4fcf4811682eb9359e1bc94`) |

Upgrades must be deliberate: re-audit against the new commit before
changing the pin. Do not track moving `main` in production.

## Runtime & install

- Python 3.11–3.12. Backends: CUDA (primary), MPS, ROCm, Intel XPU,
  CPU (all officially listed in upstream README).
- Install: `install_uv.sh` / `requirements.txt`; server started via
  `start_api_server.sh` (Linux/CUDA), `start_api_server_macos.sh`
  (Apple Silicon — forces `ACESTEP_LM_BACKEND=mlx`, port 8001),
  `start_api_server_rocm.sh`, XPU variants. Docker images exist
  (`Dockerfile`, `docker-compose.yml`).
- Server entrypoint: `python -m acestep.api_server`; default bind
  `127.0.0.1:8001` (`ACESTEP_API_HOST`/`ACESTEP_API_PORT`).
- Model weights auto-download at startup from HuggingFace Hub with
  ModelScope fallback (`acestep/model_downloader.py`).

## Models (from `acestep/model_downloader.py` + README)

DiT: `acestep-v15-turbo` (default, 2B), `acestep-v15-sft`,
`acestep-v15-base`, turbo shift/continuous/fix-inst/rl variants,
XL family `acestep-v15-xl-base` / `-xl-sft` / `-xl-turbo` (4B).
LM (optional 5Hz planner): `acestep-5Hz-lm-0.6B` / `1.7B` / `4B`.

VRAM guidance (README): 2B turbo runs in <4GB VRAM (LM disabled);
XL needs ≥12GB with offload, ≥20GB recommended. CPU-only operation is
supported but slow; A100 generates a full song in <2s, RTX 3090 <10s.

**Selection for first integration:** `acestep-v15-turbo` with
`thinking=false` (no LM) — this host is CPU-only with 15GB RAM, so the
2B turbo DiT is the only reasonable choice; XL is out of scope for
integration verification (allowed by Phase 2 rules, which require real
ACE-Step 1.5 generation, not XL specifically).

## REST API (verified in `acestep/api/http/*` at the pinned commit)

Async flow: `POST /release_task` → `task_id` → `POST /query_result`
(poll) → `GET /v1/audio?path=...` (download).

- All responses use the envelope
  `{"data": ..., "code": 200, "error": null, "timestamp": ms, "extra": null}`.
- Auth (optional): `ACESTEP_API_KEY`; clients send
  `Authorization: Bearer <key>` or `ai_token` in the body
  (`acestep/api/http/auth.py`). Empty key disables auth.

### `POST /release_task`

Request fields used by LUBER (verified in `release_task_models.py`):

| Upstream field | Type / default | Notes |
|---|---|---|
| `prompt` (alias `caption`) | str, `""` | music description |
| `lyrics` | str, `""` | empty or `[inst]`/`[instrumental]` ⇒ instrumental (`server_utils.is_instrumental`) |
| `vocal_language` | str, `"en"` | e.g. `en`, `zh`, `ja`, `ko` |
| `audio_duration` (aliases `duration`, `target_duration`) | float, null | seconds, range 10–600 |
| `audio_format` | str, `"mp3"` | `flac`,`mp3`,`opus`,`aac`,`wav`,`wav32` — WAV requested directly |
| `seed` + `use_random_seed` | int/str `-1`, bool `true` | set `use_random_seed=false` to honor `seed` |
| `model` | str, null | DiT selection, e.g. `acestep-v15-turbo` |
| `inference_steps` | int, `8` | turbo: 1–20 (8 recommended) |
| `thinking` | bool, `false` | `true` = 5Hz LM generates audio codes |
| `batch_size` | int, null (server default 2) | LUBER sends `1` |
| `use_cot_caption` / `use_cot_language` | bool, `true` | LM CoT enhancement; LUBER disables when running without LM |

There is **no** `vocal_gender` parameter upstream — vocal gender is
prompt conditioning plus the instrumental lyrics mechanism (see
AceStepPromptCompiler).

Response: `{"task_id": "<uuid>", "status": "queued", "queue_position": n}`.

### `POST /query_result`

Body `{"task_id_list": [..]}` → per task `{task_id, status, result}`;
`status`: `0` queued/running, `1` succeeded, `2` failed. `result` is a
**JSON string**; parsed, it is a list of track objects:
`file` (`/v1/audio?path=...`), `status`, `seed_value` (comma-separated
string), `dit_model`, `lm_model`, `metas` (`bpm`, `duration`,
`keyscale`, `timesignature`, `genres`), `generation_info`.

### Other endpoints

- `GET /health` → `{status:"ok", service:"ACE-Step API", version:"1.0",
  models_initialized, llm_initialized, loaded_model, loaded_lm_model}`
  (`model_service_routes.py`).
- `GET /v1/models` → `{models:[{name,is_default}...], default_model}`.
- `GET /v1/stats` → job counts, queue size, `avg_job_seconds`.
- `POST /v1/init` → load/switch DiT/LM models at runtime.
- `GET /v1/audio?path=<url-encoded>` → audio bytes.

## Known limitations / notes

- `result` being a JSON string (not object) is easy to mis-parse —
  handled inside `AceStepClient` only.
- Server default `batch_size` is 2 — must send 1 to avoid double work.
- `use_cot_caption`/`use_cot_language` default to true and may invoke
  the LM even with `thinking=false`; disabled explicitly when running
  DiT-only.
- HTTP 429 when the server queue is full; `queue_maxsize` via env.
- Audio download path (`/v1/audio?path=`) points at server-local files
  under `ACESTEP_TMPDIR` — files are transient; download promptly.

## This environment (2026-08-11 preflight)

Linux x86_64, 4 CPU cores, no GPU, 15GB RAM, ~29GB free disk.
`huggingface.co` and `modelscope.cn` are both **denied by the
organization's egress policy** (403 on proxy CONNECT), so model weights
cannot be downloaded here; real inference requires an environment with
GPU (or CPU) plus access to one of the official weight sources.
