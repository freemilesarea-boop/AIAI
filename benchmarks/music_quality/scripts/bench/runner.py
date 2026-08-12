"""Benchmark runner: drives real generations through the LUBER API.

Deliberately goes through the production HTTP path rather than calling
the provider directly, so what is measured is the same pipeline a user
gets: API → Redis → ARQ worker → AceStepProvider → ACE-Step →
post-processing → storage.

There is no mock mode. A benchmark result that did not come from the
real model is worse than no result.
"""

from __future__ import annotations

import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bench.dataset import BenchmarkPrompt
from bench.metrics import GENERATION_FAILED, measure_wav, real_time_factor
from bench.store import GenerationRecord, ResultStore

DEFAULT_API = "http://127.0.0.1:8000"
#: Refuse to start another generation below this much free disk.
MIN_FREE_DISK_GB = 6.0
POLL_INTERVAL_SECONDS = 3.0


class BenchmarkAbort(Exception):
    """Raised when the run must stop (resource guard, unreachable API)."""


@dataclass(frozen=True)
class RunConfig:
    """One point in the configuration matrix."""

    configuration_id: str
    model: str
    lm_enabled: bool
    thinking_enabled: bool
    inference_steps: int
    runtime_backend: str = "mps-mlx"


BASELINE = RunConfig(
    configuration_id="A_turbo_dit_only",
    model="acestep-v15-turbo",
    lm_enabled=False,
    thinking_enabled=False,
    inference_steps=8,
)


def free_disk_gb(path: Path | None = None) -> float:
    usage = shutil.disk_usage(str(path or Path.home()))
    return usage.free / 1024**3


def swap_used_gb() -> float | None:
    """macOS swap usage; None where unavailable."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=True
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    for token in out.split():
        if token.endswith("M") and "used" in out:
            pass
    # Format: "total = 1024.00M  used = 512.00M  free = 512.00M"
    parts = out.replace("=", " ").split()
    try:
        idx = parts.index("used")
        return float(parts[idx + 1].rstrip("M")) / 1024
    except (ValueError, IndexError):
        return None


def benchmark_id(prompt_id: str, configuration_id: str, duration: int, seed: int | None) -> str:
    """Stable id so a resumed run skips exactly what it already did."""
    seed_part = "rand" if seed is None else str(seed)
    return f"{prompt_id}__{configuration_id}__d{duration}__s{seed_part}"


def _post_json(url: str, payload: dict[str, object], headers: dict[str, str]) -> dict[str, object]:
    import json as _json

    body = _json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    for key, value in headers.items():
        request.add_header(key, value)
    with urllib.request.urlopen(request, timeout=60) as response:
        parsed: dict[str, object] = _json.loads(response.read().decode("utf-8"))
        return parsed


def _get_json(url: str) -> dict[str, object]:
    import json as _json

    with urllib.request.urlopen(url, timeout=60) as response:
        parsed: dict[str, object] = _json.loads(response.read().decode("utf-8"))
        return parsed


class BenchmarkRunner:
    def __init__(
        self,
        *,
        api_base: str = DEFAULT_API,
        storage_root: Path,
        store: ResultStore,
        benchmark_version: str,
        min_free_disk_gb: float = MIN_FREE_DISK_GB,
        generation_timeout: float = 2400.0,
    ) -> None:
        self._api = api_base.rstrip("/")
        self._storage_root = storage_root
        self._store = store
        self._version = benchmark_version
        self._min_free_disk_gb = min_free_disk_gb
        self._timeout = generation_timeout

    def check_api(self) -> None:
        try:
            health = _get_json(f"{self._api}/health")
        except (urllib.error.URLError, OSError) as exc:
            raise BenchmarkAbort(f"LUBER API unreachable at {self._api}: {exc}") from exc
        if health.get("status") != "ok":
            raise BenchmarkAbort(f"LUBER API unhealthy: {health}")

    def run_one(
        self,
        prompt: BenchmarkPrompt,
        *,
        config: RunConfig,
        duration: int,
        seed: int | None,
    ) -> GenerationRecord:
        bid = benchmark_id(prompt.prompt_id, config.configuration_id, duration, seed)

        free_gb = free_disk_gb(self._storage_root)
        if free_gb < self._min_free_disk_gb:
            raise BenchmarkAbort(
                f"free disk {free_gb:.1f} GB is below the {self._min_free_disk_gb:.1f} GB "
                "safety margin; stopping before generating more audio"
            )

        record = GenerationRecord(
            benchmark_id=bid,
            benchmark_version=self._version,
            prompt_id=prompt.prompt_id,
            genre=prompt.genre,
            language=prompt.language,
            vocal_gender=prompt.vocal_gender,
            duration_requested=duration,
            lyrics_structure=prompt.structure,
            prompt_style=prompt.prompt_style,
            prompt=prompt.prompt,
            compiled_prompt=None,
            lyrics=prompt.lyrics,
            model=config.model,
            model_version=None,
            lm_enabled=config.lm_enabled,
            thinking_enabled=config.thinking_enabled,
            inference_steps=config.inference_steps,
            seed=seed,
            configuration_id=config.configuration_id,
            runtime_backend=config.runtime_backend,
            disk_free_gb_before=round(free_gb, 2),
            swap_used_gb=swap_used_gb(),
        )

        payload: dict[str, object] = {
            "title": f"BENCH {prompt.prompt_id}",
            "prompt": prompt.prompt,
            "lyrics": prompt.lyrics,
            "vocal_gender": prompt.vocal_gender,
            "language": prompt.language,
            "duration": duration,
        }
        if seed is not None:
            payload["seed"] = seed

        started = time.monotonic()
        record.generation_started_at = datetime.now(UTC).isoformat()
        try:
            created = _post_json(
                f"{self._api}/v1/generations",
                payload,
                {"Idempotency-Key": f"bench-{bid}-{uuid.uuid4()}"},
            )
        except (urllib.error.URLError, OSError) as exc:
            record.status = "FAILED"
            record.error_code = GENERATION_FAILED
            record.metrics = {"flags": [GENERATION_FAILED], "transport_error": str(exc)[:200]}
            self._store.append(record)
            return record

        generation_id = str(created.get("generation_id", ""))
        record.generation_id = generation_id

        detail = self._poll(generation_id, deadline=started + self._timeout)
        record.generation_seconds = round(time.monotonic() - started, 2)
        record.generation_completed_at = datetime.now(UTC).isoformat()

        status = str(detail.get("status", "UNKNOWN"))
        record.status = status
        record.model_version = detail.get("model_version")  # type: ignore[assignment]
        error_code = detail.get("error_code")
        record.error_code = str(error_code) if error_code else None

        if status != "COMPLETED":
            record.metrics = {"flags": [GENERATION_FAILED]}
            self._store.append(record)
            return record

        raw_assets = detail.get("audio_assets")
        assets: list[Any] = raw_assets if isinstance(raw_assets, list) else []
        master = next(
            (a for a in assets if isinstance(a, dict) and a.get("asset_type") == "MASTER"), None
        )
        preview = next(
            (a for a in assets if isinstance(a, dict) and a.get("asset_type") == "PREVIEW"), None
        )
        if master is None:
            record.status = "FAILED"
            record.metrics = {"flags": [GENERATION_FAILED]}
            self._store.append(record)
            return record

        record.output_sha256 = str(master.get("sha256"))
        record.output_duration = float(master.get("duration") or 0.0)
        record.preview_sha256 = str(preview.get("sha256")) if preview else None
        storage_key = str(master.get("storage_key"))
        master_path = self._storage_root / storage_key
        record.output_path = storage_key

        metrics = measure_wav(master_path, requested_duration=float(duration))
        record.metrics = metrics.to_dict()
        record.real_time_factor = real_time_factor(
            record.generation_seconds or 0.0, metrics.duration_seconds
        )

        self._store.append(record)
        return record

    def _poll(self, generation_id: str, *, deadline: float) -> dict[str, object]:
        last: dict[str, object] = {"status": "UNKNOWN"}
        while time.monotonic() < deadline:
            try:
                last = _get_json(f"{self._api}/v1/generations/{generation_id}")
            except (urllib.error.URLError, OSError):
                time.sleep(POLL_INTERVAL_SECONDS)
                continue
            status = str(last.get("status", ""))
            if status in {"COMPLETED", "FAILED", "CANCELLED"}:
                return last
            time.sleep(POLL_INTERVAL_SECONDS)
        last["status"] = "TIMEOUT"
        return last
