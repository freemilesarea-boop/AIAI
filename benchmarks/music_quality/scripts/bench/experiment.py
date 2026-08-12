"""Controlled no-training A/B experiments against the ACE-Step engine.

Talks to the ACE-Step REST API directly rather than through the LUBER
pipeline. That is deliberate: the question these experiments answer is
"is this quality problem configuration-induced?", which requires
changing exactly one engine parameter at a time with everything else —
including the seed — held fixed. Routing through LUBER would add
prompt compilation and post-processing between the variable and the
measurement.

The production path is still benchmarked separately by
`bench/runner.py`; this module is for isolating causes, not for
validating the product.

Only parameters upstream actually supports are used. See
`docs/PHASE5_ACE_STEP_QUALITY_AUDIT.md` for what exists and what is
inert for turbo models (notably `guidance_scale`, which the pipeline
auto-corrects to 1.0, and `use_adg`, which is base-model only).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_ACE_STEP = "http://127.0.0.1:8001"
POLL_INTERVAL_SECONDS = 3.0


class ExperimentError(Exception):
    """Raised when the engine cannot be driven."""


@dataclass(frozen=True)
class Variant:
    """One cell of the configuration matrix.

    ``extra`` carries only upstream-supported request fields. Anything
    not set here falls back to the server's own defaults, which is what
    the Phase 5 baseline used.
    """

    variant_id: str
    description: str
    #: Use the de-duplicated compiler output instead of the legacy one.
    dedup_prompt: bool = True
    #: Send bpm / key_scale / time_signature metadata conditioning.
    send_metadata: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentRecord:
    experiment_id: str
    variant_id: str
    prompt_id: str
    genre: str
    language: str
    vocal_gender: str
    duration: int
    seed: int
    prompt_sent: str
    lyrics_sent: str
    payload: dict[str, Any]
    status: str = "PENDING"
    generation_seconds: float | None = None
    output_path: str | None = None
    output_sha256: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    structure: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _post(url: str, payload: dict[str, Any], timeout: float = 60.0) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            envelope: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError) as exc:
        raise ExperimentError(f"ACE-Step request failed: {exc}") from exc
    if envelope.get("code") != 200 or envelope.get("error"):
        raise ExperimentError(f"ACE-Step error: {envelope.get('error')}")
    data: dict[str, Any] = envelope.get("data") or {}
    return data


class EngineExperiment:
    """Drives one variant of one prompt through ACE-Step directly."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_ACE_STEP,
        model: str = "acestep-v15-turbo",
        inference_steps: int = 8,
        output_dir: Path,
        generation_timeout: float = 1800.0,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._model = model
        self._steps = inference_steps
        self._out = output_dir
        self._timeout = generation_timeout
        self._out.mkdir(parents=True, exist_ok=True)

    def health(self) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(f"{self._base}/health", timeout=15) as response:
                envelope = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError) as exc:
            raise ExperimentError(f"ACE-Step unreachable at {self._base}: {exc}") from exc
        data: dict[str, Any] = envelope.get("data") or {}
        if not data.get("models_initialized"):
            raise ExperimentError("ACE-Step has no initialized model")
        return data

    def build_payload(
        self,
        *,
        prompt: str,
        lyrics: str,
        language: str,
        duration: int,
        seed: int,
        variant: Variant,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "prompt": prompt,
            "lyrics": lyrics,
            "vocal_language": language,
            "audio_duration": float(duration),
            "audio_format": "wav",
            "model": self._model,
            "inference_steps": self._steps,
            "thinking": False,
            "batch_size": 1,
            "use_cot_caption": False,
            "use_cot_language": False,
            # Fixed seed is what makes the comparison controlled.
            "use_random_seed": False,
            "seed": seed,
        }
        if variant.send_metadata and metadata:
            for key in ("bpm", "key_scale", "time_signature"):
                if metadata.get(key) not in (None, ""):
                    payload[key] = metadata[key]
        payload.update(variant.extra)
        return payload

    def run(self, payload: dict[str, Any], destination: Path) -> tuple[float, Path]:
        started = time.monotonic()
        handle = _post(f"{self._base}/release_task", payload)
        task_id = handle.get("task_id")
        if not task_id:
            raise ExperimentError(f"no task_id returned: {handle!r}")

        deadline = started + self._timeout
        tracks: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            result = _post(f"{self._base}/query_result", {"task_id_list": [task_id]})
            entries: list[Any] = result if isinstance(result, list) else []
            if not entries:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue
            entry = entries[0]
            status = int(entry.get("status", 0))
            if status == 1:
                raw = entry.get("result")
                tracks = json.loads(raw) if isinstance(raw, str) else (raw or [])
                break
            if status == 2:
                raise ExperimentError(f"generation failed: {entry.get('result')}")
            time.sleep(POLL_INTERVAL_SECONDS)
        else:
            raise ExperimentError("generation timed out")

        if not tracks:
            raise ExperimentError("generation succeeded but returned no tracks")

        file_url = tracks[0].get("file")
        if not file_url:
            raise ExperimentError("track has no file url")

        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(f"{self._base}{file_url}", timeout=300) as response:
                destination.write_bytes(response.read())
        except (urllib.error.URLError, OSError) as exc:
            raise ExperimentError(f"audio download failed: {exc}") from exc

        return round(time.monotonic() - started, 2), destination
