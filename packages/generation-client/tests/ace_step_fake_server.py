"""In-process fake of the ACE-Step REST API for contract tests.

Implements the documented protocol at the pinned upstream commit
(docs/ACE_STEP_UPSTREAM_AUDIT.md): response envelope, /release_task,
/query_result with the JSON-*string* result field, /v1/audio, /health,
/v1/models. Used via httpx.MockTransport — no sockets, no model.
"""

from __future__ import annotations

import io
import json
import time
import uuid
import wave
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx


def _repeat_wav(source: Path, target_seconds: float) -> bytes:
    """The fixture repeated to approximately *target_seconds*.

    The same trick `MockGenerationProvider` uses, and for the same
    reason: a stand-in whose output ignores the requested duration does
    not stand in for anything.
    """
    with wave.open(str(source), "rb") as wav:
        params = wav.getparams()
        payload = wav.readframes(params.nframes)
    wanted = round(target_seconds * params.framerate)
    if wanted <= 0 or wanted == params.nframes:
        return source.read_bytes()

    repeats = -(-wanted // params.nframes)
    frame_bytes = params.sampwidth * params.nchannels
    rendered = (payload * repeats)[: wanted * frame_bytes]

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(params.nchannels)
        out.setsampwidth(params.sampwidth)
        out.setframerate(params.framerate)
        out.writeframes(rendered)
    return buffer.getvalue()


def _envelope(data: object, code: int = 200, error: str | None = None) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "data": data,
            "code": code,
            "error": error,
            "timestamp": int(time.time() * 1000),
            "extra": None,
        },
    )


class FakeAceStepServer:
    def __init__(
        self,
        audio_file: Path,
        *,
        polls_before_success: int = 1,
        fail_task: bool = False,
        fail_message: str = "synthetic upstream failure",
        models_initialized: bool = True,
        never_finish: bool = False,
    ) -> None:
        self.audio_file = audio_file
        self._requested_duration: float | None = None
        self.polls_before_success = polls_before_success
        self.fail_task = fail_task
        self.fail_message = fail_message
        self.models_initialized = models_initialized
        self.never_finish = never_finish
        self.release_payloads: list[dict[str, object]] = []
        self.auth_headers: list[str | None] = []
        self._poll_counts: dict[str, int] = {}

    def _served_duration(self) -> float:
        if self._requested_duration:
            return self._requested_duration
        with wave.open(str(self.audio_file), "rb") as wav:
            return wav.getnframes() / wav.getframerate()

    def _served_bytes(self) -> bytes:
        if self._requested_duration:
            return _repeat_wav(self.audio_file, self._requested_duration)
        return self.audio_file.read_bytes()

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.auth_headers.append(request.headers.get("Authorization"))
        path = urlparse(str(request.url)).path
        if path == "/health":
            return _envelope(
                {
                    "status": "ok",
                    "service": "ACE-Step API",
                    "version": "1.0",
                    "models_initialized": self.models_initialized,
                    "llm_initialized": False,
                    "loaded_model": "acestep-v15-turbo",
                    "loaded_lm_model": None,
                }
            )
        if path == "/v1/models":
            return _envelope(
                {
                    "models": [{"name": "acestep-v15-turbo", "is_default": True}],
                    "default_model": "acestep-v15-turbo",
                }
            )
        if path == "/release_task":
            payload = json.loads(request.content)
            self.release_payloads.append(payload)
            # A real engine returns roughly the duration it was asked
            # for, and Phase 29 measures that. A fake that always
            # returned two seconds would manufacture a duration failure
            # in every test that used it.
            self._requested_duration = float(payload.get("audio_duration") or 0.0) or None
            task_id = str(uuid.uuid4())
            self._poll_counts[task_id] = 0
            return _envelope({"task_id": task_id, "status": "queued", "queue_position": 1})
        if path == "/query_result":
            body = json.loads(request.content)
            task_id = body["task_id_list"][0]
            self._poll_counts[task_id] = self._poll_counts.get(task_id, 0) + 1
            if self.never_finish:
                return _envelope([{"task_id": task_id, "status": 0, "result": ""}])
            if self.fail_task:
                return _envelope([{"task_id": task_id, "status": 2, "result": self.fail_message}])
            if self._poll_counts[task_id] <= self.polls_before_success:
                return _envelope([{"task_id": task_id, "status": 0, "result": ""}])
            # Succeeded: result is a JSON STRING (upstream contract).
            result = json.dumps(
                [
                    {
                        "file": f"/v1/audio?path=%2Ftmp%2Fapi_audio%2F{task_id}.wav",
                        "wave": "",
                        "status": 1,
                        "seed_value": "12345,67890",
                        "dit_model": "acestep-v15-turbo",
                        "lm_model": "",
                        "metas": {
                            "bpm": 100,
                            "duration": self._served_duration(),
                            "keyscale": "",
                            "timesignature": "",
                        },
                        "generation_info": "fake",
                    }
                ]
            )
            return _envelope([{"task_id": task_id, "status": 1, "result": result}])
        if path == "/v1/audio":
            query = parse_qs(urlparse(str(request.url)).query)
            assert "path" in query, "audio download must carry ?path="
            return httpx.Response(200, content=self._served_bytes())
        return httpx.Response(404, json={"detail": "not found"})
