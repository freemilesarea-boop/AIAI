"""HTTP transport for the ACE-Step 1.5 REST API.

Implements exactly the endpoints documented upstream (docs/en/API.md at
the pinned commit): ``POST /release_task``, ``POST /query_result``,
``GET /v1/audio``, ``GET /health``, ``GET /v1/models``. Nothing else in
LUBER talks HTTP to ACE-Step.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import TracebackType
from typing import Any

import httpx

from luber_generation_client.ace_step.types import (
    AceStepHealth,
    AceStepModelList,
    AceStepQueryResult,
    AceStepTaskHandle,
    AceStepTaskStatus,
    AceStepTrack,
)


class AceStepApiError(Exception):
    """Transport-level or envelope-level ACE-Step API failure."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AceStepClient:
    """Async client for one ACE-Step API server."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        request_timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=request_timeout,
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AceStepClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    # ── envelope handling ──────────────────────────────────────────

    @staticmethod
    def _unwrap(response: httpx.Response) -> Any:
        """Validate the standard ``{data, code, error, ...}`` envelope."""
        if response.status_code != 200:
            raise AceStepApiError(
                f"ACE-Step API returned HTTP {response.status_code}",
                status_code=response.status_code,
            )
        try:
            envelope = response.json()
        except json.JSONDecodeError as exc:
            raise AceStepApiError(f"ACE-Step API returned non-JSON body: {exc}") from exc
        code = envelope.get("code")
        if code != 200 or envelope.get("error"):
            raise AceStepApiError(
                f"ACE-Step API error (code={code}): {envelope.get('error')}",
                status_code=code if isinstance(code, int) else None,
            )
        return envelope.get("data")

    # ── endpoints ──────────────────────────────────────────────────

    async def health(self) -> AceStepHealth:
        data = self._unwrap(await self._client.get("/health"))
        return AceStepHealth(
            status=str(data.get("status", "")),
            service=str(data.get("service", "")),
            version=str(data.get("version", "")),
            models_initialized=bool(data.get("models_initialized", False)),
            llm_initialized=bool(data.get("llm_initialized", False)),
            loaded_model=data.get("loaded_model"),
            loaded_lm_model=data.get("loaded_lm_model"),
        )

    async def list_models(self) -> AceStepModelList:
        data = self._unwrap(await self._client.get("/v1/models"))
        models = [str(m.get("name")) for m in data.get("models", []) if m.get("name")]
        return AceStepModelList(models=models, default_model=data.get("default_model"))

    async def submit_generation(self, payload: dict[str, Any]) -> AceStepTaskHandle:
        """POST /release_task with a payload of documented fields only."""
        data = self._unwrap(await self._client.post("/release_task", json=payload))
        return self._task_handle(data)

    async def submit_generation_with_source_audio(
        self, payload: dict[str, Any], source_audio: Path
    ) -> AceStepTaskHandle:
        """POST /release_task as multipart, uploading the source audio.

        Editing tasks need the audio itself, and upstream will not take a
        path to it: ``validate_audio_path`` rejects absolute paths outside
        the system temp directory. Uploading is also the only option that
        survives ACE-Step running on another host or LUBER's masters
        living in object storage, so it is the transport rather than a
        workaround for the path check.

        ``src_audio`` is the field name the upstream multipart parser
        reads (``form.get("ctx_audio") or form.get("src_audio")``); it
        saves the upload to its own temp file and cleans it up.

        Scalar fields travel as form values. Upstream re-parses them with
        its own coercion helpers, so they are sent as strings, and
        ``None`` is omitted entirely rather than sent as ``"None"``.
        """
        fields = {
            key: self._form_value(value) for key, value in payload.items() if value is not None
        }
        # The handle is opened and closed around the request itself, so
        # it is released on success, HTTP error, timeout and cancellation
        # alike — httpx reads the stream during ``post``.
        with source_audio.open("rb") as handle:
            files = {"src_audio": (source_audio.name, handle, "audio/wav")}
            response = await self._client.post("/release_task", data=fields, files=files)
        return self._task_handle(self._unwrap(response))

    async def submit_generation_with_reference_audio(
        self, payload: dict[str, Any], reference_audio: Path
    ) -> AceStepTaskHandle:
        """POST /release_task as multipart, uploading the reference track.

        ``ref_audio`` is a different field from ``src_audio`` and drives a
        different mechanism. Upstream reads it at
        ``release_task_request_parser.py`` — ``form.get("ref_audio") or
        form.get("reference_audio")`` — saves it to its own temp file and
        sets ``reference_audio_path``, which feeds the timbre encoder.
        ``src_audio`` instead becomes ``src_audio_path`` and feeds repaint
        or the cover sketch. Sending a reference under the source field
        would silently perform a different operation than the user asked
        for, so the two never share a code path here.

        Uploading rather than passing a path is not a convenience:
        upstream's ``validate_audio_path`` rejects absolute paths outside
        its own temp directory, and the worker may not share a filesystem
        with the engine at all.
        """
        fields = {
            key: self._form_value(value) for key, value in payload.items() if value is not None
        }
        with reference_audio.open("rb") as handle:
            files = {"ref_audio": (reference_audio.name, handle, "audio/wav")}
            response = await self._client.post("/release_task", data=fields, files=files)
        return self._task_handle(self._unwrap(response))

    @staticmethod
    def _form_value(value: Any) -> str:
        """Render one payload value for a multipart form field.

        Booleans need explicit lowercase text: ``str(False)`` is
        ``"False"``, which upstream's parser does not read as false.
        """
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    @staticmethod
    def _task_handle(data: Any) -> AceStepTaskHandle:
        task_id = data.get("task_id") if isinstance(data, dict) else None
        if not task_id:
            raise AceStepApiError(f"release_task returned no task_id: {data!r}")
        return AceStepTaskHandle(task_id=str(task_id), queue_position=data.get("queue_position"))

    async def query_generation(self, task_id: str) -> AceStepQueryResult:
        """POST /query_result for one task.

        Upstream returns ``result`` as a JSON *string* containing a list
        of track objects — parsed here so nothing downstream sees it.
        """
        data = self._unwrap(
            await self._client.post("/query_result", json={"task_id_list": [task_id]})
        )
        if not isinstance(data, list) or not data:
            raise AceStepApiError(f"query_result returned no entries for {task_id}")
        entry = data[0]
        status = AceStepTaskStatus(int(entry.get("status", 0)))
        tracks: list[AceStepTrack] = []
        error_message: str | None = None
        raw_result = entry.get("result")
        if status is AceStepTaskStatus.SUCCEEDED and raw_result:
            try:
                parsed = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
            except json.JSONDecodeError as exc:
                raise AceStepApiError(f"unparseable task result for {task_id}: {exc}") from exc
            for item in parsed:
                metas = item.get("metas") or {}
                file_url = item.get("file")
                if not file_url:
                    continue
                tracks.append(
                    AceStepTrack(
                        file_url=str(file_url),
                        seed_value=item.get("seed_value"),
                        dit_model=item.get("dit_model"),
                        lm_model=item.get("lm_model"),
                        duration=metas.get("duration"),
                    )
                )
        elif status is AceStepTaskStatus.FAILED:
            error_message = str(raw_result) if raw_result else "ACE-Step task failed"
        return AceStepQueryResult(
            task_id=str(entry.get("task_id", task_id)),
            status=status,
            tracks=tracks,
            error_message=error_message,
        )

    async def download_audio(self, file_url: str, destination: Path) -> Path:
        """GET a ``/v1/audio?path=...`` URL (as returned in task results)."""
        response = await self._client.get(file_url)
        if response.status_code != 200:
            raise AceStepApiError(
                f"audio download failed with HTTP {response.status_code}",
                status_code=response.status_code,
            )
        content = response.content
        if not content:
            raise AceStepApiError("audio download returned empty body")

        def _write() -> None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)

        await asyncio.to_thread(_write)
        return destination
