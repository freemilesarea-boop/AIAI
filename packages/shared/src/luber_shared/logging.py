"""Structured JSON logging shared by every LUBER service.

Every log line is a single JSON object so it can be shipped to a log
aggregator without extra parsing. Correlation fields (request_id,
generation_id, user_id, worker_id, model_version) are emitted whenever
they are passed via ``logging`` ``extra=`` kwargs.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

# Correlation identifiers used across the platform (see docs/ARCHITECTURE.md).
CONTEXT_FIELDS = (
    "request_id",
    "generation_id",
    "user_id",
    "worker_id",
    "model_version",
)

_RESERVED_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON documents."""

    def __init__(self, service: str) -> None:
        super().__init__()
        self._service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "service": self._service,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info and record.exc_info[0] is not None:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(service: str, level: str = "INFO") -> None:
    """Configure root logging for a service with JSON output on stdout.

    Safe to call more than once: existing handlers installed by this
    function are replaced, not duplicated.
    """
    root = logging.getLogger()
    root.setLevel(level.upper())

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(service=service))

    root.handlers = [h for h in root.handlers if not isinstance(h.formatter, JsonFormatter)]
    root.addHandler(handler)
