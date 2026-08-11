"""Typed structures for the ACE-Step REST protocol.

Shapes verified against upstream commit 6d467e4 (see
docs/ACE_STEP_UPSTREAM_AUDIT.md). Raw response dictionaries never
leave the client layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class AceStepTaskStatus(IntEnum):
    """Upstream task status integers (docs/en/API.md §3)."""

    QUEUED_OR_RUNNING = 0
    SUCCEEDED = 1
    FAILED = 2


@dataclass(frozen=True)
class AceStepHealth:
    status: str
    service: str
    version: str
    models_initialized: bool
    llm_initialized: bool
    loaded_model: str | None
    loaded_lm_model: str | None


@dataclass(frozen=True)
class AceStepModelList:
    models: list[str]
    default_model: str | None


@dataclass(frozen=True)
class AceStepTaskHandle:
    task_id: str
    queue_position: int | None = None


@dataclass(frozen=True)
class AceStepTrack:
    """One generated track inside a succeeded task result."""

    file_url: str  # relative URL: /v1/audio?path=...
    seed_value: str | None
    dit_model: str | None
    lm_model: str | None
    duration: float | None

    def first_seed(self) -> int | None:
        if not self.seed_value:
            return None
        head = self.seed_value.split(",")[0].strip()
        try:
            return int(head)
        except ValueError:
            return None


@dataclass(frozen=True)
class AceStepQueryResult:
    task_id: str
    status: AceStepTaskStatus
    tracks: list[AceStepTrack] = field(default_factory=list)
    error_message: str | None = None
