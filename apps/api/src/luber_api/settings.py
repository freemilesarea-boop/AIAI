from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field

from luber_shared import BaseServiceSettings


class ApiSettings(BaseServiceSettings):
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    # "queue" (production: dispatch to the ARQ worker) or "inline"
    # (tests/single-process dev only — never production).
    generation_execution_mode: Literal["queue", "inline"] = "queue"


@lru_cache
def get_settings() -> ApiSettings:
    return ApiSettings()
