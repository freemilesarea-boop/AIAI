from __future__ import annotations

from functools import lru_cache

from pydantic import Field

from luber_shared import BaseServiceSettings


class ApiSettings(BaseServiceSettings):
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])


@lru_cache
def get_settings() -> ApiSettings:
    return ApiSettings()
