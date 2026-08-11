"""Base settings shared by every LUBER service.

Each service subclasses :class:`BaseServiceSettings` and adds its own
fields. Values come from environment variables (or a local ``.env`` file
in development — see ``.env.example``).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class BaseServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Environment = Environment.DEVELOPMENT
    log_level: str = "INFO"
    database_url: str = "postgresql+asyncpg://luber:luber_dev_password@localhost:5432/luber"
    redis_url: str = "redis://localhost:6379/0"

    # Generation provider selection: "mock" | "ace_step"
    generation_provider: str = "mock"
    mock_fixture_path: str = "tests/fixtures/mock_generation.wav"

    # ACE-Step provider (used when generation_provider="ace_step").
    # See docs/ACE_STEP_UPSTREAM_AUDIT.md for the pinned upstream.
    ace_step_base_url: str = "http://127.0.0.1:8001"
    ace_step_api_key: str | None = None
    ace_step_model: str = "acestep-v15-turbo"
    ace_step_request_timeout: float = 60.0
    ace_step_generation_timeout: float = 600.0
    ace_step_poll_interval: float = 2.0
    ace_step_output_dir: str = "data/raw-model-output"
    ace_step_inference_steps: int = 8
    ace_step_thinking: bool = False
    # Local audio storage root; storage keys are "audio/{generation_id}/…",
    # so files land under {audio_storage_dir}/audio/. (S3-compatible
    # adapters arrive in Phase 4.)
    audio_storage_dir: str = "data"

    @property
    def is_production(self) -> bool:
        return self.environment is Environment.PRODUCTION
