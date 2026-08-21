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

    # ── Reference-audio lifecycle ─────────────────────────────────
    #
    # How long an unused reference upload is kept before cleanup may
    # remove it. 24 hours is deliberately generous: a genuinely
    # abandoned browser upload is abandoned within minutes, but a user
    # who uploads, gets pulled away and comes back the same day must not
    # find their reference gone. Short enough that storage cannot grow
    # without bound; long enough that no legitimate flow is interrupted.
    #
    # One value, read by every caller. A grace period that differed
    # between the API and the cleaner would be a deletion nobody
    # predicted.
    reference_abandonment_grace_hours: int = 24
    #: Ceiling on one cleanup invocation, so a large backlog is drained
    #: over several runs rather than in one unbounded sweep.
    reference_cleanup_batch_size: int = 200

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
    # ── Inference quality control (Phase 29) ──────────────────────
    #
    # Which candidate policy a generation runs under. STANDARD generates
    # one candidate and spends a second inference only when the first
    # one failed a measurable check. STRICT_REPRODUCIBLE makes exactly
    # one provider call and reports a quality failure as a failure —
    # someone who supplied a seed gets that seed's output or an honest
    # error, never a different song.
    #
    # See docs/INFERENCE_QUALITY_CONTROL.md for what each profile costs.
    inference_qc_policy: str = "STANDARD"

    # Set false to run generations exactly as they ran before Phase 29:
    # one provider call, no candidate QC, no retry. The escape hatch for
    # a QC regression that is rejecting good output — flip it, ship, then
    # fix the threshold rather than the other way round.
    inference_qc_enabled: bool = True

    # Where candidate audio waits while a generation decides. Not a
    # system temp directory: it has to survive a worker being killed so
    # that the queue's retry can reuse an expensive call rather than pay
    # for it twice. Removed when the generation reaches a terminal state.
    candidate_workspace_dir: str = "data/generation-candidates"

    # Audio storage backend: "local" (development) or "s3" (production,
    # any S3-compatible provider).
    storage_provider: str = "local"

    # Local adapter: storage keys are "audio/{generation_id}/…", so files
    # land under {audio_storage_dir}/audio/.
    audio_storage_dir: str = "data"

    # S3-compatible adapter (used when storage_provider="s3"). Credentials
    # come from the environment only and are never committed. Leave
    # endpoint empty for AWS S3; set it for R2/MinIO/Supabase/etc.
    storage_bucket: str = ""
    storage_region: str | None = None
    storage_endpoint: str | None = None
    storage_access_key_id: str | None = None
    storage_secret_access_key: str | None = None
    #: Required by MinIO and some self-hosted S3 gateways.
    storage_force_path_style: bool = False
    #: Optional key prefix so one bucket can serve several environments.
    storage_key_prefix: str = ""

    # --- authentication ------------------------------------------
    #: How long a session stays valid, server-side. Two weeks: long
    #: enough that a person is not logging in daily, short enough that
    #: an abandoned session on a shared machine expires by itself.
    session_lifetime_seconds: int = 14 * 24 * 60 * 60
    #: Attempts per window per client address, for signup and login
    #: separately. Ten is generous for a human and useless for a
    #: credential-stuffing run.
    auth_rate_limit_attempts: int = 10
    auth_rate_limit_window_seconds: int = 15 * 60

    @property
    def is_production(self) -> bool:
        return self.environment is Environment.PRODUCTION
