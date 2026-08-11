"""Provider factory keyed by configuration."""

from __future__ import annotations

from pathlib import Path

from luber_generation_client.ace_step import AceStepProvider, AceStepProviderConfig
from luber_generation_client.mock import MockGenerationProvider
from luber_generation_client.provider import MusicGenerationProvider
from luber_shared import BaseServiceSettings


def build_provider(provider_name: str, *, mock_fixture_path: Path) -> MusicGenerationProvider:
    """Build the mock provider (kept for tests and simple call sites)."""
    if provider_name == "mock":
        return MockGenerationProvider(mock_fixture_path)
    raise ValueError(
        f"provider {provider_name!r} requires full settings — use provider_from_settings()"
    )


def provider_from_settings(settings: BaseServiceSettings) -> MusicGenerationProvider:
    """Build the configured provider from environment-driven settings."""
    name = settings.generation_provider
    if name == "mock":
        return MockGenerationProvider(Path(settings.mock_fixture_path))
    if name == "ace_step":
        return AceStepProvider(
            AceStepProviderConfig(
                base_url=settings.ace_step_base_url,
                api_key=settings.ace_step_api_key,
                model=settings.ace_step_model,
                request_timeout=settings.ace_step_request_timeout,
                generation_timeout=settings.ace_step_generation_timeout,
                poll_interval=settings.ace_step_poll_interval,
                output_dir=Path(settings.ace_step_output_dir),
                inference_steps=settings.ace_step_inference_steps,
                thinking=settings.ace_step_thinking,
            )
        )
    raise ValueError(f"unknown generation provider: {name!r} (available: 'mock', 'ace_step')")
