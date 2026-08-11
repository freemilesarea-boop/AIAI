"""Provider factory keyed by configuration.

``ace_step`` is registered here in Phase 2; requesting it now raises
explicitly instead of pretending to work.
"""

from __future__ import annotations

from pathlib import Path

from luber_generation_client.mock import MockGenerationProvider
from luber_generation_client.provider import MusicGenerationProvider


def build_provider(provider_name: str, *, mock_fixture_path: Path) -> MusicGenerationProvider:
    if provider_name == "mock":
        return MockGenerationProvider(mock_fixture_path)
    raise ValueError(
        f"unknown or not-yet-integrated generation provider: {provider_name!r} "
        "(available: 'mock'; 'ace_step' arrives in Phase 2)"
    )
