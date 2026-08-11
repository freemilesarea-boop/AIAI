"""Provider-agnostic music generation boundary.

Business logic must never import a concrete model integration
(ACE-Step, Stable Audio, …) directly — only this package's
:class:`MusicGenerationProvider` interface.
"""

from luber_generation_client.provider import (
    GenerationRequest,
    GenerationResult,
    MusicGenerationProvider,
)

__all__ = ["GenerationRequest", "GenerationResult", "MusicGenerationProvider"]
