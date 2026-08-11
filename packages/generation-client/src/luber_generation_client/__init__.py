"""Provider-agnostic music generation boundary.

Business logic must never import a concrete model integration
(ACE-Step, Stable Audio, …) directly — only this package's
:class:`MusicGenerationProvider` interface.
"""

from luber_generation_client.errors import GenerationProviderError
from luber_generation_client.factory import build_provider
from luber_generation_client.mock import MockGenerationProvider
from luber_generation_client.provider import (
    GenerationRequest,
    GenerationResult,
    MusicGenerationProvider,
)
from luber_generation_client.queue import (
    AUDIO_QUEUE_NAME,
    GENERATION_JOB_NAME,
    GENERATION_QUEUE_NAME,
)
from luber_generation_client.service import GenerationService

__all__ = [
    "AUDIO_QUEUE_NAME",
    "GENERATION_JOB_NAME",
    "GENERATION_QUEUE_NAME",
    "GenerationProviderError",
    "GenerationRequest",
    "GenerationResult",
    "GenerationService",
    "MockGenerationProvider",
    "MusicGenerationProvider",
    "build_provider",
]
