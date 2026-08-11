"""ACE-Step 1.5 integration (transport + provider).

Everything ACE-Step-specific lives in this subpackage. The rest of
LUBER MUSIC AI only sees the ``MusicGenerationProvider`` contract —
``release_task`` / ``query_result`` / integer task statuses never leak
into GenerationService or the API layer.
"""

from luber_generation_client.ace_step.client import AceStepApiError, AceStepClient
from luber_generation_client.ace_step.compiler import AceStepPromptCompiler, CompiledAceStepInput
from luber_generation_client.ace_step.provider import AceStepProvider, AceStepProviderConfig
from luber_generation_client.ace_step.version import ACE_STEP_UPSTREAM_COMMIT, ACE_STEP_VERSION

__all__ = [
    "ACE_STEP_UPSTREAM_COMMIT",
    "ACE_STEP_VERSION",
    "AceStepApiError",
    "AceStepClient",
    "AceStepPromptCompiler",
    "AceStepProvider",
    "AceStepProviderConfig",
    "CompiledAceStepInput",
]
