"""Where the generation service meets the candidate controller.

The seam, and nothing else. The controller in `luber_inference_qc` knows
how to judge audio and decide about retries; it knows nothing about
providers, edit kinds or error codes. This module supplies exactly that
knowledge and keeps it out of both.

Three translations happen here.

**Provider errors into retryability.** Only this package knows which
`ErrorCode` means "try again" and which means "the configuration is
wrong". Getting it backwards is how a misconfigured provider burns a
whole retry budget reproducing the same error.

**Requests into expectations.** The controller compares against a
`RequestExpectation`; a `GenerationRequest` has to be turned into one,
and an edit request has to be turned into a different one — an edit's
duration is the source's, not a number the user typed.

**Tasks into policies.** Text-to-music and reference-conditioned
generation may retry; a cover or an edit runs once. That is not the
controller's judgement to make: it is a fact about what those operations
mean, and it is expressed by handing the controller a policy that
permits no retries rather than by a branch inside it.
"""

from __future__ import annotations

from luber_generation_client.editing import AudioEditRequest
from luber_generation_client.errors import GenerationProviderError
from luber_generation_client.provider import GenerationRequest
from luber_inference_qc import CandidatePolicy, RequestExpectation, profile
from luber_inference_qc.controller import ProviderCallFailed
from luber_inference_qc.policy import PolicyProfile
from luber_schemas import ErrorCode

#: Provider failures another attempt could plausibly fix. Everything
#: else is a fact about the deployment, not about this request.
RETRYABLE_PROVIDER_CODES: frozenset[str] = frozenset(
    {
        ErrorCode.GENERATION_TIMEOUT.value,
        ErrorCode.PROVIDER_BUSY.value,
        ErrorCode.INVALID_AUDIO.value,
        ErrorCode.UNKNOWN_GENERATION_ERROR.value,
    }
)

#: Failures that reproduce exactly. A missing model, a reference the
#: provider cannot honour, a machine without the memory for the job —
#: none of them changes because it was asked again.
NON_RETRYABLE_PROVIDER_CODES: frozenset[str] = frozenset(
    {
        ErrorCode.MODEL_LOAD_FAILED.value,
        ErrorCode.REFERENCE_AUDIO_UNAVAILABLE.value,
        ErrorCode.OUT_OF_MEMORY.value,
    }
)


def as_controller_failure(exc: Exception) -> ProviderCallFailed:
    """Translate a provider exception into the controller's vocabulary."""
    if isinstance(exc, GenerationProviderError):
        code = exc.error_code.value
        return ProviderCallFailed(
            str(exc),
            retryable=code in RETRYABLE_PROVIDER_CODES,
            error_code=code,
        )
    # An exception this package does not recognise is treated as
    # retryable-once rather than fatal: the budget bounds it either way,
    # and refusing to retry an unknown error would turn a transient bug
    # into a failed generation.
    return ProviderCallFailed(str(exc), retryable=True, error_code=None)


def expectation_for(request: GenerationRequest) -> RequestExpectation:
    """What QC may check about a text-to-music request."""
    return RequestExpectation(
        duration_seconds=float(request.duration_seconds),
        bpm=request.bpm,
        key_scale=request.key_scale,
        instrumental=request.instrumental,
    )


def expectation_for_edit(request: AudioEditRequest) -> RequestExpectation:
    """What QC may check about an edit.

    Deliberately thin. An edit's value is what it preserves, and the
    duration it should produce depends on the operation — an extend is
    longer than its source, a replace-range is the same length. Rather
    than encode that here and risk being wrong, the expectation states
    nothing about duration and QC checks only what is true of any audio:
    that it decoded, is not silent, did not collapse, is not clipped to
    pieces.
    """
    return RequestExpectation()


def policy_for_generation(configured: str, *, retryable_task: bool) -> CandidatePolicy:
    """The policy this generation runs under.

    A task that cannot safely be retried is given a policy that permits
    no retries, rather than a flag the controller has to remember to
    check. The controller then behaves identically for every task and
    there is one fewer branch that can be wrong.
    """
    policy = profile(configured)
    if retryable_task:
        return policy
    return policy.with_overrides(
        name=f"{policy.name}_SINGLE_ATTEMPT",
        initial_candidate_count=1,
        maximum_candidate_count=1,
        maximum_retry_rounds=0,
        maximum_total_provider_calls=1,
        retry_findings=frozenset(),
        allow_seed_variation=False,
    )


def is_retryable_task(*, edit_kind: str | None) -> bool:
    """Whether adaptive retry is safe for this operation.

    Text-to-music, with or without a reference track: yes. A retry
    produces a different song from the same request, which is what a
    retry should do, and the reference is carried unchanged.

    Cover, extend, replace-range: not in this phase. An edit's value is
    what it preserves, and a second attempt with a different seed may
    preserve differently — the product has no semantics for that yet, so
    enabling it would be a guess about what the user wanted. These still
    get QC and a trace; they simply run once.
    """
    return edit_kind is None


def strict_policy_requested(policy_name: str) -> bool:
    return policy_name.strip().upper() == PolicyProfile.STRICT_REPRODUCIBLE.value


__all__ = [
    "NON_RETRYABLE_PROVIDER_CODES",
    "RETRYABLE_PROVIDER_CODES",
    "as_controller_failure",
    "expectation_for",
    "expectation_for_edit",
    "is_retryable_task",
    "policy_for_generation",
    "strict_policy_requested",
]
