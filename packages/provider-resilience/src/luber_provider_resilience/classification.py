"""What kind of failure this was, and whether it says anything about the provider.

The whole circuit rests on one distinction: a failure that is *evidence
about the provider* versus one that merely happened while the provider
was involved. Get it wrong in one direction and a provider stays in
rotation while every request times out; get it wrong in the other and a
hundred users submitting bad reference ids take the model offline.

So the categories below are not a taxonomy for its own sake. Each one
answers "does this count?", and the answer is attached to the category
rather than decided at each call site — because the call site that
forgets is the one that poisons provider health with user error.

Three rules run through it.

**A failure the request caused is never provider evidence.** A reference
track that cannot be fetched, a request the provider refuses as
malformed: the next request from somebody else would have worked. This
is the rule that stops a bad client from opening a circuit.

**A cancellation is not a failure at all.** The user changed their mind.
Counting it would mean a UI change that made cancelling easier looked
like the model breaking.

**Quality is not availability.** Phase 29 rejecting audio means the
provider answered. A circuit is an availability device and must not be
opened by a verdict about how a song sounds — that is Phase 30's job to
report and a human's to act on.
"""

from __future__ import annotations

from enum import StrEnum

from luber_schemas import ErrorCode


class FailureCategory(StrEnum):
    """Why an attempt did not produce a delivered generation."""

    #: The provider could not be reached, or answered that it could not
    #: work. Counts toward circuit health.
    AVAILABILITY_FAILURE = "AVAILABILITY_FAILURE"

    #: The connection dropped, the socket reset, the response was
    #: malformed at the transport level. Counts: from the control
    #: plane's side a provider that cannot hold a connection is a
    #: provider that cannot be used.
    TRANSIENT_TRANSPORT_FAILURE = "TRANSIENT_TRANSPORT_FAILURE"

    #: The provider took longer than its budget. Counts. A provider that
    #: reliably times out is unavailable in every sense that matters,
    #: whatever it is doing internally.
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"

    #: The provider said "not now". Counts, but see `RATE_LIMIT_IS_SOFT`:
    #: it is the provider working correctly and declining, so the policy
    #: treats it as temporary unavailability rather than as breakage.
    PROVIDER_RATE_LIMIT = "PROVIDER_RATE_LIMIT"

    #: Credentials rejected. Counts, and is non-retryable: it reproduces
    #: on every request until somebody fixes configuration, so spending
    #: a retry budget on it buys nothing.
    PROVIDER_AUTH_FAILURE = "PROVIDER_AUTH_FAILURE"

    #: The deployment is wrong — a missing model, an unreachable base
    #: URL, a capability the provider does not have. Counts, and is
    #: non-retryable for the same reason.
    PROVIDER_CONFIGURATION_FAILURE = "PROVIDER_CONFIGURATION_FAILURE"

    #: The provider returned audio and Phase 29 refused it. Does **not**
    #: count. Availability and quality are different axes, and a circuit
    #: is an availability device.
    QUALITY_REGRESSION = "QUALITY_REGRESSION"

    #: Something on this side failed after the provider answered —
    #: post-processing, encoding, upload. Does not count: the provider
    #: did its part.
    LOCAL_VALIDATION_FAILURE = "LOCAL_VALIDATION_FAILURE"

    #: The request was the problem. Does not count, and this is the rule
    #: that stops a bad client from opening a circuit for everybody.
    USER_INPUT_FAILURE = "USER_INPUT_FAILURE"

    #: The user changed their mind. Does not count.
    CANCELLED = "CANCELLED"

    #: Nothing above fits. Counts, because an unrecognised failure from
    #: a provider is still a provider that did not deliver — and
    #: treating the unknown as harmless is how a novel outage goes
    #: unnoticed.
    UNKNOWN = "UNKNOWN"


#: Categories that are evidence about the provider.
#:
#: Written as an explicit set rather than as a method on the enum so the
#: list can be read in one place and argued with. Everything absent is
#: absent deliberately.
COUNTS_TOWARD_CIRCUIT: frozenset[str] = frozenset(
    {
        FailureCategory.AVAILABILITY_FAILURE.value,
        FailureCategory.TRANSIENT_TRANSPORT_FAILURE.value,
        FailureCategory.PROVIDER_TIMEOUT.value,
        FailureCategory.PROVIDER_RATE_LIMIT.value,
        FailureCategory.PROVIDER_AUTH_FAILURE.value,
        FailureCategory.PROVIDER_CONFIGURATION_FAILURE.value,
        FailureCategory.UNKNOWN.value,
    }
)

#: Categories that must never affect provider health, listed so the
#: guarantee is checkable rather than implied by absence.
NEVER_COUNTS: frozenset[str] = frozenset(
    {
        FailureCategory.QUALITY_REGRESSION.value,
        FailureCategory.LOCAL_VALIDATION_FAILURE.value,
        FailureCategory.USER_INPUT_FAILURE.value,
        FailureCategory.CANCELLED.value,
    }
)

#: Failures that reproduce until a human changes something. Retrying
#: them spends a budget to reproduce an error somebody already has.
NON_RETRYABLE: frozenset[str] = frozenset(
    {
        FailureCategory.PROVIDER_AUTH_FAILURE.value,
        FailureCategory.PROVIDER_CONFIGURATION_FAILURE.value,
        FailureCategory.USER_INPUT_FAILURE.value,
        FailureCategory.CANCELLED.value,
    }
)

#: A rate limit is the provider working correctly and declining. It
#: makes the provider temporarily unusable without meaning it is broken,
#: which is why the policy can weight it differently from a timeout.
RATE_LIMIT_IS_SOFT = True


#: Platform error codes mapped to resilience categories.
#:
#: This translation exists in one place because the alternative — each
#: caller deciding — is how `REFERENCE_AUDIO_UNAVAILABLE` eventually
#: gets counted as a provider failure by whichever call site was written
#: last.
_BY_ERROR_CODE: dict[str, str] = {
    ErrorCode.GENERATION_TIMEOUT.value: FailureCategory.PROVIDER_TIMEOUT.value,
    ErrorCode.PROVIDER_BUSY.value: FailureCategory.PROVIDER_RATE_LIMIT.value,
    ErrorCode.MODEL_LOAD_FAILED.value: FailureCategory.PROVIDER_CONFIGURATION_FAILURE.value,
    ErrorCode.OUT_OF_MEMORY.value: FailureCategory.AVAILABILITY_FAILURE.value,
    # The request named a reference the provider could not use. That is
    # a fact about this request, not about the provider — the next
    # request would work.
    ErrorCode.REFERENCE_AUDIO_UNAVAILABLE.value: FailureCategory.USER_INPUT_FAILURE.value,
    # Audio came back and did not survive validation. The provider
    # answered; something about what it produced was wrong. Quality,
    # not availability.
    ErrorCode.INVALID_AUDIO.value: FailureCategory.QUALITY_REGRESSION.value,
    ErrorCode.QUALITY_CHECK_FAILED.value: FailureCategory.QUALITY_REGRESSION.value,
    ErrorCode.QUALITY_RETRY_EXHAUSTED.value: FailureCategory.QUALITY_REGRESSION.value,
    # Everything after the provider answered.
    ErrorCode.UPLOAD_FAILED.value: FailureCategory.LOCAL_VALIDATION_FAILURE.value,
    ErrorCode.ENCODING_FAILED.value: FailureCategory.LOCAL_VALIDATION_FAILURE.value,
    ErrorCode.QUEUE_FAILED.value: FailureCategory.LOCAL_VALIDATION_FAILURE.value,
    ErrorCode.GENERATION_INTERRUPTED.value: FailureCategory.CANCELLED.value,
    ErrorCode.GENERATION_HAS_DERIVED_VERSIONS.value: FailureCategory.USER_INPUT_FAILURE.value,
    ErrorCode.UNKNOWN_GENERATION_ERROR.value: FailureCategory.UNKNOWN.value,
}


#: HTTP status codes, where a provider client kept one.
#:
#: Checked *before* the error code, because it is more specific: the
#: ACE-Step client collapses 429 and 401 into MODEL_LOAD_FAILED, and by
#: the time only an ErrorCode remains the difference between "try later"
#: and "your key is wrong" is gone. Classifying here recovers it without
#: changing what the user-facing failure reports.
_BY_STATUS: dict[int, str] = {
    401: FailureCategory.PROVIDER_AUTH_FAILURE.value,
    403: FailureCategory.PROVIDER_AUTH_FAILURE.value,
    404: FailureCategory.PROVIDER_CONFIGURATION_FAILURE.value,
    408: FailureCategory.PROVIDER_TIMEOUT.value,
    409: FailureCategory.AVAILABILITY_FAILURE.value,
    429: FailureCategory.PROVIDER_RATE_LIMIT.value,
    500: FailureCategory.AVAILABILITY_FAILURE.value,
    502: FailureCategory.AVAILABILITY_FAILURE.value,
    503: FailureCategory.AVAILABILITY_FAILURE.value,
    504: FailureCategory.PROVIDER_TIMEOUT.value,
}


def classify(
    *,
    error_code: str | None = None,
    status_code: int | None = None,
    cancelled: bool = False,
    timed_out: bool = False,
    transport_error: bool = False,
) -> str:
    """Which category this failure belongs to.

    The order is the point, and it runs most-certain first.

    Cancellation wins over everything: a request abandoned mid-flight
    may also have timed out, and recording that as a provider timeout
    would count the user's decision against the provider.

    A status code beats an error code because it is more specific — see
    `_BY_STATUS`.
    """
    if cancelled:
        return FailureCategory.CANCELLED.value
    if status_code is not None and status_code in _BY_STATUS:
        return _BY_STATUS[status_code]
    if status_code is not None and 500 <= status_code < 600:
        return FailureCategory.AVAILABILITY_FAILURE.value
    if status_code is not None and 400 <= status_code < 500:
        # A 4xx nobody mapped is the provider refusing this request.
        # Attributing it to the request rather than to the provider is
        # the conservative reading: a provider that answers "no" is
        # answering.
        return FailureCategory.USER_INPUT_FAILURE.value
    if timed_out:
        return FailureCategory.PROVIDER_TIMEOUT.value
    if error_code is not None and error_code in _BY_ERROR_CODE:
        return _BY_ERROR_CODE[error_code]
    if transport_error:
        return FailureCategory.TRANSIENT_TRANSPORT_FAILURE.value
    return FailureCategory.UNKNOWN.value


def counts_toward_circuit(category: str) -> bool:
    """Whether this failure is evidence about the provider."""
    return category in COUNTS_TOWARD_CIRCUIT


def is_retryable(category: str) -> bool:
    """Whether another attempt could plausibly succeed."""
    return category not in NON_RETRYABLE


def error_code_for(category: str) -> ErrorCode:
    """The platform error code a category surfaces as.

    Resilience classification is *additional* to the platform's error
    vocabulary, never a replacement: a user seeing a failure sees the
    same codes they always did. This exists only for the cases Phase 31
    itself originates — refusing a request because a circuit is open —
    where there was no provider error to carry a code of its own.
    """
    mapping = {
        FailureCategory.PROVIDER_TIMEOUT.value: ErrorCode.GENERATION_TIMEOUT,
        FailureCategory.PROVIDER_RATE_LIMIT.value: ErrorCode.PROVIDER_BUSY,
        FailureCategory.PROVIDER_AUTH_FAILURE.value: ErrorCode.MODEL_LOAD_FAILED,
        FailureCategory.PROVIDER_CONFIGURATION_FAILURE.value: ErrorCode.MODEL_LOAD_FAILED,
        FailureCategory.AVAILABILITY_FAILURE.value: ErrorCode.PROVIDER_BUSY,
        FailureCategory.TRANSIENT_TRANSPORT_FAILURE.value: ErrorCode.PROVIDER_BUSY,
        FailureCategory.USER_INPUT_FAILURE.value: ErrorCode.REFERENCE_AUDIO_UNAVAILABLE,
        FailureCategory.CANCELLED.value: ErrorCode.GENERATION_INTERRUPTED,
    }
    return mapping.get(category, ErrorCode.UNKNOWN_GENERATION_ERROR)


__all__ = [
    "COUNTS_TOWARD_CIRCUIT",
    "NEVER_COUNTS",
    "NON_RETRYABLE",
    "RATE_LIMIT_IS_SOFT",
    "FailureCategory",
    "classify",
    "counts_toward_circuit",
    "error_code_for",
    "is_retryable",
]
