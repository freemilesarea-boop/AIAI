"""Domain enums shared across API, workers, and database layers.

These values are part of the persisted contract (stored in PostgreSQL
and returned by the public API), so renames are breaking changes and
must go through a migration.
"""

from __future__ import annotations

from enum import StrEnum


class GenerationStatus(StrEnum):
    """Lifecycle of a music generation job."""

    QUEUED = "QUEUED"
    STARTING = "STARTING"
    GENERATING = "GENERATING"
    POST_PROCESSING = "POST_PROCESSING"
    UPLOADING = "UPLOADING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def is_terminal(self) -> bool:
        return self in {
            GenerationStatus.COMPLETED,
            GenerationStatus.FAILED,
            GenerationStatus.CANCELLED,
        }


class VocalGender(StrEnum):
    """User-facing vocal selection. ``INSTRUMENTAL`` means no vocals."""

    FEMALE = "female"
    MALE = "male"
    INSTRUMENTAL = "instrumental"


class AssetType(StrEnum):
    """Kind of audio asset attached to a generation.

    ``MASTER`` is the **raw** generation master: the model's output with
    nothing but format normalisation applied. It is written once, never
    overwritten by finishing, and is the source every later operation
    reads. The name predates Phase 14 and is kept because every stored
    row and every deployed client already means exactly this by it —
    renaming it would reinterpret existing data rather than describe it.

    ``FINISHED_MASTER`` is the Phase 14 finishing result. It exists only
    when the engine decided a correction was warranted, so its absence is
    normal and means "nothing to fix", not "not processed".

    Callers must never pick between the two by hand. Use
    :func:`luber_schemas.assets.select_delivery_master` for what a
    listener should hear, and
    :func:`luber_schemas.assets.select_raw_master` for what a further
    generation should be fed.
    """

    MASTER = "MASTER"
    FINISHED_MASTER = "FINISHED_MASTER"
    PREVIEW = "PREVIEW"
    STEM = "STEM"


class FinishingOutcome(StrEnum):
    """What the finishing engine did for one generation.

    Recorded durably so that "the engine looked and found nothing to do"
    stays distinguishable from "the engine never ran" (no record at all)
    and from "the engine failed" — states that an absent
    ``FINISHED_MASTER`` asset alone cannot tell apart.
    """

    FINISHED = "FINISHED"
    NO_ACTION = "NO_ACTION"
    FAILED = "FAILED"
    #: The engine corrected the audio, measured what it had produced, and
    #: judged the raw master better. Deliberately not FAILED: nothing went
    #: wrong, and the two call for opposite responses — a failure is a bug
    #: to chase, a rejection is the safeguard working. Deliberately not
    #: NO_ACTION either, because something *was* wrong and the engine
    #: could not fix it without making something else worse, which is
    #: exactly the signal worth acting on when tuning the rules.
    REJECTED = "REJECTED"


class EditKind(StrEnum):
    """How a generation was derived from another generation's audio.

    A *product* vocabulary, not the engine's. Both kinds reach ACE-Step
    as the same primitive — regenerate this time range, preserve the rest
    — and the difference is what the range means:

    ``EXTEND``
        The range begins at the end of the source, so the engine pads the
        source and generates into the padding. The song gets longer.

    ``REPLACE_RANGE``
        The range is interior. The song keeps its length and only that
        span is regenerated.

    The worker needs the distinction because it anchors the two
    differently: an extension is re-anchored to the *measured* end of the
    audio being uploaded, while a replacement uses the absolute times the
    user chose. Storing one value for both would make that routing
    guesswork.
    """

    #: ``COVER`` is a third case and not an edit at all: the engine
    #: regenerates the whole performance steered by a semantic sketch of
    #: the source, preserving none of the recording. It shares this column
    #: because the question the column answers — "how did this come from
    #: its parent?" — is the same one. It carries no time range.
    EXTEND = "EXTEND"
    REPLACE_RANGE = "REPLACE_RANGE"
    COVER = "COVER"

    @property
    def preserves_source_audio(self) -> bool:
        """Whether the parent's recording survives into the result.

        True for the repaint-backed edits, where the engine re-imposes the
        source outside the edited range. False for a cover, which
        regenerates everything. The UI uses this to avoid promising
        preservation it does not get.
        """
        return self in (EditKind.EXTEND, EditKind.REPLACE_RANGE)


class ErrorCode(StrEnum):
    """Standard machine-readable error codes returned to the frontend.

    Raw exception strings are never sent to clients.
    """

    #: Generation is switched off for this deployment. Distinct from
    #: every other code because nothing is wrong with the request, the
    #: account or the engine — the capability is not being served, and
    #: the only recourse is to wait until it is.
    GENERATION_UNAVAILABLE = "GENERATION_UNAVAILABLE"
    #: The account has used every song in its allowance period. Not a
    #: failure of the engine and not the user's input: the plan ran out.
    #: Separate from every other code because the recourse is different —
    #: wait for the period to roll, or change plan.
    GENERATION_LIMIT_REACHED = "GENERATION_LIMIT_REACHED"
    #: The account's plan does not include downloads. Raised only after
    #: ownership has already been established, so it never reveals
    #: anything about a resource the caller does not own.
    DOWNLOAD_NOT_IN_PLAN = "DOWNLOAD_NOT_IN_PLAN"
    GENERATION_TIMEOUT = "GENERATION_TIMEOUT"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    INVALID_AUDIO = "INVALID_AUDIO"
    #: Reference audio was requested but the configured provider cannot
    #: honour it, or the reference no longer exists. Never silently
    #: downgraded to an ordinary generation: a song made without the
    #: reference the user chose is a different song.
    REFERENCE_AUDIO_UNAVAILABLE = "REFERENCE_AUDIO_UNAVAILABLE"
    #: Refused a delete because other generations were derived from this
    #: one. Deleting it would leave those rows claiming to descend from
    #: nothing, so the user removes the derived versions first.
    GENERATION_HAS_DERIVED_VERSIONS = "GENERATION_HAS_DERIVED_VERSIONS"
    #: The worker was stopped, or its job cancelled, while this
    #: generation was mid-flight. Terminal only if nothing picks the job
    #: back up: the queue retries it, and the retry moves the row out of
    #: this state. Recorded rather than left mid-flight because a row
    #: still claiming GENERATING with no process behind it is a lie no
    #: operator or user can distinguish from slow progress.
    GENERATION_INTERRUPTED = "GENERATION_INTERRUPTED"
    #: The engine refused the work because its own queue is full — it
    #: answers HTTP 429 "Server busy: queue is full". Distinct from a
    #: crash or a bad request: nothing is wrong with the song, and the
    #: same request submitted later succeeds. Kept separate so the
    #: product can say "busy, try again" instead of "generation failed".
    PROVIDER_BUSY = "PROVIDER_BUSY"
    UPLOAD_FAILED = "UPLOAD_FAILED"
    ENCODING_FAILED = "ENCODING_FAILED"
    QUEUE_FAILED = "QUEUE_FAILED"
    #: Phase 29. The model produced audio and every attempt failed a
    #: measurable technical check — silent, collapsed, the wrong length,
    #: distorted at source. Kept separate from the codes above because
    #: nothing was wrong with the request or the infrastructure: the
    #: engine ran and what it made could not be delivered, and the same
    #: request submitted again may well work.
    QUALITY_CHECK_FAILED = "QUALITY_CHECK_FAILED"
    #: Phase 29. The retry budget was spent before any attempt passed.
    #: Distinct from QUALITY_CHECK_FAILED: that means nothing further
    #: would have helped, this means nothing further was tried. An
    #: operator tuning budgets needs to tell those apart.
    QUALITY_RETRY_EXHAUSTED = "QUALITY_RETRY_EXHAUSTED"
    UNKNOWN_GENERATION_ERROR = "UNKNOWN_GENERATION_ERROR"


class LineVerdict(StrEnum):
    """What a listener heard happen to one submitted lyric line.

    ``UNKNOWN`` is a real answer, not a missing one: on a dense mix a
    listener genuinely cannot always tell, and forcing a guess would
    poison the record this exists to build.
    """

    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    SKIPPED = "SKIPPED"
    DUPLICATED = "DUPLICATED"
    UNKNOWN = "UNKNOWN"


class FailureTag(StrEnum):
    """Named failure modes observed in LUBER output.

    These are the defects the human evaluator actually reports, so the
    listening tool offers them as checkboxes rather than asking for free
    text that cannot be aggregated later.
    """

    KOREAN_LINE_OMISSION = "KOREAN_LINE_OMISSION"
    LYRIC_LINE_SKIP = "LYRIC_LINE_SKIP"
    LYRIC_DUPLICATION = "LYRIC_DUPLICATION"
    TROT_LIKE_VOCAL = "TROT_LIKE_VOCAL"
    VOCAL_STYLE_OUTDATED = "VOCAL_STYLE_OUTDATED"
    EXCESSIVE_SIBILANCE = "EXCESSIVE_SIBILANCE"
    HIGH_END_OVERBOOST = "HIGH_END_OVERBOOST"
    INSTRUMENT_FIDELITY_LOW = "INSTRUMENT_FIDELITY_LOW"
    STRUCTURE_COLLAPSE = "STRUCTURE_COLLAPSE"
    MELODY_DRIFT = "MELODY_DRIFT"
    VOCAL_IDENTITY_DRIFT = "VOCAL_IDENTITY_DRIFT"
    ENDING_FAILURE = "ENDING_FAILURE"


#: Sections the full-song QA view asks about, in song order.
QA_SECTIONS: tuple[str, ...] = (
    "intro",
    "verse_1",
    "chorus",
    "verse_2",
    "bridge",
    "final_chorus",
    "outro",
)


class SupportCategory(StrEnum):
    """What a support inquiry is about.

    Stable internal identities, never display labels — the Korean text a
    user sees lives in the frontend, so renaming a category in the UI is
    not a migration.
    """

    BILLING = "BILLING"
    GENERATION = "GENERATION"
    DOWNLOAD = "DOWNLOAD"
    ACCOUNT = "ACCOUNT"
    BUG = "BUG"
    FEATURE = "FEATURE"
    OTHER = "OTHER"


class SupportStatus(StrEnum):
    """Where an inquiry is in its handling.

    Operator-owned. Nothing a customer can send changes this — there is
    no field on any request that carries it, which is why the API has no
    check for one.
    """

    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"


class UserRole(StrEnum):
    """What an account may do in the operator console.

    Stored on `users.role` and checked server-side on every admin
    request. Deliberately not derived from an email address: an
    `if email == "..."` check is a permission model that cannot be
    revoked, cannot be audited, and grants whoever registers that
    address if it ever changes hands.
    """

    USER = "USER"
    ADMIN = "ADMIN"
    SUPER_ADMIN = "SUPER_ADMIN"


#: Roles that may reach the console at all.
ADMIN_ROLES: frozenset[UserRole] = frozenset({UserRole.ADMIN, UserRole.SUPER_ADMIN})
