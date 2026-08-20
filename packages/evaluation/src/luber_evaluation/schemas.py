"""Evaluation entities, states, and the identities they must preserve.

The shape of this module follows from one fact: **a completed training
run is not evidence of quality, and a READY checkpoint is not an
accepted model.** Everything here exists to keep those two claims
separate until evidence closes the gap.

Four commitments.

*A candidate is always compared against an explicit baseline.* Nothing
is evaluated in isolation — "the audio sounds fine" is not a finding,
and without a baseline there is no way to tell an improvement from the
model having always done that.

*Identity is locked at the start.* Baseline, candidate, suite, policy
and seed set become immutable when an evaluation run begins. Changing
any of them means a new run, because a result that cites a suite which
has since changed cites nothing.

*QUALIFIED is not PRODUCTION.* Qualification means a checkpoint may
advance to promotion review. Nothing here activates a model.

*Human-required dimensions cannot be automatically satisfied.* If an
experiment's hypothesis is about vocal naturalness, no amount of
technical measurement qualifies it — the decision is
`HUMAN_REVIEW_REQUIRED`, and that is a real outcome rather than a
failure.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

EVALUATION_SCHEMA_VERSION = "luber-evaluation/1"
EVALUATION_ENGINE_VERSION = "luber-evaluation/1.0.0"


def now() -> str:
    return datetime.now(UTC).isoformat()


class EvaluationEntityKind(StrEnum):
    """Identity prefixes, following the Phase 25 convention.

    Separate prefixes rather than reused ones: seeing ``eval_…`` in a
    log line should be enough to know it is not a training run.
    """

    EVALUATION = "eval"
    REVIEW = "rev"
    HUMAN_REQUEST = "hrq"


_ID_PATTERN = re.compile(r"^(eval|rev|hrq)_[0-9a-f]{16}$")


def new_id(kind: EvaluationEntityKind) -> str:
    """A fresh identifier for *kind*. 64 bits of entropy, as Phase 25."""
    return f"{kind.value}_{secrets.token_hex(8)}"


def is_valid_id(identifier: str, kind: EvaluationEntityKind | None = None) -> bool:
    """Whether a string is a well-formed id, optionally of one kind.

    Applied at every boundary that turns an id into a path. An id from
    a command line is untrusted input.
    """
    if not _ID_PATTERN.match(identifier):
        return False
    return kind is None or identifier.startswith(f"{kind.value}_")


def digest_of(payload: Any) -> str:
    """Canonical SHA-256 over a JSON-serialisable structure."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


class EvaluationMode(StrEnum):
    """What is being judged.

    ``RAW_MODEL`` is the default and the only mode that qualifies a
    model. Phase 22 finishing corrects tonal imbalance, thickness and
    stereo instability — exactly the defects a model regression would
    show — so a finished comparison can hide the thing being looked for.

    A ``DELIVERY`` evaluation answers a different, also-real question:
    what a listener would actually receive. It never substitutes for
    the raw one, and the runner refuses to compare across modes.
    """

    RAW_MODEL = "RAW_MODEL"
    DELIVERY = "DELIVERY"


class CaseType(StrEnum):
    """Case categories, limited to what the product genuinely does.

    ``EXTEND``, ``REPLACE_RANGE`` and ``COVER`` exist because
    ``EditKind`` in the schemas package has exactly those three members.
    ``REFERENCE_CONDITIONED`` exists because the provider exposes
    ``supports_reference_audio``. Nothing else is invented.
    """

    TEXT_TO_MUSIC = "TEXT_TO_MUSIC"
    KOREAN_VOCAL = "KOREAN_VOCAL"
    ENGLISH_VOCAL = "ENGLISH_VOCAL"
    INSTRUMENTAL = "INSTRUMENTAL"
    LONG_FORM = "LONG_FORM"
    EXTEND = "EXTEND"
    REPLACE_RANGE = "REPLACE_RANGE"
    COVER = "COVER"
    REFERENCE_CONDITIONED = "REFERENCE_CONDITIONED"


class EvaluationRunStatus(StrEnum):
    DRAFT = "DRAFT"
    VALIDATING = "VALIDATING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


ALLOWED_EVALUATION_TRANSITIONS: dict[str, frozenset[str]] = {
    EvaluationRunStatus.DRAFT.value: frozenset(
        {EvaluationRunStatus.VALIDATING.value, EvaluationRunStatus.CANCELLED.value}
    ),
    EvaluationRunStatus.VALIDATING.value: frozenset(
        {
            EvaluationRunStatus.RUNNING.value,
            EvaluationRunStatus.FAILED.value,
            EvaluationRunStatus.CANCELLED.value,
        }
    ),
    EvaluationRunStatus.RUNNING.value: frozenset(
        {
            EvaluationRunStatus.COMPLETED.value,
            EvaluationRunStatus.FAILED.value,
            EvaluationRunStatus.CANCELLED.value,
        }
    ),
    EvaluationRunStatus.COMPLETED.value: frozenset(),
    EvaluationRunStatus.FAILED.value: frozenset(),
    EvaluationRunStatus.CANCELLED.value: frozenset(),
}


class QualificationOutcome(StrEnum):
    """The verdict on a candidate.

    Deliberately distinct from Phase 25's ``CandidateStatus``. That
    enum tracks where a candidate sits in the evaluation *workflow*;
    this one records what the evidence *said*. Collapsing them would
    make "we have not looked yet" and "we looked and it failed"
    indistinguishable.
    """

    PENDING = "PENDING"
    QUALIFIED = "QUALIFIED"
    REJECTED = "REJECTED"
    #: Evidence is missing or the checkpoint cannot be evaluated. Not a
    #: failure of the model — a failure to have looked properly.
    BLOCKED = "BLOCKED"
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"


class HumanReviewMode(StrEnum):
    """How much human listening a policy asks for.

    ``NONE`` is the default while infrastructure is being built. The
    project has deliberately deferred detailed listening until a
    checkpoint changes meaningfully, and requiring 41 scored dimensions
    per candidate would make the pipeline unusable — which in practice
    means it would be bypassed.
    """

    NONE = "NONE"
    #: A handful of blind A/B questions on a few cases.
    LIGHT_AB = "LIGHT_AB"
    #: The full Phase 20H rubric. Reserved for major milestones.
    FULL_BLIND = "FULL_BLIND"


class RegressionSeverity(StrEnum):
    NONE = "NONE"
    INFO = "INFO"
    MINOR = "MINOR"
    MAJOR = "MAJOR"
    CRITICAL = "CRITICAL"


class ComparisonVerdict(StrEnum):
    IMPROVED = "IMPROVED"
    UNCHANGED = "UNCHANGED"
    REGRESSED = "REGRESSED"
    #: Measured on both sides, but the difference is inside the noise
    #: the suite can resolve.
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_MEASURABLE = "NOT_MEASURABLE"


@dataclass
class ModelRef:
    """One side of a comparison, pinned.

    A baseline that could drift under an evaluation would make the
    result meaningless, so the reference records identity *and*
    digests — and the runner freezes it for the life of the run.
    """

    model_id: str
    upstream_commit: str
    #: None for the production baseline, which is the unmodified model.
    checkpoint_id: str | None = None
    checkpoint_sha256: str | None = None
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateLineage:
    """Everything a candidate inherits, preserved through evaluation.

    Carried whole rather than summarised: an evaluation that could not
    say which dataset and curation produced its candidate would be
    unable to explain a regression six months later.
    """

    candidate_id: str
    checkpoint_id: str
    run_id: str
    experiment_id: str
    base_model_id: str
    dataset_id: str = ""
    dataset_lock_sha256: str = ""
    curation_id: str = ""
    curation_lock_sha256: str = ""
    training_config_sha256: str = ""
    training_plan_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationSpec:
    """Locked inputs for one case. Identical for both sides.

    Every field a generation request accepts is pinned here, because a
    comparison where the two sides received different conditioning is
    not a comparison of models.
    """

    prompt: str
    lyrics: str = ""
    duration_seconds: float = 60.0
    language: str = "unknown"
    #: "female" / "male" / "instrumental", matching ``VocalGender``, or
    #: "unknown" when the case does not state one. Never guessed: a
    #: real backend refuses a case rather than choosing a voice for it,
    #: because a baseline sung female and a candidate sung male is not a
    #: comparison of models.
    vocal_gender: str = "unknown"
    bpm: int | None = None
    key_scale: str | None = None
    time_signature: str | None = None
    task: str = CaseType.TEXT_TO_MUSIC.value
    reference_audio_ref: str | None = None
    inference_steps: int | None = None
    guidance_scale: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationCase:
    """One thing to generate, and what may be checked about it."""

    case_id: str
    case_type: str
    spec: GenerationSpec
    #: Metrics this case supports. A case with no lyrics does not carry
    #: lyric metrics, so their absence is NOT_APPLICABLE rather than a
    #: missing measurement.
    applicable_metrics: tuple[str, ...] = ()
    #: Where this case came from — the frozen P20 set, or a suite the
    #: experiment added.
    origin: str = ""
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["spec"] = self.spec.to_dict()
        payload["applicable_metrics"] = list(self.applicable_metrics)
        payload["tags"] = list(self.tags)
        return payload


@dataclass
class SampleProvenance:
    """Where one generated sample came from. No mystery WAVs.

    Recorded for every sample so that audio a human is about to judge
    can be tied back to the exact model, checkpoint, case and seed that
    produced it — and so a later verification can prove the file has
    not been swapped.
    """

    evaluation_id: str
    case_id: str
    seed: int
    model_id: str
    checkpoint_id: str | None
    mode: str
    generation_spec_digest: str
    raw_sha256: str | None = None
    delivery_sha256: str | None = None
    artifact_ref: str | None = None
    duration_seconds: float | None = None
    #: Marks synthetic samples so they can never be mistaken for audio.
    synthetic: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HumanReviewRequest:
    """A request for listening evidence, with what it is for."""

    request_id: str
    evaluation_id: str
    mode: str
    reason: str
    case_ids: list[str] = field(default_factory=list)
    dimensions: list[str] = field(default_factory=list)
    rubric_version: str = ""
    created_at: str = field(default_factory=now)
    status: str = "PENDING"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PromotionReview:
    """An operator decision about a qualified checkpoint.

    Deliberately stops short of production. Coupling evaluation to
    runtime deployment would mean a qualification bug could change what
    users are served, and those are different risks that deserve
    different gates.
    """

    review_id: str
    candidate_id: str
    evaluation_id: str
    qualification_outcome: str
    decision: str
    decided_by: str
    rationale: str
    decided_at: str = field(default_factory=now)
    schema_version: str = EVALUATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PromotionDecisionValue(StrEnum):
    APPROVE_FOR_STAGING = "APPROVE_FOR_STAGING"
    REJECT = "REJECT"
    HOLD = "HOLD"


#: Phase 26 never emits this. Listed so the vocabulary is complete and
#: so a future phase has a name for the thing it will do.
PRODUCTION_ACTIVATION_IS_OUT_OF_SCOPE = (
    "Promotion review may approve a checkpoint for staging. Activating a model in "
    "production is a runtime deployment decision and is not made here."
)
