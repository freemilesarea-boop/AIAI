"""Human review: a small blind rubric, and the mapping kept away from it.

The project has deliberately deferred detailed listening until a
checkpoint changes meaningfully, and the Phase 20H store holds zero
scores. That is a fact this module records rather than works around.

So the design point is proportion. Requiring 41 scored dimensions per
candidate would make the pipeline unusable, and an unusable gate gets
bypassed — which is worse than a smaller gate that people actually
complete. `LIGHT_AB` asks five questions about a handful of cases.
`FULL_BLIND` remains available for a milestone and is not the default.

Blinding is structural. The package contains `A` and `B`; which is the
candidate lives in a separate mapping file the listener is never given.
Order is randomised deterministically from the evaluation seed, so the
assignment is reproducible for audit and unguessable from the package —
and nothing in a filename, an ordering or a metric hints at which side
is new.

Nothing here fabricates a score. A rubric with no responses yields an
empty result set, and a policy that required human evidence stays
unsatisfied.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from luber_evaluation.schemas import EVALUATION_SCHEMA_VERSION, HumanReviewMode, now

RUBRIC_SCHEMA_VERSION = "luber-light-ab-rubric/1"

MAPPING_FILE_NAME = "blind_mapping.json"
PACKAGE_FILE_NAME = "review_package.json"
RESPONSES_FILE_NAME = "responses.jsonl"


@dataclass(frozen=True)
class RubricQuestion:
    question_id: str
    prompt: str
    kind: str
    #: For scale questions.
    scale_low: str = ""
    scale_high: str = ""
    applies_to: str = "ALL"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: Five questions. Chosen because each maps onto a dimension no
#: automatic metric can measure, which is the only reason to spend a
#: listener's attention at all.
LIGHT_AB_QUESTIONS: tuple[RubricQuestion, ...] = (
    RubricQuestion(
        question_id="overall_preference",
        prompt="Which do you prefer overall?",
        kind="CHOICE",
    ),
    RubricQuestion(
        question_id="vocal_quality",
        prompt="Which has the more natural-sounding voice?",
        kind="CHOICE",
        applies_to="VOCAL",
    ),
    RubricQuestion(
        question_id="instrument_quality",
        prompt="Which has the more convincing instruments?",
        kind="CHOICE",
    ),
    RubricQuestion(
        question_id="prompt_fit",
        prompt="Which better matches the prompt and intended style?",
        kind="CHOICE",
    ),
    RubricQuestion(
        question_id="artifact_severity",
        prompt="Which has fewer or milder audible artifacts?",
        kind="CHOICE",
    ),
)

#: Only asked where the case has Korean lyrics; otherwise it is noise.
KOREAN_QUESTION = RubricQuestion(
    question_id="korean_pronunciation",
    prompt="Which pronounces the Korean lyrics more naturally?",
    kind="CHOICE",
    applies_to="KOREAN_VOCAL",
)


@dataclass
class LightAbRubric:
    rubric_version: str = "1"
    questions: tuple[RubricQuestion, ...] = LIGHT_AB_QUESTIONS
    schema_version: str = RUBRIC_SCHEMA_VERSION
    #: Choices a listener may give. "No preference" is a real answer,
    #: and forcing a pick would manufacture a signal.
    choices: tuple[str, ...] = ("A", "B", "NO_PREFERENCE")

    def for_case_type(self, case_type: str) -> tuple[RubricQuestion, ...]:
        questions = [
            question
            for question in self.questions
            if question.applies_to == "ALL"
            or question.applies_to == case_type
            or (question.applies_to == "VOCAL" and "VOCAL" in case_type)
        ]
        if case_type == "KOREAN_VOCAL":
            questions.append(KOREAN_QUESTION)
        return tuple(questions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "rubric_version": self.rubric_version,
            "choices": list(self.choices),
            "questions": [question.to_dict() for question in self.questions],
            "conditional_questions": [KOREAN_QUESTION.to_dict()],
        }


def _assignment(evaluation_id: str, case_id: str, seed: int, salt: int) -> bool:
    """Whether the candidate is presented as A. Deterministic, unguessable.

    Derived from a hash of the evaluation, case, seed and a per-package
    salt: reproducible for audit, and carrying no pattern a listener
    could notice across cases.
    """
    digest = hashlib.sha256(f"{evaluation_id}:{case_id}:{seed}:{salt}".encode()).digest()
    return digest[0] % 2 == 0


@dataclass
class BlindPair:
    """One comparison, as the listener sees it."""

    pair_id: str
    case_id: str
    seed: int
    prompt: str
    lyrics: str
    case_type: str
    duration_seconds: float
    a_artifact: str
    b_artifact: str
    questions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BlindMapping:
    """Which side was which. Stored apart from the package.

    Written to a separate file precisely so that handing someone the
    review package cannot accidentally hand them the answer.
    """

    evaluation_id: str
    #: pair_id -> {"A": "baseline"|"candidate", "B": ...}
    assignments: dict[str, dict[str, str]] = field(default_factory=dict)
    created_at: str = field(default_factory=now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "created_at": self.created_at,
            "assignments": {k: dict(v) for k, v in sorted(self.assignments.items())},
            "warning": "this file reveals which side is the candidate; never give it to a listener",
        }


@dataclass
class ReviewPackage:
    """What a listener receives. Contains no identity, no metrics."""

    evaluation_id: str
    mode: str
    rubric: LightAbRubric
    pairs: list[BlindPair] = field(default_factory=list)
    instructions: str = ""
    created_at: str = field(default_factory=now)
    schema_version: str = EVALUATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evaluation_id": self.evaluation_id,
            "mode": self.mode,
            "created_at": self.created_at,
            "instructions": self.instructions,
            "rubric": self.rubric.to_dict(),
            "pairs": [pair.to_dict() for pair in self.pairs],
        }


DEFAULT_INSTRUCTIONS = (
    "Listen to A and B for each pair and answer the questions. The two are from "
    "different models; which is which is not shown, deliberately. Answer on what you "
    "hear. NO_PREFERENCE is a real answer — use it rather than guessing."
)


class HumanReviewError(RuntimeError):
    """Raised when a review package or its evidence is inconsistent."""


def build_package(
    *,
    evaluation_id: str,
    cases: list[Any],
    baseline_samples: dict[tuple[str, int], str],
    candidate_samples: dict[tuple[str, int], str],
    rubric: LightAbRubric | None = None,
    mode: str = HumanReviewMode.LIGHT_AB.value,
    salt: int = 0,
    max_pairs: int | None = None,
) -> tuple[ReviewPackage, BlindMapping]:
    """Build a blinded package and its separate mapping.

    Returns both, and the caller writes them to different places. They
    are returned together because building them apart would risk them
    disagreeing, and disagreement here silently corrupts every result.
    """
    rubric = rubric or LightAbRubric()
    package = ReviewPackage(
        evaluation_id=evaluation_id,
        mode=mode,
        rubric=rubric,
        instructions=DEFAULT_INSTRUCTIONS,
    )
    mapping = BlindMapping(evaluation_id=evaluation_id)

    pairs_built = 0
    for case in sorted(cases, key=lambda c: c.case_id):
        for key in sorted(k for k in baseline_samples if k[0] == case.case_id):
            if key not in candidate_samples:
                continue
            if max_pairs is not None and pairs_built >= max_pairs:
                break
            case_id, seed = key
            candidate_is_a = _assignment(evaluation_id, case_id, seed, salt)
            pair_id = f"pair_{case_id}_{seed}"

            package.pairs.append(
                BlindPair(
                    pair_id=pair_id,
                    case_id=case_id,
                    seed=seed,
                    prompt=case.spec.prompt,
                    lyrics=case.spec.lyrics,
                    case_type=case.case_type,
                    duration_seconds=case.spec.duration_seconds,
                    a_artifact=(
                        candidate_samples[key] if candidate_is_a else baseline_samples[key]
                    ),
                    b_artifact=(
                        baseline_samples[key] if candidate_is_a else candidate_samples[key]
                    ),
                    questions=[q.to_dict() for q in rubric.for_case_type(case.case_type)],
                )
            )
            mapping.assignments[pair_id] = {
                "A": "candidate" if candidate_is_a else "baseline",
                "B": "baseline" if candidate_is_a else "candidate",
            }
            pairs_built += 1
    return package, mapping


def write_package(
    directory: Path, package: ReviewPackage, mapping: BlindMapping
) -> dict[str, Path]:
    """Write the package and mapping to *separate* files.

    The mapping goes in a sibling file rather than inside the package,
    so that sharing the package is not the same act as revealing the
    answer.
    """
    directory.mkdir(parents=True, exist_ok=True)
    package_path = directory / PACKAGE_FILE_NAME
    mapping_path = directory / MAPPING_FILE_NAME
    package_path.write_text(
        json.dumps(package.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    mapping_path.write_text(
        json.dumps(mapping.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"package": package_path, "mapping": mapping_path}


@dataclass
class HumanResponse:
    """One listener's answer to one question about one pair."""

    evaluation_id: str
    pair_id: str
    case_id: str
    question_id: str
    choice: str
    reviewer: str
    rubric_version: str
    recorded_at: str = field(default_factory=now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def record_responses(directory: Path, responses: list[HumanResponse]) -> Path:
    """Append responses. Never overwrites a prior review.

    Append-only because a second reviewer's answers are additional
    evidence, not a correction of the first — and silently replacing an
    earlier review would destroy the disagreement that is often the most
    informative thing in the data.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / RESPONSES_FILE_NAME
    with path.open("a", encoding="utf-8") as handle:
        for response in responses:
            handle.write(json.dumps(response.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")
    return path


def read_responses(directory: Path) -> list[HumanResponse]:
    path = directory / RESPONSES_FILE_NAME
    if not path.is_file():
        return []
    responses: list[HumanResponse] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        responses.append(HumanResponse(**payload))
    return responses


@dataclass
class HumanEvidence:
    """Aggregated human answers, unblinded via the mapping."""

    evaluation_id: str
    reviewed_pairs: int = 0
    reviewers: list[str] = field(default_factory=list)
    #: question_id -> {"candidate": n, "baseline": n, "no_preference": n}
    tallies: dict[str, dict[str, int]] = field(default_factory=dict)
    rubric_version: str = ""

    def preference_share(self, question_id: str) -> float | None:
        """Share of decided answers favouring the candidate.

        ``None`` when nobody answered. NO_PREFERENCE is excluded from
        the denominator: it is a real answer about the pair, not a vote
        for either side, and counting it as half would invent a
        preference.
        """
        tally = self.tallies.get(question_id)
        if not tally:
            return None
        decided = tally.get("candidate", 0) + tally.get("baseline", 0)
        if decided == 0:
            return None
        return tally["candidate"] / decided

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "reviewed_pairs": self.reviewed_pairs,
            "reviewers": sorted(set(self.reviewers)),
            "rubric_version": self.rubric_version,
            "tallies": {k: dict(sorted(v.items())) for k, v in sorted(self.tallies.items())},
            "preference_shares": {
                question: self.preference_share(question) for question in sorted(self.tallies)
            },
        }


def unblind(responses: list[HumanResponse], mapping: BlindMapping) -> HumanEvidence:
    """Turn A/B answers into candidate/baseline tallies.

    A response for a pair the mapping does not know is an error rather
    than something to skip: it means the responses and the mapping came
    from different packages, and any tally built from them would be
    fiction.
    """
    evidence = HumanEvidence(evaluation_id=mapping.evaluation_id)
    pairs: set[str] = set()

    for response in responses:
        assignment = mapping.assignments.get(response.pair_id)
        if assignment is None:
            raise HumanReviewError(
                f"response references pair {response.pair_id!r}, which this mapping does "
                "not contain; the responses and mapping are from different packages"
            )
        pairs.add(response.pair_id)
        evidence.reviewers.append(response.reviewer)
        evidence.rubric_version = response.rubric_version

        tally = evidence.tallies.setdefault(
            response.question_id, {"candidate": 0, "baseline": 0, "no_preference": 0}
        )
        if response.choice == "NO_PREFERENCE":
            tally["no_preference"] += 1
        elif response.choice in ("A", "B"):
            tally[assignment[response.choice]] += 1
        else:
            raise HumanReviewError(
                f"response choice {response.choice!r} is not one of A, B, NO_PREFERENCE"
            )

    evidence.reviewed_pairs = len(pairs)
    return evidence
