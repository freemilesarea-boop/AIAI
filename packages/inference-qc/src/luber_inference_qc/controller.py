"""The candidate loop: generate, measure, decide, select.

One place, on purpose. The alternative — a bit of QC in the service, a
retry in the worker, a threshold in the provider — is how a system ends
up retrying twice for one failure and nobody being able to say why.

The controller knows nothing about how audio is produced. It is handed a
callable that takes a seed and returns audio, so a text-to-music
generation, a cover and an edit all look identical to it, and so a test
can script a sequence of outcomes without a model. What it owns is the
*decision*: is this deliverable, is another attempt worth an inference,
and which of the survivors wins.

Four properties the loop is built around.

**The healthy path costs one call.** A first candidate with nothing
critical is selected immediately. No second candidate is generated to
compare against it, because the comparison could only say which is less
broken and neither is.

**The budget is checked before the findings are acted on.** A plan that
cannot be paid for is not made, so the trace never shows a retry that
did not happen.

**Every attempt is recorded as it happens.** The trace is handed to the
caller after each attempt, so a crash between the provider returning and
QC finishing still leaves the record that the call was made — which is
what lets resume reuse the audio instead of buying it again.

**Nothing rejected is ever delivered.** Budget exhaustion produces a
failure, not the best of a bad set. The only fallback that exists picks
among *eligible* candidates.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from luber_inference_qc.candidate import (
    CallAttribution,
    CandidateGeneration,
    CandidateStatus,
)
from luber_inference_qc.checks import RequestExpectation
from luber_inference_qc.detectors import VocalPresenceDetector
from luber_inference_qc.engine import judge
from luber_inference_qc.findings import Finding, QCFinding, Severity
from luber_inference_qc.identity import derive_seed
from luber_inference_qc.measurement import MeasurementCache
from luber_inference_qc.planner import AdaptiveRetryPlanner, RetryDecision
from luber_inference_qc.policy import Budget, CandidatePolicy
from luber_inference_qc.selector import select
from luber_inference_qc.trace import Outcome, QCTrace
from luber_inference_qc.workspace import CandidateWorkspace

logger = logging.getLogger(__name__)


class Event:
    """Structured event names, so a log line can be counted.

    Free-form log text is unsearchable at the point somebody needs to
    ask "how often did we retry last week".
    """

    CANDIDATE_STARTED = "CANDIDATE_STARTED"
    CANDIDATE_GENERATED = "CANDIDATE_GENERATED"
    CANDIDATE_RECOVERED = "CANDIDATE_RECOVERED"
    QC_STARTED = "QC_STARTED"
    QC_COMPLETED = "QC_COMPLETED"
    CANDIDATE_REJECTED = "CANDIDATE_REJECTED"
    RETRY_PLANNED = "RETRY_PLANNED"
    RETRY_STARTED = "RETRY_STARTED"
    CANDIDATE_SELECTED = "CANDIDATE_SELECTED"
    RETRY_EXHAUSTED = "RETRY_EXHAUSTED"
    PROVIDER_FAILED = "PROVIDER_FAILED"


class ProviderCallFailed(Exception):
    """A provider call that produced no audio.

    ``retryable`` is the caller's judgement, not this module's: only the
    generation client knows which of its error codes mean "try again"
    and which mean "the configuration is wrong". Getting that backwards
    is how a misconfiguration burns a whole budget.
    """

    def __init__(self, message: str, *, retryable: bool, error_code: str | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.error_code = error_code


class GeneratedAudio(Protocol):
    """What a provider call returns, as far as the controller cares."""

    audio_path: Path
    seed_used: int | None


#: Produce audio for one attempt. The seed is a request, not a promise —
#: a provider free to choose its own is passed ``None`` and reports what
#: it used.
GenerateCallable = Callable[[int | None], Awaitable[Any]]

#: Called after every attempt with the trace so far, so the caller can
#: persist it — and awaited, because persisting means a database write.
#: The whole point is that the record survives the process, so a sink
#: that could not do IO would not be one.
TraceSink = Callable[[QCTrace, Budget], Awaitable[None]]

EventSink = Callable[[str, dict[str, Any]], None]


@dataclass
class ControllerResult:
    """What the candidate phase produced."""

    trace: QCTrace
    budget: Budget
    workspace: CandidateWorkspace
    winner: CandidateGeneration | None = None
    #: The provider result for the winner, for the caller to post-process.
    winner_result: Any | None = None
    winner_path: Path | None = None
    #: Set when nothing was selected. The caller maps it to an error code.
    failure_finding: str | None = None
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def selected(self) -> bool:
        return self.winner is not None


class CandidateGenerationController:
    """Runs one generation's candidates to a decision."""

    def __init__(
        self,
        *,
        policy: CandidatePolicy,
        workspace: CandidateWorkspace,
        detector: VocalPresenceDetector | None = None,
        cache: MeasurementCache | None = None,
        on_event: EventSink | None = None,
        on_trace: TraceSink | None = None,
    ) -> None:
        self.policy = policy
        self.workspace = workspace
        self.detector = detector
        self.cache = cache or MeasurementCache()
        self.planner = AdaptiveRetryPlanner(policy)
        self._on_event = on_event
        self._on_trace = on_trace
        self._results: dict[str, Any] = {}

    def _emit(self, name: str, **payload: Any) -> None:
        if self._on_event is not None:
            self._on_event(name, payload)
        logger.info(name.lower(), extra={"event": name, **payload})

    async def _persist(self, trace: QCTrace, budget: Budget) -> None:
        if self._on_trace is not None:
            await self._on_trace(trace, budget)

    async def run(
        self,
        *,
        generation_id: str,
        request_sha256: str,
        expectation: RequestExpectation,
        generate: GenerateCallable,
        base_seed: int | None = None,
        resume_from: list[dict[str, Any]] | None = None,
    ) -> ControllerResult:
        """Generate, judge and select. Returns whichever candidate won.

        ``resume_from`` is a previous run's recorded attempts. Each one
        whose audio is still in the workspace and still matches its
        digest is reused rather than regenerated — the case this exists
        for is a worker killed after an expensive call and before the
        cheap check that follows it.
        """
        budget = Budget(policy=self.policy)
        trace = QCTrace(
            generation_id=generation_id,
            request_sha256=request_sha256,
            policy=self.policy,
            base_seed=base_seed,
        )
        started = time.monotonic()

        candidates: list[CandidateGeneration] = []
        next_seed = base_seed
        attempt_index = 0
        attribution = CallAttribution.USER_REQUEST.value
        retry_reason: str | None = None
        parent_candidate_id: str | None = None
        recovered = self._recoverable(resume_from or [])

        while True:
            # The first round may ask for several candidates at once;
            # every round after it produces exactly one.
            wanted = self.policy.initial_candidate_count if attempt_index == 0 else 1

            for _ in range(wanted):
                # `can_call`, not `exhausted`: the retry-round ceiling was
                # already checked by the planner one step earlier, and
                # the round it approved has been counted. Re-checking it
                # here would refuse that round's call.
                if budget.can_call() is not None and attempt_index > 0:
                    break

                candidate = CandidateGeneration(
                    candidate_id=f"cand_{generation_id[:8]}_{attempt_index:02d}",
                    generation_id=generation_id,
                    attempt_index=attempt_index,
                    request_sha256=request_sha256,
                    attribution=attribution,
                    seed=next_seed,
                    retry_reason=retry_reason,
                    parent_candidate_id=parent_candidate_id,
                )
                candidates.append(candidate)
                trace.add(candidate)

                await self._produce(
                    candidate=candidate,
                    generate=generate,
                    budget=budget,
                    recovered=recovered,
                    trace=trace,
                )
                await self._judge(candidate, expectation, trace, budget)

                attempt_index += 1
                next_seed = derive_seed(base_seed, attempt_index, request_sha256)

            # Selection runs on everything measured so far. A healthy
            # first candidate ends the loop here, having cost one call.
            eligible = [item for item in candidates if item.eligible]
            if eligible:
                break

            last = candidates[-1]
            plan = self.planner.plan(
                candidate=last,
                history=candidates,
                budget=budget,
                base_seed=base_seed,
                request_sha256=request_sha256,
            )
            self._emit(
                Event.RETRY_PLANNED,
                generation_id=generation_id,
                decision=plan.decision,
                reason=plan.reason,
            )
            if not plan.should_retry:
                trace.outcome = (
                    Outcome.RETRY_EXHAUSTED
                    if budget.exhausted() is not None
                    else Outcome.ALL_CANDIDATES_REJECTED
                )
                trace.outcome_detail = plan.reason
                self._emit(
                    Event.RETRY_EXHAUSTED,
                    generation_id=generation_id,
                    reason=plan.reason,
                    provider_calls=budget.provider_calls_used,
                )
                break

            budget.retry_rounds += 1
            attribution = CallAttribution.QUALITY_RETRY.value
            retry_reason = plan.reason
            parent_candidate_id = last.candidate_id
            next_seed = (
                plan.next_seed
                if plan.decision == RetryDecision.RETRY_SAME_REQUEST_NEW_SEED.value
                else base_seed
            )
            attempt_index = plan.next_attempt_index or attempt_index
            self._emit(
                Event.RETRY_STARTED,
                generation_id=generation_id,
                attempt_index=attempt_index,
                seed=next_seed,
            )

        return await self._finish(
            trace=trace,
            budget=budget,
            candidates=candidates,
            generation_id=generation_id,
            started=started,
        )

    # ── one attempt ──────────────────────────────────────────────────
    def _recoverable(self, attempts: list[dict[str, Any]]) -> dict[int, str]:
        """Attempt index → recorded digest, for the ones worth looking for."""
        return {
            int(item["attempt_index"]): str(item["raw_sha256"])
            for item in attempts
            if item.get("attempt_index") is not None and item.get("raw_sha256")
        }

    async def _produce(
        self,
        *,
        candidate: CandidateGeneration,
        generate: GenerateCallable,
        budget: Budget,
        recovered: dict[int, str],
        trace: QCTrace,
    ) -> None:
        """Get audio for one attempt, from the workspace or the provider."""
        expected = recovered.get(candidate.attempt_index)
        if expected is not None:
            path = self.workspace.recover(candidate.attempt_index, expected)
            if path is not None:
                candidate.audio_path = path
                candidate.raw_sha256 = expected
                candidate.status = CandidateStatus.GENERATED.value
                self._emit(
                    Event.CANDIDATE_RECOVERED,
                    generation_id=candidate.generation_id,
                    candidate_id=candidate.candidate_id,
                    attempt_index=candidate.attempt_index,
                )
                await self._persist(trace, budget)
                return

        self._emit(
            Event.CANDIDATE_STARTED,
            generation_id=candidate.generation_id,
            candidate_id=candidate.candidate_id,
            attempt_index=candidate.attempt_index,
            seed=candidate.seed,
            attribution=candidate.attribution,
        )
        call_started = time.monotonic()
        try:
            result = await generate(candidate.seed)
        except ProviderCallFailed as exc:
            budget.record_call(produced_candidate=False)
            candidate.provider_seconds = time.monotonic() - call_started
            candidate.status = CandidateStatus.FAILED.value
            candidate.provider_error_code = exc.error_code
            code = self._provider_finding(exc)
            candidate.findings = [
                QCFinding(
                    code=code.value,
                    severity=Severity.CRITICAL.value,
                    detail=str(exc),
                    metric="provider",
                    evidence={"error_code": exc.error_code} if exc.error_code else {},
                )
            ]
            self._emit(
                Event.PROVIDER_FAILED,
                generation_id=candidate.generation_id,
                candidate_id=candidate.candidate_id,
                retryable=exc.retryable,
                error_code=exc.error_code,
            )
            await self._persist(trace, budget)
            return

        budget.record_call(produced_candidate=True)
        candidate.provider_seconds = time.monotonic() - call_started

        path, digest = self.workspace.adopt(Path(result.audio_path), candidate.attempt_index)
        candidate.audio_path = path
        candidate.raw_sha256 = digest
        candidate.provider_request_sha256 = getattr(result, "request_sha256", None)
        # A provider that chose its own seed reports it; recording what
        # was used rather than what was asked for is what makes a trace
        # reproducible.
        if getattr(result, "seed_used", None) is not None:
            candidate.seed = result.seed_used
        candidate.status = CandidateStatus.GENERATED.value
        # Held beside the candidate rather than on it: the entity is
        # serialised into a durable trace, and a provider's result object
        # carries a local path and whatever else that provider felt like
        # returning. Neither belongs in a record read months later.
        self._results[candidate.candidate_id] = result

        self._emit(
            Event.CANDIDATE_GENERATED,
            generation_id=candidate.generation_id,
            candidate_id=candidate.candidate_id,
            seed=candidate.seed,
            provider_seconds=round(candidate.provider_seconds, 3),
        )
        await self._persist(trace, budget)

    @staticmethod
    def _provider_finding(exc: ProviderCallFailed) -> Finding:
        """Which provider failure this was.

        The caller's error code decides, because the caller is the one
        that knows. Matching on the message text is the fallback rather
        than the rule: "the request timed out" and "read timeout" are the
        same failure, and only one of them contains the word a substring
        match looks for.
        """
        code = (exc.error_code or "").upper()
        if "TIMEOUT" in code or "timeout" in str(exc).lower() or "timed out" in str(exc).lower():
            return Finding.PROVIDER_TIMEOUT
        return Finding.PROVIDER_ERROR if exc.retryable else Finding.PROVIDER_MISCONFIGURED

    async def _judge(
        self,
        candidate: CandidateGeneration,
        expectation: RequestExpectation,
        trace: QCTrace,
        budget: Budget,
    ) -> None:
        if candidate.status == CandidateStatus.FAILED.value:
            return

        self._emit(
            Event.QC_STARTED,
            generation_id=candidate.generation_id,
            candidate_id=candidate.candidate_id,
        )
        assert candidate.audio_path is not None
        judge(
            candidate,
            candidate.audio_path,
            expectation,
            detector=self.detector,
            cache=self.cache,
            sha256=candidate.raw_sha256,
        )
        self._emit(
            Event.QC_COMPLETED,
            generation_id=candidate.generation_id,
            candidate_id=candidate.candidate_id,
            status=candidate.status,
            findings=sorted(candidate.finding_codes()),
            qc_seconds=round(candidate.qc_seconds or 0.0, 3),
        )
        if not candidate.eligible:
            self._emit(
                Event.CANDIDATE_REJECTED,
                generation_id=candidate.generation_id,
                candidate_id=candidate.candidate_id,
                findings=sorted(item.code for item in candidate.critical_findings),
            )
        await self._persist(trace, budget)

    # ── the end ──────────────────────────────────────────────────────
    async def _finish(
        self,
        *,
        trace: QCTrace,
        budget: Budget,
        candidates: list[CandidateGeneration],
        generation_id: str,
        started: float,
    ) -> ControllerResult:
        budget.elapsed_seconds = time.monotonic() - started
        trace.timings = {
            "candidate_phase_seconds": budget.elapsed_seconds,
            "provider_seconds": sum(item.provider_seconds or 0.0 for item in candidates),
            "qc_seconds": sum(item.qc_seconds or 0.0 for item in candidates),
        }

        eligible = [item for item in candidates if item.eligible]
        if not eligible:
            if trace.outcome == Outcome.SELECTED:
                trace.outcome = Outcome.ALL_CANDIDATES_REJECTED
                trace.outcome_detail = "no candidate was eligible for delivery"
            await self._persist(trace, budget)
            return ControllerResult(
                trace=trace,
                budget=budget,
                workspace=self.workspace,
                failure_finding=self._dominant_failure(candidates),
                timings=trace.timings,
            )

        # `select` ranks and stamps every candidate, including the ones
        # that lost, so each carries its own reason.
        selection = select(candidates)
        trace.selection = selection
        trace.outcome = Outcome.SELECTED
        winner = next(
            item for item in candidates if item.candidate_id == selection.winner_candidate_id
        )
        trace.outcome_detail = selection.reasons.get(winner.candidate_id, "")

        self._emit(
            Event.CANDIDATE_SELECTED,
            generation_id=generation_id,
            candidate_id=winner.candidate_id,
            attempt_index=winner.attempt_index,
            provider_calls=budget.provider_calls_used,
            reason=trace.outcome_detail,
        )
        await self._persist(trace, budget)

        return ControllerResult(
            trace=trace,
            budget=budget,
            workspace=self.workspace,
            winner=winner,
            winner_result=self._results.get(winner.candidate_id),
            winner_path=winner.audio_path,
            timings=trace.timings,
        )

    @staticmethod
    def _dominant_failure(candidates: list[CandidateGeneration]) -> str | None:
        """The critical finding that best explains the failure.

        The last attempt's, because that is the one the operator will
        look at first and the one the retry chain ended on.
        """
        for candidate in reversed(candidates):
            critical = candidate.critical_findings
            if critical:
                return critical[0].code
        return None


__all__ = [
    "CandidateGenerationController",
    "ControllerResult",
    "Event",
    "GeneratedAudio",
    "ProviderCallFailed",
]
