"""GenerationService — orchestrates the generation lifecycle.

Runs inside the generation worker in production (the API process never
executes providers; it only persists and enqueues). The service walks
the full status lifecycle even though the mock provider is instant, so
the architecture already matches the real GPU path:

    QUEUED → STARTING → GENERATING → POST_PROCESSING → UPLOADING →
    COMPLETED | FAILED

All failures are translated to standard :class:`ErrorCode` values;
raw exception strings never reach clients (they are stored in
``error_message`` for operators only).
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any
from uuid import UUID

from luber_audio_utils import (
    AudioProcessingError,
    AudioStorage,
    AudioStorageError,
    WavValidationError,
    finished_master_storage_key,
    inspect_wav,
    probe_audio,
)
from luber_database import GenerationRepository
from luber_database.models.generation import Generation
from luber_generation_client.audio_to_audio import (
    AudioToAudioProvider,
    AudioToAudioRequest,
)
from luber_generation_client.candidates import (
    as_controller_failure,
    expectation_for,
    expectation_for_edit,
    is_retryable_task,
    policy_for_generation,
)
from luber_generation_client.editing import (
    AudioEditingProvider,
    AudioEditKind,
    AudioEditRequest,
)
from luber_generation_client.errors import GenerationProviderError
from luber_generation_client.postprocess import FinishingRecord, produce_delivery_assets
from luber_generation_client.provider import (
    GenerationRequest,
    GenerationResult,
    MusicGenerationProvider,
    ReferenceAudioInput,
)
from luber_generation_client.resilience import (
    GateResult,
    ResilienceGate,
    Stopwatch,
    failure_from_category,
    needs_for,
    refusal_to_failure,
)
from luber_inference_qc import Budget, CandidatePolicy, QCTrace, RequestExpectation, request_digest
from luber_inference_qc.controller import CandidateGenerationController, ControllerResult
from luber_inference_qc.workspace import CandidateWorkspace
from luber_schemas import (
    AssetType,
    EditKind,
    ErrorCode,
    GenerationStatus,
    VocalGender,
    select_raw_master,
)

logger = logging.getLogger(__name__)


def _finishing_trace_json(record: FinishingRecord) -> str:
    """Serialise the finishing decision for the durable trace.

    The plan is kept whole rather than summarised: it is the only record
    of *why* a master sounds the way it does, and a summary would have to
    guess in advance which field a future question needs.
    """
    payload: dict[str, object] = {
        "outcome": record.outcome.value,
        "finishing_version": record.finishing_version,
        "source_sha256": record.source_sha256,
    }
    if record.plan is not None:
        payload["plan"] = record.plan
    if record.verdict is not None:
        # What the engine measured in its own output and concluded. Kept
        # whole for the same reason as the plan: a rejection is only
        # actionable if the numbers behind it survive.
        payload["verdict"] = record.verdict
    if record.error is not None:
        payload["error"] = record.error
    return json.dumps(payload, sort_keys=True)


def _last_failure_detail(result: ControllerResult) -> str | None:
    """The most recent critical finding's own words."""
    for candidate in reversed(result.trace.candidates):
        for finding in candidate.critical_findings:
            return finding.detail
    return None


class QualityControlFailed(Exception):
    """Every candidate was measured and none could be delivered.

    Carries the controller result so the caller can pick the right error
    code — "nothing further would have helped" and "nothing further was
    tried" are different facts and an operator tuning budgets needs to
    tell them apart.
    """

    def __init__(self, result: ControllerResult) -> None:
        # The message an operator reads is the last thing that actually
        # went wrong, not the loop's own summary of having given up.
        # "the provider crashed" is actionable; "two consecutive
        # failures" is a description of this module.
        super().__init__(
            _last_failure_detail(result)
            or result.trace.outcome_detail
            or "no candidate was eligible"
        )
        self.result = result

    @property
    def error_code(self) -> ErrorCode:
        """The code that describes what actually went wrong.

        A provider that never produced audio did not fail a quality
        check — it failed to answer. Reporting that as
        QUALITY_CHECK_FAILED would tell an operator the model is
        producing bad songs when the truth is that it is unreachable, or
        timing out, or misconfigured, and each of those has a different
        fix.
        """
        last = self.result.trace.candidates[-1] if self.result.trace.candidates else None
        if last is not None and any(
            finding.code.startswith("PROVIDER_") for finding in last.critical_findings
        ):
            if last.provider_error_code:
                try:
                    return ErrorCode(last.provider_error_code)
                except ValueError:
                    return ErrorCode.UNKNOWN_GENERATION_ERROR
            # The provider raised something this package does not
            # recognise. It still never produced audio, so it is not a
            # quality failure.
            return ErrorCode.UNKNOWN_GENERATION_ERROR
        return (
            ErrorCode.QUALITY_RETRY_EXHAUSTED
            if self.result.trace.exhausted
            else ErrorCode.QUALITY_CHECK_FAILED
        )


class GenerationService:
    def __init__(
        self,
        repository: GenerationRepository,
        provider: MusicGenerationProvider,
        storage: AudioStorage,
        *,
        qc_policy: str = "STANDARD",
        qc_enabled: bool = True,
        candidate_workspace_dir: str = "data/generation-candidates",
        resilience: ResilienceGate | None = None,
    ) -> None:
        self._repository = repository
        self._provider = provider
        self._storage = storage
        # Phase 31. ``None`` restores the pre-Phase-31 path exactly: the
        # configured provider is called directly, no circuit is
        # consulted and no routing decision is recorded. A deployment
        # that has not enabled resilience behaves as it did.
        self._resilience = resilience
        # Phase 29. Off restores the pre-Phase-29 behaviour exactly: one
        # provider call, no candidate QC, no retry. It exists so a QC
        # regression that rejects good output can be switched off and
        # then fixed, rather than fixed under pressure.
        self._qc_enabled = qc_enabled
        self._qc_policy = qc_policy
        self._candidate_workspace_dir = Path(candidate_workspace_dir)

    async def _retract_stale_finished_master(
        self, repo: GenerationRepository, generation_id: UUID
    ) -> None:
        """Drop a finished master this run did not reproduce.

        The row goes first and the object second: a row without an object
        is a broken download, while an object without a row is unreferenced
        bytes that the generation-wide delete already cleans up.
        """
        removed = await repo.delete_audio_asset(
            generation_id, asset_type=AssetType.FINISHED_MASTER.value
        )
        if not removed:
            return
        logger.info(
            "retracted a finished master this run did not reproduce",
            extra={"generation_id": str(generation_id)},
        )
        await self._storage.delete(finished_master_storage_key(generation_id))

    async def _resolve_reference_audio(
        self, generation: Generation, stack: AsyncExitStack
    ) -> ReferenceAudioInput | None:
        """Materialise the chosen reference track for the provider.

        Refuses rather than degrades at every step. A provider that
        cannot condition on a reference, a reference row that has gone,
        an object missing from storage — each ends the generation with
        ``REFERENCE_AUDIO_UNAVAILABLE``. Generating without the reference
        would produce a song the user did not ask for while reporting
        success, which is the one outcome worse than failing.
        """
        if generation.reference_audio_id is None:
            return None

        if not self._provider.supports_reference_audio:
            raise GenerationProviderError(
                "configured provider cannot condition on reference audio",
                error_code=ErrorCode.REFERENCE_AUDIO_UNAVAILABLE,
            )

        reference = await self._repository.get_reference_audio(generation.reference_audio_id)
        if reference is None:
            raise GenerationProviderError(
                "the reference track for this generation no longer exists",
                error_code=ErrorCode.REFERENCE_AUDIO_UNAVAILABLE,
            )

        local = self._storage.local_path(reference.storage_key)
        if local is None or not local.is_file():
            try:
                data = await self._storage.open(reference.storage_key)
            except AudioStorageError as exc:
                raise GenerationProviderError(
                    "the reference track for this generation could not be read",
                    error_code=ErrorCode.REFERENCE_AUDIO_UNAVAILABLE,
                ) from exc
            handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            stack.callback(lambda: Path(handle.name).unlink(missing_ok=True))
            with handle:
                handle.write(data)
            local = Path(handle.name)

        return ReferenceAudioInput(
            reference_id=reference.id,
            audio_path=local,
            duration_seconds=reference.duration_seconds,
            sha256=reference.sha256,
        )

    async def _resolve_source_audio(self, parent: Generation, stack: AsyncExitStack) -> Path:
        """Get the parent's raw master onto local disk for the provider.

        Uses the storage backend's local path when there is one, and
        otherwise materialises the object into a temp file that the
        caller's exit stack removes. Object storage has no path, so the
        edit path cannot assume one exists.
        """
        # The *raw* master, deliberately: feeding a finished master back
        # into the model would stack finishing corrections across
        # generations, and the child gets its own finishing pass anyway.
        master = select_raw_master(list(parent.audio_assets))
        if master is None:
            raise GenerationProviderError(
                "source generation has no master audio",
                error_code=ErrorCode.INVALID_AUDIO,
            )

        local = self._storage.local_path(master.storage_key)
        if local is not None and local.is_file():
            return local

        data = await self._storage.open(master.storage_key)
        handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        stack.callback(lambda: Path(handle.name).unlink(missing_ok=True))
        with handle:
            handle.write(data)
        return Path(handle.name)

    async def _to_edit_request(
        self, generation: Generation, stack: AsyncExitStack
    ) -> AudioEditRequest:
        """Build the engine-level edit for an audio-edit generation.

        The regenerated range comes from the *measured* parent audio, not
        from any stored or requested duration: the boundary decides which
        latent frames are preserved, so it has to match the file the
        engine is actually given. A stored duration that drifted by a
        frame would silently move the seam.
        """
        if generation.parent_generation_id is None:
            raise GenerationProviderError(
                "audio edit has no parent generation",
                error_code=ErrorCode.UNKNOWN_GENERATION_ERROR,
            )
        parent = await self._repository.get_generation(generation.parent_generation_id)
        if parent is None:
            raise GenerationProviderError(
                "source generation no longer exists",
                error_code=ErrorCode.UNKNOWN_GENERATION_ERROR,
            )

        source_path = await self._resolve_source_audio(parent, stack)
        measured = probe_audio(source_path).duration_seconds
        stored_start = generation.edit_start_seconds or 0.0
        stored_end = generation.edit_end_seconds or 0.0
        if stored_end <= stored_start:
            raise GenerationProviderError(
                "audio edit has an empty range",
                error_code=ErrorCode.UNKNOWN_GENERATION_ERROR,
            )

        kind = EditKind(generation.edit_kind or EditKind.EXTEND.value)
        if kind is EditKind.EXTEND:
            # The stored start came from a duration measured at submission
            # time. Re-anchoring to the audio actually being uploaded is
            # what keeps the seam at the true end of the recording; a
            # drifted stored value would move it.
            start_seconds = measured
            end_seconds = measured + (stored_end - stored_start)
        else:
            # An interior range is the user's own choice of times, so it
            # is used as given. Only the far edge is re-checked against
            # the real audio, since a range past the end of the file
            # would silently become an extension.
            start_seconds = stored_start
            end_seconds = min(stored_end, measured)
            if end_seconds <= start_seconds:
                raise GenerationProviderError(
                    "replacement range falls outside the source audio",
                    error_code=ErrorCode.UNKNOWN_GENERATION_ERROR,
                )

        return AudioEditRequest(
            # One engine primitive serves both: regenerate this range,
            # preserve the rest. The product-level difference stays here.
            kind=AudioEditKind.REGENERATE_RANGE,
            source_audio=source_path,
            source_duration_seconds=measured,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            # Conditioning is inherited from the child row, which the API
            # populated from the parent — so an edit is described by the
            # same prompt and lyrics that produced the source.
            title=generation.title,
            prompt=generation.prompt,
            lyrics=generation.lyrics,
            vocal_gender=VocalGender(generation.vocal_gender),
            language=generation.language,
            instrumental=generation.instrumental,
            seed=generation.seed,
            bpm=generation.bpm,
            key_scale=generation.key_scale,
            time_signature=generation.time_signature,
        )

    async def _to_cover_request(
        self, generation: Generation, stack: AsyncExitStack
    ) -> AudioToAudioRequest:
        """Build the engine-neutral request for a source-conditioned cover.

        The child row already carries the target description the user
        chose and the adherence the API mapped from their chosen preset,
        so nothing here reinterprets either. The source duration is
        measured from the audio actually being uploaded.
        """
        if generation.parent_generation_id is None:
            raise GenerationProviderError(
                "cover has no source generation",
                error_code=ErrorCode.UNKNOWN_GENERATION_ERROR,
            )
        parent = await self._repository.get_generation(generation.parent_generation_id)
        if parent is None:
            raise GenerationProviderError(
                "source generation no longer exists",
                error_code=ErrorCode.UNKNOWN_GENERATION_ERROR,
            )

        source_path = await self._resolve_source_audio(parent, stack)
        measured = probe_audio(source_path).duration_seconds
        # The provider validates this against its own measured band, so an
        # out-of-range value fails rather than being quietly clamped into
        # a setting nobody calibrated.
        adherence = generation.source_adherence
        if adherence is None:
            raise GenerationProviderError(
                "cover has no recorded source adherence",
                error_code=ErrorCode.UNKNOWN_GENERATION_ERROR,
            )

        return AudioToAudioRequest(
            source_audio=source_path,
            source_duration_seconds=measured,
            title=generation.title,
            prompt=generation.prompt,
            lyrics=generation.lyrics,
            vocal_gender=VocalGender(generation.vocal_gender),
            language=generation.language,
            instrumental=generation.instrumental,
            seed=generation.seed,
            bpm=generation.bpm,
            key_scale=generation.key_scale,
            time_signature=generation.time_signature,
            source_adherence=adherence,
        )

    async def _record_cover_trace(self, generation_id: UUID, request: AudioToAudioRequest) -> None:
        """Persist what the cover provider is about to send. Best-effort."""
        try:
            provider = self._provider
            if not isinstance(provider, AudioToAudioProvider):
                return
            trace = provider.describe_audio_to_audio(request)
            if not trace:
                return
            await self._repository.record_request_trace(
                generation_id, trace=json.dumps(trace, ensure_ascii=False)
            )
        except Exception:
            logger.warning(
                "could not record provider cover trace",
                extra={"generation_id": str(generation_id)},
                exc_info=True,
            )

    def _to_provider_request(self, generation: Generation) -> GenerationRequest:
        return GenerationRequest(
            title=generation.title,
            prompt=generation.prompt,
            lyrics=generation.lyrics,
            vocal_gender=VocalGender(generation.vocal_gender),
            duration_seconds=generation.duration_requested,
            seed=generation.seed,
            language=generation.language,
            instrumental=generation.instrumental,
            bpm=generation.bpm,
            key_scale=generation.key_scale,
            time_signature=generation.time_signature,
        )

    async def _record_trace(self, generation_id: UUID, request: GenerationRequest) -> None:
        """Persist what the provider is about to send.

        Best-effort by design: a trace is diagnostic metadata, so
        failing to write it must never fail the generation the user
        actually asked for.
        """
        try:
            trace = self._provider.describe_request(request)
            if not trace:
                return
            await self._repository.record_request_trace(
                generation_id, trace=json.dumps(trace, ensure_ascii=False)
            )
        except Exception:
            logger.warning(
                "could not record provider request trace",
                extra={"generation_id": str(generation_id)},
                exc_info=True,
            )

    async def _record_edit_trace(self, generation_id: UUID, request: AudioEditRequest) -> None:
        """Persist what the editing provider is about to send.

        Same best-effort contract as the generation trace. The provider's
        ``describe_edit`` is responsible for keeping the source's path and
        bytes out of it.
        """
        try:
            provider = self._provider
            if not isinstance(provider, AudioEditingProvider):
                return
            trace = provider.describe_edit(request)
            if not trace:
                return
            await self._repository.record_request_trace(
                generation_id, trace=json.dumps(trace, ensure_ascii=False)
            )
        except Exception:
            logger.warning(
                "could not record provider edit trace",
                extra={"generation_id": str(generation_id)},
                exc_info=True,
            )

    # ── the candidate phase ──────────────────────────────────────────
    #
    # The one place Phase 29 inserts itself. Everything above is the
    # request; everything below is delivery. This is where a single
    # provider call became a loop that may make more than one.

    def _policy_for(self, generation: Generation) -> CandidatePolicy:
        return policy_for_generation(
            self._qc_policy,
            retryable_task=is_retryable_task(edit_kind=generation.edit_kind),
        )

    def _workspace_for(self, generation_id: UUID) -> CandidateWorkspace:
        return CandidateWorkspace(self._candidate_workspace_dir, str(generation_id))

    async def _produce_audio(
        self, generation: Generation, stack: AsyncExitStack
    ) -> tuple[GenerationResult | None, ControllerResult | None]:
        """Get deliverable audio, through the candidate controller or not.

        Returns the provider result and, when QC ran, the controller's
        outcome. With QC disabled the second is ``None`` and the
        behaviour is exactly what it was before Phase 29 — one call, no
        measurement, no retry.
        """
        generation_id = generation.id
        repo = self._repository

        # The request is built and traced before anything runs, exactly
        # as it was: a failed generation stays as inspectable as a
        # successful one, and the trace must not depend on QC.
        call, expectation, digest = await self._prepare_call(generation, stack)
        await repo.update_status(generation_id, GenerationStatus.GENERATING.value)

        if not self._qc_enabled:
            provider = self._provider
            if self._resilience is not None:
                # Even without QC the circuit still gates: refusing fast
                # when a provider is down is the part that has value with
                # a single provider, and it does not depend on candidates.
                decision = await self._resilience.route(needs_for(generation), attempted=())
                if not decision.permitted:
                    raise refusal_to_failure(decision)
                assert decision.selected is not None
                provider = self._resilience.provider_for(decision.selected)
                watch = Stopwatch()
                try:
                    result = await call(provider, generation.seed)
                except BaseException as exc:
                    await self._resilience.record_failure(
                        decision, exc, latency_seconds=watch.elapsed(), attempt=0
                    )
                    raise
                await self._resilience.record_success(
                    decision, latency_seconds=watch.elapsed(), attempt=0
                )
                return result, None
            return await call(provider, generation.seed), None

        policy = self._policy_for(generation)
        workspace = self._workspace_for(generation_id)

        # Declared before `persist`, which reads it. The trace is written
        # after every attempt, so the routing record has to exist by the
        # time the first one lands.
        needs = needs_for(generation)
        gate = GateResult()
        attempt_index = 0

        async def persist(trace: QCTrace, budget: Budget) -> None:
            # After every attempt, not once at the end. The record of an
            # expensive call has to survive the process that made it, or
            # a resumed job pays for it twice.
            if self._resilience is not None:
                # Routing rides along in the same document, so an
                # operator asking "why did this go where it went" reads
                # one record rather than correlating two.
                trace.resilience = gate.to_dict()
            await repo.record_inference_qc_trace(generation_id, trace=trace.to_json(budget))

        controller = CandidateGenerationController(
            policy=policy,
            workspace=workspace,
            on_trace=persist,
        )

        async def guarded(seed: int | None) -> GenerationResult:
            """One attempt, routed and recorded.

            Phase 31 lives entirely inside this closure. It adds no loop
            of its own: the controller decides *whether* there is another
            attempt and this decides *where* that attempt goes, so the
            budget stays in one place and no layer multiplies another.
            """
            nonlocal attempt_index
            index = attempt_index
            attempt_index += 1

            if self._resilience is None:
                try:
                    return await call(self._provider, seed)
                except Exception as exc:
                    raise as_controller_failure(exc) from exc

            decision = await self._resilience.route(
                needs, attempted=tuple(gate.providers_attempted)
            )
            gate.decisions.append(decision.to_dict())
            if not decision.permitted:
                # A typed refusal, not a silent substitution and not a
                # call to a provider we already know will not answer.
                raise refusal_to_failure(decision)

            assert decision.selected is not None
            if decision.selected not in gate.providers_attempted:
                gate.providers_attempted.append(decision.selected)
            if decision.fallback_used:
                gate.failovers += 1

            provider = self._resilience.provider_for(decision.selected)
            watch = Stopwatch()
            try:
                result = await call(provider, seed)
            except asyncio.CancelledError:
                # The user withdrew. Nothing was learned about the
                # provider, so the probe slot goes back and no evidence
                # is recorded — a cancellation must never look like a
                # provider failure.
                await self._resilience.abandon(decision)
                raise
            except Exception as exc:
                record, category = await self._resilience.record_failure(
                    decision, exc, latency_seconds=watch.elapsed(), attempt=index
                )
                gate.attempts.append(record)
                raise failure_from_category(category, exc) from exc

            gate.attempts.append(
                await self._resilience.record_success(
                    decision, latency_seconds=watch.elapsed(), attempt=index
                )
            )
            return result

        outcome = await controller.run(
            generation_id=str(generation_id),
            request_sha256=digest,
            expectation=expectation,
            generate=guarded,
            base_seed=generation.seed,
            resume_from=self._prior_attempts(generation),
        )

        if not outcome.selected:
            return None, outcome

        # The winner's audio is the workspace copy, not the provider's
        # own path: the provider may clean its up, and this one is the
        # file that was hashed and measured.
        assert outcome.winner_result is not None
        assert outcome.winner_path is not None
        return (
            outcome.winner_result.model_copy(update={"audio_path": outcome.winner_path}),
            outcome,
        )

    @staticmethod
    def _prior_attempts(generation: Generation) -> list[dict[str, object]]:
        """Attempts a previous execution of this job already paid for.

        Read from the durable trace, because the in-memory state of the
        process that made them is gone. Each one whose audio is still in
        the workspace and still matches its digest is reused.
        """
        if not generation.inference_qc_trace:
            return []
        try:
            trace = json.loads(generation.inference_qc_trace)
        except json.JSONDecodeError:
            return []
        attempts = trace.get("attempts")
        return attempts if isinstance(attempts, list) else []

    async def _prepare_call(
        self, generation: Generation, stack: AsyncExitStack
    ) -> tuple[Callable[[Any, int | None], Awaitable[GenerationResult]], RequestExpectation, str]:
        """Build the provider call for this task, once.

        The request is constructed and traced here rather than per
        attempt: every candidate is an attempt at the *same* request, and
        rebuilding it each time would be the one place a retry could
        silently become a different question. That property is why the
        digest below stays constant across a failover — the semantic
        request did not change, only who is being asked.

        The returned callable takes the provider as an argument rather
        than capturing one. Phase 31 may answer successive attempts with
        different providers, and a closure that had captured the first
        would quietly keep using it.
        """
        generation_id = generation.id

        if generation.edit_kind == EditKind.COVER.value:
            # A cover is source-conditioned generation, not an edit.
            # Routing it through edit() would claim repaint's
            # preservation guarantee, and routing it through generate()
            # would drop the source entirely. Neither is a legal
            # fallback: if this provider cannot do it, the run fails.
            if not isinstance(self._provider, AudioToAudioProvider):
                raise GenerationProviderError(
                    "configured provider cannot generate from audio",
                    error_code=ErrorCode.MODEL_LOAD_FAILED,
                )
            cover_request = await self._to_cover_request(generation, stack)
            await self._record_cover_trace(generation_id, cover_request)

            async def run_cover(
                provider: MusicGenerationProvider, _seed: int | None
            ) -> GenerationResult:
                # The seed is ignored: a cover runs once, so there is
                # never a second attempt for a different one to vary.
                if not isinstance(provider, AudioToAudioProvider):
                    # The router refuses a provider that cannot do this,
                    # so reaching here means the two disagree. Failing is
                    # the only safe answer: the alternative is a cover
                    # that silently became a fresh generation.
                    raise GenerationProviderError(
                        "routed provider cannot generate from audio",
                        error_code=ErrorCode.MODEL_LOAD_FAILED,
                    )
                return await provider.create_from_audio(cover_request)

            return (
                run_cover,
                RequestExpectation(),
                request_digest(cover_request, extra={"task": "cover"}),
            )

        if generation.edit_kind is not None:
            # An audio edit must reach an editing provider or fail.
            # Falling back to generate() would quietly hand the user a
            # new, unrelated song in place of the edit they asked for —
            # the exact substitution this feature exists to rule out.
            if not isinstance(self._provider, AudioEditingProvider):
                raise GenerationProviderError(
                    "configured provider cannot edit audio",
                    error_code=ErrorCode.MODEL_LOAD_FAILED,
                )
            edit_request = await self._to_edit_request(generation, stack)
            await self._record_edit_trace(generation_id, edit_request)

            async def run_edit(
                provider: MusicGenerationProvider, _seed: int | None
            ) -> GenerationResult:
                # As above: an edit runs once.
                if not isinstance(provider, AudioEditingProvider):
                    raise GenerationProviderError(
                        "routed provider cannot edit audio",
                        error_code=ErrorCode.MODEL_LOAD_FAILED,
                    )
                return await provider.edit(edit_request)

            return (
                run_edit,
                expectation_for_edit(edit_request),
                request_digest(edit_request, extra={"task": "edit"}),
            )

        reference = await self._resolve_reference_audio(generation, stack)
        request = self._to_provider_request(generation).model_copy(
            update={"reference_audio": reference}
        )
        await self._record_trace(generation_id, request)

        async def run_generate(
            provider: MusicGenerationProvider, seed: int | None
        ) -> GenerationResult:
            # The seed is the only thing a retry changes. Everything
            # else — prompt, lyrics, duration, key, and the reference
            # track — is carried across untouched, including across a
            # failover to a different provider.
            return await provider.generate(request.model_copy(update={"seed": seed}))

        return (
            run_generate,
            expectation_for(request),
            request_digest(
                request,
                extra={
                    "task": "generate",
                    "reference_sha256": reference.sha256 if reference else None,
                },
            ),
        )

    async def execute(
        self, generation_id: UUID, *, worker_id: str | None = None
    ) -> GenerationStatus:
        """Run one generation job to a terminal state. Returns that state.

        Safe to invoke more than once for the same generation. The queue
        retries a job whose worker was cancelled mid-flight, and that
        retry can arrive after the work actually finished — the window
        between :meth:`mark_completed` and the queue recording success is
        small but real. Re-running then would replace a song the user has
        already been given with different audio, so a generation that
        already reached COMPLETED is left exactly as it is.
        """
        repo = self._repository
        generation = await repo.get_generation(generation_id)
        if generation is None:
            raise LookupError(f"generation not found: {generation_id}")

        if generation.status == GenerationStatus.COMPLETED.value:
            logger.info(
                "generation already completed; skipping duplicate execution",
                extra={"generation_id": str(generation_id), "worker_id": worker_id},
            )
            return GenerationStatus.COMPLETED

        job = await repo.get_latest_job(generation_id)
        if job is not None:
            await repo.mark_job_started(
                job.id, status=GenerationStatus.STARTING.value, worker_id=worker_id
            )

        try:
            await repo.mark_started(generation_id, status=GenerationStatus.STARTING.value)

            async with AsyncExitStack() as stack:
                result, qc_result = await self._produce_audio(generation, stack)

            if qc_result is not None and not qc_result.selected:
                # Every attempt was measured and none could be
                # delivered. Raising rather than returning keeps this on
                # the same path as any other generation failure, so the
                # row, the job and the log all get the same treatment.
                raise QualityControlFailed(qc_result)

            assert result is not None

            # POST_PROCESSING and UPLOADING both happen inside
            # produce_delivery_assets; the status is advanced around it
            # so a stuck run is attributable to the right stage.
            await repo.update_status(generation_id, GenerationStatus.POST_PROCESSING.value)
            # Structural check of the raw model output before spending
            # time transcoding it.
            inspect_wav(result.audio_path)

            await repo.update_status(generation_id, GenerationStatus.UPLOADING.value)
            produced = await produce_delivery_assets(
                generation_id, result.audio_path, self._storage
            )

            for asset in produced.assets:
                await repo.create_audio_asset(
                    generation_id,
                    asset_type=asset.asset_type.value,
                    format=asset.format,
                    mime_type=asset.mime_type,
                    file_extension=asset.file_extension,
                    sample_rate=asset.sample_rate,
                    bit_depth=asset.bit_depth,
                    bitrate=asset.bitrate,
                    channels=asset.channels,
                    duration=asset.duration,
                    storage_key=asset.storage_key,
                    sha256=asset.sha256,
                    file_size=asset.file_size,
                )
            if produced.finished is None:
                # A previous attempt may have produced a finished master
                # that this one did not — a different engine version, or
                # a failure this time. Leaving the old row in place would
                # keep it winning delivery selection while pointing at an
                # object no longer backed by this run's decisions.
                await self._retract_stale_finished_master(repo, generation_id)
            await repo.record_finishing_trace(
                generation_id, trace=_finishing_trace_json(produced.finishing)
            )
            if qc_result is not None:
                # What the finishing engine did, recorded on the QC trace
                # as well, so one document answers the whole question of
                # why this file was delivered.
                qc_result.trace.finishing_outcome = produced.finishing.outcome.value
                await repo.record_inference_qc_trace(
                    generation_id, trace=qc_result.trace.to_json(qc_result.budget)
                )
            # COMPLETED only after every required asset is stored and
            # recorded — a post-processing or upload failure raises above
            # and lands in the FAILED branch instead.
            await repo.mark_completed(
                generation_id,
                status=GenerationStatus.COMPLETED.value,
                duration_actual=produced.delivery_master.duration,
                provider=result.provider,
                model_name=result.model_name,
                model_version=result.model_version,
                seed=result.seed_used,
            )
            if job is not None:
                await repo.mark_job_finished(job.id, status=GenerationStatus.COMPLETED.value)
            # Delivery has read the winner and stored it. Candidate
            # audio exists only to be measured and, for one of them,
            # delivered — so it goes. A rejected candidate's bytes are
            # deliberately not retained; the trace records the digest of
            # what was discarded.
            self._workspace_for(generation_id).cleanup()
            logger.info(
                "generation completed",
                extra={
                    "generation_id": str(generation_id),
                    "worker_id": worker_id,
                    "model_version": result.model_version,
                },
            )
            return GenerationStatus.COMPLETED
        except asyncio.CancelledError:
            # Cancellation is a BaseException, so the handler below never
            # saw it: the worker stopping mid-generation left the row
            # claiming GENERATING with nothing running behind it, and no
            # operator or user could tell that apart from slow progress.
            # Record the interruption, then re-raise so the queue keeps
            # its own retry semantics — a retry calls mark_started and
            # moves the row straight back out of this state.
            await repo.mark_failed(
                generation_id,
                status=GenerationStatus.FAILED.value,
                error_code=ErrorCode.GENERATION_INTERRUPTED.value,
                error_message="generation was interrupted before it finished",
            )
            if job is not None:
                await repo.mark_job_finished(
                    job.id,
                    status=GenerationStatus.FAILED.value,
                    error_code=ErrorCode.GENERATION_INTERRUPTED.value,
                    error_message="generation was interrupted before it finished",
                )
            # The workspace is deliberately *not* cleaned here. An
            # interrupted run is the case the queue retries, and the
            # candidate that was already paid for is exactly what the
            # retry should reuse rather than buy again.
            logger.warning(
                "generation interrupted",
                extra={"generation_id": str(generation_id), "worker_id": worker_id},
            )
            raise
        except Exception as exc:
            error_code = self._translate_error(exc)
            await repo.mark_failed(
                generation_id,
                status=GenerationStatus.FAILED.value,
                error_code=error_code.value,
                error_message=str(exc),
            )
            if job is not None:
                await repo.mark_job_finished(
                    job.id,
                    status=GenerationStatus.FAILED.value,
                    error_code=error_code.value,
                    error_message=str(exc),
                )
            # Terminal, so nothing will read the candidates again.
            self._workspace_for(generation_id).cleanup()
            logger.exception(
                "generation failed",
                extra={
                    "generation_id": str(generation_id),
                    "worker_id": worker_id,
                    "error_code": error_code.value,
                },
            )
            return GenerationStatus.FAILED

    @staticmethod
    def _translate_error(exc: Exception) -> ErrorCode:
        if isinstance(exc, QualityControlFailed):
            # "Nothing further would have helped" and "nothing further
            # was tried" are different facts, and an operator tuning
            # budgets needs to tell them apart.
            return exc.error_code
        if isinstance(exc, GenerationProviderError):
            return exc.error_code
        if isinstance(exc, WavValidationError):
            return ErrorCode.INVALID_AUDIO
        if isinstance(exc, AudioProcessingError):
            return ErrorCode.ENCODING_FAILED
        if isinstance(exc, AudioStorageError):
            return ErrorCode.UPLOAD_FAILED
        return ErrorCode.UNKNOWN_GENERATION_ERROR
