"""ARQ worker entrypoint for the generation queue.

Run with: ``arq luber_generation_worker.worker.WorkerSettings``

The worker consumes ``generation_id``s enqueued by the API and drives
:class:`GenerationService` (provider execution never happens in the API
process). Phase 1 runs the mock provider; Phase 2 registers the real
ACE-Step provider behind the same boundary.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, ClassVar

from arq.connections import RedisSettings
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from luber_audio_utils import storage_from_settings
from luber_database import (
    GenerationRepository,
    ObservabilityRepository,
    create_async_engine_from_url,
    create_session_factory,
)
from luber_database.models.generation import Generation
from luber_generation_client import GENERATION_QUEUE_NAME, GenerationService, provider_from_settings
from luber_generation_worker.singleton import (
    EXIT_ALREADY_RUNNING,
    SingleWorkerLock,
    WorkerAlreadyRunningError,
)
from luber_inference_observability.service import ingest_one
from luber_shared import BaseServiceSettings, configure_logging

logger = logging.getLogger(__name__)

QUEUE_NAME = GENERATION_QUEUE_NAME


class WorkerConfig(BaseServiceSettings):
    worker_id: str = "generation-worker-0"


async def ping(ctx: dict[str, Any], payload: str = "pong") -> str:
    """Connectivity smoke-test task."""
    logger.info(
        "ping received",
        extra={"worker_id": ctx.get("worker_id"), "payload": payload},
    )
    return payload


async def generate(ctx: dict[str, Any], generation_id: str) -> str:
    """Process one generation job end-to-end; returns the terminal status."""
    session_factory = ctx["session_factory"]
    config: WorkerConfig = ctx["config"]
    async with session_factory() as session:
        service = GenerationService(
            GenerationRepository(session),
            provider_from_settings(config),
            storage_from_settings(config),
            qc_policy=config.inference_qc_policy,
            qc_enabled=config.inference_qc_enabled,
            candidate_workspace_dir=config.candidate_workspace_dir,
        )
        status = await service.execute(uuid.UUID(generation_id), worker_id=ctx.get("worker_id"))
        await _record_observation(session, uuid.UUID(generation_id), config)
    return status.value


async def _record_observation(session: Any, generation_id: uuid.UUID, config: WorkerConfig) -> None:
    """Project this finished generation into the Phase 30 analytics table.

    Deliberately after `execute` and deliberately unable to fail it. The
    generation has already succeeded or failed by this point, and its
    outcome is recorded; losing an analytics row is a gap in a chart,
    while raising here would turn a delivered song into a failed job.

    Nothing is lost permanently either way — the scheduled ingest reads
    from a watermark and picks up anything this missed.

    This is also the only place `luber_revision` can honestly be set: the
    process writing it is the process that produced the generation. A
    backfill running next month cannot know that, and records UNKNOWN.
    """
    if not config.observability_enabled:
        return
    try:
        generation = await session.get(Generation, generation_id)
        if generation is None:
            return
        await ingest_one(
            ObservabilityRepository(session),
            generation,
            luber_revision=config.luber_revision or None,
        )
    except Exception:
        logger.warning(
            "could not record an inference observation",
            extra={"generation_id": str(generation_id)},
            exc_info=True,
        )


async def check_database(engine: AsyncEngine) -> None:
    """Fail fast, and say which dependency is missing.

    ARQ needs Redis to start at all and reports its absence loudly, but
    PostgreSQL is not touched until the first job — so without this a
    worker with a bad database URL looks perfectly healthy right up to
    the moment it drops a real generation.
    """
    async with engine.connect() as conn:
        await conn.execute(text("select 1"))


async def startup(ctx: dict[str, Any]) -> None:
    config = WorkerConfig()
    configure_logging(service="luber-generation-worker", level=config.log_level)

    # Before anything else: this machine gets one generation worker.
    # Released by the kernel when the process dies, so there is no stale
    # state to clean up after a crash.
    lock = SingleWorkerLock()
    try:
        lock.acquire()
    except WorkerAlreadyRunningError as exc:
        logger.error("refusing to start: %s", exc)
        raise SystemExit(EXIT_ALREADY_RUNNING) from exc
    ctx["singleton_lock"] = lock

    engine = create_async_engine_from_url(config.database_url)
    try:
        await check_database(engine)
    except Exception:
        # A dependency that is down is not a reason to hold the lock.
        await engine.dispose()
        lock.release()
        logger.exception("refusing to start: database unreachable")
        raise

    # ACE-Step is deliberately *not* checked. A generation submitted
    # while the engine is down fails truthfully with a stable code, so
    # refusing to start would turn a recoverable per-job failure into an
    # outage of the whole queue — including the jobs that would have
    # succeeded by the time they ran.
    ctx["config"] = config
    ctx["worker_id"] = config.worker_id
    ctx["db_engine"] = engine
    ctx["session_factory"] = create_session_factory(engine)
    logger.info(
        "generation worker started",
        extra={"worker_id": config.worker_id, "lock": str(lock.path)},
    )


async def shutdown(ctx: dict[str, Any]) -> None:
    engine = ctx.get("db_engine")
    if engine is not None:
        await engine.dispose()
    lock = ctx.get("singleton_lock")
    if lock is not None:
        lock.release()
    logger.info("generation worker stopped", extra={"worker_id": ctx.get("worker_id")})


_CONFIG = WorkerConfig()

#: The queue's timeout must sit *outside* the provider's own liveness
#: backstop, not inside it. ARQ defaults to 300s while the ACE-Step
#: provider allows at least ``ace_step_generation_timeout`` (1800s in
#: this deployment) — so the queue used to cut every slow generation
#: short, and it did so by cancelling the task. Cancellation is not a
#: provider error: nothing was recorded against the run, and the row was
#: left claiming GENERATING forever. Letting the provider time out first
#: means a slow engine produces a truthful GENERATION_TIMEOUT instead.
#: The margin covers post-processing and upload, which happen after the
#: provider returns.
JOB_TIMEOUT_SECONDS = int(_CONFIG.ace_step_generation_timeout) + 600

#: Each attempt is a full inference run. Five (the ARQ default) means a
#: job that keeps being interrupted can burn five times the compute of
#: the song it is trying to produce; two bounds that while still giving
#: an interrupted generation the one retry that actually recovers it.
MAX_TRIES = 2


class WorkerSettings:
    functions: ClassVar = [ping, generate]
    on_startup = startup
    on_shutdown = shutdown
    queue_name = QUEUE_NAME
    redis_settings = RedisSettings.from_dsn(_CONFIG.redis_url)
    max_jobs = 1  # one generation at a time per worker (GPU-bound later)
    health_check_interval = 30
    job_timeout = JOB_TIMEOUT_SECONDS
    max_tries = MAX_TRIES
