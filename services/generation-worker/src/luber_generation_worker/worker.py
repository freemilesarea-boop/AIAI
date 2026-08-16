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

from luber_audio_utils import storage_from_settings
from luber_database import (
    GenerationRepository,
    create_async_engine_from_url,
    create_session_factory,
)
from luber_generation_client import GENERATION_QUEUE_NAME, GenerationService, provider_from_settings
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
        )
        status = await service.execute(uuid.UUID(generation_id), worker_id=ctx.get("worker_id"))
    return status.value


async def startup(ctx: dict[str, Any]) -> None:
    config = WorkerConfig()
    configure_logging(service="luber-generation-worker", level=config.log_level)
    ctx["config"] = config
    ctx["worker_id"] = config.worker_id
    ctx["db_engine"] = create_async_engine_from_url(config.database_url)
    ctx["session_factory"] = create_session_factory(ctx["db_engine"])
    logger.info("generation worker started", extra={"worker_id": config.worker_id})


async def shutdown(ctx: dict[str, Any]) -> None:
    engine = ctx.get("db_engine")
    if engine is not None:
        await engine.dispose()
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
