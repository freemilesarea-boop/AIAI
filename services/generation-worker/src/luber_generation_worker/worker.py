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
from pathlib import Path
from typing import Any, ClassVar

from arq.connections import RedisSettings

from luber_audio_utils import LocalAudioStorage
from luber_database import (
    GenerationRepository,
    create_async_engine_from_url,
    create_session_factory,
)
from luber_generation_client import GENERATION_QUEUE_NAME, GenerationService, build_provider
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
            build_provider(
                config.generation_provider,
                mock_fixture_path=Path(config.mock_fixture_path),
            ),
            LocalAudioStorage(Path(config.audio_storage_dir)),
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


class WorkerSettings:
    functions: ClassVar = [ping, generate]
    on_startup = startup
    on_shutdown = shutdown
    queue_name = QUEUE_NAME
    redis_settings = RedisSettings.from_dsn(WorkerConfig().redis_url)
    max_jobs = 1  # one generation at a time per worker (GPU-bound later)
    health_check_interval = 30
