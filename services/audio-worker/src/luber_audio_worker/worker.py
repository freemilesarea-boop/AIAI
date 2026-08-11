"""ARQ worker entrypoint for the audio post-processing queue.

Run with: ``arq luber_audio_worker.worker.WorkerSettings``
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from arq.connections import RedisSettings

from luber_shared import BaseServiceSettings, configure_logging

logger = logging.getLogger(__name__)

QUEUE_NAME = "luber:audio"


class WorkerConfig(BaseServiceSettings):
    worker_id: str = "audio-worker-0"


async def ping(ctx: dict[str, Any], payload: str = "pong") -> str:
    """Connectivity smoke-test task. Replaced by real audio jobs in Phase 4."""
    logger.info(
        "ping received",
        extra={"worker_id": ctx.get("worker_id"), "payload": payload},
    )
    return payload


async def startup(ctx: dict[str, Any]) -> None:
    config = WorkerConfig()
    configure_logging(service="luber-audio-worker", level=config.log_level)
    ctx["worker_id"] = config.worker_id
    logger.info("audio worker started", extra={"worker_id": config.worker_id})


async def shutdown(ctx: dict[str, Any]) -> None:
    logger.info("audio worker stopped", extra={"worker_id": ctx.get("worker_id")})


class WorkerSettings:
    functions: ClassVar = [ping]
    on_startup = startup
    on_shutdown = shutdown
    queue_name = QUEUE_NAME
    redis_settings = RedisSettings.from_dsn(WorkerConfig().redis_url)
    max_jobs = 4
    health_check_interval = 30
