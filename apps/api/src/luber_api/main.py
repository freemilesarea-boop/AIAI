"""FastAPI application factory.

Generation endpoints (POST /v1/generations, …) arrive in Phase 1 behind
the MusicGenerationProvider boundary — no model code runs in this
process, ever.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis

from luber_api.jobs import ArqGenerationEnqueuer, InlineGenerationRunner
from luber_api.middleware import RequestIdMiddleware
from luber_api.routes.generations import router as generations_router
from luber_api.routes.health import router as health_router
from luber_api.routes.projects import assign_router as project_assign_router
from luber_api.routes.projects import router as projects_router
from luber_api.routes.reference_audio import router as reference_audio_router
from luber_api.settings import get_settings
from luber_audio_utils import storage_from_settings
from luber_database import create_async_engine_from_url, create_session_factory
from luber_generation_client import provider_from_settings
from luber_shared import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.db_engine = create_async_engine_from_url(settings.database_url)
    app.state.session_factory = create_session_factory(app.state.db_engine)
    app.state.redis = Redis.from_url(settings.redis_url)
    app.state.audio_storage = storage_from_settings(settings)
    if settings.generation_execution_mode == "inline":
        # Test/dev only: executes the provider in-process.
        app.state.enqueuer = InlineGenerationRunner(
            app.state.session_factory,
            provider_from_settings(settings),
            app.state.audio_storage,
        )
    else:
        app.state.enqueuer = ArqGenerationEnqueuer(settings.redis_url)
    try:
        yield
    finally:
        await app.state.enqueuer.close()
        await app.state.redis.aclose()
        await app.state.db_engine.dispose()


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(service="luber-api", level=settings.log_level)

    app = FastAPI(
        title="LUBER MUSIC AI API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(generations_router)
    app.include_router(reference_audio_router)
    app.include_router(projects_router)
    app.include_router(project_assign_router)
    return app


app = create_app()
