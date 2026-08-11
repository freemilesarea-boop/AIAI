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

from luber_api.middleware import RequestIdMiddleware
from luber_api.routes.health import router as health_router
from luber_api.settings import get_settings
from luber_database import create_async_engine_from_url
from luber_shared import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.db_engine = create_async_engine_from_url(settings.database_url)
    app.state.redis = Redis.from_url(settings.redis_url)
    try:
        yield
    finally:
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
    return app


app = create_app()
