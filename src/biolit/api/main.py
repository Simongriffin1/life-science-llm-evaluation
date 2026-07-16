"""FastAPI application entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from biolit.api.routers import eval as eval_router
from biolit.api.routers import hypothesis as hypothesis_router
from biolit.api.routers import retrieve as retrieve_router
from biolit.core.cache import close_redis, ping_redis
from biolit.core.config import get_settings
from biolit.core.db import dispose_engine, ping_db
from biolit.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):  # type: ignore[no-untyped-def]
    configure_logging()
    settings = get_settings()
    logger.info("BioLit starting", extra={"env": settings.app_env})
    yield
    await close_redis()
    await dispose_engine()
    logger.info("BioLit shutdown complete")


def create_app() -> FastAPI:
    """Application factory."""
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        db = await ping_db()
        redis = await ping_redis()
        status = "ok" if db.get("ok") and redis.get("ok") else "degraded"
        # Always return 200 for liveness; include dependency status in body.
        return {
            "status": status,
            "app": settings.app_name,
            "db": db,
            "redis": redis,
        }

    app.include_router(retrieve_router.router, prefix="/retrieve", tags=["retrieve"])
    app.include_router(eval_router.router, prefix="/eval", tags=["eval"])
    app.include_router(hypothesis_router.router, prefix="/hypothesize", tags=["hypothesis"])
    return app


app = create_app()
