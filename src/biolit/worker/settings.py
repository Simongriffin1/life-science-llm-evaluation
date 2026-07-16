"""arq worker settings."""

from __future__ import annotations

from arq.connections import RedisSettings

from biolit.core.config import get_settings
from biolit.worker.tasks import run_eval, run_hypothesis


def _redis_settings() -> RedisSettings:
    settings = get_settings()
    return RedisSettings.from_dsn(settings.redis_url)


class WorkerSettings:
    """arq WorkerSettings entrypoint: `arq biolit.worker.settings.WorkerSettings`."""

    functions = [run_eval, run_hypothesis]
    redis_settings = _redis_settings()
