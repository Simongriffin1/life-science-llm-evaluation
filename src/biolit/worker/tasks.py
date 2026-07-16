"""arq worker tasks."""

from __future__ import annotations

from typing import Any

from biolit.core.logging import get_logger
from biolit.eval.service import execute_eval

logger = get_logger(__name__)


async def run_eval(ctx: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Execute a full eval sweep from an arq worker."""
    logger.info("run_eval.start", extra={"config_keys": list(config.keys())})
    result = await execute_eval(config)
    return result.model_dump()


async def run_hypothesis(ctx: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Placeholder: hypothesis runs land in Phase 5."""
    logger.info("run_hypothesis stub", extra={"config_keys": list(config.keys())})
    return {"status": "not_implemented"}
