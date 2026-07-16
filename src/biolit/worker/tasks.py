"""arq worker tasks."""

from __future__ import annotations

from typing import Any

from biolit.core.logging import get_logger
from biolit.eval.service import execute_eval
from biolit.hypothesis.service import execute_hypothesis

logger = get_logger(__name__)


async def run_eval(ctx: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Execute a full eval sweep from an arq worker."""
    logger.info("run_eval.start", extra={"config_keys": list(config.keys())})
    result = await execute_eval(config)
    return result.model_dump()


async def run_hypothesis(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Execute a hypothesis run from an arq worker."""
    logger.info("run_hypothesis.start", extra={"keys": list(payload.keys())})
    result = await execute_hypothesis(
        str(payload["research_goal"]),
        payload.get("config"),
    )
    return result.model_dump()
