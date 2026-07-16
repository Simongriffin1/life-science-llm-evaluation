"""arq worker tasks (stubs until Phases 4–5)."""

from __future__ import annotations

from typing import Any

from biolit.core.logging import get_logger

logger = get_logger(__name__)


async def run_eval(ctx: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Placeholder: full eval sweeps land in Phase 4."""
    logger.info("run_eval stub", extra={"config_keys": list(config.keys())})
    return {"status": "not_implemented"}


async def run_hypothesis(ctx: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Placeholder: hypothesis runs land in Phase 5."""
    logger.info("run_hypothesis stub", extra={"config_keys": list(config.keys())})
    return {"status": "not_implemented"}
