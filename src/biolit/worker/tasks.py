"""arq worker tasks."""

from __future__ import annotations

from typing import Any

from biolit.core.logging import get_logger
from biolit.eval.service import execute_eval
from biolit.hypothesis.service import execute_hypothesis
from biolit.worker.jobs import set_job_progress

logger = get_logger(__name__)


async def run_eval(ctx: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Execute a full eval sweep from an arq worker."""
    job_id = str(ctx.get("job_id") or "")
    logger.info("run_eval.start", extra={"config_keys": list(config.keys()), "job_id": job_id})
    if job_id:
        await set_job_progress(job_id, status="running", progress=0.05, message="eval started")
    result = await execute_eval(config)
    payload = result.model_dump()
    if job_id:
        await set_job_progress(
            job_id,
            status="complete",
            progress=1.0,
            message="eval complete",
            result_ref={"run_ids": payload.get("run_ids"), "status": payload.get("status")},
        )
    return payload


async def run_hypothesis(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Execute a hypothesis run from an arq worker."""
    job_id = str(ctx.get("job_id") or "")
    logger.info("run_hypothesis.start", extra={"keys": list(payload.keys()), "job_id": job_id})
    if job_id:
        await set_job_progress(
            job_id, status="running", progress=0.05, message="hypothesis started"
        )
    result = await execute_hypothesis(
        str(payload["research_goal"]),
        payload.get("config"),
    )
    out = result.model_dump()
    if job_id:
        done = result.status in {"completed", "paused_for_review"}
        await set_job_progress(
            job_id,
            status="complete" if done else "running",
            progress=1.0 if done else 0.5,
            message=result.status,
            result_ref={"run_id": result.run_id, "status": result.status},
        )
    return out
