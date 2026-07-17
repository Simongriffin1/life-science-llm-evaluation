"""Job status API."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from biolit.core.logging import get_logger
from biolit.worker.jobs import get_job

logger = get_logger(__name__)
router = APIRouter()


@router.get("/{job_id}")
async def job_status(job_id: str) -> dict[str, Any]:
    try:
        return await get_job(job_id)
    except Exception as exc:
        logger.exception("jobs.get_failed", extra={"job_id": job_id})
        raise HTTPException(status_code=404, detail=str(exc)) from exc
