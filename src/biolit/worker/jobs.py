"""Job enqueue + status helpers backed by Redis / arq."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Literal

from arq import create_pool
from arq.connections import RedisSettings
from arq.jobs import Job, JobStatus

from biolit.core.config import get_settings
from biolit.core.logging import get_logger

logger = get_logger(__name__)

JobState = Literal["queued", "running", "complete", "failed"]

_PROGRESS_PREFIX = "biolit:job:progress:"
_META_PREFIX = "biolit:job:meta:"


async def _pool() -> Any:
    settings = get_settings()
    return await create_pool(RedisSettings.from_dsn(settings.redis_url))


async def enqueue_job(task_name: str, payload: dict[str, Any]) -> str:
    """Enqueue an arq task and return its job_id. Raises on Redis/arq failure."""
    redis = await _pool()
    try:
        job = await redis.enqueue_job(task_name, payload)
        if job is None:
            raise RuntimeError(f"Failed to enqueue {task_name}")
        job_id = str(job.job_id)
        meta = {
            "task": task_name,
            "enqueued_at": datetime.now(UTC).isoformat(),
            "payload_keys": list(payload.keys()),
        }
        await redis.set(_META_PREFIX + job_id, json.dumps(meta), ex=86400)
        await set_job_progress(job_id, status="queued", progress=0.0, message="queued")
        logger.info("job.enqueued", extra={"job_id": job_id, "task": task_name})
        return job_id
    finally:
        await redis.aclose()


async def set_job_progress(
    job_id: str,
    *,
    status: JobState,
    progress: float,
    message: str | None = None,
    result_ref: dict[str, Any] | None = None,
) -> None:
    redis = await _pool()
    try:
        body = {
            "job_id": job_id,
            "status": status,
            "progress": progress,
            "message": message,
            "result_ref": result_ref,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        await redis.set(_PROGRESS_PREFIX + job_id, json.dumps(body), ex=86400)
    finally:
        await redis.aclose()


async def get_job(job_id: str) -> dict[str, Any]:
    """Return normalized job status for GET /jobs/{id}."""
    redis = await _pool()
    try:
        progress_raw = await redis.get(_PROGRESS_PREFIX + job_id)
        meta_raw = await redis.get(_META_PREFIX + job_id)
        progress = json.loads(progress_raw) if progress_raw else {}
        meta = json.loads(meta_raw) if meta_raw else {}

        job = Job(job_id, redis)
        arq_status: JobStatus | None
        try:
            arq_status = await job.status()
        except Exception:
            arq_status = None

        status: JobState = progress.get("status") or "queued"
        result_ref = progress.get("result_ref")

        if arq_status == JobStatus.in_progress:
            status = "running"
        elif arq_status == JobStatus.complete:
            status = "complete"
            if result_ref is None:
                try:
                    result = await job.result(timeout=0)
                    if isinstance(result, dict):
                        result_ref = {
                            "run_ids": result.get("run_ids"),
                            "run_id": result.get("run_id"),
                            "status": result.get("status"),
                        }
                except Exception:
                    pass
        elif arq_status == JobStatus.not_found and not progress:
            status = "queued"

        try:
            info = await job.info()
        except Exception:
            info = None
        if info is not None and getattr(info, "success", None) is False:
            status = "failed"

        prog_val = progress.get("progress")
        prog_f = (1.0 if status == "complete" else 0.0) if prog_val is None else float(prog_val)

        return {
            "job_id": job_id,
            "status": status,
            "progress": prog_f,
            "message": progress.get("message"),
            "result_ref": result_ref,
            "meta": meta,
            "arq_status": str(arq_status) if arq_status is not None else None,
        }
    finally:
        await redis.aclose()
