"""Eval API: enqueue or synchronously run evaluation jobs."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from biolit.core.logging import get_logger
from biolit.eval.leaderboard import leaderboard
from biolit.eval.service import EvalConfig, EvalJobResult, execute_eval

logger = get_logger(__name__)
router = APIRouter()


class EvalRequest(BaseModel):
    models: list[str] = Field(min_length=1)
    datasets: list[str] = Field(min_length=1)
    mode: Literal["closed_book", "rag"] | list[Literal["closed_book", "rag"]] = "closed_book"
    retrieval: dict[str, Any] | None = None
    judge_model: str | None = None
    seed: int = 7
    limit: int | None = Field(default=20, ge=1, le=500)
    split: str = "test"
    use_cache: bool = True
    sync: bool = Field(
        default=True,
        description="If true, run inline and return results; if false, enqueue arq job.",
    )


class EvalEnqueueResponse(BaseModel):
    status: str
    job_id: str | None = None
    message: str | None = None
    result: EvalJobResult | None = None


@router.post("", response_model=EvalEnqueueResponse)
async def enqueue_eval(
    body: EvalRequest,
    background_tasks: BackgroundTasks,
) -> EvalEnqueueResponse:
    """Start an eval sweep (sync by default for local/dev; arq when sync=false)."""
    cfg = EvalConfig(
        models=body.models,
        datasets=body.datasets,
        mode=body.mode,
        retrieval=body.retrieval or {"mode": "bm25", "top_k": 8, "candidate_cap": 40},
        judge_model=body.judge_model,
        seed=body.seed,
        limit=body.limit,
        split=body.split,
        use_cache=body.use_cache,
    )

    if body.sync:
        try:
            result = await execute_eval(cfg)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("eval.sync_failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return EvalEnqueueResponse(status="completed", result=result)

    # Async path: try arq enqueue; fall back to BackgroundTasks.
    job_id: str | None = None
    try:
        from arq import create_pool
        from arq.connections import RedisSettings

        from biolit.core.config import get_settings

        settings = get_settings()
        redis = await create_pool(RedisSettings.from_dsn(settings.redis_url))
        job = await redis.enqueue_job("run_eval", cfg.model_dump())
        job_id = job.job_id if job else None
        await redis.aclose()
        return EvalEnqueueResponse(
            status="queued",
            job_id=job_id,
            message="Eval job enqueued on arq worker",
        )
    except Exception as exc:
        logger.warning("eval.arq_enqueue_failed", extra={"error": str(exc)})
        background_tasks.add_task(execute_eval, cfg)
        return EvalEnqueueResponse(
            status="queued",
            job_id=None,
            message="arq unavailable; running via FastAPI BackgroundTasks",
        )


@router.get("/leaderboard")
async def get_leaderboard(
    dataset: str | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    rows = await leaderboard(dataset=dataset, mode=mode)
    return {"rows": rows, "n": len(rows)}
