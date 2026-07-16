"""Hypothesis API router."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from biolit.core.costing import CostEstimate, estimate_hypothesis_cost
from biolit.core.logging import get_logger
from biolit.hypothesis.models import HypothesisConfig, HypothesisRunResult
from biolit.hypothesis.service import execute_hypothesis

logger = get_logger(__name__)
router = APIRouter()


class HypothesizeRequest(BaseModel):
    research_goal: str = Field(min_length=3)
    config: HypothesisConfig | dict[str, Any] | None = None
    dry_run: bool = False
    sync: bool = Field(
        default=True,
        description="Run inline when true; otherwise enqueue arq / background task.",
    )


class HypothesizeResponse(BaseModel):
    status: str
    job_id: str | None = None
    message: str | None = None
    result: HypothesisRunResult | None = None
    estimate: CostEstimate | None = None


@router.post("", response_model=HypothesizeResponse)
async def enqueue_hypothesis(
    body: HypothesizeRequest,
    background_tasks: BackgroundTasks,
) -> HypothesizeResponse:
    """PubMed-grounded hypothesis generation with Elo tournament and provenance."""
    cfg = (
        body.config
        if isinstance(body.config, HypothesisConfig)
        else HypothesisConfig.model_validate(body.config or {})
    )

    if body.dry_run:
        estimate = estimate_hypothesis_cost(
            model=cfg.model,
            n_seed=cfg.n_seed,
            evolution_rounds=cfg.evolution_rounds,
            tournament_rounds=cfg.tournament_rounds,
            max_proposals=cfg.max_proposals,
        )
        return HypothesizeResponse(
            status="dry_run",
            estimate=estimate,
            message="Cost estimate only; no hypothesis run executed",
        )

    if body.sync:
        try:
            result = await execute_hypothesis(body.research_goal, cfg)
        except Exception as exc:
            logger.exception("hypothesis.sync_failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return HypothesizeResponse(status="completed", result=result)

    try:
        from arq import create_pool
        from arq.connections import RedisSettings

        from biolit.core.config import get_settings

        settings = get_settings()
        redis = await create_pool(RedisSettings.from_dsn(settings.redis_url))
        job = await redis.enqueue_job(
            "run_hypothesis",
            {"research_goal": body.research_goal, "config": cfg.model_dump()},
        )
        job_id = job.job_id if job else None
        await redis.aclose()
        return HypothesizeResponse(
            status="queued",
            job_id=job_id,
            message="Hypothesis job enqueued on arq worker",
        )
    except Exception as exc:
        logger.warning("hypothesis.arq_enqueue_failed", extra={"error": str(exc)})
        background_tasks.add_task(execute_hypothesis, body.research_goal, cfg)
        return HypothesizeResponse(
            status="queued",
            message="arq unavailable; running via FastAPI BackgroundTasks",
        )
