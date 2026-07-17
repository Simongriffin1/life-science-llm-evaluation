"""Hypothesis API: enqueue, interactive review, resume."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from biolit.core.costing import CostEstimate, estimate_hypothesis_cost
from biolit.core.logging import get_logger
from biolit.hypothesis.feedback import ReviewDecision
from biolit.hypothesis.models import HypothesisConfig, HypothesisRunResult
from biolit.hypothesis.service import (
    execute_hypothesis,
    get_pending_review,
    resume_hypothesis,
    submit_feedback,
)
from biolit.worker.jobs import enqueue_job

logger = get_logger(__name__)
router = APIRouter()


class HypothesizeRequest(BaseModel):
    research_goal: str = Field(min_length=3)
    config: HypothesisConfig | dict[str, Any] | None = None
    dry_run: bool = False
    sync: bool = Field(
        default=False,
        description="Run inline when true; otherwise enqueue arq and return job_id.",
    )


class HypothesizeResponse(BaseModel):
    status: str
    job_id: str | None = None
    message: str | None = None
    result: HypothesisRunResult | None = None
    estimate: CostEstimate | None = None


@router.post("", response_model=HypothesizeResponse)
async def enqueue_hypothesis(body: HypothesizeRequest) -> HypothesizeResponse:
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
        return HypothesizeResponse(status=result.status, result=result)

    try:
        job_id = await enqueue_job(
            "run_hypothesis",
            {"research_goal": body.research_goal, "config": cfg.model_dump()},
        )
    except Exception as exc:
        logger.exception("hypothesis.arq_enqueue_failed")
        raise HTTPException(
            status_code=503,
            detail=f"arq enqueue failed (is the worker running?): {exc}",
        ) from exc
    return HypothesizeResponse(
        status="queued",
        job_id=job_id,
        message="Hypothesis job enqueued on arq worker",
    )


@router.get("/{run_id}/pending")
async def pending_review(run_id: str) -> dict[str, Any]:
    try:
        return await get_pending_review(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{run_id}/feedback")
async def post_feedback(run_id: str, body: ReviewDecision) -> dict[str, Any]:
    try:
        return await submit_feedback(run_id, body)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/{run_id}/resume", response_model=HypothesizeResponse)
async def resume_run(run_id: str, body: ReviewDecision | None = None) -> HypothesizeResponse:
    try:
        result = await resume_hypothesis(run_id, body)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("hypothesis.resume_http_failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return HypothesizeResponse(status=result.status, result=result)
