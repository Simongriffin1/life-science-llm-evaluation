"""Eval API: enqueue jobs, leaderboard, export, drill-down."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from biolit.core.costing import CostEstimate, estimate_eval_cost
from biolit.core.db import EvalItem, EvalRun, get_session_factory
from biolit.core.logging import get_logger
from biolit.eval.export import (
    ExportExistsError,
    export_run,
    load_run_bundle,
    render_csv,
    render_json,
)
from biolit.eval.leaderboard import leaderboard
from biolit.eval.service import EvalConfig, EvalJobResult, execute_eval
from biolit.worker.jobs import enqueue_job

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
    few_shot_k: int = Field(default=5, ge=0, le=20)
    dry_run: bool = False
    sync: bool = Field(
        default=False,
        description="If true, run inline (tests/dev); if false, enqueue arq and return job_id.",
    )


class EvalEnqueueResponse(BaseModel):
    status: str
    job_id: str | None = None
    message: str | None = None
    result: EvalJobResult | None = None
    estimate: CostEstimate | None = None


@router.post("", response_model=EvalEnqueueResponse)
async def enqueue_eval(body: EvalRequest) -> EvalEnqueueResponse:
    """Start an eval sweep — enqueue on arq by default; sync for local/tests."""
    modes = body.mode if isinstance(body.mode, list) else [body.mode]
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
        few_shot_k=body.few_shot_k,
    )

    if body.dry_run:
        estimate = estimate_eval_cost(
            models=body.models,
            datasets=body.datasets,
            modes=list(modes),
            limit=int(body.limit or 20),
            with_judge=bool(body.judge_model),
        )
        return EvalEnqueueResponse(
            status="dry_run",
            estimate=estimate,
            message="Cost estimate only; no eval executed",
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

    try:
        job_id = await enqueue_job("run_eval", cfg.model_dump())
    except Exception as exc:
        logger.exception("eval.arq_enqueue_failed")
        raise HTTPException(
            status_code=503,
            detail=f"arq enqueue failed (is the worker running?): {exc}",
        ) from exc
    return EvalEnqueueResponse(
        status="queued",
        job_id=job_id,
        message="Eval job enqueued on arq worker",
    )


@router.get("/leaderboard")
async def get_leaderboard(
    dataset: str | None = None,
    mode: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    rows = await leaderboard(dataset=dataset, mode=mode, model=model)
    return {"rows": rows, "n": len(rows)}


@router.get("/runs/{run_id}")
async def get_run_detail(
    run_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    """Drill-down: run metadata + paginated item-level rows."""
    factory = get_session_factory()
    async with factory() as session:
        run = await session.get(EvalRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown run {run_id}")
        total = await session.scalar(
            select(func.count()).select_from(EvalItem).where(EvalItem.run_id == run_id)
        )
        items = (
            (
                await session.execute(
                    select(EvalItem)
                    .where(EvalItem.run_id == run_id)
                    .order_by(EvalItem.question_id)
                    .offset(offset)
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )

    rows = [
        {
            "question_id": it.question_id,
            "prompt": it.prompt,
            "prediction": it.prediction,
            "gold": it.gold,
            "correct": it.correct,
            "retrieved_pmids": it.retrieved_pmids,
            "prompt_tokens": it.prompt_tokens,
            "completion_tokens": it.completion_tokens,
            "latency_ms": it.latency_ms,
            "judge_json": it.judge_json,
        }
        for it in items
    ]
    return {
        "run_id": run_id,
        "model": run.model,
        "mode": run.mode,
        "status": run.status,
        "metrics": run.metrics_json,
        "config": run.config_json,
        "offset": offset,
        "limit": limit,
        "total": int(total or 0),
        "items": rows,
    }


@router.get("/runs/{run_id}/export")
async def export_run_endpoint(
    run_id: str,
    format: Literal["json", "csv"] = Query(default="json", alias="format"),
) -> Response:
    """Stream an immutable export; refuses overwrite of differing content on disk."""
    try:
        bundle = await load_run_bundle(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        path = await export_run(run_id, fmt=format)
    except ExportExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    body = render_json(bundle) if format == "json" else render_csv(bundle)
    media = "application/json" if format == "json" else "text/csv"
    return Response(
        content=body,
        media_type=media,
        headers={
            "Content-Disposition": f'attachment; filename="{path.name}"',
            "X-Content-Hash": str(bundle["content_hash"]),
            "X-Export-Path": str(path),
        },
    )
