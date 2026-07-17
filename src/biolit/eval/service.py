"""Eval orchestration: create runs, execute runners, persist metrics."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from biolit.core.db import EvalDataset, EvalItem, EvalRun, get_session_factory
from biolit.core.logging import get_logger
from biolit.eval.datasets import load_dataset
from biolit.eval.metrics import compute_run_metrics
from biolit.eval.runners import EvalMode, run_item

logger = get_logger(__name__)


class EvalConfig(BaseModel):
    models: list[str]
    datasets: list[str]
    mode: EvalMode | list[EvalMode] = "closed_book"
    retrieval: dict[str, Any] | None = None
    judge_model: str | None = None
    seed: int = 7
    limit: int | None = 20
    split: str = "test"
    use_cache: bool = True
    few_shot_k: int = Field(default=5, ge=0, le=20)


class EvalJobResult(BaseModel):
    status: str
    run_ids: list[str] = Field(default_factory=list)
    leaderboard: list[dict[str, Any]] = Field(default_factory=list)
    message: str | None = None


async def _ensure_dataset_row(
    name: str,
    split: str,
    n_items: int,
    license_text: str,
    path: str | None,
) -> str:
    factory = get_session_factory()
    async with factory() as session:
        row = EvalDataset(
            id=str(uuid4()),
            name=name,
            split=split,
            n_items=n_items,
            license=license_text,
            path=path,
        )
        session.add(row)
        await session.commit()
        return row.id


async def execute_eval(config: EvalConfig | dict[str, Any]) -> EvalJobResult:
    """Run eval for each model × dataset × mode and persist immutable rows."""
    cfg = config if isinstance(config, EvalConfig) else EvalConfig.model_validate(config)
    modes: list[EvalMode] = cfg.mode if isinstance(cfg.mode, list) else [cfg.mode]

    run_ids: list[str] = []
    factory = get_session_factory()

    for dataset_name in cfg.datasets:
        info, items = load_dataset(dataset_name, split=cfg.split, limit=cfg.limit)
        dataset_id = await _ensure_dataset_row(
            info.name, info.split, info.n_items, info.license, info.path
        )

        for model in cfg.models:
            for mode in modes:
                run_id = str(uuid4())
                started = datetime.now(UTC)
                async with factory() as session:
                    session.add(
                        EvalRun(
                            id=run_id,
                            model=model,
                            dataset_id=dataset_id,
                            mode=mode,
                            config_json=cfg.model_dump(),
                            started_at=started,
                            status="running",
                        )
                    )
                    await session.commit()

                logger.info(
                    "eval.run.start",
                    extra={"run_id": run_id, "model": model, "dataset": dataset_name, "mode": mode},
                )

                item_rows: list[dict[str, Any]] = []
                try:
                    for item in items:
                        row = await run_item(
                            item,
                            model=model,
                            mode=mode,
                            dataset=dataset_name,
                            few_shot_k=cfg.few_shot_k,
                            retrieval=cfg.retrieval,
                            judge_model=cfg.judge_model if mode == "rag" else None,
                            use_cache=cfg.use_cache,
                        )
                        item_rows.append(row)

                    predictions = [str(r.get("prediction") or "") for r in item_rows]
                    golds = [str(r["gold"]) for r in item_rows]
                    retrieved = [r.get("retrieved_pmids") or [] for r in item_rows]
                    gold_pmids = [r.get("gold_pmids") or [] for r in item_rows]
                    grounded = []
                    for r in item_rows:
                        g = (r.get("judge_json") or {}).get("groundedness") or {}
                        if g.get("groundedness") is not None:
                            grounded.append(float(g["groundedness"]))
                    metrics = compute_run_metrics(
                        predictions=predictions,
                        golds=golds,
                        retrieved_pmids=retrieved if mode == "rag" else None,
                        gold_pmids=gold_pmids if mode == "rag" else None,
                        groundedness_scores=grounded or None,
                        item_stats=item_rows,
                        k=int((cfg.retrieval or {}).get("top_k", 8)),
                        few_shot_k=cfg.few_shot_k,
                    )

                    async with factory() as session:
                        for row in item_rows:
                            session.add(
                                EvalItem(
                                    id=str(uuid4()),
                                    run_id=run_id,
                                    question_id=row["question_id"],
                                    prompt=row["prompt"],
                                    prediction=row["prediction"],
                                    gold=row["gold"],
                                    correct=row["correct"],
                                    retrieved_pmids=row.get("retrieved_pmids") or None,
                                    latency_ms=row.get("latency_ms"),
                                    prompt_tokens=row.get("prompt_tokens"),
                                    completion_tokens=row.get("completion_tokens"),
                                    judge_json=row.get("judge_json"),
                                )
                            )
                        run = await session.get(EvalRun, run_id)
                        if run is not None:
                            run.status = "completed"
                            run.finished_at = datetime.now(UTC)
                            run.metrics_json = metrics
                        await session.commit()

                    run_ids.append(run_id)
                    logger.info(
                        "eval.run.done",
                        extra={"run_id": run_id, "accuracy": metrics.get("accuracy")},
                    )
                except Exception as exc:
                    logger.exception("eval.run.failed", extra={"run_id": run_id})
                    async with factory() as session:
                        run = await session.get(EvalRun, run_id)
                        if run is not None:
                            run.status = "failed"
                            run.finished_at = datetime.now(UTC)
                            run.metrics_json = {"error": str(exc)}
                        await session.commit()
                    raise

    from biolit.eval.leaderboard import leaderboard

    board = await leaderboard()
    return EvalJobResult(status="completed", run_ids=run_ids, leaderboard=board)
