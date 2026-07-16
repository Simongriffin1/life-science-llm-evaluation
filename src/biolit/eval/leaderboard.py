"""Leaderboard queries over persisted eval_runs."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from biolit.core.db import EvalDataset, EvalRun, get_session_factory


async def leaderboard(
    *,
    dataset: str | None = None,
    mode: str | None = None,
) -> list[dict[str, Any]]:
    """
    Return comparable rows: model × dataset × mode with metrics.

    Also attaches rag_vs_closed_book deltas when both modes exist for a model/dataset.
    """
    factory = get_session_factory()
    async with factory() as session:
        stmt = (
            select(EvalRun, EvalDataset)
            .join(EvalDataset, EvalRun.dataset_id == EvalDataset.id)
            .where(EvalRun.status == "completed")
            .order_by(EvalDataset.name, EvalRun.model, EvalRun.mode)
        )
        if dataset:
            stmt = stmt.where(EvalDataset.name == dataset)
        if mode:
            stmt = stmt.where(EvalRun.mode == mode)
        result = await session.execute(stmt)
        rows = result.all()

    table: list[dict[str, Any]] = []
    for run, ds in rows:
        metrics = run.metrics_json or {}
        table.append(
            {
                "run_id": run.id,
                "model": run.model,
                "dataset": ds.name,
                "split": ds.split,
                "mode": run.mode,
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "recall_at_k": metrics.get("recall_at_k"),
                "mrr": metrics.get("mrr"),
                "ndcg_at_k": metrics.get("ndcg_at_k"),
                "groundedness": metrics.get("groundedness"),
                "cost": metrics.get("cost"),
                "metrics": metrics,
            }
        )

    # Attach closed_book → rag accuracy delta per model/dataset.
    closed: dict[tuple[str, str], float] = {}
    for row in table:
        if row["mode"] == "closed_book" and row["accuracy"] is not None:
            closed[(row["model"], row["dataset"])] = float(row["accuracy"])
    for row in table:
        if row["mode"] == "rag" and row["accuracy"] is not None:
            base = closed.get((row["model"], row["dataset"]))
            if base is not None:
                row["rag_vs_closed_book_delta"] = float(row["accuracy"]) - base
            else:
                row["rag_vs_closed_book_delta"] = None
        else:
            row["rag_vs_closed_book_delta"] = None
    return table
