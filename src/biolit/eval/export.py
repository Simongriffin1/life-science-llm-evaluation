"""Immutable export of eval runs to JSON and CSV."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import select

from biolit.core.db import EvalDataset, EvalItem, EvalRun, get_session_factory

ExportFormat = Literal["json", "csv"]


class ExportExistsError(FileExistsError):
    """Raised when refusing to overwrite an existing export artifact."""


async def load_run_bundle(run_id: str) -> dict[str, Any]:
    """Load run config, metrics, dataset info, and item rows."""
    factory = get_session_factory()
    async with factory() as session:
        run = await session.get(EvalRun, run_id)
        if run is None:
            raise KeyError(f"Unknown eval run {run_id}")
        ds = await session.get(EvalDataset, run.dataset_id)
        items = (
            (
                await session.execute(
                    select(EvalItem).where(EvalItem.run_id == run_id).order_by(EvalItem.question_id)
                )
            )
            .scalars()
            .all()
        )

    item_rows: list[dict[str, Any]] = []
    for it in items:
        judge = it.judge_json or {}
        grounded = None
        if isinstance(judge, dict):
            g = judge.get("groundedness") or {}
            if isinstance(g, dict) and g.get("groundedness") is not None:
                grounded = g.get("groundedness")
        item_rows.append(
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
                "groundedness": grounded,
                "judge_json": it.judge_json,
            }
        )

    content_hash = _item_set_hash(item_rows)
    return {
        "run_id": run_id,
        "model": run.model,
        "mode": run.mode,
        "status": run.status,
        "dataset": {
            "id": ds.id if ds else None,
            "name": ds.name if ds else None,
            "split": ds.split if ds else None,
            "license": ds.license if ds else None,
        },
        "config": run.config_json or {},
        "metrics": run.metrics_json or {},
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "items": item_rows,
        "content_hash": content_hash,
    }


def _item_set_hash(items: list[dict[str, Any]]) -> str:
    material = json.dumps(items, sort_keys=True, default=str)
    return hashlib.sha256(material.encode()).hexdigest()


def export_filename(run_id: str, content_hash: str, fmt: ExportFormat) -> str:
    short = content_hash[:8]
    return f"eval_{run_id}_{short}.{fmt}"


def render_json(bundle: dict[str, Any]) -> str:
    return json.dumps(bundle, indent=2, sort_keys=True, default=str) + "\n"


def render_csv(bundle: dict[str, Any]) -> str:
    fieldnames = [
        "question_id",
        "prediction",
        "gold",
        "correct",
        "retrieved_pmids",
        "prompt_tokens",
        "completion_tokens",
        "latency_ms",
        "groundedness",
        "prompt",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in bundle.get("items") or []:
        out = dict(row)
        pmids = out.get("retrieved_pmids") or []
        out["retrieved_pmids"] = ";".join(str(p) for p in pmids)
        writer.writerow(out)
    return buf.getvalue()


async def export_run(
    run_id: str,
    *,
    fmt: ExportFormat = "json",
    out_dir: Path | str | None = None,
) -> Path:
    """
    Write an immutable export file. Filename includes run_id + short content hash.
    Refuses to overwrite an existing path.
    """
    bundle = await load_run_bundle(run_id)
    content_hash = str(bundle["content_hash"])
    name = export_filename(run_id, content_hash, fmt)
    directory = Path(out_dir) if out_dir else Path("data") / "exports"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    if path.exists():
        # Identical content is fine — verify hash matches and return existing.
        existing = path.read_text(encoding="utf-8")
        if fmt == "json":
            prior = json.loads(existing)
            if prior.get("content_hash") == content_hash:
                return path
        raise ExportExistsError(f"Export exists and differs: {path}")

    body = render_json(bundle) if fmt == "json" else render_csv(bundle)
    path.write_text(body, encoding="utf-8")
    return path
