"""Eval dataset loaders and shared Item schema."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from biolit.core.logging import get_logger

logger = get_logger(__name__)

DatasetName = Literal["pubmedqa", "bioasq", "medqa", "medmcqa", "mmlu_med"]

DATASET_LICENSES: dict[str, str] = {
    "pubmedqa": "MIT (qiaojin/PubMedQA) — check redistribution terms",
    "bioasq": "BioASQ challenge terms — non-commercial research use; record license",
    "medqa": "MedQA-USMLE (Jin et al.) — check source license before redistribution",
    "medmcqa": "MedMCQA (Pal et al.) — check source license before redistribution",
    "mmlu_med": "MMLU medical subsets (Hendrycks et al.) — MIT-style; verify subset terms",
}


class EvalItem(BaseModel):
    """Normalized benchmark item."""

    id: str
    question: str
    options: dict[str, str] | None = None
    gold: str
    gold_pmids: list[str] = Field(default_factory=list)
    context: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class DatasetInfo(BaseModel):
    name: str
    split: str
    n_items: int
    license: str
    path: str | None = None


def _repo_data_root() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "datasets"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalize_pubmedqa(row: dict[str, Any], idx: int) -> EvalItem:
    decision = str(row.get("final_decision") or row.get("answer") or row.get("gold") or "").lower()
    contexts = row.get("context")
    context_text: str | None = None
    if isinstance(contexts, dict):
        parts = contexts.get("contexts") or []
        context_text = "\n".join(str(p) for p in parts)
    elif isinstance(contexts, list):
        context_text = "\n".join(str(p) for p in contexts)
    elif isinstance(contexts, str):
        context_text = contexts
    pmid = str(row.get("pubid") or row.get("pmid") or "")
    return EvalItem(
        id=str(row.get("id") or pmid or f"pubmedqa-{idx}"),
        question=str(row.get("question") or ""),
        options={"yes": "yes", "no": "no", "maybe": "maybe"},
        gold=decision,
        gold_pmids=[pmid] if pmid else [],
        context=context_text,
        meta={"license": DATASET_LICENSES["pubmedqa"]},
    )


def _normalize_mcqa(row: dict[str, Any], idx: int, name: str) -> EvalItem:
    options = row.get("options")
    if isinstance(options, list):
        options = {chr(ord("A") + i): str(o) for i, o in enumerate(options)}
    elif not isinstance(options, dict):
        options = None
    gold = str(row.get("answer_idx") or row.get("answer") or row.get("gold") or "")
    if options and gold not in options and gold.upper() in {k.upper() for k in options}:
        gold = next(k for k in options if k.upper() == gold.upper())
    pmids = row.get("gold_pmids") or row.get("pmids") or []
    return EvalItem(
        id=str(row.get("id") or f"{name}-{idx}"),
        question=str(row.get("question") or ""),
        options={str(k): str(v) for k, v in options.items()} if options else None,
        gold=gold,
        gold_pmids=[str(p) for p in pmids],
        context=row.get("context"),
        meta={"license": DATASET_LICENSES.get(name, "unknown")},
    )


def _normalize_bioasq(row: dict[str, Any], idx: int) -> EvalItem:
    qtype = str(row.get("type") or row.get("qtype") or "yesno").lower()
    gold = row.get("exact_answer") or row.get("gold") or row.get("answer") or ""
    if isinstance(gold, list):
        gold = "; ".join(str(x) for x in gold)
    options = {"yes": "yes", "no": "no"} if qtype in {"yesno", "yes/no"} else None
    pmids = row.get("documents") or row.get("gold_pmids") or []
    return EvalItem(
        id=str(row.get("id") or f"bioasq-{idx}"),
        question=str(row.get("question") or row.get("body") or ""),
        options=options,
        gold=str(gold).lower() if options else str(gold),
        gold_pmids=[str(p).split("/")[-1] for p in pmids],
        context=None,
        meta={"type": qtype, "license": DATASET_LICENSES["bioasq"]},
    )


_NORMALIZERS: dict[str, Callable[[dict[str, Any], int], EvalItem]] = {
    "pubmedqa": _normalize_pubmedqa,
    "bioasq": _normalize_bioasq,
    "medqa": lambda r, i: _normalize_mcqa(r, i, "medqa"),
    "medmcqa": lambda r, i: _normalize_mcqa(r, i, "medmcqa"),
    "mmlu_med": lambda r, i: _normalize_mcqa(r, i, "mmlu_med"),
}


def load_dataset(
    name: str,
    *,
    split: str = "test",
    limit: int | None = None,
    path: Path | str | None = None,
) -> tuple[DatasetInfo, list[EvalItem]]:
    """
    Load a MIRAGE-family dataset into EvalItem rows.

    Looks under ``data/datasets/{name}/`` for ``{split}.jsonl`` or ``sample.jsonl``.
    """
    if name not in DATASET_LICENSES:
        raise ValueError(f"Unknown dataset {name!r}. Supported: {sorted(DATASET_LICENSES)}")

    root = Path(path) if path else _repo_data_root() / name
    candidates = [
        root / f"{split}.jsonl",
        root / "sample.jsonl",
        root / f"{split}.json",
    ]
    file_path = next((p for p in candidates if p.exists()), None)
    if file_path is None:
        raise FileNotFoundError(
            f"No local data for {name}/{split}. Place JSONL at {root / 'sample.jsonl'} "
            f"(license: {DATASET_LICENSES[name]})"
        )

    raw: list[dict[str, Any]]
    if file_path.suffix == ".jsonl":
        raw = _load_jsonl(file_path)
    else:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            raw = list(payload.values()) if payload else []
            # PubMedQA official sometimes uses id->record map
            if raw and not isinstance(raw[0], dict):
                raw = [payload]
            elif payload and all(isinstance(v, dict) for v in payload.values()):
                raw = [{"id": k, **v} for k, v in payload.items()]
        else:
            raw = list(payload)

    normalizer = _NORMALIZERS[name]
    items = [normalizer(row, i) for i, row in enumerate(raw)]
    if limit is not None:
        items = items[:limit]

    info = DatasetInfo(
        name=name,
        split=split,
        n_items=len(items),
        license=DATASET_LICENSES[name],
        path=str(file_path),
    )
    logger.info(
        "eval.dataset_loaded",
        extra={"dataset": name, "n": len(items), "path": str(file_path)},
    )
    return info, items
