"""MIRAGE-style prompt templates with few-shot exemplars."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from biolit.eval.datasets import EvalItem

ANSWER_DELIMITER = "The answer is"

DatasetKind = Literal["mcqa", "ynm"]

MCQA_DATASETS = frozenset({"medqa", "medmcqa", "mmlu_med"})
YNM_DATASETS = frozenset({"pubmedqa", "bioasq"})


def dataset_kind(dataset: str) -> DatasetKind:
    if dataset in YNM_DATASETS:
        return "ynm"
    return "mcqa"


def fewshot_path(dataset: str) -> Path:
    root = Path(__file__).resolve().parents[3] / "data" / "fewshot"
    return root / f"{dataset}.json"


def load_fewshot(dataset: str, k: int) -> list[dict[str, Any]]:
    """Load up to ``k`` exemplars from ``data/fewshot/<dataset>.json``."""
    if k <= 0:
        return []
    path = fewshot_path(dataset)
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return []
    return [ex for ex in raw[:k] if isinstance(ex, dict)]


def _format_options(options: dict[str, str] | None) -> str:
    if not options:
        return ""
    lines = ["Options:"]
    for key, val in options.items():
        lines.append(f"  ({key}) {val}")
    return "\n".join(lines)


def _format_exemplar(ex: dict[str, Any], *, kind: DatasetKind) -> str:
    q = str(ex.get("question") or "").strip()
    gold = str(ex.get("gold") or "").strip()
    options = ex.get("options")
    parts = ["--- Example ---", f"Question: {q}"]
    if isinstance(options, dict) and options:
        parts.append(_format_options({str(k): str(v) for k, v in options.items()}))
    if kind == "ynm":
        parts.append(f"{ANSWER_DELIMITER} ({gold.lower()})")
    else:
        parts.append(f"{ANSWER_DELIMITER} ({gold.upper()})")
    return "\n".join(parts)


def build_prompt(
    item: EvalItem,
    *,
    dataset: str,
    mode: str,
    few_shot_k: int = 0,
    snippets: list[str] | None = None,
) -> str:
    """
    Build a MIRAGE-style prompt.

    Instructs the model to end with ``The answer is (X)``. Few-shot blocks are
    marked with ``--- Example ---`` so tests / analytics can detect them.
    """
    kind = dataset_kind(dataset)
    if kind == "ynm":
        label_help = (
            "Answer with exactly one of: yes, no, maybe. "
            f'End your response with "{ANSWER_DELIMITER} (yes)" '
            f"(or no / maybe), using that delimiter verbatim."
        )
    else:
        label_help = (
            "Answer with a single option letter (A/B/C/D/…). "
            f'End your response with "{ANSWER_DELIMITER} (X)" '
            "where X is the letter, using that delimiter verbatim."
        )

    parts: list[str] = [
        "You are a careful biomedical reasoning assistant.",
        "Think step by step, then commit to one final label.",
        label_help,
        "",
    ]

    exemplars = load_fewshot(dataset, few_shot_k)
    for ex in exemplars:
        parts.append(_format_exemplar(ex, kind=kind))
        parts.append("")

    parts.append("Your response:")
    parts.append(f"Question: {item.question}")
    if item.options:
        parts.append(_format_options(item.options))

    if mode == "rag" and snippets:
        parts.append("")
        parts.append("Retrieved evidence:")
        for i, snip in enumerate(snippets, start=1):
            parts.append(f"[{i}] {snip}")

    parts.append("")
    parts.append(f'Remember: finish with "{ANSWER_DELIMITER} (X)" on its own line.')
    return "\n".join(parts)
