"""Eval metrics: accuracy, F1, retrieval ranking, groundedness helpers, cost aggregates."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


def normalize_label(value: str) -> str:
    return " ".join(value.strip().lower().split())


def accuracy(predictions: list[str], golds: list[str]) -> float:
    if not golds:
        return 0.0
    correct = sum(
        1
        for p, g in zip(predictions, golds, strict=True)
        if normalize_label(p) == normalize_label(g)
    )
    return correct / len(golds)


def macro_f1(predictions: list[str], golds: list[str]) -> float:
    """Macro-averaged F1 over label set in golds ∪ predictions."""
    if not golds:
        return 0.0
    labels = sorted({normalize_label(x) for x in golds + predictions})
    f1s: list[float] = []
    pred_n = [normalize_label(p) for p in predictions]
    gold_n = [normalize_label(g) for g in golds]
    for label in labels:
        tp = sum(1 for p, g in zip(pred_n, gold_n, strict=True) if p == label and g == label)
        fp = sum(1 for p, g in zip(pred_n, gold_n, strict=True) if p == label and g != label)
        fn = sum(1 for p, g in zip(pred_n, gold_n, strict=True) if p != label and g == label)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec))
    return sum(f1s) / len(f1s) if f1s else 0.0


def recall_at_k(retrieved: list[list[str]], gold_pmids: list[list[str]], k: int) -> float:
    if not gold_pmids:
        return 0.0
    scores: list[float] = []
    for docs, gold in zip(retrieved, gold_pmids, strict=True):
        if not gold:
            continue
        hit = len(set(docs[:k]) & set(gold))
        scores.append(hit / len(set(gold)))
    return sum(scores) / len(scores) if scores else 0.0


def mrr(retrieved: list[list[str]], gold_pmids: list[list[str]]) -> float:
    if not gold_pmids:
        return 0.0
    scores: list[float] = []
    for docs, gold in zip(retrieved, gold_pmids, strict=True):
        if not gold:
            continue
        gold_set = set(gold)
        rr = 0.0
        for i, pmid in enumerate(docs, start=1):
            if pmid in gold_set:
                rr = 1.0 / i
                break
        scores.append(rr)
    return sum(scores) / len(scores) if scores else 0.0


def ndcg_at_k(retrieved: list[list[str]], gold_pmids: list[list[str]], k: int) -> float:
    if not gold_pmids:
        return 0.0

    def dcg(rels: list[int]) -> float:
        return sum(rel / math.log2(i + 1) for i, rel in enumerate(rels, start=1))

    scores: list[float] = []
    for docs, gold in zip(retrieved, gold_pmids, strict=True):
        if not gold:
            continue
        gold_set = set(gold)
        rels = [1 if d in gold_set else 0 for d in docs[:k]]
        ideal = sorted(rels, reverse=True)
        idcg = dcg(ideal)
        scores.append(0.0 if idcg == 0 else dcg(rels) / idcg)
    return sum(scores) / len(scores) if scores else 0.0


def aggregate_cost(items: list[dict[str, Any]]) -> dict[str, float]:
    prompt_tokens = sum(int(i.get("prompt_tokens") or 0) for i in items)
    completion_tokens = sum(int(i.get("completion_tokens") or 0) for i in items)
    latencies = [float(i["latency_ms"]) for i in items if i.get("latency_ms") is not None]
    return {
        "prompt_tokens": float(prompt_tokens),
        "completion_tokens": float(completion_tokens),
        "total_tokens": float(prompt_tokens + completion_tokens),
        "latency_ms_mean": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "n_items": float(len(items)),
    }


def compute_run_metrics(
    *,
    predictions: list[str],
    golds: list[str],
    retrieved_pmids: list[list[str]] | None = None,
    gold_pmids: list[list[str]] | None = None,
    groundedness_scores: list[float] | None = None,
    item_stats: list[dict[str, Any]] | None = None,
    k: int = 8,
) -> dict[str, Any]:
    """Aggregate metrics for one eval run."""
    metrics: dict[str, Any] = {
        "accuracy": accuracy(predictions, golds),
        "macro_f1": macro_f1(predictions, golds),
        "label_counts": dict(Counter(normalize_label(g) for g in golds)),
    }
    if retrieved_pmids is not None and gold_pmids is not None:
        metrics["recall_at_k"] = recall_at_k(retrieved_pmids, gold_pmids, k)
        metrics["mrr"] = mrr(retrieved_pmids, gold_pmids)
        metrics["ndcg_at_k"] = ndcg_at_k(retrieved_pmids, gold_pmids, k)
    if groundedness_scores:
        metrics["groundedness"] = sum(groundedness_scores) / len(groundedness_scores)
    if item_stats:
        metrics["cost"] = aggregate_cost(item_stats)
    return metrics
