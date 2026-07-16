"""Unit tests for eval metrics."""

from biolit.eval.metrics import (
    accuracy,
    compute_run_metrics,
    macro_f1,
    mrr,
    ndcg_at_k,
    recall_at_k,
)


def test_accuracy_and_macro_f1() -> None:
    preds = ["yes", "no", "maybe", "yes"]
    gold = ["yes", "no", "yes", "yes"]
    assert accuracy(preds, gold) == 0.75
    assert 0.0 < macro_f1(preds, gold) <= 1.0


def test_retrieval_metrics() -> None:
    retrieved = [["1", "2", "3"], ["9", "8", "7"]]
    gold = [["2"], ["7", "5"]]
    assert recall_at_k(retrieved, gold, k=2) == 0.5  # first misses@2? wait 2 is at rank2
    # first: hit at rank 2 → recall 1.0; second: 7 at rank 3, @k=2 → 0 → mean 0.5
    assert abs(recall_at_k(retrieved, gold, k=2) - 0.5) < 1e-9
    assert mrr(retrieved, gold) == (0.5 + (1 / 3)) / 2
    assert 0.0 <= ndcg_at_k(retrieved, gold, k=3) <= 1.0


def test_compute_run_metrics_bundle() -> None:
    metrics = compute_run_metrics(
        predictions=["A", "B"],
        golds=["A", "B"],
        retrieved_pmids=[["1", "2"], ["3"]],
        gold_pmids=[["1"], ["9"]],
        groundedness_scores=[1.0, 0.0],
        item_stats=[
            {"prompt_tokens": 10, "completion_tokens": 5, "latency_ms": 100},
            {"prompt_tokens": 20, "completion_tokens": 5, "latency_ms": 200},
        ],
        k=2,
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["groundedness"] == 0.5
    assert metrics["cost"]["total_tokens"] == 40.0
