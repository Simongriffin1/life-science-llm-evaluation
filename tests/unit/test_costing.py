"""Unit tests for dry-run costing helpers."""

from biolit.core.costing import estimate_eval_cost, estimate_hypothesis_cost


def test_estimate_eval_scales_with_modes() -> None:
    one = estimate_eval_cost(
        models=["gpt-4o-mini"],
        datasets=["pubmedqa"],
        modes=["closed_book"],
        limit=10,
    )
    two = estimate_eval_cost(
        models=["gpt-4o-mini"],
        datasets=["pubmedqa"],
        modes=["closed_book", "rag"],
        limit=10,
    )
    assert two.n_llm_calls == one.n_llm_calls * 2
    assert two.est_cost_usd >= one.est_cost_usd


def test_estimate_hypothesis_positive() -> None:
    est = estimate_hypothesis_cost(model="gpt-4o-mini", n_seed=4, evolution_rounds=1)
    assert est.n_llm_calls >= 6
    assert est.est_total_tokens > 0
    assert "Rough pre-flight" in est.note
