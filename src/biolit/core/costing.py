"""Dry-run cost / token estimation for eval and hypothesis sweeps."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# Approximate USD per 1M tokens (list prices; update as needed). Unknown models → gpt-4o-mini rates.
_RATES: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-opus-4-8": {"input": 15.00, "output": 75.00},
}


class CostEstimate(BaseModel):
    n_llm_calls: int
    est_prompt_tokens: int
    est_completion_tokens: int
    est_total_tokens: int
    est_cost_usd: float
    assumptions: dict[str, Any] = Field(default_factory=dict)
    note: str = (
        "Rough pre-flight estimate only. Actual spend depends on prompt length, "
        "caching, retries, and provider pricing."
    )


def _rate_for(model: str) -> dict[str, float]:
    if model in _RATES:
        return _RATES[model]
    for key, rates in _RATES.items():
        if key in model:
            return rates
    return _RATES["gpt-4o-mini"]


def _usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = _rate_for(model)
    return (prompt_tokens / 1_000_000) * rates["input"] + (completion_tokens / 1_000_000) * rates[
        "output"
    ]


def usd_for_usage(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Public wrapper for pricing actual token usage."""
    return round(_usd(model, prompt_tokens, completion_tokens), 6)


def estimate_eval_cost(
    *,
    models: list[str],
    datasets: list[str],
    modes: list[str],
    limit: int,
    prompt_tokens_per_item: int = 800,
    completion_tokens_per_item: int = 200,
    rag_extra_prompt_tokens: int = 600,
    judge_tokens: tuple[int, int] = (400, 150),
    with_judge: bool = False,
) -> CostEstimate:
    """Estimate cost for an eval sweep before running it."""
    n_items = limit * len(datasets)
    n_modes = len(modes)
    n_calls = 0
    prompt = 0
    completion = 0
    cost = 0.0

    for model in models:
        for mode in modes:
            ptok = prompt_tokens_per_item + (rag_extra_prompt_tokens if mode == "rag" else 0)
            ctok = completion_tokens_per_item
            calls = n_items
            n_calls += calls
            prompt += ptok * calls
            completion += ctok * calls
            cost += _usd(model, ptok * calls, ctok * calls)
            if with_judge and mode == "rag":
                jp, jc = judge_tokens
                n_calls += calls
                prompt += jp * calls
                completion += jc * calls
                cost += _usd(model, jp * calls, jc * calls)

    return CostEstimate(
        n_llm_calls=n_calls,
        est_prompt_tokens=prompt,
        est_completion_tokens=completion,
        est_total_tokens=prompt + completion,
        est_cost_usd=round(cost, 4),
        assumptions={
            "models": models,
            "datasets": datasets,
            "modes": modes,
            "limit_per_dataset": limit,
            "n_modes": n_modes,
            "prompt_tokens_per_item": prompt_tokens_per_item,
            "completion_tokens_per_item": completion_tokens_per_item,
            "with_judge": with_judge,
        },
    )


def estimate_hypothesis_cost(
    *,
    model: str,
    n_seed: int,
    evolution_rounds: int,
    tournament_rounds: int = 2,
    max_proposals: int = 3,
    prompt_tokens_per_call: int = 1200,
    completion_tokens_per_call: int = 400,
) -> CostEstimate:
    """
    Estimate LLM calls for generate + reflect + rank + evolve + meta-review.

    Tournament pairs ≈ n/2 per ranking pass; ranking passes ≈ evolution_rounds + 1.
    """
    # generate: 1, reflect: n_seed, rank passes: evo+1, evolve: evolution_rounds, meta: 1
    n_active = max(n_seed, 2)
    rank_passes = evolution_rounds + 1
    pairs_per_pass = max(n_active // 2, 1)
    # Cap growth loosely after evolution
    n_calls = (
        1  # generate
        + n_seed  # reflect
        + rank_passes * pairs_per_pass  # tournament judges
        + evolution_rounds  # evolve
        + 1  # meta-review
    )
    # tournament_rounds is advisory in current graph; fold into assumptions
    prompt = prompt_tokens_per_call * n_calls
    completion = completion_tokens_per_call * n_calls
    cost = _usd(model, prompt, completion)
    return CostEstimate(
        n_llm_calls=n_calls,
        est_prompt_tokens=prompt,
        est_completion_tokens=completion,
        est_total_tokens=prompt + completion,
        est_cost_usd=round(cost, 4),
        assumptions={
            "model": model,
            "n_seed": n_seed,
            "evolution_rounds": evolution_rounds,
            "tournament_rounds": tournament_rounds,
            "max_proposals": max_proposals,
            "rank_passes": rank_passes,
            "pairs_per_pass": pairs_per_pass,
        },
    )
