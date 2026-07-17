"""Phase 7b/7c: few-shot effect, export immutability, leaderboard filters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from biolit.core.db import init_db
from biolit.eval.export import export_run, load_run_bundle
from biolit.eval.leaderboard import leaderboard
from biolit.eval.service import EvalConfig, execute_eval
from biolit.retrieval.service import RetrieveResult, ScoredDocument


@pytest.fixture
async def eval_mocks() -> Any:
    await init_db()

    async def fake_complete(
        model: str, messages: list[dict[str, str]], **_k: Any
    ) -> dict[str, Any]:
        content = messages[-1]["content"]
        # When few-shot examples are present, answer correctly with delimiter;
        # when absent, emit unparseable text for half the items.
        has_fewshot = "--- Example" in content
        q = content
        letter = "A"
        if (
            "Hypokalemia" in q
            or "loop diuretics" in q
            or "Kernig" in q
            or "Warfarin" in q
            or "INR" in q
        ):
            letter = "B"
        if has_fewshot:
            body = f"Reasoning.\nThe answer is ({letter})"
        else:
            body = "I am not sure how to format this."
        return {
            "content": body,
            "model": model,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "finish_reason": "stop",
        }

    async def fake_retrieve(query: str, **_k: Any) -> RetrieveResult:
        return RetrieveResult(
            query=query,
            mode="bm25",
            top_k=2,
            documents=[
                ScoredDocument(
                    pmid="1",
                    title="t",
                    abstract="a",
                    rank=1,
                    score=1.0,
                    scores={"bm25": 1.0},
                )
            ],
        )

    with (
        patch("biolit.eval.runners.complete", new=AsyncMock(side_effect=fake_complete)),
        patch("biolit.eval.runners._cached_complete", new=AsyncMock(side_effect=fake_complete)),
        patch("biolit.eval.runners.retrieve", new=AsyncMock(side_effect=fake_retrieve)),
        patch("biolit.core.cache.cache_get", new=AsyncMock(return_value=None)),
        patch("biolit.core.cache.cache_set", new=AsyncMock(return_value=True)),
    ):
        yield


@pytest.mark.asyncio
async def test_few_shot_k_changes_accuracy(eval_mocks: None, tmp_path: Path) -> None:
    zero = await execute_eval(
        EvalConfig(
            models=["m"],
            datasets=["medqa"],
            mode="closed_book",
            limit=20,
            few_shot_k=0,
            use_cache=False,
        )
    )
    five = await execute_eval(
        EvalConfig(
            models=["m"],
            datasets=["medqa"],
            mode="closed_book",
            limit=20,
            few_shot_k=5,
            use_cache=False,
        )
    )
    z_run = zero.run_ids[0]
    f_run = five.run_ids[0]
    z_bundle = await load_run_bundle(z_run)
    f_bundle = await load_run_bundle(f_run)
    assert z_bundle["metrics"]["few_shot_k"] == 0
    assert f_bundle["metrics"]["few_shot_k"] == 5
    assert "unparsed_rate" in z_bundle["metrics"]
    # few-shot compliant outputs should parse; zero-shot mock is unparseable
    assert z_bundle["metrics"]["unparsed_rate"] > f_bundle["metrics"]["unparsed_rate"]
    assert f_bundle["metrics"]["accuracy"] != z_bundle["metrics"].get("accuracy")


@pytest.mark.asyncio
async def test_export_json_csv_stable_hash(eval_mocks: None, tmp_path: Path) -> None:
    result = await execute_eval(
        EvalConfig(
            models=["m"],
            datasets=["medqa"],
            mode="closed_book",
            limit=5,
            few_shot_k=3,
            use_cache=False,
        )
    )
    run_id = result.run_ids[0]
    p1 = await export_run(run_id, fmt="json", out_dir=tmp_path)
    p2 = await export_run(run_id, fmt="json", out_dir=tmp_path)
    assert p1 == p2
    data1 = json.loads(p1.read_text())
    data2 = json.loads(p2.read_text())
    assert data1["content_hash"] == data2["content_hash"]
    csv_path = await export_run(run_id, fmt="csv", out_dir=tmp_path)
    assert csv_path.exists()
    assert "question_id" in csv_path.read_text().splitlines()[0]


@pytest.mark.asyncio
async def test_leaderboard_model_filter(eval_mocks: None) -> None:
    await execute_eval(
        EvalConfig(
            models=["alpha-model", "beta-model"],
            datasets=["medqa"],
            mode="closed_book",
            limit=3,
            few_shot_k=3,
            use_cache=False,
        )
    )
    rows = await leaderboard(model="alpha-model", dataset="medqa")
    assert rows
    assert all(r["model"] == "alpha-model" for r in rows)
