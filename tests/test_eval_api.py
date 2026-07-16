"""Eval harness integration tests with mocked LLM + retrieval."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from biolit.api.main import create_app
from biolit.core.db import init_db
from biolit.eval.runners import parse_answer
from biolit.eval.service import EvalConfig, execute_eval
from biolit.retrieval.service import RetrieveResult, ScoredDocument


def test_parse_answer_extracts_label() -> None:
    text = "Reasoning...\nAnswer: yes"
    assert parse_answer(text, {"yes": "yes", "no": "no", "maybe": "maybe"}) == "yes"


@pytest.fixture
async def eval_env() -> AsyncGenerator[None, None]:
    await init_db()

    async def fake_complete(
        model: str, messages: list[dict[str, str]], **_k: Any
    ) -> dict[str, Any]:
        content = messages[-1]["content"]
        # Deterministic: echo gold-ish answers from question id embedded in prompt when present.
        # For pubmedqa sample, cycle yes/no/maybe from question number.
        import re

        m = re.search(r"intervention (\d+)", content)
        idx = int(m.group(1)) if m else 0
        ans = ["yes", "no", "maybe"][idx % 3]
        # Model "good" always correct; model "bad" always wrong.
        if model == "bad-model":
            ans = ["no", "maybe", "yes"][idx % 3]
        return {
            "content": f"Thinking...\nAnswer: {ans}",
            "model": model,
            "prompt_tokens": 12,
            "completion_tokens": 4,
            "total_tokens": 16,
            "finish_reason": "stop",
        }

    async def fake_retrieve(query: str, **_k: Any) -> RetrieveResult:
        return RetrieveResult(
            query=query,
            mode="bm25",
            top_k=2,
            use_index=False,
            documents=[
                ScoredDocument(
                    pmid="30000000",
                    title="Evidence paper",
                    abstract="Supports the intervention claim.",
                    rank=1,
                    score=1.0,
                    scores={"bm25": 1.0},
                    highlight="**intervention**",
                )
            ],
        )

    with (
        patch("biolit.eval.runners.complete", new=AsyncMock(side_effect=fake_complete)),
        patch("biolit.eval.runners._cached_complete", new=AsyncMock(side_effect=fake_complete)),
        patch("biolit.eval.judge.complete", new=AsyncMock(side_effect=fake_complete)),
        patch("biolit.eval.runners.retrieve", new=AsyncMock(side_effect=fake_retrieve)),
        patch("biolit.core.cache.cache_get", new=AsyncMock(return_value=None)),
        patch("biolit.core.cache.cache_set", new=AsyncMock(return_value=True)),
    ):
        yield


@pytest.mark.asyncio
async def test_execute_eval_two_models_two_modes(eval_env: None) -> None:
    result = await execute_eval(
        EvalConfig(
            models=["good-model", "bad-model"],
            datasets=["pubmedqa"],
            mode=["closed_book", "rag"],
            limit=20,
            retrieval={"mode": "bm25", "top_k": 2, "candidate_cap": 5},
            judge_model=None,
            use_cache=False,
        )
    )
    assert result.status == "completed"
    assert len(result.run_ids) == 4  # 2 models × 2 modes
    # Leaderboard should include rag deltas for both models
    rag_rows = [r for r in result.leaderboard if r["mode"] == "rag" and r["dataset"] == "pubmedqa"]
    assert len(rag_rows) >= 2
    for row in rag_rows:
        assert "rag_vs_closed_book_delta" in row
        assert row["accuracy"] is not None

    good_closed = next(
        r for r in result.leaderboard if r["model"] == "good-model" and r["mode"] == "closed_book"
    )
    assert good_closed["accuracy"] == 1.0


@pytest.mark.asyncio
async def test_post_eval_and_leaderboard(eval_env: None) -> None:
    app = create_app()
    with (
        patch("biolit.api.main.ping_db", new=AsyncMock(return_value={"ok": True})),
        patch("biolit.api.main.ping_redis", new=AsyncMock(return_value={"ok": True})),
        patch("biolit.api.main.close_redis", new=AsyncMock()),
        patch("biolit.api.main.dispose_engine", new=AsyncMock()),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/eval",
                json={
                    "models": ["good-model", "bad-model"],
                    "datasets": ["pubmedqa"],
                    "mode": ["closed_book", "rag"],
                    "limit": 20,
                    "sync": True,
                    "retrieval": {"mode": "bm25", "top_k": 2, "candidate_cap": 5},
                },
            )
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["status"] == "completed"
            assert body["result"]["run_ids"]
            board = await client.get("/eval/leaderboard")
            assert board.status_code == 200
            rows = board.json()["rows"]
            assert any(r.get("rag_vs_closed_book_delta") is not None for r in rows)
