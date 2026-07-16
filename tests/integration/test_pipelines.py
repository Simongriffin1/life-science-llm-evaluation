"""Integration tests: retrieval → eval(RAG) and retrieval → hypothesis."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from biolit.core.costing import estimate_eval_cost, estimate_hypothesis_cost
from biolit.core.db import init_db
from biolit.eval.service import EvalConfig, execute_eval
from biolit.hypothesis.models import HypothesisConfig
from biolit.hypothesis.service import execute_hypothesis
from biolit.ingest.models import PubMedDocument
from biolit.retrieval.service import RetrieveResult, ScoredDocument, retrieve


@pytest.fixture
async def integration_env() -> AsyncGenerator[None, None]:
    await init_db()

    docs = [
        PubMedDocument(
            pmid="7001",
            title="TREM2 and microglial phagocytosis",
            abstract="TREM2 signaling promotes phagocytosis of amyloid in microglia.",
        ),
        PubMedDocument(
            pmid="7002",
            title="Ferroptosis pathway in ALS",
            abstract="Lipid peroxidation markers rise in ALS motor neurons.",
        ),
    ]

    async def fake_retrieve(query: str, **kwargs: Any) -> RetrieveResult:
        return RetrieveResult(
            query=query,
            mode=str(kwargs.get("mode") or "bm25"),
            top_k=2,
            use_index=False,
            documents=[
                ScoredDocument(
                    pmid=d.pmid,
                    title=d.title,
                    abstract=d.abstract,
                    rank=i + 1,
                    score=1.0 - i * 0.1,
                    scores={"bm25": 1.0 - i * 0.1},
                    highlight=f"**{(d.title or '')[:40]}**",
                )
                for i, d in enumerate(docs)
            ],
        )

    async def fake_complete(
        model: str,
        messages: list[dict[str, str]],
        **_k: Any,
    ) -> dict[str, Any]:
        system = messages[0]["content"].lower()
        user = messages[-1]["content"]
        if "json array" in system and "hypotheses" in system:
            content = """[
              {"statement":"Boosting TREM2 reduces pathology via phagocytosis",
               "rationale":"Grounded in PMID 7001",
               "mechanism":"TREM2-DAP12 signaling",
               "experiment":"Agonist antibody in microglia culture",
               "falsification":"If phagocytosis is unchanged, reject",
               "pmid_citations":["7001"]}
            ]"""
        elif "critique" in system:
            content = '{"reflection":"ok","novelty":0.6,"feasibility":0.7,"keep":true}'
        elif "judging two hypotheses" in system:
            content = '{"winner":"A","rationale":"better evidence"}'
        elif (
            "evolve" in system
            or "offspring" in user.lower()
            or "research agenda" in system
            or "falsification criterion" in system
        ):
            content = "[]"
        else:
            # Eval CoT: answer yes for pubmedqa-style
            import re

            m = re.search(r"intervention (\d+)", user)
            idx = int(m.group(1)) if m else 0
            ans = ["yes", "no", "maybe"][idx % 3]
            if model == "bad-model":
                ans = ["no", "maybe", "yes"][idx % 3]
            content = f"Reasoning.\nAnswer: {ans}"
        return {
            "content": content,
            "model": model,
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "total_tokens": 20,
            "finish_reason": "stop",
        }

    with (
        patch("biolit.retrieval.service.PubMedClient") as mock_client_cls,
        patch(
            "biolit.eval.runners.retrieve",
            new=AsyncMock(side_effect=fake_retrieve),
        ),
        patch(
            "biolit.hypothesis.agents.generation.retrieve",
            new=AsyncMock(side_effect=fake_retrieve),
        ),
        patch(
            "biolit.eval.runners._cached_complete",
            new=AsyncMock(side_effect=fake_complete),
        ),
        patch(
            "biolit.hypothesis.agents.generation.complete",
            new=AsyncMock(side_effect=fake_complete),
        ),
        patch(
            "biolit.hypothesis.agents.reflection.complete",
            new=AsyncMock(side_effect=fake_complete),
        ),
        patch(
            "biolit.hypothesis.agents.evolution.complete",
            new=AsyncMock(side_effect=fake_complete),
        ),
        patch(
            "biolit.hypothesis.agents.meta_review.complete",
            new=AsyncMock(side_effect=fake_complete),
        ),
        patch(
            "biolit.hypothesis.tournament.complete",
            new=AsyncMock(side_effect=fake_complete),
        ),
        patch("biolit.retrieval.service._persist", new=AsyncMock(return_value="q")),
    ):
        # Live retrieve() path for retrieval→X tests uses PubMedClient
        instance = AsyncMock()
        instance.search_documents = AsyncMock(
            return_value=[
                PubMedDocument(
                    pmid=d.pmid,
                    title=d.title,
                    abstract=d.abstract,
                )
                for d in docs
            ]
        )
        instance.aclose = AsyncMock()
        mock_client_cls.return_value = instance
        yield


@pytest.mark.asyncio
async def test_retrieval_feeds_eval_rag(integration_env: None) -> None:
    # Retrieval produces scored docs
    ret = await retrieve("TREM2 microglia", mode="bm25", top_k=2, persist=False)
    assert ret.documents
    assert ret.documents[0].pmid

    # Eval RAG consumes retrieval (mocked to same corpus)
    result = await execute_eval(
        EvalConfig(
            models=["good-model"],
            datasets=["pubmedqa"],
            mode="rag",
            limit=5,
            retrieval={"mode": "bm25", "top_k": 2, "candidate_cap": 5},
            use_cache=False,
        )
    )
    assert result.status == "completed"
    assert result.run_ids
    row = next(r for r in result.leaderboard if r["mode"] == "rag")
    assert row["accuracy"] is not None


@pytest.mark.asyncio
async def test_retrieval_feeds_hypothesis(integration_env: None) -> None:
    ret = await retrieve("ferroptosis ALS", mode="bm25", top_k=2, persist=False)
    assert len(ret.documents) >= 1

    result = await execute_hypothesis(
        "Modulate ferroptosis in ALS with a repurposable drug",
        HypothesisConfig(
            n_seed=1,
            evolution_rounds=0,
            max_proposals=1,
            retrieval_mode="bm25",
        ),
    )
    assert result.status == "completed"
    assert result.proposals
    assert all(p.has_provenance() for p in result.proposals)


def test_dry_run_cost_estimates() -> None:
    ev = estimate_eval_cost(
        models=["gpt-4o-mini"],
        datasets=["pubmedqa"],
        modes=["closed_book", "rag"],
        limit=20,
    )
    assert ev.n_llm_calls == 40
    assert ev.est_cost_usd > 0

    hyp = estimate_hypothesis_cost(
        model="gpt-4o-mini",
        n_seed=6,
        evolution_rounds=1,
    )
    assert hyp.n_llm_calls > 0
    assert hyp.est_total_tokens > 0
