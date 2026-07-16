"""Hypothesis engine unit/integration tests with mocked LLM + retrieval."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from biolit.api.main import create_app
from biolit.core.db import Evidence, Hypothesis, get_session_factory, init_db
from biolit.hypothesis.models import HypothesisConfig
from biolit.hypothesis.service import execute_hypothesis
from biolit.hypothesis.tournament import update_elo
from biolit.ingest.models import PubMedDocument
from biolit.retrieval.service import RetrieveResult, ScoredDocument


def test_elo_winner_gains_rating() -> None:
    winner, loser = update_elo(1000.0, 1000.0, k=32)
    assert winner > 1000.0
    assert loser < 1000.0


@pytest.fixture
async def hyp_env() -> AsyncGenerator[None, None]:
    await init_db()

    docs = [
        PubMedDocument(
            pmid="5001",
            title="Ferroptosis in ALS motor neurons",
            abstract="Iron-dependent lipid peroxidation contributes to motor neuron death in ALS.",
        ),
        PubMedDocument(
            pmid="5002",
            title="Approved drug modulates GPX4",
            abstract="An FDA-approved compound increases GPX4 activity in neurons.",
        ),
    ]

    async def fake_retrieve(query: str, **_k: Any) -> RetrieveResult:
        return RetrieveResult(
            query=query,
            mode="bm25",
            top_k=2,
            documents=[
                ScoredDocument(
                    pmid=d.pmid,
                    title=d.title,
                    abstract=d.abstract,
                    rank=i + 1,
                    score=1.0,
                    scores={"bm25": 1.0},
                )
                for i, d in enumerate(docs)
            ],
        )

    async def fake_complete(
        model: str, messages: list[dict[str, str]], **_k: Any
    ) -> dict[str, Any]:
        content = messages[-1]["content"].lower()
        system = messages[0]["content"].lower()
        if "json array" in system and "hypotheses" in system:
            body = """[
              {"statement":"Drug X reduces ferroptosis in ALS motor neurons via GPX4",
               "rationale":"Literature links ferroptosis to ALS",
               "mechanism":"GPX4 upregulation limits lipid peroxidation",
               "experiment":"Treat iPSC motor neurons with Drug X and measure lipid ROS",
               "falsification":"If lipid ROS is unchanged, reject the hypothesis",
               "pmid_citations":["5001","5002"]},
              {"statement":"Drug Y blocks iron uptake to limit ferroptosis in ALS",
               "rationale":"Iron dependence of ferroptosis",
               "mechanism":"Reduced labile iron pool",
               "experiment":"Measure Fe2+ after Drug Y in ALS neurons",
               "falsification":"If Fe2+ is unchanged, reject",
               "pmid_citations":["5001"]}
            ]"""
        elif "critique" in system:
            body = '{"reflection":"Solid and testable","novelty":0.7,"feasibility":0.8,"keep":true}'
        elif "judging two hypotheses" in system:
            body = '{"winner":"A","rationale":"Clearer mechanism and falsification"}'
        elif "evolve" in system or "offspring" in content:
            body = """[
              {"statement":"Drug X + iron chelation synergistically limits ferroptosis in ALS",
               "mechanism":"GPX4 boost plus lower Fe2+",
               "experiment":"Combinatorial treatment in ALS iPSC neurons",
               "falsification":"No synergy on lipid ROS rejects the claim",
               "parent_ids":[],
               "pmid_citations":["5001","5002"]}
            ]"""
        elif "research agenda" in system or "falsification criterion" in system:
            body = """[
              {"id":"PLACEHOLDER",
               "statement":"Drug X reduces ferroptosis in ALS motor neurons via GPX4",
               "mechanism":"GPX4 upregulation limits lipid peroxidation",
               "experiment":"Treat iPSC motor neurons with Drug X and measure lipid ROS",
               "falsification":"If lipid ROS is unchanged under Drug X, reject the hypothesis"}
            ]"""
            # id patched below after generation — meta_review uses existing ids;
            # for robustness return empty and let fallback kick in when id missing
            body = "[]"
        else:
            body = '{"ok":true}'
        return {
            "content": body,
            "model": model,
            "prompt_tokens": 20,
            "completion_tokens": 40,
            "total_tokens": 60,
            "finish_reason": "stop",
        }

    with (
        patch(
            "biolit.hypothesis.agents.generation.retrieve",
            new=AsyncMock(side_effect=fake_retrieve),
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
    ):
        yield


@pytest.mark.asyncio
async def test_hypothesis_run_requires_evidence(hyp_env: None) -> None:
    result = await execute_hypothesis(
        "Identify a repurposable approved drug that modulates ferroptosis in ALS motor neurons",
        HypothesisConfig(
            n_seed=2,
            tournament_rounds=1,
            evolution_rounds=1,
            max_proposals=2,
            model="gpt-4o-mini",
            retrieval_mode="bm25",
        ),
    )
    assert result.status == "completed"
    assert result.proposals
    for p in result.proposals:
        assert p.mechanism
        assert p.experiment
        assert p.falsification
        assert p.elo is not None
        assert p.has_provenance()
        assert len(p.evidence) >= 1
        assert all(e.pmid for e in p.evidence)

    factory = get_session_factory()
    async with factory() as session:
        hyps = (await session.execute(select(Hypothesis))).scalars().all()
        assert hyps
        for h in hyps:
            ev = (
                (await session.execute(select(Evidence).where(Evidence.hypothesis_id == h.id)))
                .scalars()
                .all()
            )
            assert len(ev) >= 1


@pytest.mark.asyncio
async def test_post_hypothesize(hyp_env: None) -> None:
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
                "/hypothesize",
                json={
                    "research_goal": "Modulate ferroptosis in ALS with a repurposable drug",
                    "sync": True,
                    "config": {
                        "n_seed": 2,
                        "tournament_rounds": 1,
                        "evolution_rounds": 0,
                        "max_proposals": 2,
                    },
                },
            )
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["status"] == "completed"
            proposals = body["result"]["proposals"]
            assert proposals
            assert all(p["evidence"] for p in proposals)
            assert all(p["falsification"] for p in proposals)
