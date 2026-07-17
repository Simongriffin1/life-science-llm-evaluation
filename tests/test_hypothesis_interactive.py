"""Interactive human-feedback gate for hypothesis runs."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from biolit.core.db import init_db
from biolit.hypothesis.feedback import ReviewDecision, apply_review_decision
from biolit.hypothesis.models import EvidenceRef, HypothesisConfig, HypothesisDraft
from biolit.hypothesis.service import (
    execute_hypothesis,
    get_pending_review,
    resume_hypothesis,
    submit_feedback,
)
from biolit.ingest.models import PubMedDocument
from biolit.retrieval.service import RetrieveResult, ScoredDocument


def test_apply_review_reject_and_redirect() -> None:
    parent = HypothesisDraft(
        id="p1",
        statement="Parent",
        evidence=[EvidenceRef(pmid="1", snippet="s")],
    )
    child = HypothesisDraft(
        id="c1",
        statement="Child",
        parent_id="p1",
        evidence=[EvidenceRef(pmid="1", snippet="s")],
    )
    all_hyps, rejected, notes = apply_review_decision(
        [parent, child],
        ReviewDecision(
            actions=[
                {"id": "p1", "action": "reject"},
                {"id": "c1", "action": "redirect", "note": "focus on ferroptosis"},
            ]
        ),
    )
    assert "p1" in rejected and "c1" in rejected
    assert all(h.status == "rejected" for h in all_hyps)
    assert any("ferroptosis" in n for n in notes)


@pytest.fixture
async def hyp_interactive_env() -> AsyncGenerator[None, None]:
    await init_db()

    docs = [
        PubMedDocument(
            pmid="5001",
            title="Ferroptosis in ALS motor neurons",
            abstract="Iron-dependent lipid peroxidation contributes to motor neuron death.",
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
        content = messages[-1]["content"]
        system = messages[0]["content"].lower()
        if "json array" in system and "hypotheses" in system:
            body = """[
              {"statement":"Drug X reduces ferroptosis in ALS via GPX4",
               "rationale":"Lit links ferroptosis to ALS",
               "mechanism":"GPX4 upregulation",
               "experiment":"Treat neurons with Drug X",
               "falsification":"If lipid ROS unchanged, reject",
               "pmid_citations":["5001","5002"]},
              {"statement":"Drug Y blocks iron uptake in ALS",
               "rationale":"Iron dependence",
               "mechanism":"Lower labile iron",
               "experiment":"Measure Fe2+ after Drug Y",
               "falsification":"If Fe2+ unchanged, reject",
               "pmid_citations":["5001"]}
            ]"""
        elif "critique" in system:
            body = '{"reflection":"Solid","novelty":0.7,"feasibility":0.8,"keep":true}'
        elif "judging two hypotheses" in system:
            body = '{"winner":"A","rationale":"Clearer mechanism"}'
        elif "evolve" in system or "offspring" in content.lower():
            steered = "steering constraints" in content.lower() or "ferroptosis" in content.lower()
            stmt = (
                "Steered: prioritize ferroptosis with Drug X"
                if steered
                else "Drug X + chelation synergy"
            )
            body = f"""[
              {{"statement":"{stmt}",
               "mechanism":"GPX4 plus lower Fe2+",
               "experiment":"Combinatorial treatment",
               "falsification":"No synergy rejects",
               "parent_ids":[],
               "pmid_citations":["5001","5002"]}}
            ]"""
        elif "research agenda" in system or "falsification criterion" in system:
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
async def test_interactive_pause_feedback_resume(hyp_interactive_env: None) -> None:
    paused = await execute_hypothesis(
        "Modulate ferroptosis in ALS",
        HypothesisConfig(
            n_seed=2,
            evolution_rounds=1,
            max_proposals=1,
            interactive=True,
            retrieval_mode="bm25",
            model="gpt-4o-mini",
        ),
    )
    assert paused.status == "paused_for_review"
    pending = await get_pending_review(paused.run_id)
    hyps = pending["hypotheses"]
    assert len(hyps) >= 2
    reject_id = hyps[0]["id"]
    keep_id = hyps[1]["id"]

    decision = ReviewDecision(
        actions=[
            {"id": reject_id, "action": "reject"},
            {
                "id": keep_id,
                "action": "redirect",
                "note": "Prioritize ferroptosis mechanisms",
            },
        ]
    )
    await submit_feedback(paused.run_id, decision)
    mid = await resume_hypothesis(paused.run_id, decision)
    # After evolve, interactive pauses again before meta-review
    assert mid.status in {"paused_for_review", "completed"}
    if mid.status == "paused_for_review":
        pending2 = await get_pending_review(paused.run_id)
        actions = [{"id": h["id"], "action": "accept"} for h in pending2["hypotheses"]]
        final = await resume_hypothesis(paused.run_id, ReviewDecision(actions=actions))
    else:
        final = mid
    assert final.status == "completed"
    assert reject_id in (final.budget_used.get("rejected_ids") or [])
    notes = final.budget_used.get("steering_notes") or []
    assert any("ferroptosis" in n.lower() for n in notes)
    assert final.proposals
    # Rejected lineage must not appear as a proposal statement source id
    assert all(p.id != reject_id for p in final.proposals)
