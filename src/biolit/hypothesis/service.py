"""Hypothesis run orchestration and persistence."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from biolit.core.db import Hypothesis, HypothesisRun, TournamentMatch, get_session_factory
from biolit.core.llm import track_token_usage
from biolit.core.logging import get_logger
from biolit.hypothesis.graph import build_hypothesis_graph
from biolit.hypothesis.models import HypothesisConfig, HypothesisDraft, HypothesisRunResult
from biolit.hypothesis.provenance import enforce_cite_only, persist_evidence, require_provenance

logger = get_logger(__name__)


async def _persist_hypotheses(
    run_id: str,
    drafts: list[HypothesisDraft],
) -> None:
    ordered = sorted(drafts, key=lambda d: (d.generation, d.id))
    factory = get_session_factory()
    async with factory() as session:
        for d in ordered:
            if not d.has_provenance():
                continue
            session.add(
                Hypothesis(
                    id=d.id,
                    run_id=run_id,
                    statement=d.statement,
                    rationale=d.rationale,
                    mechanism=d.mechanism,
                    experiment=d.experiment,
                    falsification=d.falsification,
                    novelty=d.novelty,
                    feasibility=d.feasibility,
                    elo=d.elo,
                    generation=d.generation,
                    parent_id=d.parent_id,
                    cluster_id=d.cluster_id,
                    status=d.status,
                )
            )
        await session.flush()
        await session.commit()

    for d in ordered:
        if d.has_provenance():
            await persist_evidence(d.id, d.evidence)


async def execute_hypothesis(
    research_goal: str,
    config: HypothesisConfig | dict[str, Any] | None = None,
) -> HypothesisRunResult:
    """Run the LangGraph hypothesis engine and persist proposals with evidence."""
    cfg = (
        config
        if isinstance(config, HypothesisConfig)
        else HypothesisConfig.model_validate(config or {})
    )
    run_id = str(uuid4())
    factory = get_session_factory()
    async with factory() as session:
        session.add(
            HypothesisRun(
                id=run_id,
                research_goal=research_goal,
                config_json=cfg.model_dump(),
                status="running",
            )
        )
        await session.commit()

    graph = build_hypothesis_graph()
    initial = {
        "research_goal": research_goal,
        "config": cfg.model_dump(),
        "run_id": run_id,
        "hypotheses": [],
        "proposals": [],
        "matches": [],
        "evolution_round": 0,
        "tournament_round": 0,
        "human_feedback": cfg.human_feedback,
        "budget_used": {},
        "allowed_pmids": [],
    }
    try:
        with track_token_usage(cfg.max_total_tokens) as usage:
            final = await graph.ainvoke(initial)
        allowed = {str(p) for p in (final.get("allowed_pmids") or [])}
        proposals = [HypothesisDraft.model_validate(p) for p in (final.get("proposals") or [])]
        proposals = require_provenance(proposals)
        proposals = enforce_cite_only(proposals, allowed, strict=True)
        all_hyps = [HypothesisDraft.model_validate(h) for h in (final.get("hypotheses") or [])]
        # Persist all active/proposal hypotheses that have evidence.
        to_store = [
            h
            for h in all_hyps + proposals
            if h.has_provenance() and h.status in {"active", "proposal", "duplicate"}
        ]
        # Dedupe by id
        uniq: dict[str, HypothesisDraft] = {h.id: h for h in to_store}
        # Ensure proposals are marked and stored
        for p in proposals:
            uniq[p.id] = p
        await _persist_hypotheses(run_id, list(uniq.values()))

        # Persist tournament matches now that hypotheses exist.
        match_rows = list(final.get("matches") or [])
        if match_rows:
            async with factory() as session:
                for m in match_rows:
                    session.add(
                        TournamentMatch(
                            id=m["id"],
                            run_id=run_id,
                            hyp_a=m["hyp_a"],
                            hyp_b=m["hyp_b"],
                            winner=m.get("winner"),
                            judge_rationale=m.get("judge_rationale"),
                            round=int(m.get("round") or 0),
                        )
                    )
                await session.commit()

        # Verify invariant: every proposal has evidence + cite-only + unvalidated_lead
        for p in proposals:
            if not p.has_provenance():
                raise RuntimeError(f"Proposal {p.id} missing evidence")
            if not p.unvalidated_lead:
                raise RuntimeError(f"Proposal {p.id} missing unvalidated_lead=true")
            illegal = sorted(p.cited_pmids() - allowed)
            if illegal:
                raise RuntimeError(f"Proposal {p.id} cites non-retrieved PMIDs: {illegal}")

        budget = dict(final.get("budget_used") or {})
        budget["n_matches"] = len(final.get("matches") or [])
        budget["n_hypotheses"] = len(uniq)
        budget["n_proposals"] = len(proposals)
        budget["prompt_tokens"] = usage.prompt_tokens
        budget["completion_tokens"] = usage.completion_tokens
        budget["total_tokens"] = usage.total_tokens
        budget["llm_calls"] = usage.calls

        async with factory() as session:
            run = await session.get(HypothesisRun, run_id)
            if run is not None:
                run.status = "completed"
                run.finished_at = datetime.now(UTC)
                run.budget_used = budget
            await session.commit()

        return HypothesisRunResult(
            run_id=run_id,
            status="completed",
            research_goal=research_goal,
            proposals=proposals,
            n_hypotheses=len(uniq),
            n_matches=int(budget.get("n_matches") or 0),
            budget_used=budget,
            retrieved_pmids=sorted(allowed),
        )
    except Exception as exc:
        logger.exception("hypothesis.run_failed", extra={"run_id": run_id})
        async with factory() as session:
            run = await session.get(HypothesisRun, run_id)
            if run is not None:
                run.status = "failed"
                run.finished_at = datetime.now(UTC)
                run.budget_used = {"error": str(exc)}
            await session.commit()
        raise
