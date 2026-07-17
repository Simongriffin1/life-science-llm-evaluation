"""Hypothesis run orchestration, interactive pause/resume, and persistence."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from langgraph.types import Command

from biolit.core.db import Hypothesis, HypothesisRun, TournamentMatch, get_session_factory
from biolit.core.llm import track_token_usage
from biolit.core.logging import get_logger
from biolit.hypothesis.feedback import ReviewDecision
from biolit.hypothesis.graph import build_hypothesis_graph, thread_config
from biolit.hypothesis.models import HypothesisConfig, HypothesisDraft, HypothesisRunResult
from biolit.hypothesis.provenance import enforce_cite_only, persist_evidence, require_provenance

logger = get_logger(__name__)

_GRAPH = build_hypothesis_graph()


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


async def _mark_run(
    run_id: str,
    *,
    status: str,
    budget: dict[str, Any] | None = None,
) -> None:
    factory = get_session_factory()
    async with factory() as session:
        run = await session.get(HypothesisRun, run_id)
        if run is not None:
            run.status = status
            if budget is not None:
                run.budget_used = budget
            if status in {"completed", "failed"}:
                run.finished_at = datetime.now(UTC)
            await session.commit()


def _interrupt_payload(final: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = final.get("__interrupt__") or []
    if not interrupts:
        return None
    first = interrupts[0]
    value = getattr(first, "value", first)
    return dict(value) if isinstance(value, dict) else {"raw": value}


async def _finalize_completed(
    run_id: str,
    research_goal: str,
    final: dict[str, Any],
    usage: Any,
) -> HypothesisRunResult:
    allowed = {str(p) for p in (final.get("allowed_pmids") or [])}
    proposals = [HypothesisDraft.model_validate(p) for p in (final.get("proposals") or [])]
    proposals = require_provenance(proposals)
    proposals = enforce_cite_only(proposals, allowed, strict=True)
    all_hyps = [HypothesisDraft.model_validate(h) for h in (final.get("hypotheses") or [])]
    to_store = [
        h
        for h in all_hyps + proposals
        if h.has_provenance() and h.status in {"active", "proposal", "duplicate", "rejected"}
    ]
    uniq: dict[str, HypothesisDraft] = {h.id: h for h in to_store}
    for p in proposals:
        uniq[p.id] = p
    await _persist_hypotheses(run_id, list(uniq.values()))

    match_rows = list(final.get("matches") or [])
    known_ids = set(uniq.keys())
    match_rows = [
        m for m in match_rows if m.get("hyp_a") in known_ids and m.get("hyp_b") in known_ids
    ]
    if match_rows:
        factory = get_session_factory()
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
    budget["prompt_tokens"] = getattr(usage, "prompt_tokens", 0)
    budget["completion_tokens"] = getattr(usage, "completion_tokens", 0)
    budget["total_tokens"] = getattr(usage, "total_tokens", 0)
    budget["llm_calls"] = getattr(usage, "calls", 0)
    budget["steering_notes"] = list(final.get("steering_notes") or [])
    budget["rejected_ids"] = list(final.get("rejected_ids") or [])

    await _mark_run(run_id, status="completed", budget=budget)
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


async def execute_hypothesis(
    research_goal: str,
    config: HypothesisConfig | dict[str, Any] | None = None,
) -> HypothesisRunResult:
    """Run the LangGraph hypothesis engine (pauses when interactive)."""
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

    initial: dict[str, Any] = {
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
        "steering_notes": [],
        "rejected_ids": [],
        "pending_review": [],
    }
    tcfg = thread_config(run_id)
    try:
        with track_token_usage(cfg.max_total_tokens) as usage:
            final = await _GRAPH.ainvoke(initial, tcfg)
        pause = _interrupt_payload(final)
        if pause is not None:
            budget = {
                **(final.get("budget_used") or {}),
                "pending_review": pause,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "llm_calls": usage.calls,
            }
            await _mark_run(run_id, status="paused_for_review", budget=budget)
            pending = list(pause.get("hypotheses") or [])
            return HypothesisRunResult(
                run_id=run_id,
                status="paused_for_review",
                research_goal=research_goal,
                proposals=[],
                n_hypotheses=len(pending),
                n_matches=len(final.get("matches") or []),
                budget_used=budget,
                retrieved_pmids=list(final.get("allowed_pmids") or []),
            )
        return await _finalize_completed(run_id, research_goal, final, usage)
    except Exception as exc:
        logger.exception("hypothesis.run_failed", extra={"run_id": run_id})
        await _mark_run(run_id, status="failed", budget={"error": str(exc)})
        raise


async def get_pending_review(run_id: str) -> dict[str, Any]:
    """Return ranked hypotheses awaiting human review for a paused run."""
    factory = get_session_factory()
    async with factory() as session:
        run = await session.get(HypothesisRun, run_id)
        if run is None:
            raise KeyError(f"Unknown hypothesis run {run_id}")
        budget = run.budget_used or {}
        pending = budget.get("pending_review") or {}
        return {
            "run_id": run_id,
            "status": run.status,
            "research_goal": run.research_goal,
            "tournament_round": pending.get("tournament_round"),
            "evolution_round": pending.get("evolution_round"),
            "hypotheses": pending.get("hypotheses") or [],
        }


async def submit_feedback(run_id: str, decision: ReviewDecision | dict[str, Any]) -> dict[str, Any]:
    """Store feedback for the current pause (applied on resume)."""
    parsed = (
        decision
        if isinstance(decision, ReviewDecision)
        else ReviewDecision.model_validate(decision)
    )
    factory = get_session_factory()
    async with factory() as session:
        run = await session.get(HypothesisRun, run_id)
        if run is None:
            raise KeyError(f"Unknown hypothesis run {run_id}")
        if run.status != "paused_for_review":
            raise RuntimeError(f"Run {run_id} is not paused_for_review (status={run.status})")
        budget = dict(run.budget_used or {})
        budget["pending_feedback"] = parsed.model_dump()
        run.budget_used = budget
        await session.commit()
    return {"run_id": run_id, "status": "feedback_recorded", "n_actions": len(parsed.actions)}


async def resume_hypothesis(
    run_id: str, decision: ReviewDecision | dict[str, Any] | None = None
) -> HypothesisRunResult:
    """Resume an interactive run from its LangGraph checkpoint."""
    factory = get_session_factory()
    async with factory() as session:
        run = await session.get(HypothesisRun, run_id)
        if run is None:
            raise KeyError(f"Unknown hypothesis run {run_id}")
        if run.status != "paused_for_review":
            raise RuntimeError(f"Run {run_id} is not paused_for_review")
        budget = dict(run.budget_used or {})
        stored = budget.get("pending_feedback")
        research_goal = run.research_goal
        cfg = HypothesisConfig.model_validate(run.config_json or {})

    if decision is None:
        if not stored:
            raise RuntimeError("No feedback recorded; POST /feedback first or pass decision")
        decision = stored
    parsed = (
        decision
        if isinstance(decision, ReviewDecision)
        else ReviewDecision.model_validate(decision)
    )

    await _mark_run(run_id, status="running")
    tcfg = thread_config(run_id)
    try:
        with track_token_usage(cfg.max_total_tokens) as usage:
            final = await _GRAPH.ainvoke(Command(resume=parsed.model_dump()), tcfg)
        pause = _interrupt_payload(final)
        if pause is not None:
            budget = {
                **(final.get("budget_used") or {}),
                "pending_review": pause,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "llm_calls": usage.calls,
            }
            await _mark_run(run_id, status="paused_for_review", budget=budget)
            return HypothesisRunResult(
                run_id=run_id,
                status="paused_for_review",
                research_goal=research_goal,
                proposals=[],
                n_hypotheses=len(pause.get("hypotheses") or []),
                n_matches=len(final.get("matches") or []),
                budget_used=budget,
                retrieved_pmids=list(final.get("allowed_pmids") or []),
            )
        return await _finalize_completed(run_id, research_goal, final, usage)
    except Exception as exc:
        logger.exception("hypothesis.resume_failed", extra={"run_id": run_id})
        await _mark_run(run_id, status="failed", budget={"error": str(exc)})
        raise
