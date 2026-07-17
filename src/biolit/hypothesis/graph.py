"""LangGraph hypothesis engine: generate → reflect → rank → [review] → evolve → … → meta-review."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from biolit.core.logging import get_logger
from biolit.hypothesis.agents.evolution import evolve
from biolit.hypothesis.agents.generation import generate_seeds
from biolit.hypothesis.agents.meta_review import meta_review
from biolit.hypothesis.agents.proximity import cluster_hypotheses, ranking_pool
from biolit.hypothesis.agents.reflection import reflect_all
from biolit.hypothesis.feedback import ReviewDecision, apply_review_decision
from biolit.hypothesis.models import HypothesisConfig, HypothesisDraft
from biolit.hypothesis.provenance import enforce_cite_only, require_provenance
from biolit.hypothesis.tournament import run_tournament_round

logger = get_logger(__name__)

# Process-local checkpointer — required for interactive interrupt / resume.
_CHECKPOINTER = MemorySaver()


class HypothesisState(TypedDict, total=False):
    research_goal: str
    config: dict[str, Any]
    run_id: str
    hypotheses: list[dict[str, Any]]
    proposals: list[dict[str, Any]]
    matches: list[dict[str, Any]]
    evolution_round: int
    tournament_round: int
    human_feedback: str | None
    budget_used: dict[str, Any]
    allowed_pmids: list[str]
    steering_notes: list[str]
    rejected_ids: list[str]
    pending_review: list[dict[str, Any]]


def _cfg(state: HypothesisState) -> HypothesisConfig:
    return HypothesisConfig.model_validate(state.get("config") or {})


def _hyps(state: HypothesisState) -> list[HypothesisDraft]:
    return [HypothesisDraft.model_validate(h) for h in state.get("hypotheses") or []]


def _dump(hyps: list[HypothesisDraft]) -> list[dict[str, Any]]:
    return [h.model_dump() for h in hyps]


def _allowed(state: HypothesisState) -> set[str]:
    return {str(p) for p in (state.get("allowed_pmids") or [])}


def _goal_with_steering(state: HypothesisState) -> str:
    goal = state["research_goal"]
    feedback = state.get("human_feedback") or _cfg(state).human_feedback
    parts = [goal]
    if feedback:
        parts.append(f"Human feedback: {feedback}")
    notes = state.get("steering_notes") or []
    if notes:
        parts.append("Steering constraints:\n" + "\n".join(notes))
    return "\n".join(parts)


async def node_generate(state: HypothesisState) -> dict[str, Any]:
    cfg = _cfg(state)
    drafts, docs = await generate_seeds(state["research_goal"], cfg)
    allowed = {d.pmid for d in docs}
    drafts = require_provenance(drafts)
    drafts = enforce_cite_only(drafts, allowed, strict=True)
    return {
        "hypotheses": _dump(drafts),
        "allowed_pmids": sorted(allowed),
        "budget_used": {
            **(state.get("budget_used") or {}),
            "n_generated": len(drafts),
            "n_retrieved": len(docs),
        },
    }


async def node_reflect(state: HypothesisState) -> dict[str, Any]:
    cfg = _cfg(state)
    reflected = await reflect_all(
        _hyps(state),
        research_goal=_goal_with_steering(state),
        config=cfg,
    )
    reflected = require_provenance(reflected)
    reflected = enforce_cite_only(reflected, _allowed(state), strict=True)
    clustered = cluster_hypotheses(reflected)
    return {"hypotheses": _dump(clustered)}


async def node_rank(state: HypothesisState) -> dict[str, Any]:
    cfg = _cfg(state)
    pool = ranking_pool(_hyps(state))
    round_idx = int(state.get("tournament_round") or 0) + 1
    updated, matches = await run_tournament_round(
        pool,
        research_goal=_goal_with_steering(state),
        model=cfg.judge_model or cfg.model,
        run_id=state["run_id"],
        round_idx=round_idx,
        elo_k=cfg.elo_k,
        persist=False,
    )
    by_id = {h.id: h for h in _hyps(state)}
    for u in updated:
        by_id[u.id] = u
    all_matches = list(state.get("matches") or []) + matches
    return {
        "hypotheses": _dump(list(by_id.values())),
        "matches": all_matches,
        "tournament_round": round_idx,
    }


async def node_review(state: HypothesisState) -> dict[str, Any]:
    """Pause for human accept/reject/redirect when config.interactive is true."""
    cfg = _cfg(state)
    if not cfg.interactive:
        return {}

    pool = ranking_pool(_hyps(state))
    payload = {
        "run_id": state["run_id"],
        "tournament_round": state.get("tournament_round"),
        "evolution_round": state.get("evolution_round"),
        "hypotheses": [
            {
                "id": h.id,
                "statement": h.statement,
                "elo": h.elo,
                "generation": h.generation,
                "parent_id": h.parent_id,
                "evidence": [e.model_dump() for e in h.evidence],
            }
            for h in pool
        ],
    }
    decision_raw = interrupt(payload)
    decision = ReviewDecision.model_validate(decision_raw or {})
    kept, rejected, notes = apply_review_decision(
        _hyps(state),
        decision,
        steering_notes=list(state.get("steering_notes") or []),
    )
    prev_rejected = list(state.get("rejected_ids") or [])
    return {
        "hypotheses": _dump(kept),
        "steering_notes": notes,
        "rejected_ids": sorted(set(prev_rejected) | set(rejected)),
        "pending_review": [],
        "budget_used": {
            **(state.get("budget_used") or {}),
            "last_review": {
                "rejected": rejected,
                "steering_notes": notes,
                "n_actions": len(decision.actions),
            },
        },
    }


async def node_evolve(state: HypothesisState) -> dict[str, Any]:
    cfg = _cfg(state)
    evo_round = int(state.get("evolution_round") or 0) + 1
    parents = ranking_pool(_hyps(state))[:4]
    kids = await evolve(
        parents,
        research_goal=_goal_with_steering(state),
        config=cfg,
        generation=evo_round,
    )
    kids = require_provenance(kids)
    kids = enforce_cite_only(kids, _allowed(state), strict=True)
    rejected = set(state.get("rejected_ids") or [])
    kids = [k for k in kids if not (k.parent_id and k.parent_id in rejected)]
    merged = _hyps(state) + kids
    clustered = cluster_hypotheses(merged)
    return {
        "hypotheses": _dump(clustered),
        "evolution_round": evo_round,
    }


async def node_meta_review(state: HypothesisState) -> dict[str, Any]:
    cfg = _cfg(state)
    proposals = await meta_review(
        _hyps(state),
        research_goal=_goal_with_steering(state),
        config=cfg,
    )
    proposals = require_provenance(proposals)
    proposals = enforce_cite_only(proposals, _allowed(state), strict=True)
    return {"proposals": _dump(proposals)}


def _after_rank(state: HypothesisState) -> str:
    if _cfg(state).interactive:
        return "review"
    return _should_evolve(state)


def _should_evolve(state: HypothesisState) -> str:
    cfg = _cfg(state)
    done = int(state.get("evolution_round") or 0)
    if done < cfg.evolution_rounds:
        return "evolve"
    return "meta_review"


def build_hypothesis_graph(*, checkpointer: Any | None = None) -> Any:
    graph: StateGraph[HypothesisState] = StateGraph(HypothesisState)
    graph.add_node("generate", node_generate)
    graph.add_node("reflect", node_reflect)
    graph.add_node("rank", node_rank)
    graph.add_node("review", node_review)
    graph.add_node("evolve", node_evolve)
    graph.add_node("meta_review", node_meta_review)

    graph.set_entry_point("generate")
    graph.add_edge("generate", "reflect")
    graph.add_edge("reflect", "rank")
    graph.add_conditional_edges(
        "rank",
        _after_rank,
        {"review": "review", "evolve": "evolve", "meta_review": "meta_review"},
    )
    graph.add_conditional_edges(
        "review",
        _should_evolve,
        {"evolve": "evolve", "meta_review": "meta_review"},
    )
    graph.add_edge("evolve", "rank")
    graph.add_edge("meta_review", END)
    return graph.compile(checkpointer=checkpointer or _CHECKPOINTER)


def thread_config(run_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": run_id}}
