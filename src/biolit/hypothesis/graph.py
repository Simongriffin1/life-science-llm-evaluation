"""LangGraph hypothesis engine: generate → reflect → rank → evolve → … → meta-review."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from biolit.core.logging import get_logger
from biolit.hypothesis.agents.evolution import evolve
from biolit.hypothesis.agents.generation import generate_seeds
from biolit.hypothesis.agents.meta_review import meta_review
from biolit.hypothesis.agents.proximity import cluster_hypotheses, ranking_pool
from biolit.hypothesis.agents.reflection import reflect_all
from biolit.hypothesis.models import HypothesisConfig, HypothesisDraft
from biolit.hypothesis.provenance import enforce_cite_only, require_provenance
from biolit.hypothesis.tournament import run_tournament_round

logger = get_logger(__name__)


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


def _cfg(state: HypothesisState) -> HypothesisConfig:
    return HypothesisConfig.model_validate(state.get("config") or {})


def _hyps(state: HypothesisState) -> list[HypothesisDraft]:
    return [HypothesisDraft.model_validate(h) for h in state.get("hypotheses") or []]


def _dump(hyps: list[HypothesisDraft]) -> list[dict[str, Any]]:
    return [h.model_dump() for h in hyps]


def _allowed(state: HypothesisState) -> set[str]:
    return {str(p) for p in (state.get("allowed_pmids") or [])}


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
    feedback = state.get("human_feedback") or cfg.human_feedback
    goal = state["research_goal"]
    if feedback:
        goal = f"{goal}\nHuman feedback: {feedback}"
    reflected = await reflect_all(_hyps(state), research_goal=goal, config=cfg)
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
        research_goal=state["research_goal"],
        model=cfg.judge_model or cfg.model,
        run_id=state["run_id"],
        round_idx=round_idx,
        elo_k=cfg.elo_k,
        persist=False,  # hypotheses are written after the graph completes
    )
    # Merge Elo updates back into full hypothesis list.
    by_id = {h.id: h for h in _hyps(state)}
    for u in updated:
        by_id[u.id] = u
    all_matches = list(state.get("matches") or []) + matches
    return {
        "hypotheses": _dump(list(by_id.values())),
        "matches": all_matches,
        "tournament_round": round_idx,
    }


async def node_evolve(state: HypothesisState) -> dict[str, Any]:
    cfg = _cfg(state)
    evo_round = int(state.get("evolution_round") or 0) + 1
    parents = ranking_pool(_hyps(state))[:4]
    kids = await evolve(
        parents,
        research_goal=state["research_goal"],
        config=cfg,
        generation=evo_round,
    )
    kids = require_provenance(kids)
    kids = enforce_cite_only(kids, _allowed(state), strict=True)
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
        research_goal=state["research_goal"],
        config=cfg,
    )
    proposals = require_provenance(proposals)
    proposals = enforce_cite_only(proposals, _allowed(state), strict=True)
    return {"proposals": _dump(proposals)}


def _should_evolve(state: HypothesisState) -> str:
    cfg = _cfg(state)
    done = int(state.get("evolution_round") or 0)
    if done < cfg.evolution_rounds:
        return "evolve"
    return "meta_review"


def build_hypothesis_graph() -> Any:
    graph: StateGraph[HypothesisState] = StateGraph(HypothesisState)
    graph.add_node("generate", node_generate)
    graph.add_node("reflect", node_reflect)
    graph.add_node("rank", node_rank)
    graph.add_node("evolve", node_evolve)
    graph.add_node("meta_review", node_meta_review)

    graph.set_entry_point("generate")
    graph.add_edge("generate", "reflect")
    graph.add_edge("reflect", "rank")
    graph.add_conditional_edges(
        "rank",
        _should_evolve,
        {"evolve": "evolve", "meta_review": "meta_review"},
    )
    graph.add_edge("evolve", "rank")
    graph.add_edge("meta_review", END)
    return graph.compile()
