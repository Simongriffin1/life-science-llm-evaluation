"""Pairwise Elo tournament with LLM judge."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from biolit.core.db import TournamentMatch, get_session_factory
from biolit.core.llm import complete
from biolit.core.logging import get_logger
from biolit.hypothesis.json_utils import extract_json_object
from biolit.hypothesis.models import HypothesisDraft

logger = get_logger(__name__)


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_elo(
    rating_winner: float,
    rating_loser: float,
    *,
    k: float = 32.0,
) -> tuple[float, float]:
    exp_w = expected_score(rating_winner, rating_loser)
    exp_l = expected_score(rating_loser, rating_winner)
    return rating_winner + k * (1.0 - exp_w), rating_loser + k * (0.0 - exp_l)


async def judge_pair(
    hyp_a: HypothesisDraft,
    hyp_b: HypothesisDraft,
    *,
    research_goal: str,
    model: str,
) -> dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a biomedical scientist judging two hypotheses for a research goal. "
                "Prefer testable, novel, mechanistically clear, well-evidenced ideas. "
                'Respond ONLY with JSON: {"winner": "A"|"B", "rationale": "<short>"}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Research goal: {research_goal}\n\n"
                f"Hypothesis A:\n{hyp_a.statement}\nMechanism: {hyp_a.mechanism}\n"
                f"Evidence PMIDs: {[e.pmid for e in hyp_a.evidence]}\n\n"
                f"Hypothesis B:\n{hyp_b.statement}\nMechanism: {hyp_b.mechanism}\n"
                f"Evidence PMIDs: {[e.pmid for e in hyp_b.evidence]}\n"
            ),
        },
    ]
    result = await complete(model, messages, temperature=0.0, max_tokens=400)
    try:
        parsed = extract_json_object(result["content"])
    except Exception:
        parsed = {"winner": "A", "rationale": "parse_failed_default_A"}
    winner = str(parsed.get("winner", "A")).upper()
    if winner not in {"A", "B"}:
        winner = "A"
    return {
        "winner": winner,
        "rationale": str(parsed.get("rationale") or ""),
        "prompt_tokens": result.get("prompt_tokens", 0),
        "completion_tokens": result.get("completion_tokens", 0),
    }


async def run_tournament_round(
    hypotheses: list[HypothesisDraft],
    *,
    research_goal: str,
    model: str,
    run_id: str,
    round_idx: int,
    elo_k: float = 32.0,
    persist: bool = True,
) -> tuple[list[HypothesisDraft], list[dict[str, Any]]]:
    """
    Pair adjacent hypotheses (by current Elo), judge, update Elo, optionally persist matches.
    """
    if len(hypotheses) < 2:
        return hypotheses, []

    ordered = sorted(hypotheses, key=lambda h: (-h.elo, h.id))
    matches: list[dict[str, Any]] = []
    by_id = {h.id: h for h in ordered}

    pairs = [(ordered[i], ordered[i + 1]) for i in range(0, len(ordered) - 1, 2)]
    for hyp_a, hyp_b in pairs:
        verdict = await judge_pair(hyp_a, hyp_b, research_goal=research_goal, model=model)
        winner_id = hyp_a.id if verdict["winner"] == "A" else hyp_b.id
        loser_id = hyp_b.id if winner_id == hyp_a.id else hyp_a.id
        w = by_id[winner_id]
        loser = by_id[loser_id]
        new_w, new_l = update_elo(w.elo, loser.elo, k=elo_k)
        w.elo = new_w
        loser.elo = new_l
        match = {
            "id": str(uuid4()),
            "hyp_a": hyp_a.id,
            "hyp_b": hyp_b.id,
            "winner": winner_id,
            "judge_rationale": verdict["rationale"],
            "round": round_idx,
        }
        matches.append(match)

    if persist and matches:
        factory = get_session_factory()
        async with factory() as session:
            for m in matches:
                session.add(
                    TournamentMatch(
                        id=m["id"],
                        run_id=run_id,
                        hyp_a=m["hyp_a"],
                        hyp_b=m["hyp_b"],
                        winner=m["winner"],
                        judge_rationale=m["judge_rationale"],
                        round=m["round"],
                    )
                )
            await session.commit()

    updated = sorted(by_id.values(), key=lambda h: (-h.elo, h.id))
    logger.info(
        "hypothesis.tournament_round",
        extra={"round": round_idx, "n_matches": len(matches)},
    )
    return updated, matches
