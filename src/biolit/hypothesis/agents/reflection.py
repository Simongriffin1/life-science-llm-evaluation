"""Reflection agent: critique hypotheses for correctness, novelty, testability."""

from __future__ import annotations

from biolit.core.llm import complete
from biolit.hypothesis.json_utils import extract_json_object
from biolit.hypothesis.models import HypothesisConfig, HypothesisDraft


async def reflect_on(
    hypothesis: HypothesisDraft,
    *,
    research_goal: str,
    config: HypothesisConfig,
) -> HypothesisDraft:
    messages = [
        {
            "role": "system",
            "content": (
                "Critique a biomedical hypothesis. Respond ONLY with JSON: "
                '{"reflection":"...","novelty":0to1,"feasibility":0to1,'
                '"keep":true/false,"revised_statement":null|"..."}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Goal: {research_goal}\nHypothesis: {hypothesis.statement}\n"
                f"Mechanism: {hypothesis.mechanism}\n"
                f"Falsification: {hypothesis.falsification}\n"
                f"Evidence: {[e.model_dump() for e in hypothesis.evidence]}\n"
            ),
        },
    ]
    llm = await complete(config.model, messages, temperature=0.0, max_tokens=500)
    try:
        parsed = extract_json_object(llm["content"])
    except Exception:
        parsed = {"reflection": llm["content"][:400], "keep": True}

    if parsed.get("revised_statement"):
        hypothesis.statement = str(parsed["revised_statement"]).strip()
    hypothesis.reflection = str(parsed.get("reflection") or "")
    if parsed.get("novelty") is not None:
        hypothesis.novelty = float(parsed["novelty"])
    if parsed.get("feasibility") is not None:
        hypothesis.feasibility = float(parsed["feasibility"])
    if parsed.get("keep") is False:
        hypothesis.status = "rejected"
    return hypothesis


async def reflect_all(
    hypotheses: list[HypothesisDraft],
    *,
    research_goal: str,
    config: HypothesisConfig,
) -> list[HypothesisDraft]:
    out: list[HypothesisDraft] = []
    for hyp in hypotheses:
        updated = await reflect_on(hyp, research_goal=research_goal, config=config)
        if updated.status != "rejected" and updated.has_provenance():
            out.append(updated)
    return out
