"""Meta-review agent: synthesize surviving hypotheses into research proposals."""

from __future__ import annotations

from biolit.core.llm import complete
from biolit.hypothesis.json_utils import extract_json_array
from biolit.hypothesis.models import HypothesisConfig, HypothesisDraft


async def meta_review(
    hypotheses: list[HypothesisDraft],
    *,
    research_goal: str,
    config: HypothesisConfig,
) -> list[HypothesisDraft]:
    """Select and polish top proposals; preserve evidence trails."""
    ranked = sorted(
        [h for h in hypotheses if h.status == "active" and h.has_provenance()],
        key=lambda h: (-h.elo, h.id),
    )[: max(config.max_proposals * 2, config.max_proposals)]
    if not ranked:
        return []

    summary = [
        {
            "id": h.id,
            "elo": h.elo,
            "statement": h.statement,
            "mechanism": h.mechanism,
            "experiment": h.experiment,
            "falsification": h.falsification,
            "pmids": [e.pmid for e in h.evidence],
        }
        for h in ranked
    ]
    messages = [
        {
            "role": "system",
            "content": (
                "You are a principal investigator writing a short research agenda. "
                "Pick the best hypotheses and tighten mechanism, experiment, and an explicit "
                "falsification criterion. Respond ONLY with JSON array of "
                '{"id":"<existing id>","statement","mechanism","experiment","falsification"}.'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Goal: {research_goal}\nCandidates:\n{summary}\n"
                f"Return at most {config.max_proposals} proposals using existing ids."
            ),
        },
    ]
    llm = await complete(config.model, messages, temperature=0.2, max_tokens=1500)
    try:
        raw_items = extract_json_array(llm["content"])
    except Exception:
        return ranked[: config.max_proposals]

    by_id = {h.id: h for h in ranked}
    proposals: list[HypothesisDraft] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        hid = str(item.get("id") or "")
        base = by_id.get(hid)
        if base is None:
            continue
        base.statement = str(item.get("statement") or base.statement)
        base.mechanism = str(item.get("mechanism") or base.mechanism or "")
        base.experiment = str(item.get("experiment") or base.experiment or "")
        base.falsification = str(item.get("falsification") or base.falsification or "")
        base.status = "proposal"
        if base.has_provenance() and base.falsification:
            proposals.append(base)
        if len(proposals) >= config.max_proposals:
            break

    if not proposals:
        for h in ranked[: config.max_proposals]:
            if not h.falsification:
                h.falsification = (
                    "If the predicted measurable effect is absent under the proposed experiment, "
                    "reject the hypothesis."
                )
            h.status = "proposal"
            proposals.append(h)
    return proposals
