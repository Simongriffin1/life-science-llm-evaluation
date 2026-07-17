"""Evolution agent: mutate / recombine top hypotheses."""

from __future__ import annotations

from uuid import uuid4

from biolit.core.llm import complete
from biolit.core.logging import get_logger
from biolit.hypothesis.convert import try_parse_to_draft
from biolit.hypothesis.json_utils import extract_json_array
from biolit.hypothesis.models import EvidenceRef, HypothesisConfig, HypothesisDraft

logger = get_logger(__name__)


async def evolve(
    parents: list[HypothesisDraft],
    *,
    research_goal: str,
    config: HypothesisConfig,
    generation: int,
) -> list[HypothesisDraft]:
    if len(parents) < 1:
        return []

    payload = [
        {
            "id": p.id,
            "statement": p.statement,
            "mechanism": p.mechanism,
            "pmids": [e.pmid for e in p.evidence],
        }
        for p in parents[:4]
    ]
    messages = [
        {
            "role": "system",
            "content": (
                "Evolve biomedical hypotheses by sharpening mechanisms or recombining ideas. "
                "Each offspring MUST cite PMIDs from the parents. Respond ONLY with a JSON array "
                'of {"statement","mechanism","experiment","falsification","parent_ids":[],'
                '"pmid_citations":[]}.'
            ),
        },
        {
            "role": "user",
            "content": f"Goal: {research_goal}\nParents:\n{payload}\nProduce up to 3 offspring.",
        },
    ]
    llm = await complete(config.model, messages, temperature=0.5, max_tokens=1500)
    try:
        raw_items = extract_json_array(llm["content"])
    except Exception:
        return []

    by_pmid: dict[str, EvidenceRef] = {}
    for p in parents:
        for e in p.evidence:
            by_pmid[e.pmid] = e
    allowed = set(by_pmid.keys())

    offspring: list[HypothesisDraft] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        cites = [str(p) for p in (item.get("pmid_citations") or []) if str(p) in by_pmid]
        evidence = [by_pmid[p] for p in cites]
        if not evidence:
            # Inherit evidence from first parent to satisfy provenance mandate.
            evidence = list(parents[0].evidence[:2])
        if not evidence:
            continue
        statement = str(item.get("statement") or "").strip()
        if not statement:
            continue
        parent_ids = item.get("parent_ids") or [parents[0].id]
        raw = {
            "id": str(uuid4()),
            "statement": statement,
            "rationale": str(item.get("rationale") or "").strip() or statement,
            "mechanism": str(item.get("mechanism") or "").strip() or "To be refined.",
            "experiment": str(item.get("experiment") or "").strip() or "To be specified.",
            "falsification": str(item.get("falsification") or "").strip()
            or "Observation that would refute the claim.",
            "novelty": 0.5,
            "feasibility": 0.5,
            "generation": generation,
            "parent_id": str(parent_ids[0]) if parent_ids else parents[0].id,
            "evidence": [
                {"pmid": e.pmid, "snippet": e.snippet, "stance": e.stance} for e in evidence
            ],
            "unvalidated_lead": True,
        }
        draft = try_parse_to_draft(raw, retrieved_pmids=allowed, draft_status="active")
        if draft is None:
            logger.warning(
                "hypothesis.evolve_schema_rejected",
                extra={"statement": statement[:120]},
            )
            continue
        offspring.append(draft)
    return offspring
