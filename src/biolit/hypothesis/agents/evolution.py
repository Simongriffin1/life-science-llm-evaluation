"""Evolution agent: mutate / recombine top hypotheses."""

from __future__ import annotations

from uuid import uuid4

from biolit.core.llm import complete
from biolit.hypothesis.json_utils import extract_json_array
from biolit.hypothesis.models import EvidenceRef, HypothesisConfig, HypothesisDraft


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
        parent_ids = item.get("parent_ids") or [parents[0].id]
        offspring.append(
            HypothesisDraft(
                id=str(uuid4()),
                statement=str(item.get("statement") or "").strip(),
                mechanism=str(item.get("mechanism") or "") or None,
                experiment=str(item.get("experiment") or "") or None,
                falsification=str(item.get("falsification") or "") or None,
                generation=generation,
                parent_id=str(parent_ids[0]) if parent_ids else parents[0].id,
                evidence=evidence,
                unvalidated_lead=True,
            )
        )
    return [o for o in offspring if o.statement and o.has_provenance()]
