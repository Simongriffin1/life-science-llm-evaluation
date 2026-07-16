"""Generation agent: retrieve literature and propose grounded seed hypotheses."""

from __future__ import annotations

from typing import Literal, cast
from uuid import uuid4

from biolit.core.llm import complete
from biolit.core.logging import get_logger
from biolit.hypothesis.json_utils import extract_json_array
from biolit.hypothesis.models import EvidenceRef, HypothesisConfig, HypothesisDraft
from biolit.hypothesis.provenance import evidence_from_retrieval, upsert_documents
from biolit.ingest.models import PubMedDocument
from biolit.retrieval.service import retrieve

logger = get_logger(__name__)


async def generate_seeds(
    research_goal: str,
    config: HypothesisConfig,
) -> tuple[list[HypothesisDraft], list[PubMedDocument]]:
    """Ground generation in retrieval; only keep hypotheses with evidence."""
    mode = cast(Literal["bm25", "dense", "hybrid"], config.retrieval_mode)
    result = await retrieve(
        research_goal,
        top_k=config.top_k_retrieve,
        mode=mode,
        candidate_cap=config.candidate_cap,
        use_index=config.use_index,
        persist=True,
    )
    docs = [
        PubMedDocument(
            pmid=d.pmid,
            title=d.title,
            abstract=d.abstract,
            authors=d.authors,
            journal=d.journal,
            pub_date=d.pub_date,
            mesh_terms=d.mesh_terms,
            doi=d.doi,
        )
        for d in result.documents
    ]
    await upsert_documents(docs)

    bibliography = "\n".join(
        f"- PMID {d.pmid}: {d.title or ''} — {(d.abstract or '')[:240]}" for d in docs
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a biomedical hypothesis generator. Propose testable hypotheses "
                "ONLY using the provided PMIDs. Every hypothesis MUST cite at least one PMID. "
                "Respond ONLY with a JSON array of objects: "
                '{"statement","rationale","mechanism","experiment","falsification",'
                '"pmid_citations":["..."]}.'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Research goal: {research_goal}\n\nLiterature:\n{bibliography}\n\n"
                f"Propose {config.n_seed} distinct hypotheses."
            ),
        },
    ]
    llm = await complete(config.model, messages, temperature=0.4, max_tokens=2000)
    try:
        raw_items = extract_json_array(llm["content"])
    except Exception:
        logger.warning("hypothesis.generate_parse_failed")
        raw_items = []

    allowed = {d.pmid for d in docs}
    default_evidence = evidence_from_retrieval(docs, limit=3)
    drafts: list[HypothesisDraft] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        cites = [str(p) for p in (item.get("pmid_citations") or []) if str(p) in allowed]
        evidence: list[EvidenceRef] = []
        by_pmid = {d.pmid: d for d in docs}
        for pmid in cites:
            doc = by_pmid[pmid]
            snippet = f"{doc.title or ''}. {(doc.abstract or '')[:240]}".strip()
            evidence.append(EvidenceRef(pmid=pmid, snippet=snippet or pmid, stance="supports"))
        if not evidence and default_evidence:
            evidence = list(default_evidence[:2])
        if not evidence:
            continue
        drafts.append(
            HypothesisDraft(
                id=str(uuid4()),
                statement=str(item.get("statement") or "").strip(),
                rationale=str(item.get("rationale") or "") or None,
                mechanism=str(item.get("mechanism") or "") or None,
                experiment=str(item.get("experiment") or "") or None,
                falsification=str(item.get("falsification") or "") or None,
                generation=0,
                evidence=evidence,
                unvalidated_lead=True,
            )
        )

    drafts = [d for d in drafts if d.statement and d.has_provenance()]
    logger.info("hypothesis.generated", extra={"n": len(drafts)})
    return drafts, docs
