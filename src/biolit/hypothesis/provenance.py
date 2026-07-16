"""Mandatory provenance: evidence rows must exist for every hypothesis."""

from __future__ import annotations

from uuid import uuid4

from sqlalchemy.dialects.postgresql import insert

from biolit.core.db import Document, Evidence, get_session_factory
from biolit.core.logging import get_logger
from biolit.hypothesis.models import EvidenceRef, HypothesisDraft, Stance
from biolit.ingest.models import PubMedDocument

logger = get_logger(__name__)


async def upsert_documents(documents: list[PubMedDocument]) -> None:
    if not documents:
        return
    factory = get_session_factory()
    async with factory() as session:
        for doc in documents:
            stmt = (
                insert(Document)
                .values(
                    pmid=doc.pmid,
                    title=doc.title,
                    abstract=doc.abstract,
                    authors=doc.authors or None,
                    journal=doc.journal,
                    pub_date=doc.pub_date,
                    mesh_terms=doc.mesh_terms or None,
                    doi=doc.doi,
                    source=doc.source,
                    raw_json=doc.raw_json,
                )
                .on_conflict_do_update(
                    index_elements=["pmid"],
                    set_={
                        "title": doc.title,
                        "abstract": doc.abstract,
                        "authors": doc.authors or None,
                        "journal": doc.journal,
                        "pub_date": doc.pub_date,
                        "mesh_terms": doc.mesh_terms or None,
                        "doi": doc.doi,
                        "raw_json": doc.raw_json,
                    },
                )
            )
            await session.execute(stmt)
        await session.commit()


def evidence_from_retrieval(
    documents: list[PubMedDocument],
    *,
    stance: Stance = "supports",
    limit: int = 3,
) -> list[EvidenceRef]:
    """Build evidence refs from retrieved docs (title/abstract snippets)."""
    refs: list[EvidenceRef] = []
    for doc in documents[:limit]:
        snippet = doc.title or ""
        if doc.abstract:
            snippet = f"{doc.title or ''}. {doc.abstract[:280]}".strip()
        if not snippet:
            continue
        refs.append(
            EvidenceRef(
                pmid=doc.pmid,
                snippet=snippet[:500],
                stance=stance,
            )
        )
    return refs


def require_provenance(hypotheses: list[HypothesisDraft]) -> list[HypothesisDraft]:
    """Drop any hypothesis lacking evidence — provenance is mandatory."""
    kept = [h for h in hypotheses if h.has_provenance()]
    dropped = len(hypotheses) - len(kept)
    if dropped:
        logger.warning("hypothesis.dropped_no_provenance", extra={"n": dropped})
    return kept


class CiteOnlyViolation(ValueError):
    """Raised when a hypothesis cites a PMID outside the retrieval set."""

    def __init__(self, hypothesis_id: str, illegal_pmids: list[str]) -> None:
        self.hypothesis_id = hypothesis_id
        self.illegal_pmids = illegal_pmids
        super().__init__(
            f"Hypothesis {hypothesis_id} cites PMIDs outside retrieval set: {illegal_pmids}"
        )


def enforce_cite_only(
    hypotheses: list[HypothesisDraft],
    allowed_pmids: set[str] | list[str],
    *,
    strict: bool = True,
) -> list[HypothesisDraft]:
    """
    Cite-only: every evidence PMID must be in the retrieval set for this run.

    When ``strict`` is True (default), illegal citations fail closed.
    Always stamps ``unvalidated_lead=True`` on kept drafts.
    """
    allowed = {str(p) for p in allowed_pmids}
    kept: list[HypothesisDraft] = []
    for h in hypotheses:
        illegal = sorted({e.pmid for e in h.evidence if e.pmid not in allowed})
        if illegal:
            if strict:
                raise CiteOnlyViolation(h.id, illegal)
            legal = [e for e in h.evidence if e.pmid in allowed]
            if not legal:
                logger.warning(
                    "hypothesis.dropped_no_cite_only_evidence",
                    extra={"hypothesis_id": h.id},
                )
                continue
            h = h.model_copy(update={"evidence": legal})
        kept.append(h.model_copy(update={"unvalidated_lead": True}))
    return kept


async def persist_evidence(hypothesis_id: str, evidence: list[EvidenceRef]) -> int:
    """Write evidence rows for a persisted hypothesis. Returns count written."""
    if not evidence:
        raise ValueError("Cannot persist hypothesis without evidence rows")
    factory = get_session_factory()
    async with factory() as session:
        for ev in evidence:
            session.add(
                Evidence(
                    id=str(uuid4()),
                    hypothesis_id=hypothesis_id,
                    pmid=ev.pmid,
                    snippet=ev.snippet,
                    stance=ev.stance,
                )
            )
        await session.commit()
    return len(evidence)
