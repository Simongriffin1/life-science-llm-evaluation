"""Retrieval service: esearch → fetch → BM25 + dense → RRF → rerank → persist."""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy.dialects.postgresql import insert

from biolit.core.config import get_settings
from biolit.core.db import Document, RetrievalHit, RetrievalQuery, get_session_factory
from biolit.core.logging import get_logger
from biolit.ingest.models import PubMedDocument
from biolit.ingest.pubmed_client import PubMedClient
from biolit.retrieval.bm25 import rank_bm25
from biolit.retrieval.dense_medcpt import rank_dense
from biolit.retrieval.fusion import reciprocal_rank_fusion
from biolit.retrieval.rerank import rerank
from biolit.retrieval.types import RankedHit

logger = get_logger(__name__)

RetrievalMode = Literal["bm25", "dense", "hybrid"]


class RetrieveFilters(BaseModel):
    date_from: str | None = None
    date_to: str | None = None
    mesh: list[str] | None = None
    journal: str | None = None


class ScoredDocument(BaseModel):
    pmid: str
    title: str | None = None
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    journal: str | None = None
    pub_date: date | None = None
    mesh_terms: list[str] = Field(default_factory=list)
    doi: str | None = None
    rank: int
    score: float
    scores: dict[str, float] = Field(default_factory=dict)
    highlight: str | None = None


class RetrieveResult(BaseModel):
    query_id: str | None = None
    query: str
    mode: RetrievalMode
    top_k: int
    use_index: bool = False
    documents: list[ScoredDocument]


def highlight_spans(query: str, text: str | None, *, max_len: int = 280) -> str | None:
    """Return a snippet with matched query terms marked ``**like this**``."""
    if not text:
        return None
    tokens = [t.lower() for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2]
    if not tokens:
        snippet = text[:max_len].rstrip()
        return snippet + ("…" if len(text) > max_len else "")

    pattern = re.compile(
        r"(" + "|".join(re.escape(t) for t in sorted(set(tokens), key=len, reverse=True)) + r")",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        start = max(0, match.start() - 60)
        end = min(len(text), match.end() + max_len - 60)
        window = text[start:end]
    else:
        start = 0
        window = text[:max_len]

    highlighted = pattern.sub(lambda m: f"**{m.group(0)}**", window)
    prefix = "…" if start > 0 else ""
    suffix = "…" if start + len(window) < len(text) else ""
    return f"{prefix}{highlighted}{suffix}"


def _bm25_to_ranked(hits: list[Any]) -> list[RankedHit]:
    return [
        RankedHit(
            pmid=h.pmid,
            score=float(h.score),
            rank=h.rank,
            retriever="bm25",
            scores={"bm25": float(h.score)},
        )
        for h in hits
    ]


async def _persist(
    query: str,
    filters: dict[str, Any] | None,
    mode: str,
    top_k: int,
    documents: list[PubMedDocument],
    final_hits: list[RankedHit],
    highlights: dict[str, str | None],
) -> str | None:
    """Upsert documents and write retrieval_queries / retrieval_hits. Returns query_id."""
    settings = get_settings()
    try:
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

            q_row = RetrievalQuery(
                query=query,
                filters_json=filters,
                mode=mode,
                top_k=top_k,
            )
            session.add(q_row)
            await session.flush()

            for hit in final_hits:
                session.add(
                    RetrievalHit(
                        query_id=q_row.id,
                        pmid=hit.pmid,
                        retriever=hit.retriever,
                        score=hit.score,
                        rank=hit.rank,
                        highlight=highlights.get(hit.pmid),
                    )
                )
            await session.commit()
            return q_row.id
    except Exception:
        logger.exception(
            "retrieval.persist_failed",
            extra={"database_url": settings.database_url.split("@")[-1]},
        )
        return None


async def retrieve(
    query: str,
    *,
    filters: RetrieveFilters | dict[str, Any] | None = None,
    top_k: int | None = None,
    mode: RetrievalMode = "hybrid",
    candidate_cap: int | None = None,
    persist: bool = True,
    use_index: bool | None = None,
    pubmed_client: PubMedClient | None = None,
) -> RetrieveResult:
    """
    Hybrid PubMed retrieval pipeline.

    Modes:
      - bm25: lexical only
      - dense: MedCPT dense only
      - hybrid: BM25 + dense → RRF → MedCPT cross-encoder rerank

    When ``use_index`` is True, candidates and the dense leg come from the
    persistent pgvector corpus (no live E-utilities). Falls back to the live
    path if the index is empty.
    """
    from biolit.ingest.indexer import (
        indexed_chunk_count,
        load_all_indexed_documents,
        load_documents_by_pmids,
        rank_dense_from_index,
    )

    settings = get_settings()
    k = top_k if top_k is not None else settings.retrieval_top_k
    cap = candidate_cap if candidate_cap is not None else settings.retrieval_candidate_cap

    filt = (
        filters
        if isinstance(filters, RetrieveFilters)
        else RetrieveFilters.model_validate(filters or {})
    )
    filters_json = filt.model_dump(exclude_none=True)

    # Auto: use the persistent index when one exists and the caller asked for auto.
    if use_index is None:
        use_index = False
    if use_index and await indexed_chunk_count() == 0:
        logger.warning("use_index requested but no chunks found; falling back to live path")
        use_index = False

    documents: list[PubMedDocument]
    if use_index:
        if mode == "dense":
            dense_hits = await rank_dense_from_index(query, top_k=k)
            documents = await load_documents_by_pmids([h.pmid for h in dense_hits])
            by_pmid = {d.pmid: d for d in documents}
            final_hits = dense_hits
        else:
            documents = await load_all_indexed_documents(limit=cap)
            if not documents:
                return RetrieveResult(query=query, mode=mode, top_k=k, use_index=True, documents=[])
            by_pmid = {d.pmid: d for d in documents}
            if mode == "bm25":
                final_hits = _bm25_to_ranked(rank_bm25(query, documents, top_k=k))
            else:
                bm25_hits = _bm25_to_ranked(rank_bm25(query, documents))
                dense_hits = await rank_dense_from_index(query, top_k=len(documents))
                fused = reciprocal_rank_fusion(
                    {"bm25": bm25_hits, "dense": dense_hits},
                    top_k=min(len(documents), max(k * 2, k)),
                )
                # Ensure docs cover fused PMIDs (dense may surface extras).
                missing = [h.pmid for h in fused if h.pmid not in by_pmid]
                if missing:
                    extra = await load_documents_by_pmids(missing)
                    for doc in extra:
                        by_pmid[doc.pmid] = doc
                        documents.append(doc)
                final_hits = await rerank(query, fused, documents, top_k=k)
    else:
        owns_client = pubmed_client is None
        client = pubmed_client or PubMedClient()
        try:
            documents = await client.search_documents(
                query,
                retmax=cap,
                mindate=filt.date_from,
                maxdate=filt.date_to,
                mesh=filt.mesh,
                journal=filt.journal,
            )
        finally:
            if owns_client:
                await client.aclose()

        if not documents:
            return RetrieveResult(query=query, mode=mode, top_k=k, use_index=False, documents=[])

        by_pmid = {d.pmid: d for d in documents}

        if mode == "bm25":
            final_hits = _bm25_to_ranked(rank_bm25(query, documents, top_k=k))
        elif mode == "dense":
            final_hits = await rank_dense(query, documents, top_k=k)
        else:
            bm25_hits = _bm25_to_ranked(rank_bm25(query, documents))
            dense_hits = await rank_dense(query, documents)
            fused = reciprocal_rank_fusion(
                {"bm25": bm25_hits, "dense": dense_hits},
                top_k=min(len(documents), max(k * 2, k)),
            )
            final_hits = await rerank(query, fused, documents, top_k=k)

    highlights = {
        hit.pmid: highlight_spans(query, by_pmid[hit.pmid].text_for_lexical())
        for hit in final_hits
        if hit.pmid in by_pmid
    }

    query_id: str | None = None
    if persist:
        query_id = await _persist(
            query,
            filters_json or None,
            mode,
            k,
            list(by_pmid.values()),
            final_hits,
            highlights,
        )

    scored: list[ScoredDocument] = []
    for hit in final_hits:
        maybe_doc = by_pmid.get(hit.pmid)
        if maybe_doc is None:
            continue
        scored.append(
            ScoredDocument(
                pmid=maybe_doc.pmid,
                title=maybe_doc.title,
                abstract=maybe_doc.abstract,
                authors=maybe_doc.authors,
                journal=maybe_doc.journal,
                pub_date=maybe_doc.pub_date,
                mesh_terms=maybe_doc.mesh_terms,
                doi=maybe_doc.doi,
                rank=hit.rank,
                score=hit.score,
                scores=hit.scores,
                highlight=highlights.get(hit.pmid),
            )
        )

    logger.info(
        "retrieval.done",
        extra={
            "mode": mode,
            "use_index": use_index,
            "n_docs": len(documents),
            "returned": len(scored),
        },
    )
    return RetrieveResult(
        query_id=query_id,
        query=query,
        mode=mode,
        top_k=k,
        use_index=use_index,
        documents=scored,
    )
