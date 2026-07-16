"""Scoped corpus ingest into Postgres + pgvector for persistent dense retrieval."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from sqlalchemy import delete, func, select, text
from sqlalchemy.dialects.postgresql import insert

from biolit.core.config import get_settings
from biolit.core.db import Chunk, Document, get_session_factory
from biolit.core.logging import get_logger
from biolit.ingest.chunking import chunks_for_document
from biolit.ingest.models import PubMedDocument
from biolit.ingest.pubmed_client import PubMedClient
from biolit.retrieval.dense_medcpt import DenseBackend, get_dense_backend
from biolit.retrieval.types import RankedHit

logger = get_logger(__name__)


@dataclass(frozen=True)
class IngestResult:
    n_documents: int
    n_chunks: int
    pmids: list[str]


async def ensure_vector_index() -> None:
    """Create an HNSW cosine index on chunks.embedding if missing."""
    factory = get_session_factory()
    async with factory() as session:
        await session.execute(
            text(
                "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx "
                "ON chunks USING hnsw (embedding vector_cosine_ops)"
            )
        )
        await session.commit()
    logger.info("pgvector HNSW index ensured")


async def indexed_chunk_count() -> int:
    """Return number of stored chunks with embeddings."""
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(func.count()).select_from(Chunk).where(Chunk.embedding.is_not(None))
        )
        return int(result.scalar_one())


async def upsert_documents(documents: list[PubMedDocument]) -> None:
    """Upsert PubMed documents into the documents table."""
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


async def embed_and_store_chunks(
    documents: list[PubMedDocument],
    *,
    backend: DenseBackend | None = None,
    replace_existing: bool = True,
    batch_size: int = 16,
) -> int:
    """
    Chunk documents, embed with the MedCPT article encoder, store in pgvector.

    Returns the number of chunks written.
    """
    if not documents:
        return 0

    b = backend or get_dense_backend()
    settings = get_settings()
    dim = settings.embedding_dim

    prepared: list[tuple[str, str, str]] = []  # pmid, section, text
    for doc in documents:
        for section, text_piece in chunks_for_document(doc.title, doc.abstract):
            prepared.append((doc.pmid, section, text_piece))

    if not prepared:
        return 0

    texts = [t for _, _, t in prepared]
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vectors = await asyncio.to_thread(b.encode_articles, batch)
        for row in vectors:
            vec = row.tolist()
            if len(vec) != dim:
                # Fake backends in tests may use smaller dims; pad/truncate.
                vec = vec + [0.0] * (dim - len(vec)) if len(vec) < dim else vec[:dim]
            embeddings.append(vec)

    factory = get_session_factory()
    async with factory() as session:
        if replace_existing:
            pmids = list({p for p, _, _ in prepared})
            await session.execute(delete(Chunk).where(Chunk.pmid.in_(pmids)))

        for (pmid, section, text_piece), emb in zip(prepared, embeddings, strict=True):
            session.add(
                Chunk(
                    id=str(uuid4()),
                    pmid=pmid,
                    section=section,
                    text=text_piece,
                    embedding=emb,
                )
            )
        await session.commit()

    await ensure_vector_index()
    logger.info(
        "index.chunks_stored",
        extra={"n_docs": len(documents), "n_chunks": len(prepared)},
    )
    return len(prepared)


async def ingest_corpus(
    *,
    mesh: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    journal: str | None = None,
    query: str | None = None,
    retmax: int = 200,
    backend: DenseBackend | None = None,
    pubmed_client: PubMedClient | None = None,
) -> IngestResult:
    """
    Pull a scoped PubMed corpus, upsert documents, chunk+embed into pgvector.

    Scope is defined by MeSH / date window / journal (and optional free-text query).
    """
    search_query = query or (mesh if mesh else "biomedical")
    mesh_list = [mesh] if mesh else None

    owns = pubmed_client is None
    client = pubmed_client or PubMedClient()
    try:
        documents = await client.search_documents(
            search_query,
            retmax=retmax,
            mindate=date_from,
            maxdate=date_to,
            mesh=mesh_list,
            journal=journal,
        )
    finally:
        if owns:
            await client.aclose()

    await upsert_documents(documents)
    n_chunks = await embed_and_store_chunks(documents, backend=backend)
    return IngestResult(
        n_documents=len(documents),
        n_chunks=n_chunks,
        pmids=[d.pmid for d in documents],
    )


async def load_documents_by_pmids(pmids: list[str]) -> list[PubMedDocument]:
    """Load stored documents as PubMedDocument models."""
    if not pmids:
        return []
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(select(Document).where(Document.pmid.in_(pmids)))
        rows = result.scalars().all()
    by_pmid = {r.pmid: r for r in rows}
    docs: list[PubMedDocument] = []
    for pmid in pmids:
        row = by_pmid.get(pmid)
        if row is None:
            continue
        docs.append(
            PubMedDocument(
                pmid=row.pmid,
                title=row.title,
                abstract=row.abstract,
                authors=list(row.authors or []),
                journal=row.journal,
                pub_date=row.pub_date,
                mesh_terms=list(row.mesh_terms or []),
                doi=row.doi,
                source=row.source or "pubmed",
                raw_json=row.raw_json,
            )
        )
    return docs


async def rank_dense_from_index(
    query: str,
    *,
    top_k: int,
    backend: DenseBackend | None = None,
) -> list[RankedHit]:
    """
    Dense retrieval against the persistent pgvector index.

    Aggregates chunk hits to document level using max similarity per PMID.
    """
    b = backend or get_dense_backend()
    q_vec = await asyncio.to_thread(b.encode_query, query)
    q_list = q_vec.tolist()
    settings = get_settings()
    dim = settings.embedding_dim
    if len(q_list) < dim:
        q_list = q_list + [0.0] * (dim - len(q_list))
    elif len(q_list) > dim:
        q_list = q_list[:dim]

    # Fetch a wider chunk pool then aggregate to documents.
    chunk_limit = max(top_k * 4, top_k)
    factory = get_session_factory()
    async with factory() as session:
        # cosine distance: lower is better; similarity = 1 - distance
        distance = Chunk.embedding.cosine_distance(q_list)
        stmt = (
            select(Chunk.pmid, Chunk.text, distance.label("dist"))
            .where(Chunk.embedding.is_not(None))
            .order_by(distance)
            .limit(chunk_limit)
        )
        result = await session.execute(stmt)
        rows = result.all()

    best: dict[str, float] = {}
    for pmid, _text, dist in rows:
        sim = 1.0 - float(dist)
        prev = best.get(pmid)
        if prev is None or sim > prev:
            best[pmid] = sim

    ordered = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    return [
        RankedHit(
            pmid=pmid,
            score=score,
            rank=rank,
            retriever="dense_index",
            scores={"dense": score, "dense_index": score},
        )
        for rank, (pmid, score) in enumerate(ordered, start=1)
    ]


async def load_all_indexed_documents(*, limit: int | None = None) -> list[PubMedDocument]:
    """Load documents that have at least one chunk (for BM25 over the indexed corpus)."""
    factory = get_session_factory()
    async with factory() as session:
        stmt = (
            select(Document)
            .where(Document.pmid.in_(select(Chunk.pmid).distinct()))
            .order_by(Document.pmid)
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        result = await session.execute(stmt)
        rows = result.scalars().all()
    return [
        PubMedDocument(
            pmid=row.pmid,
            title=row.title,
            abstract=row.abstract,
            authors=list(row.authors or []),
            journal=row.journal,
            pub_date=row.pub_date,
            mesh_terms=list(row.mesh_terms or []),
            doi=row.doi,
            source=row.source or "pubmed",
            raw_json=row.raw_json,
        )
        for row in rows
    ]


def ingest_result_dict(result: IngestResult) -> dict[str, Any]:
    return {
        "n_documents": result.n_documents,
        "n_chunks": result.n_chunks,
        "pmids": result.pmids,
    }
