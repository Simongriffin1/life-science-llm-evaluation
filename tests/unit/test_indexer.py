"""Chunk + embed + pgvector round-trip tests (requires local Postgres)."""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import delete, select

from biolit.core.db import Chunk, Document, dispose_engine, init_db
from biolit.ingest.indexer import (
    embed_and_store_chunks,
    load_documents_by_pmids,
    rank_dense_from_index,
    upsert_documents,
)
from biolit.ingest.models import PubMedDocument


class _HashDense:
    """Deterministic bag-of-words encoder padded to 768 dims."""

    vocab = ["trem2", "microglia", "phagocytosis", "cancer", "trial"]

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(768, dtype=np.float32)
        low = text.lower()
        for i, w in enumerate(self.vocab):
            if w in low:
                v[i] = 1.0
        n = float(np.linalg.norm(v))
        return v / n if n else v

    def encode_query(self, query: str) -> np.ndarray:
        return self._vec(query)

    def encode_articles(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts], axis=0)


@pytest.fixture
async def clean_index() -> None:
    await init_db()
    from biolit.core.db import (
        Evidence,
        Hypothesis,
        HypothesisRun,
        RetrievalHit,
        RetrievalQuery,
        TournamentMatch,
        get_session_factory,
    )

    factory = get_session_factory()
    async with factory() as session:
        await session.execute(delete(TournamentMatch))
        await session.execute(delete(Evidence))
        await session.execute(
            Hypothesis.__table__.update().values(parent_id=None)  # type: ignore[attr-defined]
        )
        await session.execute(delete(Hypothesis))
        await session.execute(delete(HypothesisRun))
        await session.execute(delete(RetrievalHit))
        await session.execute(delete(RetrievalQuery))
        await session.execute(delete(Chunk))
        await session.execute(delete(Document))
        await session.commit()
    yield
    await dispose_engine()


@pytest.mark.asyncio
async def test_chunk_embed_store_and_search_roundtrip(clean_index: None) -> None:
    docs = [
        PubMedDocument(
            pmid="9001",
            title="Cancer chemotherapy trial",
            abstract="Oncology results from a breast cancer trial.",
        ),
        PubMedDocument(
            pmid="9002",
            title="TREM2 signaling in microglia",
            abstract="TREM2 modulates microglial phagocytosis in Alzheimer disease.",
        ),
        PubMedDocument(
            pmid="9003",
            title="Microglial activation",
            abstract="A general review of microglia biology.",
        ),
    ]
    await upsert_documents(docs)
    n_chunks = await embed_and_store_chunks(docs, backend=_HashDense())
    assert n_chunks >= 3

    from biolit.core.db import get_session_factory

    factory = get_session_factory()
    async with factory() as session:
        count = await session.execute(select(Chunk))
        stored = count.scalars().all()
    assert all(c.embedding is not None for c in stored)
    assert {c.pmid for c in stored} == {"9001", "9002", "9003"}

    hits = await rank_dense_from_index(
        "TREM2 microglia phagocytosis",
        top_k=2,
        backend=_HashDense(),
    )
    assert hits[0].pmid == "9002"
    assert hits[0].score > hits[1].score

    loaded = await load_documents_by_pmids(["9002", "9001"])
    assert [d.pmid for d in loaded] == ["9002", "9001"]
    assert loaded[0].title and "TREM2" in loaded[0].title
