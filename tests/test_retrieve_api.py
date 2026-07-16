"""API-level retrieval tests with mocked PubMed + MedCPT backends."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from biolit.api.main import create_app
from biolit.ingest.models import PubMedDocument
from biolit.retrieval.dense_medcpt import set_dense_backend
from biolit.retrieval.rerank import set_rerank_backend


class _FakeDense:
    def encode_query(self, query: str) -> np.ndarray:
        vocab = ["trem2", "microglia", "phagocytosis", "cancer"]
        v = np.zeros(len(vocab), dtype=np.float32)
        for i, w in enumerate(vocab):
            if w in query.lower():
                v[i] = 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_articles(self, texts: list[str]) -> np.ndarray:
        vocab = ["trem2", "microglia", "phagocytosis", "cancer"]
        rows = []
        for text in texts:
            v = np.zeros(len(vocab), dtype=np.float32)
            low = text.lower()
            for i, w in enumerate(vocab):
                if w in low:
                    v[i] = 1.0
            n = np.linalg.norm(v)
            rows.append(v / n if n else v)
        return np.stack(rows, axis=0)


class _FakeRerank:
    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        for _q, passage in pairs:
            low = passage.lower()
            score = 0.0
            if "trem2" in low:
                score += 3.0
            if "phagocytosis" in low:
                score += 2.0
            if "microglia" in low:
                score += 1.0
            scores.append(score)
        return scores


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    app = create_app()
    docs = [
        PubMedDocument(
            pmid="100",
            title="Chemotherapy cancer trial",
            abstract="Results from an oncology trial.",
        ),
        PubMedDocument(
            pmid="200",
            title="TREM2 signaling in microglia",
            abstract="TREM2 modulates microglial phagocytosis in Alzheimer disease.",
        ),
        PubMedDocument(
            pmid="300",
            title="Microglial activation review",
            abstract="A general review of microglia without mentioning TREM2.",
        ),
    ]

    class FakePubMed:
        async def search_documents(self, *_a: Any, **_k: Any) -> list[PubMedDocument]:
            return docs

        async def aclose(self) -> None:
            return None

    set_dense_backend(_FakeDense())
    set_rerank_backend(_FakeRerank())

    with (
        patch("biolit.api.main.ping_db", new=AsyncMock(return_value={"ok": True})),
        patch("biolit.api.main.ping_redis", new=AsyncMock(return_value={"ok": True})),
        patch("biolit.api.main.close_redis", new=AsyncMock()),
        patch("biolit.api.main.dispose_engine", new=AsyncMock()),
        patch("biolit.retrieval.service.PubMedClient", return_value=FakePubMed()),
        patch(
            "biolit.retrieval.service._persist",
            new=AsyncMock(return_value="test-query-id"),
        ),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac

    set_dense_backend(None)
    set_rerank_backend(None)


@pytest.mark.asyncio
async def test_retrieve_hybrid_returns_scores_and_highlights(client: AsyncClient) -> None:
    response = await client.post(
        "/retrieve",
        json={
            "query": "TREM2 microglia phagocytosis",
            "top_k": 3,
            "mode": "hybrid",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "hybrid"
    assert len(body["documents"]) == 3
    top = body["documents"][0]
    assert top["pmid"] == "200"
    assert "rerank" in top["scores"]
    assert "bm25" in top["scores"] or "rrf" in top["scores"]
    assert top["highlight"]
    assert "**" in top["highlight"]


@pytest.mark.asyncio
async def test_hybrid_reranks_vs_bm25_only(client: AsyncClient) -> None:
    bm25 = await client.post(
        "/retrieve",
        json={"query": "TREM2 microglia phagocytosis", "top_k": 3, "mode": "bm25"},
    )
    hybrid = await client.post(
        "/retrieve",
        json={"query": "TREM2 microglia phagocytosis", "top_k": 3, "mode": "hybrid"},
    )
    assert bm25.status_code == 200 and hybrid.status_code == 200
    bm25_docs = bm25.json()["documents"]
    hybrid_docs = hybrid.json()["documents"]
    assert bm25_docs[0]["scores"].keys() == {"bm25"}
    assert "rerank" in hybrid_docs[0]["scores"]
    # Both should surface the TREM2 paper first for this query, but scores differ.
    assert bm25_docs[0]["pmid"] == "200"
    assert hybrid_docs[0]["pmid"] == "200"
    assert bm25_docs[0]["score"] != hybrid_docs[0]["score"]
