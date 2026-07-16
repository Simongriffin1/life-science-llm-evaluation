"""Tests for RRF fusion and cross-encoder rerank logic."""

from __future__ import annotations

import numpy as np
import pytest

from biolit.ingest.models import PubMedDocument
from biolit.retrieval.fusion import reciprocal_rank_fusion
from biolit.retrieval.rerank import rerank
from biolit.retrieval.types import RankedHit


def test_rrf_prefers_consensus() -> None:
    # a is strong on BM25 only; b is strong on both → b should win RRF.
    bm25 = [
        RankedHit(pmid="a", score=10.0, rank=1, retriever="bm25", scores={"bm25": 10.0}),
        RankedHit(pmid="b", score=5.0, rank=2, retriever="bm25", scores={"bm25": 5.0}),
        RankedHit(pmid="c", score=1.0, rank=3, retriever="bm25", scores={"bm25": 1.0}),
    ]
    dense = [
        RankedHit(pmid="b", score=0.9, rank=1, retriever="dense", scores={"dense": 0.9}),
        RankedHit(pmid="c", score=0.5, rank=2, retriever="dense", scores={"dense": 0.5}),
        RankedHit(pmid="a", score=0.1, rank=3, retriever="dense", scores={"dense": 0.1}),
    ]
    fused = reciprocal_rank_fusion({"bm25": bm25, "dense": dense})
    assert [h.pmid for h in fused] == ["b", "a", "c"]
    assert fused[0].scores["rrf"] > fused[1].scores["rrf"]
    assert "bm25" in fused[0].scores and "dense" in fused[0].scores


def test_rrf_empty() -> None:
    assert reciprocal_rank_fusion({}) == []


class _FakeRerank:
    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        mapping = {"3. gamma": 9.0, "1. alpha": 5.0, "2. beta": 1.0}
        return [mapping.get(p, 0.0) for _, p in pairs]


@pytest.mark.asyncio
async def test_rerank_reorders_by_cross_encoder_score() -> None:
    docs = [
        PubMedDocument(pmid="1", title="1", abstract="alpha"),
        PubMedDocument(pmid="2", title="2", abstract="beta"),
        PubMedDocument(pmid="3", title="3", abstract="gamma"),
    ]
    candidates = [
        RankedHit(pmid="1", score=0.5, rank=1, retriever="rrf", scores={"rrf": 0.5}),
        RankedHit(pmid="2", score=0.4, rank=2, retriever="rrf", scores={"rrf": 0.4}),
        RankedHit(pmid="3", score=0.3, rank=3, retriever="rrf", scores={"rrf": 0.3}),
    ]
    out = await rerank("q", candidates, docs, top_k=3, backend=_FakeRerank())
    assert [h.pmid for h in out] == ["3", "1", "2"]
    assert out[0].scores["rerank"] == 9.0
    assert out[0].rank == 1


class _FakeDense:
    """Deterministic bag-of-words dense backend for service tests."""

    def encode_query(self, query: str) -> np.ndarray:
        vocab = ["trem2", "microglia", "phagocytosis", "cancer", "trial"]
        v = np.zeros(len(vocab), dtype=np.float32)
        for i, w in enumerate(vocab):
            if w in query.lower():
                v[i] = 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_articles(self, texts: list[str]) -> np.ndarray:
        vocab = ["trem2", "microglia", "phagocytosis", "cancer", "trial"]
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


@pytest.mark.asyncio
async def test_rank_dense_with_fake_backend() -> None:
    from biolit.retrieval.dense_medcpt import rank_dense

    docs = [
        PubMedDocument(pmid="1", title="Cancer trial", abstract="Oncology results"),
        PubMedDocument(
            pmid="2",
            title="TREM2 in microglia",
            abstract="TREM2 modulates microglial phagocytosis",
        ),
    ]
    hits = await rank_dense("TREM2 microglia", docs, backend=_FakeDense())
    assert hits[0].pmid == "2"
    assert hits[0].score > hits[1].score
