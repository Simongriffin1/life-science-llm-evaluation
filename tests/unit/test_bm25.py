"""Unit tests for BM25 lexical ranking."""

from __future__ import annotations

from biolit.ingest.models import PubMedDocument
from biolit.retrieval.bm25 import rank_bm25, tokenize


def test_tokenize_lowercases() -> None:
    assert tokenize("TREM2 Microglia") == ["trem2", "microglia"]


def test_rank_bm25_orders_by_relevance() -> None:
    docs = [
        PubMedDocument(
            pmid="1",
            title="Unrelated oncology trial",
            abstract="Chemotherapy response in breast cancer patients.",
        ),
        PubMedDocument(
            pmid="2",
            title="TREM2 signaling in microglia",
            abstract="TREM2 modulates microglial phagocytosis in Alzheimer's disease.",
        ),
        PubMedDocument(
            pmid="3",
            title="Microglial activation",
            abstract="General review of microglia without TREM2.",
        ),
    ]
    hits = rank_bm25("TREM2 microglia phagocytosis", docs, top_k=3)
    assert [h.pmid for h in hits] == ["2", "3", "1"]
    assert hits[0].score > hits[1].score >= hits[2].score
    assert hits[0].rank == 1


def test_rank_bm25_empty_corpus() -> None:
    assert rank_bm25("anything", []) == []
