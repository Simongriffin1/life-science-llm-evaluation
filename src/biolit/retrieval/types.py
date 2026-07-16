"""Shared ranking hit types for lexical / dense / fused / reranked results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RankedHit:
    """A single ranked document from one or more retrievers."""

    pmid: str
    score: float
    rank: int
    retriever: str
    scores: dict[str, float] = field(default_factory=dict)
