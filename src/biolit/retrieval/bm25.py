"""BM25 lexical ranking over title + abstract of a candidate document set."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from biolit.ingest.models import PubMedDocument

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Simple alphanumeric tokenizer suitable for biomedical titles/abstracts."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass(frozen=True)
class BM25Hit:
    pmid: str
    score: float
    rank: int


def rank_bm25(
    query: str,
    documents: list[PubMedDocument],
    *,
    top_k: int | None = None,
) -> list[BM25Hit]:
    """
    Score `documents` with BM25Okapi over title+abstract.

    Returns ranked hits (highest score first). Documents with empty text are scored 0
    and sorted stably after scored docs.
    """
    if not documents:
        return []

    corpus_tokens = [tokenize(doc.text_for_lexical()) for doc in documents]
    # BM25Okapi requires a non-empty corpus; empty-token docs still occupy a slot.
    bm25 = BM25Okapi(corpus_tokens if any(corpus_tokens) else [[""]] * len(documents))
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens) if query_tokens else [0.0] * len(documents)

    indexed = sorted(
        enumerate(scores),
        key=lambda pair: (-float(pair[1]), pair[0]),
    )
    limit = top_k if top_k is not None else len(indexed)
    hits: list[BM25Hit] = []
    for rank, (idx, score) in enumerate(indexed[:limit], start=1):
        hits.append(
            BM25Hit(
                pmid=documents[idx].pmid,
                score=float(score),
                rank=rank,
            )
        )
    return hits
