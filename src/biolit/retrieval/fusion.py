"""Reciprocal Rank Fusion (RRF) over ranked lists."""

from __future__ import annotations

from biolit.retrieval.types import RankedHit

# RRF-2: k=60 is the classic constant; MIRAGE uses BM25+MedCPT fusion.
DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: dict[str, list[RankedHit]],
    *,
    k: int = DEFAULT_RRF_K,
    top_k: int | None = None,
) -> list[RankedHit]:
    """
    Fuse multiple ranked lists with RRF.

    `rankings` maps retriever name -> ordered hits (rank 1 = best).
    Returns fused hits sorted by descending RRF score, with per-retriever
    scores preserved in `scores`.
    """
    if not rankings:
        return []

    rrf_scores: dict[str, float] = {}
    per_retriever: dict[str, dict[str, float]] = {}
    raw_scores: dict[str, dict[str, float]] = {}

    for retriever, hits in rankings.items():
        for hit in hits:
            contrib = 1.0 / (k + hit.rank)
            rrf_scores[hit.pmid] = rrf_scores.get(hit.pmid, 0.0) + contrib
            per_retriever.setdefault(hit.pmid, {})[retriever] = contrib
            raw_scores.setdefault(hit.pmid, {})[retriever] = hit.score

    ordered = sorted(rrf_scores.items(), key=lambda kv: (-kv[1], kv[0]))
    limit = top_k if top_k is not None else len(ordered)
    fused: list[RankedHit] = []
    for rank, (pmid, score) in enumerate(ordered[:limit], start=1):
        scores = dict(raw_scores.get(pmid, {}))
        scores["rrf"] = score
        for r_name, contrib in per_retriever.get(pmid, {}).items():
            scores[f"rrf_{r_name}"] = contrib
        fused.append(
            RankedHit(
                pmid=pmid,
                score=score,
                rank=rank,
                retriever="rrf",
                scores=scores,
            )
        )
    return fused
