"""Proximity / near-duplicate clustering before tournament ranking."""

from __future__ import annotations

import hashlib
import re

from biolit.hypothesis.models import HypothesisDraft


def _normalize(text: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return " ".join(tokens)


def cluster_hypotheses(hypotheses: list[HypothesisDraft]) -> list[HypothesisDraft]:
    """
    Assign cluster_id by statement fingerprint; keep highest-Elo member per cluster
    for ranking, but return all with cluster labels.
    """
    clusters: dict[str, str] = {}
    for hyp in hypotheses:
        key = hashlib.sha1(_normalize(hyp.statement).encode()).hexdigest()[:12]
        # Soft merge: identical normalized statements share a cluster.
        clusters[hyp.id] = key
        hyp.cluster_id = key

    # Prefer unique clusters for tournament entry: keep best Elo per cluster.
    best: dict[str, HypothesisDraft] = {}
    for hyp in hypotheses:
        cid = hyp.cluster_id or hyp.id
        prev = best.get(cid)
        if prev is None or hyp.elo > prev.elo:
            best[cid] = hyp
    # Mark non-representatives as clustered duplicates (still retained in full list).
    reps = {h.id for h in best.values()}
    for hyp in hypotheses:
        if hyp.id not in reps:
            hyp.status = "duplicate"
    return hypotheses


def ranking_pool(hypotheses: list[HypothesisDraft]) -> list[HypothesisDraft]:
    """Active, non-duplicate hypotheses with provenance."""
    return [h for h in hypotheses if h.status == "active" and h.has_provenance()]
