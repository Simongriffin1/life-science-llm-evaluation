"""Bridge between schema-level Hypothesis and LangGraph HypothesisDraft."""

from __future__ import annotations

from typing import Any

from biolit.hypothesis.models import EvidenceRef, HypothesisDraft
from biolit.hypothesis.schemas import (
    Hypothesis,
    HypothesisStatus,
    parse_hypothesis,
)

_DRAFT_STATUS_TO_SCHEMA: dict[str, HypothesisStatus] = {
    "active": HypothesisStatus.alive,
    "proposal": HypothesisStatus.alive,
    "accepted": HypothesisStatus.accepted,
    "rejected": HypothesisStatus.rejected,
    "duplicate": HypothesisStatus.superseded,
    "superseded": HypothesisStatus.superseded,
}

_SCHEMA_STATUS_TO_DRAFT: dict[HypothesisStatus, str] = {
    HypothesisStatus.alive: "active",
    HypothesisStatus.accepted: "accepted",
    HypothesisStatus.rejected: "rejected",
    HypothesisStatus.superseded: "duplicate",
}


def draft_to_hypothesis(draft: HypothesisDraft) -> Hypothesis:
    """Validate a runtime draft into a structural Hypothesis (no retrieval context)."""
    return Hypothesis.model_validate(
        {
            "id": draft.id,
            "statement": draft.statement,
            "rationale": (draft.rationale or draft.statement).strip() or draft.statement,
            "mechanism": (draft.mechanism or "To be refined.").strip() or "To be refined.",
            "experiment": (draft.experiment or "To be specified.").strip() or "To be specified.",
            "falsification": (draft.falsification or "").strip() or "Not specified.",
            "novelty": float(draft.novelty if draft.novelty is not None else 0.5),
            "feasibility": float(draft.feasibility if draft.feasibility is not None else 0.5),
            "elo": draft.elo,
            "generation": draft.generation,
            "parent_id": draft.parent_id,
            "cluster_id": draft.cluster_id,
            "status": _DRAFT_STATUS_TO_SCHEMA.get(draft.status, HypothesisStatus.alive),
            "evidence": [
                {
                    "pmid": e.pmid,
                    "snippet": e.snippet,
                    "stance": e.stance,
                }
                for e in draft.evidence
            ],
            "unvalidated_lead": True,
        }
    )


def hypothesis_to_draft(
    hyp: Hypothesis,
    *,
    status: str | None = None,
    reflection: str | None = None,
) -> HypothesisDraft:
    """Convert a structural Hypothesis into a LangGraph draft."""
    draft_status = status or _SCHEMA_STATUS_TO_DRAFT.get(hyp.status, "active")
    return HypothesisDraft(
        id=hyp.id or "",
        statement=hyp.statement,
        rationale=hyp.rationale,
        mechanism=hyp.mechanism,
        experiment=hyp.experiment,
        falsification=hyp.falsification,
        novelty=hyp.novelty,
        feasibility=hyp.feasibility,
        elo=hyp.elo,
        generation=hyp.generation,
        parent_id=hyp.parent_id,
        cluster_id=hyp.cluster_id,
        status=draft_status,
        evidence=[
            EvidenceRef(pmid=e.pmid, snippet=e.snippet, stance=e.stance.value) for e in hyp.evidence
        ],
        reflection=reflection,
        unvalidated_lead=True,
    )


def try_parse_to_draft(
    raw: dict[str, Any],
    *,
    retrieved_pmids: set[str] | list[str],
    draft_status: str = "active",
) -> HypothesisDraft | None:
    """Parse LLM JSON into a draft; return None on structural/cite-only failure."""
    try:
        hyp = parse_hypothesis(raw, retrieved_pmids)
    except Exception:
        return None
    return hypothesis_to_draft(hyp, status=draft_status)
