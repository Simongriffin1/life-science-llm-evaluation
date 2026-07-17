"""Apply human review actions to hypothesis drafts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from biolit.core.logging import get_logger
from biolit.hypothesis.models import HypothesisDraft

logger = get_logger(__name__)

ActionKind = Literal["accept", "reject", "redirect"]


class HypothesisAction(BaseModel):
    id: str
    action: ActionKind
    note: str | None = None


class ReviewDecision(BaseModel):
    actions: list[HypothesisAction] = Field(default_factory=list)


def apply_review_decision(
    hypotheses: list[HypothesisDraft],
    decision: ReviewDecision | dict[str, Any],
    *,
    steering_notes: list[str] | None = None,
) -> tuple[list[HypothesisDraft], list[str], list[str]]:
    """
    Apply accept | reject | redirect.

    reject drops the hypothesis (and later descendants via parent_id checks).
    redirect appends a steering note for subsequent prompts.
    Returns (kept_hypotheses, rejected_ids, updated_steering_notes).
    """
    parsed = (
        decision
        if isinstance(decision, ReviewDecision)
        else ReviewDecision.model_validate(decision)
    )
    notes = list(steering_notes or [])
    rejected: set[str] = set()
    by_id = {h.id: h for h in hypotheses}

    for act in parsed.actions:
        if act.id not in by_id:
            logger.warning("hypothesis.feedback_unknown_id", extra={"id": act.id})
            continue
        if act.action == "reject":
            rejected.add(act.id)
            by_id[act.id].status = "rejected"
        elif act.action == "redirect":
            note = (act.note or "").strip()
            if note:
                notes.append(f"[redirect {act.id}] {note}")
        elif act.action == "accept":
            by_id[act.id].status = "active"

    # Drop rejected lineages (self + descendants by parent_id chain)
    if rejected:
        changed = True
        while changed:
            changed = False
            for h in list(by_id.values()):
                if h.id in rejected:
                    continue
                if h.parent_id and h.parent_id in rejected:
                    rejected.add(h.id)
                    h.status = "rejected"
                    changed = True

    kept = [h for h in by_id.values() if h.id not in rejected and h.status != "rejected"]
    # Retain rejected drafts (status already set) so tournament FKs / audit stay valid.
    rejected_drafts = [h for h in by_id.values() if h.id in rejected]
    return kept + rejected_drafts, sorted(rejected), notes
