"""Hypothesis schemas with schema-level cite-only enforcement.

Every hypothesis must carry at least one evidence item, an explicit
falsification criterion, and the ``unvalidated_lead`` flag. When a hypothesis
is validated with a retrieved-PMID context, its citations are required to be a
subset of the retrieved set: a claim cannot cite a paper the retriever never
surfaced. This is the structural guarantee behind the smoke test in Phase 7a.

Usage
-----
Construct with the retrieval context so cite-only is enforced at parse time::

    hyp = parse_hypothesis(raw_json, retrieved_pmids={"12345", "67890"})

or check an already-built hypothesis::

    stray = enforce_cite_only(hyp, retrieved_pmids)
    if stray:
        ...  # reject the hypothesis; do not persist it
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

# Key under which the allowed PMID set is passed via `model_validate(..., context=...)`.
RETRIEVED_PMIDS: str = "retrieved_pmids"


class Stance(StrEnum):
    """How a cited source relates to the hypothesis."""

    supports = "supports"
    contradicts = "contradicts"
    context = "context"


class HypothesisStatus(StrEnum):
    alive = "alive"
    accepted = "accepted"
    rejected = "rejected"
    superseded = "superseded"


class Evidence(BaseModel):
    """A single provenance link: a verbatim snippet from a cited PMID."""

    model_config = ConfigDict(frozen=True)

    pmid: str = Field(..., pattern=r"^\d+$", description="PubMed ID, digits only.")
    snippet: str = Field(..., min_length=1, description="Verbatim span from the cited abstract.")
    stance: Stance

    @field_validator("pmid", mode="before")
    @classmethod
    def _coerce_pmid(cls, v: object) -> str:
        # Accept ints or padded strings from upstream parsers; normalize to a bare digit string.
        return str(v).strip()


class Hypothesis(BaseModel):
    """A single hypothesis. Cannot exist without provenance.

    Structural invariants enforced regardless of context:
      - at least one Evidence item (no provenance -> no hypothesis)
      - a non-empty falsification criterion
      - unvalidated_lead pinned True (cannot be omitted or flipped)

    Additional invariant when a retrieval context is supplied:
      - every cited PMID is a member of the retrieved set (cite-only)
    """

    id: str | None = None
    statement: str = Field(..., min_length=1)
    rationale: str = Field(..., min_length=1)
    mechanism: str = Field(..., min_length=1)
    experiment: str = Field(..., min_length=1, description="A concrete, runnable test.")
    falsification: str = Field(
        ..., min_length=1, description="The observation that would refute this hypothesis."
    )
    novelty: float = Field(..., ge=0.0, le=1.0)
    feasibility: float = Field(..., ge=0.0, le=1.0)
    elo: float = 1400.0
    generation: int = Field(0, ge=0)
    parent_id: str | None = None
    cluster_id: str | None = None
    status: HypothesisStatus = HypothesisStatus.alive
    evidence: list[Evidence] = Field(
        ..., min_length=1, description="Mandatory provenance. At least one item."
    )
    unvalidated_lead: Literal[True] = Field(
        True,
        description="Pinned. Every proposal leaves the system labeled as an unvalidated lead.",
    )

    @property
    def cited_pmids(self) -> set[str]:
        return {e.pmid for e in self.evidence}

    @model_validator(mode="after")
    def _cite_only(self, info: ValidationInfo) -> Hypothesis:
        context = info.context or {}
        retrieved = context.get(RETRIEVED_PMIDS)
        if retrieved is None:
            # No retrieval set supplied: structural checks above still apply,
            # but subset enforcement is deferred to enforce_cite_only().
            return self
        allowed = {str(p).strip() for p in retrieved}
        stray = sorted(self.cited_pmids - allowed)
        if stray:
            raise ValueError(
                f"cite-only violation: evidence cites PMIDs outside the retrieved set: {stray}"
            )
        return self


class MetaReviewOutput(BaseModel):
    """Final meta-review payload: the ranked proposals a run returns."""

    research_goal: str = Field(..., min_length=1)
    proposals: list[Hypothesis] = Field(..., min_length=1)
    unvalidated_lead: Literal[True] = True


def enforce_cite_only(hyp: Hypothesis, retrieved_pmids: Iterable[str]) -> list[str]:
    """Return the sorted list of cited PMIDs that fall outside the retrieved set.

    An empty list means the hypothesis satisfies cite-only. Use this in the
    service layer as a guard before persistence; a non-empty result should
    cause the hypothesis to be dropped and logged, never stored.
    """
    allowed = {str(p).strip() for p in retrieved_pmids}
    return sorted(hyp.cited_pmids - allowed)


def parse_hypothesis(raw: dict[str, Any], retrieved_pmids: Iterable[str]) -> Hypothesis:
    """Validate raw model output into a Hypothesis with cite-only enforced.

    Raises pydantic.ValidationError if provenance is missing, the falsification
    criterion is empty, or any citation is outside ``retrieved_pmids``.
    """
    return Hypothesis.model_validate(
        raw, context={RETRIEVED_PMIDS: {str(p).strip() for p in retrieved_pmids}}
    )
