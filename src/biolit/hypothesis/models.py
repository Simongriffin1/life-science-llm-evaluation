"""In-memory hypothesis models used by the LangGraph engine."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Stance = Literal["supports", "contradicts", "context"]


class EvidenceRef(BaseModel):
    pmid: str
    snippet: str
    stance: Stance = "supports"


class HypothesisDraft(BaseModel):
    id: str
    statement: str
    rationale: str | None = None
    mechanism: str | None = None
    experiment: str | None = None
    falsification: str | None = None
    novelty: float | None = None
    feasibility: float | None = None
    elo: float = 1000.0
    generation: int = 0
    parent_id: str | None = None
    cluster_id: str | None = None
    status: str = "active"
    evidence: list[EvidenceRef] = Field(default_factory=list)
    reflection: str | None = None
    unvalidated_lead: bool = True

    def has_provenance(self) -> bool:
        return len(self.evidence) > 0

    def cited_pmids(self) -> set[str]:
        return {e.pmid for e in self.evidence}


class HypothesisConfig(BaseModel):
    n_seed: int = 6
    tournament_rounds: int = 2
    evolution_rounds: int = 1
    top_k_retrieve: int = 8
    retrieval_mode: str = "bm25"
    candidate_cap: int = 40
    use_index: bool = False
    model: str = "gpt-4o-mini"
    judge_model: str | None = None
    elo_k: float = 32.0
    max_proposals: int = 3
    human_feedback: str | None = None
    max_total_tokens: int | None = 80_000
    interactive: bool = False


class HypothesisRunResult(BaseModel):
    run_id: str
    status: str
    research_goal: str
    proposals: list[HypothesisDraft] = Field(default_factory=list)
    n_hypotheses: int = 0
    n_matches: int = 0
    budget_used: dict[str, Any] = Field(default_factory=dict)
    retrieved_pmids: list[str] = Field(default_factory=list)
