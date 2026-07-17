"""Schema-level cite-only and provenance invariants."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from biolit.hypothesis.convert import draft_to_hypothesis, hypothesis_to_draft
from biolit.hypothesis.models import EvidenceRef, HypothesisDraft
from biolit.hypothesis.schemas import (
    Evidence,
    Hypothesis,
    Stance,
    enforce_cite_only,
    parse_hypothesis,
)


def _valid_raw(*, pmid: str = "12345") -> dict:
    return {
        "id": "h1",
        "statement": "TREM2 modulates microglial phagocytosis.",
        "rationale": "Supported by cited abstracts.",
        "mechanism": "Receptor signaling alters phagocytic uptake.",
        "experiment": "Knockdown TREM2 in iPSC microglia and measure amyloid uptake.",
        "falsification": "No change in uptake after knockdown.",
        "novelty": 0.6,
        "feasibility": 0.7,
        "evidence": [
            {
                "pmid": pmid,
                "snippet": "TREM2 is required for microglial phagocytosis.",
                "stance": "supports",
            }
        ],
        "unvalidated_lead": True,
    }


def test_missing_evidence_fails_validation() -> None:
    raw = _valid_raw()
    raw["evidence"] = []
    with pytest.raises(ValidationError):
        Hypothesis.model_validate(raw)


def test_parse_hypothesis_rejects_out_of_set_pmid() -> None:
    with pytest.raises(ValidationError):
        parse_hypothesis(_valid_raw(pmid="99999"), retrieved_pmids={"12345"})


def test_enforce_cite_only_returns_stray_pmids() -> None:
    hyp = Hypothesis.model_validate(_valid_raw(pmid="99999"))
    stray = enforce_cite_only(hyp, {"12345", "22222"})
    assert stray == ["99999"]


def test_enforce_cite_only_empty_when_subset() -> None:
    hyp = parse_hypothesis(_valid_raw(pmid="12345"), retrieved_pmids={"12345", "67890"})
    assert enforce_cite_only(hyp, {"12345", "67890"}) == []


def test_draft_schema_round_trip_preserves_evidence() -> None:
    draft = HypothesisDraft(
        id="h1",
        statement="TREM2 modulates microglial phagocytosis.",
        rationale="Supported by cited abstracts.",
        mechanism="Receptor signaling alters phagocytic uptake.",
        experiment="Knockdown TREM2 in iPSC microglia and measure amyloid uptake.",
        falsification="No change in uptake after knockdown.",
        novelty=0.6,
        feasibility=0.7,
        evidence=[
            EvidenceRef(
                pmid="12345",
                snippet="TREM2 is required for microglial phagocytosis.",
                stance="supports",
            )
        ],
        unvalidated_lead=True,
    )
    hyp = draft_to_hypothesis(draft)
    assert isinstance(hyp.evidence[0], Evidence)
    assert hyp.evidence[0].pmid == "12345"
    assert hyp.unvalidated_lead is True
    back = hypothesis_to_draft(hyp)
    assert back.id == "h1"
    assert back.cited_pmids() == {"12345"}
    assert back.evidence[0].stance == Stance.supports.value
    assert back.unvalidated_lead is True
