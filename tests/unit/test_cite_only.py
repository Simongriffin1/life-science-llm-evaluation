"""Unit tests for cite-only provenance enforcement."""

from __future__ import annotations

import pytest

from biolit.hypothesis.models import EvidenceRef, HypothesisDraft
from biolit.hypothesis.provenance import CiteOnlyViolation, enforce_cite_only


def _draft(*, pmid: str, hid: str = "h1") -> HypothesisDraft:
    return HypothesisDraft(
        id=hid,
        statement="Test statement",
        evidence=[EvidenceRef(pmid=pmid, snippet="snip", stance="supports")],
        unvalidated_lead=False,
    )


def test_enforce_cite_only_accepts_subset() -> None:
    kept = enforce_cite_only([_draft(pmid="111")], {"111", "222"}, strict=True)
    assert len(kept) == 1
    assert kept[0].unvalidated_lead is True
    assert kept[0].cited_pmids() == {"111"}


def test_enforce_cite_only_strict_rejects_foreign_pmid() -> None:
    with pytest.raises(CiteOnlyViolation) as exc:
        enforce_cite_only([_draft(pmid="999")], {"111"}, strict=True)
    assert "999" in exc.value.illegal_pmids


def test_enforce_cite_only_non_strict_filters() -> None:
    mixed = HypothesisDraft(
        id="h2",
        statement="Mixed cites",
        evidence=[
            EvidenceRef(pmid="111", snippet="ok", stance="supports"),
            EvidenceRef(pmid="999", snippet="bad", stance="supports"),
        ],
    )
    kept = enforce_cite_only([mixed], {"111"}, strict=False)
    assert len(kept) == 1
    assert kept[0].cited_pmids() == {"111"}
    assert kept[0].unvalidated_lead is True
