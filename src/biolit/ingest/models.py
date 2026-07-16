"""Pydantic models for PubMed ingest / retrieval candidates."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field


class PubMedDocument(BaseModel):
    """Parsed PubMed article aligned with the `documents` table schema."""

    pmid: str
    title: str | None = None
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    journal: str | None = None
    pub_date: date | None = None
    mesh_terms: list[str] = Field(default_factory=list)
    doi: str | None = None
    source: str = "pubmed"
    raw_json: dict[str, Any] | None = None

    def text_for_lexical(self) -> str:
        """Title + abstract string used by BM25."""
        parts = [p for p in (self.title, self.abstract) if p]
        return " ".join(parts)
