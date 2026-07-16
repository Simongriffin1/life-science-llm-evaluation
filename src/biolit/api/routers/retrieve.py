"""Retrieval API router."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from biolit.retrieval.service import RetrieveFilters, RetrieveResult, retrieve

router = APIRouter()


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    filters: RetrieveFilters | None = None
    top_k: int = Field(default=20, ge=1, le=100)
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    candidate_cap: int | None = Field(default=None, ge=1, le=500)
    use_index: bool = False


@router.post("", response_model=RetrieveResult)
async def retrieve_endpoint(body: RetrieveRequest) -> RetrieveResult:
    """Hybrid PubMed retrieval (BM25 + MedCPT + RRF + cross-encoder rerank)."""
    try:
        return await retrieve(
            body.query,
            filters=body.filters,
            top_k=body.top_k,
            mode=body.mode,
            candidate_cap=body.candidate_cap,
            use_index=body.use_index,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
