"""Retrieval API router (stub until Phase 2)."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()


class RetrieveRequest(BaseModel):
    query: str
    filters: dict[str, object] | None = None
    top_k: int = Field(default=20, ge=1, le=100)
    mode: str = Field(default="hybrid", pattern="^(bm25|dense|hybrid)$")


class RetrieveResponse(BaseModel):
    status: str = "not_implemented"
    message: str = "Retrieval lands in Phase 1–2"


@router.post("")
async def retrieve(_body: RetrieveRequest) -> RetrieveResponse:
    """Stub: hybrid retrieval arrives in Phases 1–2."""
    return RetrieveResponse()
