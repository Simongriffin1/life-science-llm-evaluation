"""Hypothesis API router (stub until Phase 5)."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HypothesizeRequest(BaseModel):
    research_goal: str
    config: dict[str, object] | None = None


class HypothesizeResponse(BaseModel):
    status: str = "not_implemented"
    message: str = "Hypothesis engine lands in Phase 5"


@router.post("")
async def enqueue_hypothesis(_body: HypothesizeRequest) -> HypothesizeResponse:
    """Stub: hypothesis engine arrives in Phase 5."""
    return HypothesizeResponse()
