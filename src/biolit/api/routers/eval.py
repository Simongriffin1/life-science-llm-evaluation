"""Eval API router (stub until Phase 4)."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()


class EvalRequest(BaseModel):
    models: list[str]
    datasets: list[str]
    mode: str = Field(default="closed_book", pattern="^(closed_book|rag)$")
    retrieval: dict[str, object] | None = None
    judge_model: str | None = None
    seed: int = 7


class EvalResponse(BaseModel):
    status: str = "not_implemented"
    message: str = "Eval harness lands in Phase 4"


@router.post("")
async def enqueue_eval(_body: EvalRequest) -> EvalResponse:
    """Stub: eval harness arrives in Phase 4."""
    return EvalResponse()
