"""MedCPT cross-encoder reranking of fused candidates."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

from biolit.core.config import get_settings
from biolit.core.logging import get_logger
from biolit.ingest.models import PubMedDocument
from biolit.retrieval.types import RankedHit

logger = get_logger(__name__)

CROSS_ENCODER_ID = "ncbi/MedCPT-Cross-Encoder"


class RerankBackend(Protocol):
    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]: ...


class MedCPTCrossEncoderBackend:
    """Lazy-loaded MedCPT cross-encoder."""

    def __init__(self, device: str | None = None) -> None:
        settings = get_settings()
        self.device = device or settings.medcpt_device
        self._tokenizer: Any = None
        self._model: Any = None
        self._loaded = False

    def _ensure_loaded_sync(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "MedCPT cross-encoder requires the retrieval extra: `uv sync --extra retrieval`."
            ) from exc

        logger.info(
            "MedCPT first load: downloading cross-encoder from Hugging Face if not cached "
            f"({CROSS_ENCODER_ID}); this can take several minutes",
            extra={"device": self.device},
        )
        from biolit.retrieval.torch_compat import from_pretrained_medcpt

        self._tokenizer = AutoTokenizer.from_pretrained(CROSS_ENCODER_ID)
        self._model = from_pretrained_medcpt(AutoModelForSequenceClassification, CROSS_ENCODER_ID)

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU")
            self.device = "cpu"

        self._model.to(self.device).eval()
        self._loaded = True
        logger.info("MedCPT cross-encoder ready")

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        self._ensure_loaded_sync()
        if not pairs:
            return []
        import torch

        queries = [p[0] for p in pairs]
        passages = [p[1] for p in pairs]
        with torch.no_grad():
            encoded = self._tokenizer(
                queries,
                passages,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            logits = self._model(**encoded).logits.view(-1)
            return [float(x) for x in logits.cpu().tolist()]


_rerank_backend: RerankBackend | None = None


def get_rerank_backend() -> RerankBackend:
    global _rerank_backend
    if _rerank_backend is None:
        _rerank_backend = MedCPTCrossEncoderBackend()
    return _rerank_backend


def set_rerank_backend(backend: RerankBackend | None) -> None:
    global _rerank_backend
    _rerank_backend = backend


def _passage(doc: PubMedDocument) -> str:
    title = doc.title or ""
    abstract = doc.abstract or ""
    if title and abstract:
        return f"{title}. {abstract}"
    return title or abstract


def _rerank_sync(
    query: str,
    candidates: list[RankedHit],
    documents_by_pmid: dict[str, PubMedDocument],
    *,
    top_k: int | None,
    backend: RerankBackend,
) -> list[RankedHit]:
    if not candidates:
        return []
    pairs: list[tuple[str, str]] = []
    for hit in candidates:
        doc = documents_by_pmid.get(hit.pmid)
        pairs.append((query, _passage(doc) if doc else ""))
    scores = backend.score_pairs(pairs)
    rescored: list[RankedHit] = []
    for hit, score in zip(candidates, scores, strict=True):
        merged_scores = dict(hit.scores)
        merged_scores["rerank"] = float(score)
        rescored.append(
            RankedHit(
                pmid=hit.pmid,
                score=float(score),
                rank=hit.rank,
                retriever="rerank",
                scores=merged_scores,
            )
        )
    rescored.sort(key=lambda h: (-h.score, h.pmid))
    limit = top_k if top_k is not None else len(rescored)
    out: list[RankedHit] = []
    for rank, hit in enumerate(rescored[:limit], start=1):
        out.append(
            RankedHit(
                pmid=hit.pmid,
                score=hit.score,
                rank=rank,
                retriever="rerank",
                scores=hit.scores,
            )
        )
    return out


async def rerank(
    query: str,
    candidates: list[RankedHit],
    documents: list[PubMedDocument],
    *,
    top_k: int | None = None,
    backend: RerankBackend | None = None,
) -> list[RankedHit]:
    """Rerank fused candidates with MedCPT cross-encoder (threadpool)."""
    b = backend or get_rerank_backend()
    by_pmid = {d.pmid: d for d in documents}
    return await asyncio.to_thread(
        _rerank_sync,
        query,
        candidates,
        by_pmid,
        top_k=top_k,
        backend=b,
    )
