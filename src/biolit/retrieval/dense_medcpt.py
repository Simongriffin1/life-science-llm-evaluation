"""MedCPT dense retrieval (query + article encoders)."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from biolit.core.config import get_settings
from biolit.core.logging import get_logger
from biolit.ingest.models import PubMedDocument
from biolit.retrieval.types import RankedHit

logger = get_logger(__name__)

QUERY_ENCODER_ID = "ncbi/MedCPT-Query-Encoder"
ARTICLE_ENCODER_ID = "ncbi/MedCPT-Article-Encoder"


class DenseBackend(Protocol):
    """Encodable backend used by dense ranking (real MedCPT or test double)."""

    def encode_query(self, query: str) -> NDArray[np.floating[Any]]: ...

    def encode_articles(self, texts: list[str]) -> NDArray[np.floating[Any]]: ...


class MedCPTBackend:
    """Lazy-loaded MedCPT query/article encoders via Hugging Face transformers."""

    def __init__(self, device: str | None = None) -> None:
        settings = get_settings()
        self.device = device or settings.medcpt_device
        self._query_tokenizer: Any = None
        self._query_model: Any = None
        self._article_tokenizer: Any = None
        self._article_model: Any = None
        self._lock = asyncio.Lock()
        self._loaded = False

    def _ensure_loaded_sync(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "MedCPT requires the retrieval extra: "
                "`uv sync --extra retrieval` (torch + transformers)."
            ) from exc

        logger.info(
            "MedCPT first load: downloading encoders from Hugging Face if not cached "
            f"({QUERY_ENCODER_ID}, {ARTICLE_ENCODER_ID}); this can take several minutes",
            extra={"device": self.device},
        )
        from biolit.retrieval.torch_compat import from_pretrained_medcpt

        self._query_tokenizer = AutoTokenizer.from_pretrained(QUERY_ENCODER_ID)  # type: ignore[no-untyped-call,unused-ignore]
        self._query_model = from_pretrained_medcpt(AutoModel, QUERY_ENCODER_ID)
        self._article_tokenizer = AutoTokenizer.from_pretrained(ARTICLE_ENCODER_ID)  # type: ignore[no-untyped-call,unused-ignore]
        self._article_model = from_pretrained_medcpt(AutoModel, ARTICLE_ENCODER_ID)

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU")
            self.device = "cpu"

        self._query_model.to(self.device).eval()
        self._article_model.to(self.device).eval()
        self._loaded = True
        logger.info("MedCPT encoders ready")

    def encode_query(self, query: str) -> NDArray[np.floating[Any]]:
        self._ensure_loaded_sync()
        import torch

        with torch.no_grad():
            encoded = self._query_tokenizer(
                query,
                return_tensors="pt",
                truncation=True,
                max_length=64,
                padding=True,
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            out = self._query_model(**encoded)
            # CLS embedding
            emb = out.last_hidden_state[:, 0, :]
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            return np.asarray(emb.cpu().numpy()[0], dtype=np.float32)

    def encode_articles(self, texts: list[str]) -> NDArray[np.floating[Any]]:
        self._ensure_loaded_sync()
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)
        import torch

        with torch.no_grad():
            encoded = self._article_tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            out = self._article_model(**encoded)
            emb = out.last_hidden_state[:, 0, :]
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            return np.asarray(emb.cpu().numpy(), dtype=np.float32)


_backend: DenseBackend | None = None


def get_dense_backend() -> DenseBackend:
    """Process-wide MedCPT backend singleton."""
    global _backend
    if _backend is None:
        _backend = MedCPTBackend()
    return _backend


def set_dense_backend(backend: DenseBackend | None) -> None:
    """Override the dense backend (tests). Pass None to reset."""
    global _backend
    _backend = backend


def _article_text(doc: PubMedDocument) -> str:
    title = doc.title or ""
    abstract = doc.abstract or ""
    if title and abstract:
        return f"{title}\n{abstract}"
    return title or abstract


def _rank_dense_sync(
    query: str,
    documents: list[PubMedDocument],
    *,
    top_k: int | None,
    backend: DenseBackend,
) -> list[RankedHit]:
    if not documents:
        return []
    q = backend.encode_query(query)
    texts = [_article_text(d) for d in documents]
    article_embs = backend.encode_articles(texts)
    # Cosine similarity (== dot product for L2-normalized vectors)
    scores = article_embs @ q
    indexed = sorted(enumerate(scores.tolist()), key=lambda p: (-float(p[1]), p[0]))
    limit = top_k if top_k is not None else len(indexed)
    hits: list[RankedHit] = []
    for rank, (idx, score) in enumerate(indexed[:limit], start=1):
        hits.append(
            RankedHit(
                pmid=documents[idx].pmid,
                score=float(score),
                rank=rank,
                retriever="dense",
                scores={"dense": float(score)},
            )
        )
    return hits


async def rank_dense(
    query: str,
    documents: list[PubMedDocument],
    *,
    top_k: int | None = None,
    backend: DenseBackend | None = None,
) -> list[RankedHit]:
    """Async dense ranking; model inference runs in a threadpool."""
    b = backend or get_dense_backend()
    return await asyncio.to_thread(_rank_dense_sync, query, documents, top_k=top_k, backend=b)
