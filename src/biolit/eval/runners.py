"""Closed-book and RAG eval runners with LLM response caching."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Literal

from biolit.core.cache import cache_get, cache_set
from biolit.core.llm import complete
from biolit.core.logging import get_logger
from biolit.eval.datasets import EvalItem
from biolit.eval.judge import judge_groundedness
from biolit.eval.metrics import normalize_label
from biolit.eval.parse import parse_answer
from biolit.eval.templates import build_prompt
from biolit.retrieval.service import retrieve

logger = get_logger(__name__)

EvalMode = Literal["closed_book", "rag"]


def _prompt_hash(model: str, messages: list[dict[str, str]]) -> str:
    material = json.dumps({"model": model, "messages": messages}, sort_keys=True)
    return hashlib.sha256(material.encode()).hexdigest()


async def _cached_complete(
    model: str,
    messages: list[dict[str, str]],
    *,
    use_cache: bool = True,
) -> dict[str, Any]:
    key = f"llm:{_prompt_hash(model, messages)}"
    if use_cache:
        cached = await cache_get(key)
        if cached:
            payload = json.loads(cached)
            assert isinstance(payload, dict)
            return dict(payload)
    result = await complete(model, messages, temperature=0.0)
    # Drop non-serializable raw response before caching.
    store = {k: v for k, v in result.items() if k != "raw"}
    if use_cache:
        await cache_set(key, json.dumps(store))
    return store


async def run_item(
    item: EvalItem,
    *,
    model: str,
    mode: EvalMode,
    dataset: str = "pubmedqa",
    few_shot_k: int = 0,
    retrieval: dict[str, Any] | None = None,
    judge_model: str | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Run one eval item; returns a row suitable for eval_items persistence."""
    snippets: list[str] = []
    retrieved_pmids: list[str] = []
    if mode == "rag":
        ret_cfg = retrieval or {}
        result = await retrieve(
            item.question,
            top_k=int(ret_cfg.get("top_k", 8)),
            mode=ret_cfg.get("mode", "bm25"),
            candidate_cap=int(ret_cfg.get("candidate_cap", 40)),
            use_index=bool(ret_cfg.get("use_index", False)),
            persist=False,
        )
        for doc in result.documents:
            retrieved_pmids.append(doc.pmid)
            snip = doc.highlight or (doc.title or "")
            if doc.abstract:
                snip = f"{doc.title or ''}. {doc.abstract[:400]}"
            snippets.append(snip.strip())

    prompt = build_prompt(
        item,
        dataset=dataset,
        mode=mode,
        few_shot_k=few_shot_k,
        snippets=snippets or None,
    )
    messages = [{"role": "user", "content": prompt}]
    t0 = time.perf_counter()
    llm = await _cached_complete(model, messages, use_cache=use_cache)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    parsed = parse_answer(llm["content"], item.options)
    prediction = parsed.label or ""
    # Unparsed is not scored as wrong — correct stays None.
    if parsed.unparsed:
        correct: bool | None = None
    else:
        correct = normalize_label(prediction) == normalize_label(item.gold)

    judge_json: dict[str, Any] = {
        "parse": {
            "unparsed": parsed.unparsed,
            "confidence": parsed.confidence,
            "raw": parsed.raw,
        }
    }
    if mode == "rag" and judge_model and snippets:
        grounded = await judge_groundedness(
            answer=llm["content"],
            snippets=snippets,
            model=judge_model,
        )
        judge_json["groundedness"] = grounded

    return {
        "question_id": item.id,
        "prompt": prompt,
        "prediction": prediction if not parsed.unparsed else None,
        "gold": item.gold,
        "correct": correct,
        "unparsed": parsed.unparsed,
        "parse_confidence": parsed.confidence,
        "retrieved_pmids": retrieved_pmids,
        "latency_ms": latency_ms,
        "prompt_tokens": int(llm.get("prompt_tokens") or 0),
        "completion_tokens": int(llm.get("completion_tokens") or 0),
        "judge_json": judge_json,
        "raw_completion": llm.get("content"),
        "gold_pmids": item.gold_pmids,
    }
