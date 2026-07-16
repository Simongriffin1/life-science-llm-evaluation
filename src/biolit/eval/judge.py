"""LLM-as-judge helpers for open-ended answers and groundedness."""

from __future__ import annotations

import json
import re
from typing import Any

from biolit.core.llm import complete
from biolit.core.logging import get_logger

logger = get_logger(__name__)


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("expected object", text, 0)
    return dict(parsed)


async def judge_open_ended(
    *,
    question: str,
    prediction: str,
    gold: str,
    model: str,
) -> dict[str, Any]:
    """Rubric judge: score 0–1 correctness for open-ended items."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a biomedical exam grader. Score the prediction against the gold "
                "answer. Respond ONLY with JSON: "
                '{"score": <0to1 float>, "correct": <bool>, "rationale": "<short>"}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\nGold: {gold}\nPrediction: {prediction}\n"
                "Score semantic correctness."
            ),
        },
    ]
    result = await complete(model, messages, temperature=0.0, max_tokens=300)
    try:
        parsed = _extract_json(result["content"])
    except (json.JSONDecodeError, TypeError):
        logger.warning("judge.parse_failed", extra={"content": result["content"][:200]})
        parsed = {"score": 0.0, "correct": False, "rationale": "parse_failed"}
    parsed["prompt_tokens"] = result["prompt_tokens"]
    parsed["completion_tokens"] = result["completion_tokens"]
    return parsed


async def judge_groundedness(
    *,
    answer: str,
    snippets: list[str],
    model: str,
) -> dict[str, Any]:
    """
    Estimate fraction of answer claims entailed by retrieved snippets.

    Returns JSON with groundedness in [0, 1].
    """
    joined = "\n---\n".join(snippets[:8]) if snippets else "(no snippets)"
    messages = [
        {
            "role": "system",
            "content": (
                "You assess whether an answer is grounded in provided literature snippets. "
                "Respond ONLY with JSON: "
                '{"groundedness": <0to1 float>, "rationale": "<short>"}'
            ),
        },
        {
            "role": "user",
            "content": f"Snippets:\n{joined}\n\nAnswer:\n{answer}\n",
        },
    ]
    result = await complete(model, messages, temperature=0.0, max_tokens=300)
    try:
        parsed = _extract_json(result["content"])
    except (json.JSONDecodeError, TypeError):
        parsed = {"groundedness": 0.0, "rationale": "parse_failed"}
    parsed["prompt_tokens"] = result["prompt_tokens"]
    parsed["completion_tokens"] = result["completion_tokens"]
    return parsed
