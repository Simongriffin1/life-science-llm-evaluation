"""LiteLLM-backed LLM gateway. The ONLY place provider calls happen."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import litellm

from biolit.core.config import get_settings
from biolit.core.logging import get_logger

logger = get_logger(__name__)

# Soft-disable litellm's own verbose logging; we use structured logs.
litellm.suppress_debug_info = True


class TokenBudgetExceeded(RuntimeError):
    """Raised when cumulative LLM tokens exceed a hard budget."""


@dataclass
class TokenUsageTracker:
    max_total_tokens: int | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    by_model: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def record(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.calls += 1
        bucket = self.by_model.setdefault(model, {"prompt": 0, "completion": 0, "calls": 0})
        bucket["prompt"] += prompt_tokens
        bucket["completion"] += completion_tokens
        bucket["calls"] += 1
        if self.max_total_tokens is not None and self.total_tokens > self.max_total_tokens:
            raise TokenBudgetExceeded(
                f"Token budget exceeded: used {self.total_tokens} > max {self.max_total_tokens}"
            )


_usage_tracker: ContextVar[TokenUsageTracker | None] = ContextVar("llm_usage_tracker", default=None)


@contextmanager
def track_token_usage(max_total_tokens: int | None = None) -> Iterator[TokenUsageTracker]:
    """Accumulate token usage for nested ``complete`` calls; optionally enforce a hard cap."""
    tracker = TokenUsageTracker(max_total_tokens=max_total_tokens)
    token = _usage_tracker.set(tracker)
    try:
        yield tracker
    finally:
        _usage_tracker.reset(token)


def _langfuse_enabled() -> bool:
    settings = get_settings()
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


def _maybe_configure_langfuse() -> None:
    """Wire Langfuse callbacks into LiteLLM when keys are present."""
    if not _langfuse_enabled():
        return
    settings = get_settings()
    # LiteLLM reads these env vars; set them so tracing works without a global .env load.
    import os

    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key or "")
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key or "")
    os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)
    if "langfuse" not in (litellm.success_callback or []):
        litellm.success_callback = [*(litellm.success_callback or []), "langfuse"]
        litellm.failure_callback = [*(litellm.failure_callback or []), "langfuse"]


async def complete(
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    **params: Any,
) -> dict[str, Any]:
    """
    Async chat completion via LiteLLM.

    Returns a dict with content, model, and token usage. Never call provider SDKs elsewhere.
    """
    _maybe_configure_langfuse()
    settings = get_settings()
    resolved_model = model or settings.default_llm_model

    kwargs: dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
        "temperature": temperature,
        **params,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    logger.info(
        "llm.complete",
        extra={"model": resolved_model, "n_messages": len(messages)},
    )

    response = await litellm.acompletion(**kwargs)

    choice = response.choices[0]
    content = choice.message.content or ""
    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens))

    result: dict[str, Any] = {
        "content": content,
        "model": resolved_model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "finish_reason": getattr(choice, "finish_reason", None),
        "raw": response,
    }
    tracker = _usage_tracker.get()
    if tracker is not None:
        tracker.record(resolved_model, prompt_tokens, completion_tokens)
    logger.info(
        "llm.complete.done",
        extra={
            "model": resolved_model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    )
    return result
