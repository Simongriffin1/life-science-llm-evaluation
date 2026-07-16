"""LiteLLM-backed LLM gateway. The ONLY place provider calls happen."""

from __future__ import annotations

from typing import Any

import litellm

from biolit.core.config import get_settings
from biolit.core.logging import get_logger

logger = get_logger(__name__)

# Soft-disable litellm's own verbose logging; we use structured logs.
litellm.suppress_debug_info = True


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
    logger.info(
        "llm.complete.done",
        extra={
            "model": resolved_model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    )
    return result
