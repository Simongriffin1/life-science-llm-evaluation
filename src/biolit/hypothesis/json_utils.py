"""JSON helpers for hypothesis LLM calls."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(text: str) -> dict[str, Any]:
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


def extract_json_array(text: str) -> list[Any]:
    text = text.strip()
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, list):
        raise json.JSONDecodeError("expected array", text, 0)
    return list(parsed)
