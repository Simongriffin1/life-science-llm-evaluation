"""Robust answer extraction for MIRAGE-style eval completions."""

from __future__ import annotations

import re
from typing import NamedTuple

from biolit.eval.metrics import normalize_label
from biolit.eval.templates import ANSWER_DELIMITER

# Primary: The answer is (X)  / The answer is X
_DELIMITER_RE = re.compile(
    rf"{re.escape(ANSWER_DELIMITER)}\s*\(?\s*([A-Za-z]+|yes|no|maybe)\s*\)?",
    flags=re.IGNORECASE,
)
# Last standalone option letter on its own line or trailed by punctuation
_LAST_LETTER_RE = re.compile(
    r"(?:^|\n)\s*(?:\(?([A-Da-d])\)?[.:)]?\s*$)",
    flags=re.MULTILINE,
)
_LAST_YNM_RE = re.compile(
    r"(?:^|\n)\s*(?:\(?\s*(yes|no|maybe)\s*\)?[.:]?\s*$)",
    flags=re.IGNORECASE | re.MULTILINE,
)


class ParseResult(NamedTuple):
    label: str | None
    confidence: float
    raw: str
    unparsed: bool


def _canonicalize(candidate: str, options: dict[str, str] | None) -> str | None:
    """Map a raw candidate onto an option key; None if it cannot be matched."""
    token = candidate.strip().strip("`\"'").strip()
    if not token:
        return None
    if options:
        keys_upper = {k.upper(): k for k in options}
        # Single letter → option key
        if len(token) == 1 and token.upper() in keys_upper:
            return keys_upper[token.upper()]
        # Exact key
        for key in options:
            if normalize_label(key) == normalize_label(token):
                return key
        # Match option text
        for key, val in options.items():
            if normalize_label(val) == normalize_label(token):
                return key
        # Leading letter "A)" / "A."
        m = re.match(r"^([A-Za-z])\b", token)
        if m and m.group(1).upper() in keys_upper:
            return keys_upper[m.group(1).upper()]
        return None
    # No options: return normalized token as-is (open-ended); still "parsed"
    return normalize_label(token) or None


def parse_answer(text: str, options: dict[str, str] | None = None) -> ParseResult:
    """
    Extract a label from a model completion.

    Primary: regex on ``The answer is (X)``.
    Fallbacks: last standalone option letter / yes|no|maybe; normalized label match.
    Never silently coerce — if nothing matches, ``unparsed=True`` and ``label=None``.
    """
    raw = text or ""
    stripped = raw.strip()
    if not stripped:
        return ParseResult(label=None, confidence=0.0, raw=raw, unparsed=True)

    # 1) Delimiter matches (prefer last occurrence)
    matches = list(_DELIMITER_RE.finditer(stripped))
    if matches:
        cand = matches[-1].group(1)
        label = _canonicalize(cand, options)
        if label is not None:
            return ParseResult(label=label, confidence=1.0, raw=raw, unparsed=False)

    # 2) Last standalone letter / ynm on its own line
    if options:
        option_keys = {k.upper() for k in options}
        ynm = {"YES", "NO", "MAYBE"} & option_keys or (
            {"YES", "NO", "MAYBE"}
            if all(k.lower() in {"yes", "no", "maybe"} for k in options)
            else set()
        )
        if ynm or all(k.lower() in {"yes", "no", "maybe"} for k in options):
            ynm_hits = list(_LAST_YNM_RE.finditer(stripped))
            if ynm_hits:
                label = _canonicalize(ynm_hits[-1].group(1), options)
                if label is not None:
                    return ParseResult(label=label, confidence=0.7, raw=raw, unparsed=False)
        letter_hits = list(_LAST_LETTER_RE.finditer(stripped))
        if letter_hits:
            label = _canonicalize(letter_hits[-1].group(1), options)
            if label is not None:
                return ParseResult(label=label, confidence=0.6, raw=raw, unparsed=False)

        # 3) Normalized label match anywhere in the last non-empty line
        last_line = next(
            (ln.strip() for ln in reversed(stripped.splitlines()) if ln.strip()),
            "",
        )
        label = _canonicalize(last_line, options)
        if label is not None:
            return ParseResult(label=label, confidence=0.4, raw=raw, unparsed=False)
        # Scan whole text for unique option-key occurrence
        found: list[str] = []
        for key in options:
            if re.search(rf"\b{re.escape(key)}\b", stripped, flags=re.IGNORECASE):
                found.append(key)
        if len(found) == 1:
            return ParseResult(label=found[0], confidence=0.3, raw=raw, unparsed=False)
        # Unique option *text* mention (e.g. "my choice is two")
        text_hits: list[str] = []
        norm_body = normalize_label(stripped)
        for key, val in options.items():
            nv = normalize_label(val)
            if nv and nv in norm_body:
                text_hits.append(key)
        if len(text_hits) == 1:
            return ParseResult(label=text_hits[0], confidence=0.35, raw=raw, unparsed=False)

    return ParseResult(label=None, confidence=0.0, raw=raw, unparsed=True)
