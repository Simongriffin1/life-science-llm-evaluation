"""Text chunking helpers for abstract indexing."""

from __future__ import annotations


def chunk_text(
    text: str,
    *,
    max_chars: int = 1200,
    overlap: int = 150,
) -> list[str]:
    """
    Split text into overlapping character windows.

    Prefers paragraph boundaries when present; falls back to fixed windows.
    """
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if len(paragraphs) > 1:
        chunks: list[str] = []
        buf = ""
        for para in paragraphs:
            candidate = f"{buf}\n{para}".strip() if buf else para
            if len(candidate) <= max_chars:
                buf = candidate
                continue
            if buf:
                chunks.append(buf)
            if len(para) <= max_chars:
                buf = para
            else:
                chunks.extend(_window(para, max_chars=max_chars, overlap=overlap))
                buf = ""
        if buf:
            chunks.append(buf)
        return chunks

    return _window(cleaned, max_chars=max_chars, overlap=overlap)


def _window(text: str, *, max_chars: int, overlap: int) -> list[str]:
    step = max(1, max_chars - overlap)
    out: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start += step
    return [c for c in out if c]


def chunks_for_document(title: str | None, abstract: str | None) -> list[tuple[str, str]]:
    """
    Build (section, text) chunks for a PubMed document.

    Returns at least a title chunk when present, plus abstract windows.
    """
    out: list[tuple[str, str]] = []
    if title and title.strip():
        out.append(("title", title.strip()))
    if abstract and abstract.strip():
        for i, piece in enumerate(chunk_text(abstract)):
            # Prepend title to the first abstract chunk for MedCPT-style context.
            if i == 0 and title:
                piece = f"{title.strip()}. {piece}"
            out.append(("abstract", piece))
    if not out and title:
        out.append(("title", title.strip()))
    return out
