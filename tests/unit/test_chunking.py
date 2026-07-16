"""Tests for abstract chunking."""

from biolit.ingest.chunking import chunk_text, chunks_for_document


def test_chunk_text_short_passthrough() -> None:
    assert chunk_text("hello world") == ["hello world"]


def test_chunk_text_windows() -> None:
    text = "a" * 2500
    chunks = chunk_text(text, max_chars=1000, overlap=100)
    assert len(chunks) >= 2
    assert all(len(c) <= 1000 for c in chunks)


def test_chunks_for_document_includes_title_and_abstract() -> None:
    parts = chunks_for_document("TREM2 review", "Abstract paragraph one.\n\nParagraph two.")
    assert parts[0] == ("title", "TREM2 review")
    assert any(section == "abstract" for section, _ in parts)
    assert parts[1][1].startswith("TREM2 review")
