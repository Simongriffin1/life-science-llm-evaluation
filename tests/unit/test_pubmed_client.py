"""Unit tests for PubMed parsers and client (mocked HTTP)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from biolit.core.config import get_settings
from biolit.ingest.parsers import (
    parse_efetch_xml,
    parse_esearch_pmids,
    parse_esummary_json,
)
from biolit.ingest.pubmed_client import PubMedClient

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "pubmed"


def test_parse_esearch_pmids() -> None:
    payload = json.loads((FIXTURES / "esearch_trem2.json").read_text())
    pmids = parse_esearch_pmids(payload)
    assert pmids == ["42458498", "42437587", "42396649"]


def test_parse_esummary_json() -> None:
    payload = json.loads((FIXTURES / "esummary_trem2.json").read_text())
    docs = parse_esummary_json(payload)
    assert len(docs) == 3
    assert docs[0].pmid == "42458498"
    assert docs[0].title
    assert docs[0].authors


def test_parse_efetch_xml() -> None:
    xml_text = (FIXTURES / "efetch_trem2.xml").read_text()
    docs = parse_efetch_xml(xml_text)
    assert len(docs) == 3
    by_pmid = {d.pmid: d for d in docs}
    assert "42458498" in by_pmid
    doc = by_pmid["42458498"]
    assert doc.title and "TREM2" in doc.title
    assert doc.abstract and len(doc.abstract) > 100
    assert doc.doi
    assert doc.journal
    assert doc.authors


@pytest.mark.asyncio
async def test_pubmed_client_esearch_and_efetch_mocked() -> None:
    esearch_body = (FIXTURES / "esearch_trem2.json").read_text()
    efetch_body = (FIXTURES / "efetch_trem2.xml").read_text()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params.get("tool")
        assert request.url.params.get("email")
        path = request.url.path
        if path.endswith("esearch.fcgi"):
            return httpx.Response(200, text=esearch_body)
        if path.endswith("efetch.fcgi"):
            return httpx.Response(200, text=efetch_body)
        return httpx.Response(404, text="not found")

    transport = httpx.MockTransport(handler)
    settings = get_settings().model_copy(
        update={
            "ncbi_base_url": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
            "ncbi_tool": "biolit-test",
            "ncbi_email": "test@example.com",
        }
    )
    async with httpx.AsyncClient(transport=transport) as http:
        client = PubMedClient(settings=settings, client=http, use_cache=False)
        with (
            patch("biolit.ingest.pubmed_client.cache_get", new=AsyncMock(return_value=None)),
            patch("biolit.ingest.pubmed_client.cache_set", new=AsyncMock(return_value=True)),
        ):
            pmids = await client.esearch("TREM2 microglia", retmax=3)
            assert pmids == ["42458498", "42437587", "42396649"]
            docs = await client.efetch(pmids)
            assert len(docs) == 3
            assert docs[0].pmid == "42458498"


@pytest.mark.asyncio
async def test_pubmed_client_retries_on_429() -> None:
    esearch_body = (FIXTURES / "esearch_trem2.json").read_text()
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, text=esearch_body)

    transport = httpx.MockTransport(handler)
    settings = get_settings().model_copy(
        update={"ncbi_base_url": "https://example.test/entrez/eutils"}
    )
    async with httpx.AsyncClient(transport=transport) as http:
        client = PubMedClient(settings=settings, client=http, use_cache=False)
        with (
            patch("biolit.ingest.pubmed_client.cache_get", new=AsyncMock(return_value=None)),
            patch("biolit.ingest.pubmed_client.cache_set", new=AsyncMock(return_value=True)),
            patch("biolit.ingest.pubmed_client.asyncio.sleep", new=AsyncMock()),
        ):
            pmids = await client.esearch("TREM2", retmax=3)
        assert pmids == ["42458498", "42437587", "42396649"]
        assert calls["n"] == 3
