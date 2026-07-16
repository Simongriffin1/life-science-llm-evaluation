"""Async NCBI E-utilities client with retries, auth params, and Redis caching."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any
from urllib.parse import urlencode

import httpx

from biolit.core.cache import cache_get, cache_set
from biolit.core.config import Settings, get_settings
from biolit.core.logging import get_logger
from biolit.ingest.models import PubMedDocument
from biolit.ingest.parsers import parse_efetch_xml, parse_esearch_count, parse_esearch_pmids

logger = get_logger(__name__)

_MAX_RETRIES = 5
_BASE_BACKOFF_S = 0.5


class PubMedClientError(RuntimeError):
    """Raised when NCBI E-utilities calls fail after retries."""


class PubMedClient:
    """Thin async wrapper around esearch / esummary / efetch."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        use_cache: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client
        self._owns_client = client is None
        self.use_cache = use_cache

    async def __aenter__(self) -> PubMedClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _auth_params(self) -> dict[str, str]:
        params: dict[str, str] = {
            "tool": self.settings.ncbi_tool,
            "email": self.settings.ncbi_email,
        }
        if self.settings.ncbi_api_key:
            params["api_key"] = self.settings.ncbi_api_key
        return params

    def _cache_key(self, endpoint: str, params: dict[str, Any]) -> str:
        # Stable key: exclude auth from hash identity but include tool for safety.
        material = urlencode(sorted((k, str(v)) for k, v in params.items() if k != "api_key"))
        digest = hashlib.sha256(f"{endpoint}?{material}".encode()).hexdigest()
        return f"ncbi:{endpoint}:{digest}"

    async def _request(
        self,
        endpoint: str,
        params: dict[str, Any],
        *,
        expect_json: bool,
    ) -> str:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
            self._owns_client = True

        merged = {**self._auth_params(), **params}
        cache_key = self._cache_key(endpoint, merged)
        if self.use_cache:
            cached = await cache_get(cache_key)
            if cached is not None:
                logger.debug("ncbi.cache_hit", extra={"endpoint": endpoint})
                return cached

        url = f"{self.settings.ncbi_base_url.rstrip('/')}/{endpoint}"
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = await self._client.get(url, params=merged)
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise PubMedClientError(
                        f"NCBI {endpoint} HTTP {response.status_code}: {response.text[:200]}"
                    )
                response.raise_for_status()
                text = response.text
                if expect_json:
                    # Validate JSON early so bad payloads don't poison the cache.
                    json.loads(text)
                if self.use_cache:
                    await cache_set(cache_key, text)
                return text
            except (httpx.HTTPError, PubMedClientError, json.JSONDecodeError) as exc:
                last_error = exc
                sleep_s = _BASE_BACKOFF_S * (2**attempt)
                logger.warning(
                    "ncbi.retry",
                    extra={
                        "endpoint": endpoint,
                        "attempt": attempt + 1,
                        "sleep_s": sleep_s,
                        "error": str(exc),
                    },
                )
                await asyncio.sleep(sleep_s)

        raise PubMedClientError(f"NCBI {endpoint} failed after retries: {last_error}")

    async def esearch(
        self,
        query: str,
        *,
        retmax: int | None = None,
        mindate: str | None = None,
        maxdate: str | None = None,
        datetype: str = "pdat",
        extra_term: str | None = None,
    ) -> list[str]:
        """Search PubMed; return candidate PMIDs (capped by settings.retrieval_candidate_cap)."""
        cap = retmax if retmax is not None else self.settings.retrieval_candidate_cap
        term = query if not extra_term else f"({query}) AND ({extra_term})"
        params: dict[str, Any] = {
            "db": "pubmed",
            "term": term,
            "retmax": cap,
            "retmode": "json",
        }
        if mindate:
            params["mindate"] = mindate
            params["datetype"] = datetype
        if maxdate:
            params["maxdate"] = maxdate
            params["datetype"] = datetype

        text = await self._request("esearch.fcgi", params, expect_json=True)
        payload = json.loads(text)
        pmids = parse_esearch_pmids(payload)
        count = parse_esearch_count(payload)
        logger.info(
            "ncbi.esearch",
            extra={"query": query, "count": count, "returned": len(pmids)},
        )
        return pmids

    async def esummary(self, pmids: list[str]) -> list[PubMedDocument]:
        """Fetch document summaries for PMIDs (batched)."""
        if not pmids:
            return []
        from biolit.ingest.parsers import parse_esummary_json

        docs: list[PubMedDocument] = []
        for batch in _batches(pmids, 200):
            text = await self._request(
                "esummary.fcgi",
                {"db": "pubmed", "id": ",".join(batch), "retmode": "json"},
                expect_json=True,
            )
            docs.extend(parse_esummary_json(text))
        return docs

    async def efetch(self, pmids: list[str]) -> list[PubMedDocument]:
        """Fetch full MEDLINE XML and parse into PubMedDocument rows (batched)."""
        if not pmids:
            return []
        docs: list[PubMedDocument] = []
        for batch in _batches(pmids, 200):
            text = await self._request(
                "efetch.fcgi",
                {"db": "pubmed", "id": ",".join(batch), "retmode": "xml"},
                expect_json=False,
            )
            docs.extend(parse_efetch_xml(text))
        logger.info("ncbi.efetch", extra={"n_pmids": len(pmids), "n_docs": len(docs)})
        return docs

    async def search_documents(
        self,
        query: str,
        *,
        retmax: int | None = None,
        mindate: str | None = None,
        maxdate: str | None = None,
        mesh: list[str] | None = None,
        journal: str | None = None,
    ) -> list[PubMedDocument]:
        """esearch → efetch pipeline returning parsed documents."""
        extra_parts: list[str] = []
        if mesh:
            for term in mesh:
                extra_parts.append(f'"{term}"[MeSH Terms]')
        if journal:
            extra_parts.append(f'"{journal}"[Journal]')
        extra_term = " AND ".join(extra_parts) if extra_parts else None
        pmids = await self.esearch(
            query,
            retmax=retmax,
            mindate=mindate,
            maxdate=maxdate,
            extra_term=extra_term,
        )
        return await self.efetch(pmids)


def _batches(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]
