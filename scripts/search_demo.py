#!/usr/bin/env python3
"""Live PubMed BM25 demo. Skips if NCBI_API_KEY is not set."""

from __future__ import annotations

import argparse
import asyncio

from biolit.core.config import get_settings
from biolit.core.logging import configure_logging, get_logger
from biolit.ingest.pubmed_client import PubMedClient
from biolit.retrieval.bm25 import rank_bm25


async def run(query: str, top_k: int) -> int:
    configure_logging()
    logger = get_logger(__name__)
    settings = get_settings()

    if not settings.ncbi_api_key:
        logger.warning(
            "NCBI_API_KEY not set — skipping live search. "
            "Set it in .env to run this demo against PubMed."
        )
        return 0

    retmax = min(100, settings.retrieval_candidate_cap)
    async with PubMedClient() as client:
        docs = await client.search_documents(query, retmax=retmax)

    if not docs:
        logger.warning("No documents returned for query=%s", query)
        return 1

    hits = rank_bm25(query, docs, top_k=top_k)
    by_pmid = {d.pmid: d for d in docs}
    print(f"\nTop {len(hits)} BM25 results for: {query!r}\n")
    for hit in hits:
        doc = by_pmid[hit.pmid]
        title = (doc.title or "").replace("\n", " ")
        print(f"{hit.rank:2d}. PMID {hit.pmid}  score={hit.score:.4f}")
        print(f"    {title}")
        if doc.journal:
            print(f"    {doc.journal} ({doc.pub_date})")
        print()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="PubMed BM25 search demo")
    parser.add_argument("query", nargs="?", default="TREM2 microglia")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args.query, args.top_k)))


if __name__ == "__main__":
    main()
