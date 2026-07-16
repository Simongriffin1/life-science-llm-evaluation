#!/usr/bin/env python3
"""Ingest a scoped PubMed corpus into the persistent pgvector index.

Example:
  uv run python scripts/ingest.py --mesh Microglia --from 2018-01-01 --retmax 50
"""

from __future__ import annotations

import argparse
import asyncio

from biolit.core.config import get_settings
from biolit.core.db import init_db
from biolit.core.logging import configure_logging, get_logger
from biolit.ingest.indexer import ingest_corpus, ingest_result_dict


async def run(args: argparse.Namespace) -> int:
    configure_logging()
    logger = get_logger(__name__)
    settings = get_settings()

    if not settings.ncbi_api_key and not args.allow_no_key:
        logger.warning(
            "NCBI_API_KEY not set — refusing live ingest. "
            "Pass --allow-no-key to proceed at the unauthenticated rate limit, "
            "or set NCBI_API_KEY in .env."
        )
        return 0

    await init_db()
    result = await ingest_corpus(
        mesh=args.mesh,
        date_from=args.date_from,
        date_to=args.date_to,
        journal=args.journal,
        query=args.query,
        retmax=args.retmax,
    )
    payload = ingest_result_dict(result)
    logger.info("ingest.complete", extra=payload)
    print(
        f"Ingested {result.n_documents} documents → {result.n_chunks} chunks "
        f"(mesh={args.mesh!r}, from={args.date_from!r})"
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a scoped PubMed corpus into pgvector")
    parser.add_argument("--mesh", type=str, default=None, help='MeSH term, e.g. "Microglia"')
    parser.add_argument(
        "--from",
        dest="date_from",
        type=str,
        default=None,
        help="Start date YYYY-MM-DD",
    )
    parser.add_argument(
        "--to",
        dest="date_to",
        type=str,
        default=None,
        help="End date YYYY-MM-DD",
    )
    parser.add_argument("--journal", type=str, default=None)
    parser.add_argument("--query", type=str, default=None, help="Optional free-text query")
    parser.add_argument("--retmax", type=int, default=200, help="Max PubMed hits to ingest")
    parser.add_argument(
        "--allow-no-key",
        action="store_true",
        help="Allow ingest without NCBI_API_KEY (slower, rate-limited)",
    )
    args = parser.parse_args()
    if not any([args.mesh, args.journal, args.query, args.date_from]):
        parser.error("Provide at least one of --mesh / --journal / --query / --from")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
