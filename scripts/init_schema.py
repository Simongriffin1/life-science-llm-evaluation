"""Initialize BioLit database schema (pgvector + all tables)."""

from __future__ import annotations

import asyncio

from biolit.core.db import init_db
from biolit.core.logging import configure_logging, get_logger


async def main() -> None:
    configure_logging()
    logger = get_logger(__name__)
    await init_db()
    logger.info("Schema ready")


if __name__ == "__main__":
    asyncio.run(main())
