"""BioLit — biomedical literature retrieval, eval, and hypothesis generation."""

__version__ = "0.1.0"


def main() -> None:
    """CLI entrypoint placeholder."""
    from biolit.core.logging import configure_logging, get_logger

    configure_logging()
    logger = get_logger(__name__)
    logger.info("biolit %s — use uvicorn biolit.api.main:app", __version__)
