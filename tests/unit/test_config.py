"""Unit tests for settings loading."""

from biolit.core.config import Settings, get_settings


def test_settings_defaults() -> None:
    get_settings.cache_clear()
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
    )
    assert settings.app_name == "biolit"
    assert settings.embedding_dim == 768
    assert settings.ncbi_tool == "biolit"
    get_settings.cache_clear()
