"""BioLit application settings loaded from environment / .env."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for BioLit. Nothing subsystem-specific is hard-coded elsewhere."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_name: str = "biolit"
    app_env: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # NCBI E-utilities
    ncbi_api_key: str | None = None
    ncbi_email: str = "biolit@example.com"
    ncbi_tool: str = "biolit"
    ncbi_base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    # Database
    database_url: str = Field(
        default="postgresql+psycopg://biolit:biolit@localhost:5432/biolit",
        description="Async SQLAlchemy URL (psycopg driver)",
    )

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600

    # LLM providers (LiteLLM reads these; listed here so .env.example is complete)
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    default_llm_model: str = "gpt-4o-mini"

    # Langfuse
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "http://localhost:3000"

    # Retrieval defaults
    retrieval_top_k: int = 20
    retrieval_candidate_cap: int = 500
    medcpt_device: str = "cpu"

    # Embedding dimension for MedCPT article encoder
    embedding_dim: int = 768


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
