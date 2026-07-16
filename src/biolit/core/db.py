"""Async SQLAlchemy engine, session factory, and ORM models matching the design data model."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import date, datetime
from typing import Any
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from biolit.core.config import get_settings
from biolit.core.logging import get_logger

logger = get_logger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


class Base(DeclarativeBase):
    """Declarative base for all BioLit tables."""


# ---------------------------------------------------------------------------
# Documents / chunks
# ---------------------------------------------------------------------------


class Document(Base):
    __tablename__ = "documents"

    pmid: Mapped[str] = mapped_column(String(32), primary_key=True)
    title: Mapped[str | None] = mapped_column(Text)
    abstract: Mapped[str | None] = mapped_column(Text)
    authors: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    journal: Mapped[str | None] = mapped_column(Text)
    pub_date: Mapped[date | None] = mapped_column(Date)
    mesh_terms: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    doi: Mapped[str | None] = mapped_column(String(128))
    source: Mapped[str | None] = mapped_column(String(64), default="pubmed")
    raw_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    chunks: Mapped[list[Chunk]] = relationship(back_populates="document")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    pmid: Mapped[str] = mapped_column(String(32), ForeignKey("documents.pmid", ondelete="CASCADE"))
    section: Mapped[str | None] = mapped_column(String(64))
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(768))

    document: Mapped[Document] = relationship(back_populates="chunks")


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class RetrievalQuery(Base):
    __tablename__ = "retrieval_queries"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    query: Mapped[str] = mapped_column(Text)
    filters_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    mode: Mapped[str] = mapped_column(String(32))
    top_k: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    hits: Mapped[list[RetrievalHit]] = relationship(back_populates="query_row")


class RetrievalHit(Base):
    __tablename__ = "retrieval_hits"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    query_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("retrieval_queries.id", ondelete="CASCADE")
    )
    pmid: Mapped[str] = mapped_column(String(32), ForeignKey("documents.pmid"))
    retriever: Mapped[str] = mapped_column(String(64))
    score: Mapped[float] = mapped_column(Float)
    rank: Mapped[int] = mapped_column(Integer)
    highlight: Mapped[str | None] = mapped_column(Text)

    query_row: Mapped[RetrievalQuery] = relationship(back_populates="hits")


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


class EvalDataset(Base):
    __tablename__ = "eval_datasets"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    name: Mapped[str] = mapped_column(String(128))
    split: Mapped[str] = mapped_column(String(64))
    n_items: Mapped[int] = mapped_column(Integer)
    license: Mapped[str | None] = mapped_column(Text)
    path: Mapped[str | None] = mapped_column(Text)

    runs: Mapped[list[EvalRun]] = relationship(back_populates="dataset")


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    model: Mapped[str] = mapped_column(String(256))
    dataset_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("eval_datasets.id"))
    mode: Mapped[str] = mapped_column(String(32))
    config_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metrics_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(32), default="pending")

    dataset: Mapped[EvalDataset] = relationship(back_populates="runs")
    items: Mapped[list[EvalItem]] = relationship(back_populates="run")


class EvalItem(Base):
    __tablename__ = "eval_items"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    run_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("eval_runs.id", ondelete="CASCADE")
    )
    question_id: Mapped[str] = mapped_column(String(128))
    prompt: Mapped[str] = mapped_column(Text)
    prediction: Mapped[str | None] = mapped_column(Text)
    gold: Mapped[str | None] = mapped_column(Text)
    correct: Mapped[bool | None] = mapped_column(Boolean)
    retrieved_pmids: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    latency_ms: Mapped[float | None] = mapped_column(Float)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    judge_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    run: Mapped[EvalRun] = relationship(back_populates="items")


# ---------------------------------------------------------------------------
# Hypothesis
# ---------------------------------------------------------------------------


class HypothesisRun(Base):
    __tablename__ = "hypothesis_runs"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    research_goal: Mapped[str] = mapped_column(Text)
    config_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    budget_used: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    hypotheses: Mapped[list[Hypothesis]] = relationship(back_populates="run")


class Hypothesis(Base):
    __tablename__ = "hypotheses"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    run_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("hypothesis_runs.id", ondelete="CASCADE")
    )
    statement: Mapped[str] = mapped_column(Text)
    rationale: Mapped[str | None] = mapped_column(Text)
    mechanism: Mapped[str | None] = mapped_column(Text)
    experiment: Mapped[str | None] = mapped_column(Text)
    falsification: Mapped[str | None] = mapped_column(Text)
    novelty: Mapped[float | None] = mapped_column(Float)
    feasibility: Mapped[float | None] = mapped_column(Float)
    elo: Mapped[float] = mapped_column(Float, default=1000.0)
    generation: Mapped[int] = mapped_column(Integer, default=0)
    parent_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False), ForeignKey("hypotheses.id"))
    cluster_id: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="active")

    run: Mapped[HypothesisRun] = relationship(back_populates="hypotheses")
    evidence_rows: Mapped[list[Evidence]] = relationship(back_populates="hypothesis")


class Evidence(Base):
    __tablename__ = "evidence"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    hypothesis_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("hypotheses.id", ondelete="CASCADE")
    )
    pmid: Mapped[str] = mapped_column(String(32), ForeignKey("documents.pmid"))
    snippet: Mapped[str] = mapped_column(Text)
    stance: Mapped[str] = mapped_column(String(32))  # supports | contradicts | context

    hypothesis: Mapped[Hypothesis] = relationship(back_populates="evidence_rows")


class TournamentMatch(Base):
    __tablename__ = "tournament_matches"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    run_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("hypothesis_runs.id", ondelete="CASCADE")
    )
    hyp_a: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("hypotheses.id"))
    hyp_b: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("hypotheses.id"))
    winner: Mapped[str | None] = mapped_column(UUID(as_uuid=False), ForeignKey("hypotheses.id"))
    judge_rationale: Mapped[str | None] = mapped_column(Text)
    round: Mapped[int] = mapped_column(Integer)


# ---------------------------------------------------------------------------
# Engine / session
# ---------------------------------------------------------------------------


def get_engine() -> AsyncEngine:
    """Return (and lazily create) the shared async engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            echo=settings.app_env == "development",
            pool_pre_ping=True,
        )
        logger.info("DB engine created")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the shared async session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            expire_on_commit=False,
            class_=AsyncSession,
        )
    return _session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI-friendly session dependency."""
    factory = get_session_factory()
    async with factory() as session:
        yield session


async def init_db() -> None:
    """Create pgvector extension and all tables. Idempotent."""
    from sqlalchemy import text

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database schema initialized")


async def dispose_engine() -> None:
    """Dispose the shared engine (tests / shutdown)."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None


async def ping_db() -> dict[str, Any]:
    """Health-check database connectivity."""
    from sqlalchemy import text

    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
