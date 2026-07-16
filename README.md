# BioLit — Phase 0 scaffold

Biomedical literature **retrieval**, **LLM evaluation**, and **hypothesis generation** over a shared PubMed retrieval core.

> Hypotheses are unvalidated research leads. Keep a human in the loop; provenance is mandatory.

## Quick start (Phase 0)

```bash
# 1. Copy env and install
cp .env.example .env
uv sync --extra dev
# For MedCPT dense retrieval (Phase 2+): uv sync --extra dev --extra retrieval

# 2. Infra (start Docker Desktop first)
docker compose up -d
# wait until postgres + redis are healthy

# 3. Create schema (pgvector + tables)
uv run python scripts/init_schema.py

# 4. API
uv run uvicorn biolit.api.main:app --reload --host 0.0.0.0 --port 8000

# 5. Smoke check
curl -s http://localhost:8000/health
```

## Checks

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
uv run pytest
```

## Stack

Python 3.11+, uv, FastAPI, Postgres+pgvector, Redis, arq, LiteLLM, LangGraph, Langfuse.
Retrievers (later phases): BM25 + MedCPT + RRF + cross-encoder rerank.

## NCBI

Set `NCBI_EMAIL` and preferably `NCBI_API_KEY` in `.env`. Always honor E-utilities rate limits.

## Dataset licenses

PubMedQA, BioASQ, MedQA, MedMCQA, MMLU-Med each carry their own terms — recorded in `eval_datasets` when Phase 4 lands. Do not redistribute without checking rights.
