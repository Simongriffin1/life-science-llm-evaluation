# BioLit

Biomedical literature **retrieval**, **LLM evaluation**, and **hypothesis generation** over a shared PubMed retrieval core.

> **Responsible use:** Hypotheses are unvalidated research leads, not findings. Keep a human in the loop. Provenance (PMID + snippet + stance) is mandatory. Do not treat outputs as experimental validation.

## What you get

| Surface | Endpoint / UI | Purpose |
|---|---|---|
| Retrieval | `POST /retrieve` | Hybrid PubMed search (BM25 ± MedCPT ± rerank) with highlights |
| Eval | `POST /eval`, `GET /eval/leaderboard` | Closed-book and RAG benchmarks (MIRAGE-family) |
| Hypothesis | `POST /hypothesize` | Literature-grounded proposals with Elo ranking + evidence trails |
| Dashboard | Streamlit | Thin UI over all three |

## Setup

```bash
# 1. Environment
cp .env.example .env
# Edit .env — set at least:
#   NCBI_EMAIL=you@example.com
#   NCBI_API_KEY=...          # https://www.ncbi.nlm.nih.gov/account/settings/
#   OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY / GOOGLE_API_KEY

# 2. Install
uv sync --extra dev --extra frontend
# Optional MedCPT dense/rerank:
uv sync --extra dev --extra frontend --extra retrieval

# 3. Infra (Docker Desktop must be running)
docker compose up -d
# wait until postgres + redis are healthy

# 4. Schema
uv run python scripts/init_schema.py

# 5. API
uv run uvicorn biolit.api.main:app --reload --host 0.0.0.0 --port 8000

# 6. Dashboard (separate terminal)
BIOLIT_API_URL=http://127.0.0.1:8000 uv run streamlit run frontend/app.py
```

Open the Streamlit URL, then exercise **Retrieve**, **Eval** (start with dry-run), and **Hypothesis** (dry-run first).

## Quick API checks

```bash
curl -s http://127.0.0.1:8000/health

curl -s -X POST http://127.0.0.1:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"TREM2 microglia","mode":"bm25","top_k":5,"candidate_cap":20}'

# Cost estimate only
curl -s -X POST http://127.0.0.1:8000/eval \
  -H 'Content-Type: application/json' \
  -d '{"models":["gpt-4o-mini"],"datasets":["pubmedqa"],"mode":["closed_book","rag"],"limit":20,"dry_run":true}'

curl -s -X POST http://127.0.0.1:8000/hypothesize \
  -H 'Content-Type: application/json' \
  -d '{"research_goal":"Modulate ferroptosis in ALS","dry_run":true,"config":{"n_seed":4}}'
```

## Scoped corpus ingest (optional)

```bash
uv run python scripts/ingest.py --mesh Microglia --from 2020-01-01 --retmax 50
# Then retrieve with use_index=true
```

## Quality checks

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
uv run pytest
```

## NCBI

Register at [NCBI](https://www.ncbi.nlm.nih.gov/account/), create an API key under account settings, and set `NCBI_TOOL`, `NCBI_EMAIL`, and `NCBI_API_KEY`. Honor rate limits; Redis caches E-utilities responses.

## Dataset licenses

Bundled `data/datasets/pubmedqa/sample.jsonl` is a **synthetic** 20-item slice for harness testing, not redistributable PubMedQA content.

Real PubMedQA, BioASQ, MedQA, MedMCQA, and MMLU-Med each carry their own terms. Record licenses in `eval_datasets` and check redistribution rights before shipping anything public. Place JSONL under `data/datasets/{name}/sample.jsonl` or `{split}.jsonl`.

## Stack

Python 3.11+, uv, FastAPI, Postgres+pgvector, Redis, arq, LiteLLM, LangGraph, Langfuse, Streamlit.
Retrievers: BM25 + MedCPT (optional) + RRF + cross-encoder rerank.
