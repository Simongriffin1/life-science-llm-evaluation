# BioLit — Agent Instructions

## What this is
A biomedical research system with three subsystems over a shared retrieval core:
1. Retrieval — hybrid PubMed retrieval (BM25 + MedCPT dense + cross-encoder rerank, RRF fusion).
2. Eval — a reproducible harness benchmarking LLMs on MIRAGE-family datasets, closed-book and RAG.
3. Hypothesis — a LangGraph multi-agent PubMed-to-hypothesis engine with an Elo tournament and mandatory evidence provenance.

## Golden rules
- Retrieval is a shared library. Eval and Hypothesis import `biolit.retrieval.service`; they never re-implement PubMed access.
- All LLM calls go through `biolit.core.llm`. Never import a provider SDK (anthropic, openai, ...) anywhere else.
- Nothing hard-coded. Model names, top_k, budgets, paths, and dataset locations come from config (`biolit.core.config`) or per-run YAML.
- Every hypothesis persists `evidence` rows (pmid + snippet + stance). No provenance, no hypothesis.
- Long operations (full eval sweeps, hypothesis runs) run as arq jobs, not in the request thread.

## Stack (do not substitute without asking)
Python 3.11+, uv, FastAPI, Postgres + pgvector, Redis, arq, LiteLLM, LangGraph, Langfuse.
Retrievers: rank_bm25; MedCPT (ncbi/MedCPT-Query-Encoder, -Article-Encoder, -Cross-Encoder). Fusion: RRF.

## Conventions
- Type hints everywhere; Pydantic for all IO models and settings.
- `ruff` for lint+format, `mypy` for types, `pytest` for tests. All must pass before a phase is done.
- Async I/O for network calls (httpx, async db). CPU-bound model inference may be sync, run in a threadpool from async paths.
- Structured logging via `biolit.core.logging`. No bare `print`.
- Small, composable modules matching the repo tree in the design doc. No god-files.

## Definition of done for any phase
Code + tests + a runnable check (test, curl, or script) that demonstrates the phase works. ruff, mypy, pytest green.
