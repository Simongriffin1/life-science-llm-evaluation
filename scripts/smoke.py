#!/usr/bin/env python3
"""
Phase 7a — live end-to-end smoke.

Fails closed: real NCBI + LLM keys required; no mocks.
Usage: uv run python scripts/smoke.py --max-usd 0.50
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass, field

from sqlalchemy import func, select


@dataclass
class StepResult:
    name: str
    ok: bool
    detail: str
    seconds: float
    tokens: int = 0
    usd: float = 0.0
    extras: list[str] = field(default_factory=list)


def _fmt_usd(v: float) -> str:
    return f"${v:.4f}"


def _report(steps: list[StepResult], total_usd: float) -> int:
    width = 72
    print("\n" + "=" * width)
    print("BIO LIT LIVE SMOKE REPORT")
    print("=" * width)
    failed = 0
    for s in steps:
        status = "PASS" if s.ok else "FAIL"
        if not s.ok:
            failed += 1
        line = (
            f"[{status}] {s.name:<22} {s.seconds:6.1f}s  "
            f"tokens={s.tokens:<6} spend={_fmt_usd(s.usd)}"
        )
        print(line)
        print(f"       {s.detail}")
        for extra in s.extras:
            print(f"       {extra}")
    print("-" * width)
    overall = "PASS" if failed == 0 else "FAIL"
    print(f"OVERALL: {overall}  steps_failed={failed}  total_est_spend={_fmt_usd(total_usd)}")
    print("=" * width + "\n")
    return 0 if failed == 0 else 1


def preflight() -> StepResult:
    t0 = time.perf_counter()
    from biolit.core.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    missing: list[str] = []
    if not settings.ncbi_api_key:
        missing.append("NCBI_API_KEY")
    # Settings default is a placeholder; require a real NCBI_EMAIL from .env/env.
    if not settings.ncbi_email or settings.ncbi_email == "biolit@example.com":
        missing.append("NCBI_EMAIL")

    llm_keys = {
        "OPENAI_API_KEY": settings.openai_api_key,
        "ANTHROPIC_API_KEY": settings.anthropic_api_key,
        "GOOGLE_API_KEY": settings.google_api_key,
    }
    if not any(llm_keys.values()):
        missing.append("OPENAI_API_KEY|ANTHROPIC_API_KEY|GOOGLE_API_KEY")

    if missing:
        return StepResult(
            name="1.preflight",
            ok=False,
            detail=(
                f"Missing or empty required keys: {', '.join(missing)}. "
                "Set them in .env — smoke does not fall back to mocks."
            ),
            seconds=time.perf_counter() - t0,
        )
    present_llm = [k for k, v in llm_keys.items() if v]
    return StepResult(
        name="1.preflight",
        ok=True,
        detail=f"NCBI + LLM keys present ({', '.join(present_llm)})",
        seconds=time.perf_counter() - t0,
    )


async def step_retrieve() -> StepResult:
    t0 = time.perf_counter()
    from biolit.core.logging import get_logger
    from biolit.retrieval.service import retrieve

    log = get_logger("smoke.retrieve")
    query = "TREM2 microglial phagocytosis Alzheimer"
    log.info(
        "Hybrid retrieve starting — MedCPT encoders download on first run "
        "(ncbi/MedCPT-Query-Encoder, Article-Encoder, Cross-Encoder); may take minutes"
    )
    print(
        "\n[retrieve] First hybrid run may download MedCPT encoders from Hugging Face; "
        "please wait...\n"
    )

    bm25 = await retrieve(
        query,
        mode="bm25",
        top_k=10,
        candidate_cap=40,
        persist=False,
        use_index=False,
    )
    hybrid = await retrieve(
        query,
        mode="hybrid",
        top_k=10,
        candidate_cap=40,
        persist=True,
        use_index=False,
    )
    elapsed = time.perf_counter() - t0

    if len(hybrid.documents) < 5:
        return StepResult(
            name="2.retrieve",
            ok=False,
            detail=f"Expected >=5 docs, got {len(hybrid.documents)}",
            seconds=elapsed,
        )

    bm25_order = [d.pmid for d in bm25.documents]
    hybrid_order = [d.pmid for d in hybrid.documents]
    if bm25_order == hybrid_order:
        return StepResult(
            name="2.retrieve",
            ok=False,
            detail=(
                f"Dense reranking did not change order vs bm25-only (both={hybrid_order[:5]}...)"
            ),
            seconds=elapsed,
        )

    top3 = hybrid.documents[:3]
    extras = []
    for d in top3:
        extras.append(
            f"#{d.rank} PMID {d.pmid} score={d.score:.4f} scores={d.scores} "
            f"| {(d.title or '')[:70]}"
        )
    has_rerank = any("rerank" in d.scores for d in hybrid.documents)
    if not has_rerank:
        return StepResult(
            name="2.retrieve",
            ok=False,
            detail="Hybrid results missing rerank scores",
            seconds=elapsed,
            extras=extras,
        )

    return StepResult(
        name="2.retrieve",
        ok=True,
        detail=(
            f"{len(hybrid.documents)} docs; order changed vs bm25 "
            f"(bm25[0]={bm25_order[0]}, hybrid[0]={hybrid_order[0]})"
        ),
        seconds=elapsed,
        extras=extras,
    )


async def step_eval_dry_run(max_usd: float, model: str) -> tuple[StepResult, float]:
    t0 = time.perf_counter()
    from biolit.core.costing import estimate_eval_cost

    estimate = estimate_eval_cost(
        models=[model],
        datasets=["pubmedqa"],
        modes=["closed_book", "rag"],
        limit=5,
    )
    elapsed = time.perf_counter() - t0
    ok = estimate.est_cost_usd <= max_usd
    return (
        StepResult(
            name="3.eval_dry_run",
            ok=ok,
            detail=(
                f"estimate={_fmt_usd(estimate.est_cost_usd)} "
                f"calls={estimate.n_llm_calls} tokens≈{estimate.est_total_tokens} "
                f"threshold={_fmt_usd(max_usd)}" + ("" if ok else " — EXCEEDS --max-usd")
            ),
            seconds=elapsed,
            tokens=estimate.est_total_tokens,
            usd=estimate.est_cost_usd,
        ),
        estimate.est_cost_usd,
    )


async def step_eval_live(model: str) -> StepResult:
    t0 = time.perf_counter()
    from biolit.core.costing import usd_for_usage
    from biolit.core.db import EvalItem, EvalRun, get_session_factory
    from biolit.core.llm import track_token_usage
    from biolit.eval.service import EvalConfig, execute_eval

    with track_token_usage() as usage:
        result = await execute_eval(
            EvalConfig(
                models=[model],
                datasets=["pubmedqa"],
                mode=["closed_book", "rag"],
                limit=5,
                retrieval={"mode": "bm25", "top_k": 5, "candidate_cap": 20},
                use_cache=False,
            )
        )
    elapsed = time.perf_counter() - t0
    spend = usd_for_usage(model, usage.prompt_tokens, usage.completion_tokens)

    if result.status != "completed" or len(result.run_ids) < 2:
        return StepResult(
            name="4.eval_live",
            ok=False,
            detail=f"Eval incomplete: status={result.status} run_ids={result.run_ids}",
            seconds=elapsed,
            tokens=usage.total_tokens,
            usd=spend,
        )

    factory = get_session_factory()
    async with factory() as session:
        n_items = await session.scalar(
            select(func.count()).select_from(EvalItem).where(EvalItem.run_id.in_(result.run_ids))
        )
        runs = (
            (await session.execute(select(EvalRun).where(EvalRun.id.in_(result.run_ids))))
            .scalars()
            .all()
        )

    if int(n_items or 0) < 10:
        return StepResult(
            name="4.eval_live",
            ok=False,
            detail=f"Expected >=10 eval_items (5×2), got {n_items}",
            seconds=elapsed,
            tokens=usage.total_tokens,
            usd=spend,
        )

    by_mode: dict[str, float | None] = {}
    for run in runs:
        metrics = run.metrics_json or {}
        by_mode[run.mode] = metrics.get("accuracy")

    cb = by_mode.get("closed_book")
    rag = by_mode.get("rag")
    if cb is None or rag is None:
        return StepResult(
            name="4.eval_live",
            ok=False,
            detail=f"Missing accuracy for modes: {by_mode}",
            seconds=elapsed,
            tokens=usage.total_tokens,
            usd=spend,
        )
    delta = float(rag) - float(cb)
    return StepResult(
        name="4.eval_live",
        ok=True,
        detail=(
            f"persisted {n_items} eval_items; "
            f"closed_book={cb:.2f} rag={rag:.2f} delta(rag-cb)={delta:+.2f}"
        ),
        seconds=elapsed,
        tokens=usage.total_tokens,
        usd=spend,
    )


async def step_hypothesize(model: str) -> StepResult:
    t0 = time.perf_counter()
    from biolit.core.costing import usd_for_usage
    from biolit.core.db import Evidence, Hypothesis, get_session_factory
    from biolit.hypothesis.models import HypothesisConfig
    from biolit.hypothesis.service import execute_hypothesis

    result = await execute_hypothesis(
        "How does TREM2 modulation of microglial phagocytosis affect Alzheimer pathology?",
        HypothesisConfig(
            n_seed=4,
            tournament_rounds=1,
            evolution_rounds=1,
            max_proposals=2,
            top_k_retrieve=8,
            retrieval_mode="bm25",
            candidate_cap=30,
            model=model,
            max_total_tokens=60_000,
        ),
    )
    elapsed = time.perf_counter() - t0
    budget = result.budget_used or {}
    tokens = int(budget.get("total_tokens") or 0)
    spend = usd_for_usage(
        model,
        int(budget.get("prompt_tokens") or 0),
        int(budget.get("completion_tokens") or 0),
    )

    allowed = set(result.retrieved_pmids)
    if not result.proposals:
        return StepResult(
            name="5.hypothesize",
            ok=False,
            detail="Expected >=1 proposal",
            seconds=elapsed,
            tokens=tokens,
            usd=spend,
        )

    problems: list[str] = []
    for p in result.proposals:
        if not p.unvalidated_lead:
            problems.append(f"{p.id}: missing unvalidated_lead=true")
        if not p.evidence:
            problems.append(f"{p.id}: no evidence rows")
            continue
        cited = p.cited_pmids()
        illegal = sorted(cited - allowed)
        if illegal:
            problems.append(f"{p.id}: cites non-retrieved PMIDs {illegal}")
        if not cited.issubset(allowed):
            problems.append(f"{p.id}: cite-only subset check failed")

    # DB evidence must also be cite-only
    factory = get_session_factory()
    async with factory() as session:
        hyps = (
            (await session.execute(select(Hypothesis).where(Hypothesis.run_id == result.run_id)))
            .scalars()
            .all()
        )
        for h in hyps:
            if h.status != "proposal":
                continue
            rows = (
                (await session.execute(select(Evidence).where(Evidence.hypothesis_id == h.id)))
                .scalars()
                .all()
            )
            if not rows:
                problems.append(f"db {h.id}: no evidence rows")
            bad = [e.pmid for e in rows if e.pmid not in allowed]
            if bad:
                problems.append(f"db {h.id}: evidence PMIDs outside retrieval {bad}")

    if problems:
        return StepResult(
            name="5.hypothesize",
            ok=False,
            detail="; ".join(problems),
            seconds=elapsed,
            tokens=tokens,
            usd=spend,
        )

    extras = [
        f"retrieved_pmids={len(allowed)} proposals={len(result.proposals)} "
        f"llm_calls={budget.get('llm_calls')}"
    ]
    for p in result.proposals:
        extras.append(
            f"proposal cites={sorted(p.cited_pmids())} "
            f"unvalidated_lead={p.unvalidated_lead} | {p.statement[:80]}"
        )
    return StepResult(
        name="5.hypothesize",
        ok=True,
        detail=(
            f"{len(result.proposals)} proposal(s); cite-only holds "
            f"(all evidence PMIDs ⊆ retrieved set of {len(allowed)})"
        ),
        seconds=elapsed,
        tokens=tokens,
        usd=spend,
        extras=extras,
    )


async def async_main(max_usd: float) -> int:
    from biolit.core.config import get_settings
    from biolit.core.db import init_db
    from biolit.core.logging import configure_logging

    configure_logging()
    get_settings.cache_clear()
    settings = get_settings()
    model = settings.default_llm_model

    steps: list[StepResult] = []
    total_usd = 0.0

    # 1. Preflight
    pf = preflight()
    steps.append(pf)
    if not pf.ok:
        return _report(steps, total_usd)

    await init_db()

    # 2. Retrieve
    steps.append(await step_retrieve())
    if not steps[-1].ok:
        return _report(steps, total_usd)

    # 3. Eval dry-run (estimate only — not added to actual spend)
    dry, _dry_usd = await step_eval_dry_run(max_usd, model)
    steps.append(dry)
    if not dry.ok:
        return _report(steps, total_usd)

    # 4. Eval live
    ev = await step_eval_live(model)
    steps.append(ev)
    total_usd += ev.usd
    if not ev.ok:
        return _report(steps, total_usd)

    # 5. Hypothesize
    hyp = await step_hypothesize(model)
    steps.append(hyp)
    total_usd += hyp.usd

    if total_usd >= 1.0:
        steps.append(
            StepResult(
                name="6.spend_cap",
                ok=False,
                detail=f"Actual LLM spend {_fmt_usd(total_usd)} exceeds soft $1.00 acceptance cap",
                seconds=0.0,
                usd=total_usd,
            )
        )

    return _report(steps, total_usd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BioLit live end-to-end smoke (fails closed)")
    parser.add_argument(
        "--max-usd",
        type=float,
        default=0.50,
        help="Hard-fail eval dry-run if estimate exceeds this USD (default 0.50)",
    )
    args = parser.parse_args(argv)
    try:
        return asyncio.run(async_main(args.max_usd))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:  # noqa: BLE001 — top-level smoke report
        print(f"\nSMOKE ABORTED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
