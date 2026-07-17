"""Streamlit MVP dashboard for BioLit (Retrieve / Eval / Hypothesis)."""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
import streamlit as st

API_URL = os.environ.get("BIOLIT_API_URL", "http://127.0.0.1:8000").rstrip("/")


def _client() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=300.0)


def _health() -> dict[str, Any]:
    with _client() as client:
        r = client.get("/health")
        r.raise_for_status()
        return r.json()


def _poll_job(job_id: str, *, label: str = "job") -> dict[str, Any]:
    """Poll GET /jobs/{id} until complete/failed or timeout."""
    placeholder = st.empty()
    deadline = time.time() + 600
    with _client() as client:
        while time.time() < deadline:
            r = client.get(f"/jobs/{job_id}")
            if r.status_code != 200:
                placeholder.error(r.text)
                return {"status": "failed", "message": r.text}
            body = r.json()
            status = body.get("status")
            prog = float(body.get("progress") or 0)
            placeholder.info(f"{label}: {status} ({prog:.0%}) — {body.get('message') or ''}")
            if status in {"complete", "failed"}:
                return body
            time.sleep(1.5)
    return {"status": "failed", "message": "timed out waiting for job"}


st.set_page_config(page_title="BioLit", layout="wide")
st.title("BioLit")
st.caption(
    "Biomedical retrieval · LLM evaluation · hypothesis generation. "
    "Hypotheses are unvalidated leads — keep a human in the loop; provenance is mandatory."
)

try:
    health = _health()
    st.sidebar.success(f"API {health.get('status', '?')} @ {API_URL}")
except Exception as exc:
    st.sidebar.error(f"API unreachable at {API_URL}: {exc}")
    st.stop()

tab_retrieve, tab_eval, tab_hyp = st.tabs(["Retrieve", "Eval", "Hypothesis"])

with tab_retrieve:
    st.subheader("PubMed retrieval")
    q = st.text_input("Query", value="TREM2 microglia phagocytosis")
    col1, col2, col3 = st.columns(3)
    with col1:
        mode = st.selectbox("Mode", ["bm25", "dense", "hybrid"], index=0)
    with col2:
        top_k = st.number_input("top_k", min_value=1, max_value=50, value=10)
    with col3:
        use_index = st.checkbox("Use persistent index", value=False)
    candidate_cap = st.slider("Candidate cap", 10, 200, 40)

    if st.button("Search", type="primary"):
        with st.spinner("Retrieving…"):
            with _client() as client:
                r = client.post(
                    "/retrieve",
                    json={
                        "query": q,
                        "mode": mode,
                        "top_k": int(top_k),
                        "use_index": use_index,
                        "candidate_cap": int(candidate_cap),
                    },
                )
            if r.status_code != 200:
                st.error(r.text)
            else:
                data = r.json()
                st.write(
                    f"Returned **{len(data.get('documents', []))}** docs "
                    f"(mode=`{data.get('mode')}`, use_index=`{data.get('use_index')}`)"
                )
                for doc in data.get("documents", []):
                    with st.expander(
                        f"#{doc['rank']} PMID {doc['pmid']} · score={doc['score']:.4f}"
                    ):
                        st.markdown(f"**{doc.get('title') or '(no title)'}**")
                        if doc.get("highlight"):
                            st.markdown(doc["highlight"])
                        st.json(doc.get("scores") or {})
                        if doc.get("abstract"):
                            st.write(doc["abstract"][:800])

with tab_eval:
    st.subheader("LLM evaluation")
    models = st.text_input("Models (comma-separated)", value="gpt-4o-mini")
    datasets = st.multiselect(
        "Datasets",
        ["pubmedqa", "bioasq", "medqa", "medmcqa", "mmlu_med"],
        default=["pubmedqa"],
    )
    modes = st.multiselect("Modes", ["closed_book", "rag"], default=["closed_book", "rag"])
    limit = st.number_input("Items per dataset", min_value=1, max_value=200, value=20)
    few_shot_k = st.number_input("few_shot_k", min_value=0, max_value=10, value=5)
    dry_run = st.checkbox("Dry-run cost estimate only", value=True)
    run_sync = st.checkbox("Run sync in API (dev)", value=False, key="eval_sync")

    if st.button("Run eval", type="primary"):
        model_list = [m.strip() for m in models.split(",") if m.strip()]
        payload = {
            "models": model_list,
            "datasets": datasets,
            "mode": modes,
            "limit": int(limit),
            "few_shot_k": int(few_shot_k),
            "sync": run_sync,
            "dry_run": dry_run,
            "retrieval": {"mode": "bm25", "top_k": 5, "candidate_cap": 30},
        }
        with st.spinner("Working…"), _client() as client:
            r = client.post("/eval", json=payload)
        if r.status_code != 200:
            st.error(r.text)
        else:
            body = r.json()
            if body.get("status") == "dry_run":
                st.info("Dry-run estimate")
                st.json(body.get("estimate"))
            elif body.get("job_id"):
                job = _poll_job(body["job_id"], label="eval")
                st.json(job)
            else:
                st.success(f"Completed runs: {body.get('result', {}).get('run_ids')}")
                st.dataframe(body.get("result", {}).get("leaderboard") or [])

    if st.button("Refresh leaderboard"):
        with _client() as client:
            r = client.get("/eval/leaderboard")
        if r.status_code == 200:
            st.dataframe(r.json().get("rows") or [])

with tab_hyp:
    st.subheader("Hypothesis generation")
    st.warning(
        "Outputs are unvalidated research leads, not findings. "
        "Every proposal must carry an evidence trail; review before acting."
    )
    goal = st.text_area(
        "Research goal",
        value=(
            "Identify a repurposable approved drug that modulates ferroptosis in ALS motor neurons"
        ),
        height=100,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        n_seed = st.number_input("n_seed", 2, 20, 4)
    with c2:
        evo = st.number_input("evolution_rounds", 0, 5, 1)
    with c3:
        max_prop = st.number_input("max_proposals", 1, 10, 3)
    model = st.text_input("Model", value="gpt-4o-mini")
    interactive = st.checkbox("Interactive review gate", value=False)
    hyp_dry = st.checkbox("Dry-run cost estimate only", value=True, key="hyp_dry")
    hyp_sync = st.checkbox("Run sync in API (dev)", value=False, key="hyp_sync")

    if st.button("Generate hypotheses", type="primary"):
        payload = {
            "research_goal": goal,
            "sync": hyp_sync,
            "dry_run": hyp_dry,
            "config": {
                "n_seed": int(n_seed),
                "evolution_rounds": int(evo),
                "max_proposals": int(max_prop),
                "model": model,
                "retrieval_mode": "bm25",
                "candidate_cap": 40,
                "interactive": interactive,
            },
        }
        with st.spinner("Running hypothesis engine…"), _client() as client:
            r = client.post("/hypothesize", json=payload)
        if r.status_code != 200:
            st.error(r.text)
        else:
            body = r.json()
            if body.get("status") == "dry_run":
                st.info("Dry-run estimate")
                st.json(body.get("estimate"))
            elif body.get("job_id"):
                job = _poll_job(body["job_id"], label="hypothesis")
                st.json(job)
                ref = (job.get("result_ref") or {}) if isinstance(job, dict) else {}
                if ref.get("run_id"):
                    st.session_state["hyp_run_id"] = ref["run_id"]
            else:
                result = body.get("result") or {}
                st.session_state["hyp_run_id"] = result.get("run_id")
                st.success(
                    f"Status {body.get('status')} · run {result.get('run_id')} · "
                    f"{len(result.get('proposals') or [])} proposals"
                )
                for i, p in enumerate(result.get("proposals") or [], start=1):
                    with st.expander(f"Proposal {i} · Elo {p.get('elo', 0):.1f}"):
                        st.markdown(f"**{p.get('statement')}**")
                        st.write(p.get("mechanism"))
                        st.write(p.get("experiment"))
                        st.write(p.get("falsification"))
                        st.json(p.get("evidence") or [])

    run_id = st.text_input("Run ID (for review)", value=st.session_state.get("hyp_run_id") or "")
    if run_id and st.button("Load pending review"):
        with _client() as client:
            r = client.get(f"/hypothesize/{run_id}/pending")
        if r.status_code != 200:
            st.error(r.text)
        else:
            pending = r.json()
            st.json(pending)
            hyps = pending.get("hypotheses") or []
            if hyps:
                reject_id = st.selectbox("Reject hypothesis", [h["id"] for h in hyps])
                redirect_note = st.text_input("Redirect note (optional)")
                if st.button("Submit feedback + resume"):
                    actions = [{"id": reject_id, "action": "reject"}]
                    if redirect_note.strip() and hyps:
                        other = next(h for h in hyps if h["id"] != reject_id)
                        actions.append(
                            {
                                "id": other["id"],
                                "action": "redirect",
                                "note": redirect_note.strip(),
                            }
                        )
                    with _client() as client:
                        fr = client.post(
                            f"/hypothesize/{run_id}/feedback",
                            json={"actions": actions},
                        )
                        rr = client.post(
                            f"/hypothesize/{run_id}/resume",
                            json={"actions": actions},
                        )
                    st.write(fr.json())
                    st.write(rr.json())
