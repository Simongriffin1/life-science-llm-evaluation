"use client";

import { useEffect, useMemo, useState } from "react";
import { EloRow } from "@/components/EloRow";
import { EvidenceTrail } from "@/components/EvidenceTrail";
import { JobStatus } from "@/components/JobStatus";
import {
  enqueueHypothesis,
  getPending,
  postFeedback,
  resumeHypothesis,
} from "@/lib/api";
import { useJob } from "@/lib/useJob";
import type { HypothesisDraft, PendingReview } from "@/lib/types";

export default function HypothesizePage() {
  const [goal, setGoal] = useState(
    "How does TREM2 modulation of microglial phagocytosis affect Alzheimer pathology?",
  );
  const [nSeed, setNSeed] = useState(4);
  const [tournamentRounds, setTournamentRounds] = useState(1);
  const [evolutionRounds, setEvolutionRounds] = useState(1);
  const [budget, setBudget] = useState(60000);
  const [interactive, setInteractive] = useState(true);
  const [jobId, setJobId] = useState<string | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [resultStatus, setResultStatus] = useState<string | null>(null);
  const [hypotheses, setHypotheses] = useState<HypothesisDraft[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [pending, setPending] = useState<PendingReview | null>(null);
  const [redirectNote, setRedirectNote] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const { job, error: jobError, done } = useJob(jobId);

  const selected = useMemo(
    () => hypotheses.find((h) => h.id === selectedId) || null,
    [hypotheses, selectedId],
  );

  const ranked = useMemo(
    () => [...hypotheses].sort((a, b) => b.elo - a.elo || a.id.localeCompare(b.id)),
    [hypotheses],
  );

  useEffect(() => {
    if (!done || !job) return;
    const ref = job.result_ref;
    const rid = ref?.run_id;
    if (!rid) return;
    setRunId(rid);
    setResultStatus(ref?.status || job.status);
    if (ref?.status === "paused_for_review") {
      void getPending(rid)
        .then((p) => {
          setPending(p);
          const drafts: HypothesisDraft[] = (p.hypotheses || []).map((h) => ({
            id: h.id,
            statement: h.statement,
            elo: h.elo,
            generation: h.generation,
            parent_id: h.parent_id,
            status: "active",
            evidence: h.evidence || [],
            unvalidated_lead: true,
          }));
          setHypotheses(drafts);
          if (drafts[0]) setSelectedId(drafts[0].id);
        })
        .catch((err) => setError(err instanceof Error ? err.message : String(err)));
    }
  }, [done, job]);

  const launch = async () => {
    setBusy(true);
    setError(null);
    setPending(null);
    setHypotheses([]);
    try {
      const res = await enqueueHypothesis({
        research_goal: goal,
        sync: false,
        config: {
          n_seed: nSeed,
          tournament_rounds: tournamentRounds,
          evolution_rounds: evolutionRounds,
          max_total_tokens: budget,
          interactive,
          retrieval_mode: "bm25",
          candidate_cap: 30,
          max_proposals: 3,
        },
      });
      if (res.job_id) {
        setJobId(res.job_id);
      } else if (res.result) {
        setRunId(res.result.run_id);
        setResultStatus(res.result.status);
        setHypotheses(res.result.proposals || []);
        if (res.result.proposals?.[0]) setSelectedId(res.result.proposals[0].id);
      } else {
        setError(res.message || "No job_id — start the arq worker");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const applyReview = async (action: "accept" | "reject" | "redirect") => {
    if (!runId || !selectedId) return;
    setBusy(true);
    setError(null);
    try {
      const actions: Array<{ id: string; action: "accept" | "reject" | "redirect"; note?: string }> =
        action === "redirect"
          ? [
              {
                id: selectedId,
                action: "redirect",
                note: redirectNote || "Steer toward clearer mechanism",
              },
            ]
          : [{ id: selectedId, action }];
      // Accept all others when accepting one; reject only selected
      if (action === "accept" && pending) {
        for (const h of pending.hypotheses) {
          if (h.id !== selectedId) {
            actions.push({ id: h.id, action: "accept" });
          }
        }
      }
      await postFeedback(runId, actions);
      const resumed = await resumeHypothesis(runId, actions);
      setResultStatus(resumed.status);
      if (resumed.result?.status === "paused_for_review") {
        const p = await getPending(runId);
        setPending(p);
        const drafts: HypothesisDraft[] = (p.hypotheses || []).map((h) => ({
          id: h.id,
          statement: h.statement,
          elo: h.elo,
          generation: h.generation,
          parent_id: h.parent_id,
          status: "active",
          evidence: h.evidence || [],
          unvalidated_lead: true,
        }));
        setHypotheses(drafts);
      } else if (resumed.result) {
        setPending(null);
        setHypotheses(resumed.result.proposals || []);
        if (resumed.result.proposals?.[0]) {
          setSelectedId(resumed.result.proposals[0].id);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const paused = resultStatus === "paused_for_review";

  return (
    <div className="mx-auto max-w-6xl space-y-4">
      <div>
        <h1 className="text-lg font-medium text-ink">Hypothesize</h1>
        <p className="text-sm text-ink-muted">
          Literature-grounded proposals with Elo ranking and mandatory evidence.
        </p>
      </div>

      <section className="space-y-3 rounded-card border border-border bg-card p-4">
        <textarea
          value={goal}
          onChange={(e) => setGoal(e.target.value)}
          rows={3}
          className="w-full rounded border border-border bg-inset px-3 py-2 text-sm outline-none focus:border-accent"
          placeholder="Research goal"
        />
        <div className="flex flex-wrap gap-3 text-xs text-ink-secondary">
          <label className="flex items-center gap-1">
            n_seed
            <input
              type="number"
              min={2}
              max={20}
              value={nSeed}
              onChange={(e) => setNSeed(Number(e.target.value))}
              className="w-14 rounded border border-border bg-inset px-1.5 py-1 font-mono"
            />
          </label>
          <label className="flex items-center gap-1">
            tournament
            <input
              type="number"
              min={1}
              max={5}
              value={tournamentRounds}
              onChange={(e) => setTournamentRounds(Number(e.target.value))}
              className="w-14 rounded border border-border bg-inset px-1.5 py-1 font-mono"
            />
          </label>
          <label className="flex items-center gap-1">
            evolution
            <input
              type="number"
              min={0}
              max={5}
              value={evolutionRounds}
              onChange={(e) => setEvolutionRounds(Number(e.target.value))}
              className="w-14 rounded border border-border bg-inset px-1.5 py-1 font-mono"
            />
          </label>
          <label className="flex items-center gap-1">
            token budget
            <input
              type="number"
              min={5000}
              step={1000}
              value={budget}
              onChange={(e) => setBudget(Number(e.target.value))}
              className="w-24 rounded border border-border bg-inset px-1.5 py-1 font-mono"
            />
          </label>
          <label className="flex items-center gap-1.5">
            <input
              type="checkbox"
              checked={interactive}
              onChange={(e) => setInteractive(e.target.checked)}
            />
            interactive review
          </label>
          <button
            type="button"
            disabled={busy || goal.trim().length < 3}
            onClick={() => void launch()}
            className="rounded border border-accent bg-accent px-3 py-1.5 font-medium text-white disabled:opacity-50 dark:text-page"
          >
            {busy ? "Working…" : "Launch"}
          </button>
        </div>
      </section>

      {error ? (
        <div className="rounded-card border border-danger/40 bg-card px-3 py-2 text-sm text-danger">
          {error}
        </div>
      ) : null}
      <JobStatus job={job} error={jobError} />

      {paused ? (
        <div className="rounded-card border border-warning/50 bg-card px-3 py-2 text-sm text-warning">
          Paused for review — accept, reject, or redirect the selected hypothesis, then resume.
        </div>
      ) : null}

      <div className="grid gap-4 lg:grid-cols-2">
        <section className="space-y-2">
          <h2 className="text-sm font-medium text-ink">Ranked by Elo</h2>
          <div className="space-y-2">
            {ranked.map((h, i) => (
              <EloRow
                key={h.id}
                rank={i + 1}
                elo={h.elo}
                statement={h.statement}
                generation={h.generation}
                parentId={h.parent_id}
                selected={h.id === selectedId}
                onClick={() => setSelectedId(h.id)}
              />
            ))}
            {!ranked.length ? (
              <p className="text-sm text-ink-muted">No hypotheses yet.</p>
            ) : null}
          </div>
        </section>

        <section className="space-y-3 rounded-card border border-border bg-card p-4">
          {selected ? (
            <>
              <div className="flex flex-wrap items-center gap-2">
                <span className="rounded border border-warning px-1.5 py-0.5 text-xs font-medium text-warning">
                  Unvalidated lead
                </span>
                <span className="font-mono text-xs text-ink-muted">
                  Elo {selected.elo.toFixed(1)}
                </span>
              </div>
              <h2 className="text-sm font-medium text-ink">{selected.statement}</h2>
              {selected.mechanism ? (
                <div>
                  <div className="text-xs font-medium text-ink-muted">Mechanism</div>
                  <p className="text-sm text-ink-secondary">{selected.mechanism}</p>
                </div>
              ) : null}
              {selected.experiment ? (
                <div>
                  <div className="text-xs font-medium text-ink-muted">Experiment</div>
                  <p className="text-sm text-ink-secondary">{selected.experiment}</p>
                </div>
              ) : null}
              {selected.falsification ? (
                <div>
                  <div className="text-xs font-medium text-ink-muted">Falsification</div>
                  <p className="text-sm text-ink-secondary">{selected.falsification}</p>
                </div>
              ) : null}
              <div>
                <div className="mb-2 text-xs font-medium text-ink-muted">Evidence trail</div>
                <EvidenceTrail evidence={selected.evidence || []} />
              </div>

              {paused ? (
                <div className="space-y-2 border-t border-border pt-3">
                  <div className="flex flex-wrap gap-2">
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() => void applyReview("accept")}
                      className="rounded border border-success px-2 py-1 text-xs font-medium text-success hover:bg-inset"
                    >
                      Accept
                    </button>
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() => void applyReview("reject")}
                      className="rounded border border-danger px-2 py-1 text-xs font-medium text-danger hover:bg-inset"
                    >
                      Reject
                    </button>
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() => void applyReview("redirect")}
                      className="rounded border border-warning px-2 py-1 text-xs font-medium text-warning hover:bg-inset"
                    >
                      Redirect
                    </button>
                  </div>
                  <input
                    value={redirectNote}
                    onChange={(e) => setRedirectNote(e.target.value)}
                    placeholder="Redirect note (steering constraint)"
                    className="w-full rounded border border-border bg-inset px-2 py-1.5 text-xs"
                  />
                </div>
              ) : null}
            </>
          ) : (
            <p className="text-sm text-ink-muted">Select a hypothesis to inspect evidence.</p>
          )}
        </section>
      </div>
    </div>
  );
}
