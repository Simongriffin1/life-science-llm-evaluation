"use client";

import { useEffect, useMemo, useState } from "react";
import { DeltaBar } from "@/components/DeltaBar";
import { FilterChips } from "@/components/FilterChips";
import { JobStatus } from "@/components/JobStatus";
import {
  enqueueEval,
  exportRunUrl,
  getEvalRun,
  getLeaderboard,
} from "@/lib/api";
import { useJob } from "@/lib/useJob";
import type { LeaderboardRow } from "@/lib/types";

const MODEL_CHIPS = [
  { id: "gpt-4o-mini", label: "gpt-4o-mini" },
  { id: "gpt-4o", label: "gpt-4o" },
];

const DATASETS = [
  { id: "pubmedqa", label: "pubmedqa" },
  { id: "medqa", label: "medqa" },
  { id: "medmcqa", label: "medmcqa" },
  { id: "mmlu_med", label: "mmlu_med" },
  { id: "bioasq", label: "bioasq" },
];

const MODES = [
  { id: "closed_book", label: "closed book" },
  { id: "rag", label: "rag" },
];

function toggle(list: string[], id: string, multi: boolean): string[] {
  if (!multi) return [id];
  return list.includes(id) ? list.filter((x) => x !== id) : [...list, id];
}

type ModelAgg = {
  model: string;
  closed: number | null;
  rag: number | null;
  delta: number | null;
  groundedness: number | null;
  tokens: number | null;
  runIdClosed?: string;
  runIdRag?: string;
};

export default function EvaluatePage() {
  const [models, setModels] = useState(["gpt-4o-mini"]);
  const [datasets, setDatasets] = useState(["medqa"]);
  const [modes, setModes] = useState(["closed_book", "rag"]);
  const [seed, setSeed] = useState(7);
  const [fewShotK, setFewShotK] = useState(5);
  const [limit, setLimit] = useState(10);
  const [dryRun, setDryRun] = useState(false);
  const [estimate, setEstimate] = useState<unknown>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [rows, setRows] = useState<LeaderboardRow[]>([]);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [items, setItems] = useState<Array<Record<string, unknown>>>([]);
  const [busy, setBusy] = useState(false);

  const { job, error: jobError, done } = useJob(jobId);

  const refreshBoard = async () => {
    const board = await getLeaderboard({ dataset: datasets[0] });
    setRows(board.rows);
  };

  useEffect(() => {
    void refreshBoard().catch(() => undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!done || !job || job.status === "failed") return;
    void refreshBoard().catch((err) =>
      setError(err instanceof Error ? err.message : String(err)),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [done, job?.status]);

  const aggs: ModelAgg[] = useMemo(() => {
    const byModel = new Map<string, ModelAgg>();
    for (const r of rows) {
      const cur = byModel.get(r.model) || {
        model: r.model,
        closed: null,
        rag: null,
        delta: null,
        groundedness: null,
        tokens: null,
      };
      if (r.mode === "closed_book") {
        cur.closed = r.accuracy;
        cur.runIdClosed = r.run_id;
      }
      if (r.mode === "rag") {
        cur.rag = r.accuracy;
        cur.delta = r.rag_vs_closed_book_delta ?? null;
        cur.groundedness = r.groundedness ?? null;
        cur.runIdRag = r.run_id;
      }
      const tokens = r.cost?.total_tokens;
      if (typeof tokens === "number") cur.tokens = (cur.tokens || 0) + tokens;
      byModel.set(r.model, cur);
    }
    return Array.from(byModel.values());
  }, [rows]);

  const launch = async () => {
    setBusy(true);
    setError(null);
    setEstimate(null);
    try {
      const res = await enqueueEval({
        models,
        datasets,
        mode: modes,
        seed,
        few_shot_k: fewShotK,
        limit,
        dry_run: dryRun,
        sync: false,
        retrieval: { mode: "bm25", top_k: 5, candidate_cap: 20 },
      });
      if (res.status === "dry_run") {
        setEstimate(res.estimate);
      } else if (res.job_id) {
        setJobId(res.job_id);
      } else {
        setError(res.message || "No job_id returned — is the arq worker running?");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const openRun = async (runId: string) => {
    setSelectedRun(runId);
    const detail = await getEvalRun(runId, 0, 50);
    setItems(detail.items);
  };

  return (
    <div className="mx-auto max-w-6xl space-y-4">
      <div>
        <h1 className="text-lg font-medium text-ink">Evaluate</h1>
        <p className="text-sm text-ink-muted">
          Launch MIRAGE-family runs, poll jobs, compare closed vs rag.
        </p>
      </div>

      <section className="space-y-3 rounded-card border border-border bg-card p-4">
        <div>
          <div className="mb-1 text-xs font-medium text-ink-muted">Models</div>
          <FilterChips
            chips={MODEL_CHIPS}
            selected={models}
            onToggle={(id) => setModels(toggle(models, id, true))}
          />
        </div>
        <div>
          <div className="mb-1 text-xs font-medium text-ink-muted">Dataset</div>
          <FilterChips
            chips={DATASETS}
            selected={datasets}
            multi={false}
            onToggle={(id) => setDatasets([id])}
          />
        </div>
        <div>
          <div className="mb-1 text-xs font-medium text-ink-muted">Modes</div>
          <FilterChips
            chips={MODES}
            selected={modes}
            onToggle={(id) => setModes(toggle(modes, id, true))}
          />
        </div>
        <div className="flex flex-wrap gap-3 text-xs text-ink-secondary">
          <label className="flex items-center gap-1">
            seed
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
              className="w-16 rounded border border-border bg-inset px-1.5 py-1 font-mono"
            />
          </label>
          <label className="flex items-center gap-1">
            few-shot k
            <input
              type="number"
              min={0}
              max={10}
              value={fewShotK}
              onChange={(e) => setFewShotK(Number(e.target.value))}
              className="w-14 rounded border border-border bg-inset px-1.5 py-1 font-mono"
            />
          </label>
          <label className="flex items-center gap-1">
            limit
            <input
              type="number"
              min={1}
              max={100}
              value={limit}
              onChange={(e) => setLimit(Number(e.target.value))}
              className="w-14 rounded border border-border bg-inset px-1.5 py-1 font-mono"
            />
          </label>
          <label className="flex items-center gap-1.5">
            <input
              type="checkbox"
              checked={dryRun}
              onChange={(e) => setDryRun(e.target.checked)}
            />
            dry-run estimate
          </label>
          <button
            type="button"
            disabled={busy || !models.length || !datasets.length || !modes.length}
            onClick={() => void launch()}
            className="rounded border border-accent bg-accent px-3 py-1.5 font-medium text-white disabled:opacity-50 dark:text-page"
          >
            {busy ? "Launching…" : dryRun ? "Estimate" : "Launch job"}
          </button>
        </div>
      </section>

      {error ? (
        <div className="rounded-card border border-danger/40 bg-card px-3 py-2 text-sm text-danger">
          {error}
        </div>
      ) : null}
      <JobStatus job={job} error={jobError} />
      {estimate ? (
        <pre className="overflow-auto rounded-card border border-border bg-inset p-3 font-mono text-xs">
          {JSON.stringify(estimate, null, 2)}
        </pre>
      ) : null}

      <section className="space-y-3">
        <h2 className="text-sm font-medium text-ink">Leaderboard</h2>
        <div className="overflow-x-auto rounded-card border border-border bg-card">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-border text-xs text-ink-muted">
              <tr>
                <th className="px-3 py-2 font-medium">Model</th>
                <th className="px-3 py-2 font-medium">Closed</th>
                <th className="px-3 py-2 font-medium">Rag</th>
                <th className="px-3 py-2 font-medium text-accent">Δrag</th>
                <th className="px-3 py-2 font-medium">Groundedness</th>
                <th className="px-3 py-2 font-medium">Tokens</th>
                <th className="px-3 py-2 font-medium">$</th>
              </tr>
            </thead>
            <tbody>
              {aggs.map((a) => (
                <tr
                  key={a.model}
                  className="cursor-pointer border-b border-border last:border-0 hover:bg-inset"
                  onClick={() => {
                    const id = a.runIdRag || a.runIdClosed;
                    if (id) void openRun(id);
                  }}
                >
                  <td className="px-3 py-2">{a.model}</td>
                  <td className="px-3 py-2 font-mono text-xs">
                    {a.closed == null ? "—" : a.closed.toFixed(2)}
                  </td>
                  <td className="px-3 py-2 font-mono text-xs">
                    {a.rag == null ? "—" : a.rag.toFixed(2)}
                  </td>
                  <td className="px-3 py-2 font-mono text-xs font-medium text-accent">
                    {a.delta == null
                      ? "—"
                      : `${a.delta >= 0 ? "+" : ""}${a.delta.toFixed(2)}`}
                  </td>
                  <td className="px-3 py-2 font-mono text-xs">
                    {a.groundedness == null ? "—" : a.groundedness.toFixed(2)}
                  </td>
                  <td className="px-3 py-2 font-mono text-xs">
                    {a.tokens == null ? "—" : a.tokens}
                  </td>
                  <td className="px-3 py-2 font-mono text-xs text-ink-muted">—</td>
                </tr>
              ))}
              {!aggs.length ? (
                <tr>
                  <td colSpan={7} className="px-3 py-4 text-ink-muted">
                    No completed runs yet.
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          {aggs
            .filter((a) => a.closed != null && a.rag != null)
            .map((a) => (
              <div
                key={`bar-${a.model}`}
                className="rounded-card border border-border bg-card p-3"
              >
                <DeltaBar
                  label={a.model}
                  closed={a.closed as number}
                  rag={a.rag as number}
                />
              </div>
            ))}
        </div>
      </section>

      {selectedRun ? (
        <section className="space-y-2 rounded-card border border-border bg-card p-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h2 className="text-sm font-medium text-ink">
              Drill-down{" "}
              <span className="font-mono text-xs text-ink-muted">{selectedRun}</span>
            </h2>
            <div className="flex gap-2">
              <a
                className="rounded border border-border px-2 py-1 text-xs font-medium hover:bg-inset"
                href={exportRunUrl(selectedRun, "json")}
              >
                Export json
              </a>
              <a
                className="rounded border border-border px-2 py-1 text-xs font-medium hover:bg-inset"
                href={exportRunUrl(selectedRun, "csv")}
              >
                Export csv
              </a>
            </div>
          </div>
          <ul className="max-h-80 space-y-2 overflow-auto">
            {items.map((it) => (
              <li
                key={String(it.question_id)}
                className="rounded border border-border bg-inset px-2 py-2 text-xs"
              >
                <div className="font-mono text-ink-muted">{String(it.question_id)}</div>
                <div className="text-ink-secondary">
                  pred{" "}
                  <span className="font-mono">{String(it.prediction ?? "—")}</span> · gold{" "}
                  <span className="font-mono">{String(it.gold ?? "—")}</span> ·{" "}
                  {it.correct === true
                    ? "correct"
                    : it.correct === false
                      ? "wrong"
                      : "unparsed"}
                </div>
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </div>
  );
}
