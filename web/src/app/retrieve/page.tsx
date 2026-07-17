"use client";

import { useMemo, useState } from "react";
import { FilterChips } from "@/components/FilterChips";
import { ScoreBreakdown } from "@/components/ScoreBreakdown";
import { retrieve } from "@/lib/api";
import type { RetrievalMode, RetrieveResult } from "@/lib/types";

const MODES: { id: RetrievalMode; label: string }[] = [
  { id: "bm25", label: "bm25" },
  { id: "dense", label: "dense" },
  { id: "hybrid", label: "hybrid" },
];

function yearFromDate(d: string | null): string {
  if (!d) return "—";
  return d.slice(0, 4);
}

function HighlightTitle({ text }: { text: string }) {
  // highlight comes as markdown-ish **term** from API
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return (
    <span>
      {parts.map((p, i) => {
        if (p.startsWith("**") && p.endsWith("**")) {
          return (
            <mark key={i} className="bg-accent-soft text-ink">
              {p.slice(2, -2)}
            </mark>
          );
        }
        return <span key={i}>{p}</span>;
      })}
    </span>
  );
}

export default function RetrievePage() {
  const [query, setQuery] = useState("TREM2 microglial phagocytosis Alzheimer");
  const [mode, setMode] = useState<RetrievalMode>("hybrid");
  const [topK, setTopK] = useState(10);
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [mesh, setMesh] = useState("");
  const [journal, setJournal] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RetrieveResult | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);

  const activeFilters = useMemo(() => {
    const chips: string[] = [];
    if (dateFrom) chips.push(`from ${dateFrom}`);
    if (dateTo) chips.push(`to ${dateTo}`);
    if (mesh) chips.push(`mesh ${mesh}`);
    if (journal) chips.push(`journal ${journal}`);
    return chips;
  }, [dateFrom, dateTo, mesh, journal]);

  const onSearch = async () => {
    setLoading(true);
    setError(null);
    const t0 = performance.now();
    try {
      const filters: {
        date_from?: string;
        date_to?: string;
        mesh?: string[];
        journal?: string;
      } = {};
      if (dateFrom) filters.date_from = dateFrom;
      if (dateTo) filters.date_to = dateTo;
      if (mesh.trim()) filters.mesh = mesh.split(",").map((s) => s.trim()).filter(Boolean);
      if (journal.trim()) filters.journal = journal.trim();
      const res = await retrieve({
        query,
        mode,
        top_k: topK,
        candidate_cap: 40,
        filters: Object.keys(filters).length ? filters : undefined,
      });
      setResult(res);
      setLatencyMs(performance.now() - t0);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-5xl space-y-4">
      <div>
        <h1 className="text-lg font-medium text-ink">Retrieve</h1>
        <p className="text-sm text-ink-muted">
          Live PubMed hybrid search with lexical, dense, and rerank scores.
        </p>
      </div>

      <section className="rounded-card border border-border bg-card p-4 space-y-3">
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          rows={2}
          className="w-full resize-y rounded border border-border bg-inset px-3 py-2 text-sm text-ink outline-none focus:border-accent"
          placeholder="Query"
        />
        <div className="flex flex-wrap items-center gap-3">
          <FilterChips
            chips={MODES}
            selected={[mode]}
            multi={false}
            onToggle={(id) => setMode(id as RetrievalMode)}
          />
          <label className="flex items-center gap-1.5 text-xs text-ink-secondary">
            top_k
            <input
              type="number"
              min={1}
              max={50}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="w-16 rounded border border-border bg-inset px-1.5 py-1 font-mono text-xs"
            />
          </label>
          <button
            type="button"
            onClick={() => void onSearch()}
            disabled={loading || !query.trim()}
            className="rounded border border-accent bg-accent px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50 dark:text-page"
          >
            {loading ? "Searching…" : "Search"}
          </button>
        </div>
        <div className="grid gap-2 sm:grid-cols-4">
          <input
            type="date"
            value={dateFrom}
            onChange={(e) => setDateFrom(e.target.value)}
            className="rounded border border-border bg-inset px-2 py-1.5 text-xs"
            aria-label="Date from"
          />
          <input
            type="date"
            value={dateTo}
            onChange={(e) => setDateTo(e.target.value)}
            className="rounded border border-border bg-inset px-2 py-1.5 text-xs"
            aria-label="Date to"
          />
          <input
            value={mesh}
            onChange={(e) => setMesh(e.target.value)}
            placeholder="MeSH (comma-separated)"
            className="rounded border border-border bg-inset px-2 py-1.5 text-xs"
          />
          <input
            value={journal}
            onChange={(e) => setJournal(e.target.value)}
            placeholder="Journal"
            className="rounded border border-border bg-inset px-2 py-1.5 text-xs"
          />
        </div>
        {activeFilters.length ? (
          <p className="text-xs text-ink-muted">Filters: {activeFilters.join(" · ")}</p>
        ) : null}
      </section>

      {error ? (
        <div className="rounded-card border border-danger/40 bg-card px-3 py-2 text-sm text-danger">
          {error}
        </div>
      ) : null}

      {result ? (
        <>
          <div className="flex flex-wrap gap-x-4 gap-y-1 rounded-card border border-border bg-inset px-3 py-2 font-mono text-xs text-ink-secondary">
            <span>candidates ≤40</span>
            <span>mode {result.mode}</span>
            <span>fusion {result.mode === "hybrid" ? "rrf" : "—"}</span>
            <span>rerank {result.mode === "hybrid" ? "on" : "off"}</span>
            <span>returned {result.documents.length}</span>
            <span>latency {latencyMs != null ? `${Math.round(latencyMs)}ms` : "—"}</span>
            <span>cache via redis</span>
          </div>

          <ul className="space-y-2">
            {result.documents.map((doc) => (
              <li
                key={doc.pmid}
                className="rounded-card border border-border bg-card px-3 py-3"
              >
                <div className="mb-1 flex items-baseline justify-between gap-3">
                  <span className="font-mono text-xs text-ink-muted">#{doc.rank}</span>
                  <ScoreBreakdown scores={doc.scores} />
                </div>
                <h2 className="text-sm font-medium text-ink">
                  {doc.highlight ? (
                    <HighlightTitle text={doc.highlight} />
                  ) : (
                    doc.title || "(no title)"
                  )}
                </h2>
                <p className="mt-1 font-mono text-xs text-ink-muted">
                  {doc.journal || "—"} · {yearFromDate(doc.pub_date)} ·{" "}
                  <a
                    href={`https://pubmed.ncbi.nlm.nih.gov/${doc.pmid}/`}
                    target="_blank"
                    rel="noreferrer"
                    className="text-accent hover:underline"
                  >
                    PMID {doc.pmid}
                  </a>
                </p>
                {doc.abstract ? (
                  <p className="mt-2 line-clamp-3 text-xs text-ink-secondary">
                    {doc.abstract}
                  </p>
                ) : null}
              </li>
            ))}
          </ul>
        </>
      ) : null}
    </div>
  );
}
