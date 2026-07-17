type Props = {
  closed: number;
  rag: number;
  label?: string;
};

/** Closed-book base + retrieval-gain (or loss) segment. */
export function DeltaBar({ closed, rag, label }: Props) {
  const closedPct = Math.max(0, Math.min(100, closed * 100));
  const delta = rag - closed;
  const gainPct = Math.max(0, Math.min(100 - closedPct, delta * 100));
  const lossPct = Math.max(0, Math.min(closedPct, -delta * 100));

  return (
    <div className="space-y-1">
      {label ? <div className="text-xs text-ink-muted">{label}</div> : null}
      <div className="flex h-2 w-full overflow-hidden rounded border border-border bg-inset">
        <div
          className="h-full bg-ink-muted/40"
          style={{ width: `${closedPct - lossPct}%` }}
          title={`Closed ${closed.toFixed(2)}`}
        />
        {delta >= 0 ? (
          <div
            className="h-full bg-success"
            style={{ width: `${gainPct}%` }}
            title={`Δrag +${delta.toFixed(2)}`}
          />
        ) : (
          <div
            className="h-full bg-danger"
            style={{ width: `${lossPct}%` }}
            title={`Δrag ${delta.toFixed(2)}`}
          />
        )}
      </div>
      <div className="flex justify-between font-mono text-xs text-ink-secondary">
        <span>cb {closed.toFixed(2)}</span>
        <span className={delta >= 0 ? "text-success" : "text-danger"}>
          Δ {delta >= 0 ? "+" : ""}
          {delta.toFixed(2)}
        </span>
        <span>rag {rag.toFixed(2)}</span>
      </div>
    </div>
  );
}
