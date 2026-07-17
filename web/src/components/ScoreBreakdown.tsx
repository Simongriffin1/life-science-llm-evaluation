type Props = {
  scores: Record<string, number>;
};

function pick(scores: Record<string, number>, keys: string[]): number | null {
  for (const k of keys) {
    if (typeof scores[k] === "number") return scores[k];
  }
  return null;
}

export function ScoreBreakdown({ scores }: Props) {
  const lexical = pick(scores, ["bm25", "lexical", "L"]);
  const dense = pick(scores, ["dense", "D"]);
  const rerank = pick(scores, ["rerank", "R", "cross"]);

  const cell = (label: string, value: number | null) => (
    <span className="inline-flex items-baseline gap-1 font-mono text-xs text-ink-secondary">
      <span className="text-ink-muted">{label}</span>
      <span>{value == null ? "—" : value.toFixed(3)}</span>
    </span>
  );

  return (
    <div className="flex flex-wrap gap-3" title="Lexical / dense / rerank">
      {cell("L", lexical)}
      {cell("D", dense)}
      {cell("R", rerank)}
    </div>
  );
}
