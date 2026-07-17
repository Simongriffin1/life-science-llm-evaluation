type Props = {
  rank: number;
  elo: number;
  statement: string;
  generation: number;
  selected?: boolean;
  onClick?: () => void;
  parentId?: string | null;
};

export function EloRow({
  rank,
  elo,
  statement,
  generation,
  selected,
  onClick,
  parentId,
}: Props) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`w-full rounded-card border px-3 py-2.5 text-left transition-colors ${
        selected
          ? "border-accent bg-accent-soft"
          : "border-border bg-card hover:bg-inset"
      }`}
    >
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <span className="font-mono text-xs text-ink-muted">#{rank}</span>
        <span className="font-mono text-sm font-medium text-ink">
          {elo.toFixed(1)}
        </span>
      </div>
      <p className="text-sm text-ink">{statement}</p>
      <p className="mt-1 font-mono text-xs text-ink-muted">
        gen {generation}
        {parentId ? ` · parent ${parentId.slice(0, 8)}` : ""}
      </p>
    </button>
  );
}
