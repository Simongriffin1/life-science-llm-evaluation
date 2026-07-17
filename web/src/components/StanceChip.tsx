type Stance = "supports" | "contradicts" | "context" | string;

const STYLES: Record<string, string> = {
  supports: "border-success text-success",
  contradicts: "border-danger text-danger",
  context: "border-ink-muted text-ink-muted",
};

export function StanceChip({ stance }: { stance: Stance }) {
  const cls = STYLES[stance] || STYLES.context;
  return (
    <span
      className={`inline-flex rounded border px-1.5 py-0.5 text-xs font-medium capitalize ${cls}`}
    >
      {stance}
    </span>
  );
}
