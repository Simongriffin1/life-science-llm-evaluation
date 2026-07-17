type Stance = "supports" | "contradicts" | "context" | string;

const STYLES: Record<string, { color: string; bg: string }> = {
  supports: { color: "var(--success)", bg: "var(--success-bg)" },
  contradicts: { color: "var(--danger)", bg: "var(--danger-bg)" },
  context: { color: "var(--ink-2)", bg: "var(--inset)" },
};

export function StanceChip({ stance }: { stance: Stance }) {
  const s = STYLES[stance] || STYLES.context;
  return (
    <span
      className="inline-flex rounded px-1.5 py-0.5 font-mono text-xs font-medium"
      style={{ color: s.color, background: s.bg }}
    >
      {stance}
    </span>
  );
}
