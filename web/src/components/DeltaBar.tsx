import * as React from "react";

/**
 * DeltaBar visualizes a model's closed-book score and its retrieval-augmented
 * score as a single bar: a neutral base up to the lower of the two, then a
 * colored segment for the difference. Positive delta (RAG helps) reads success;
 * negative delta (RAG hurts) reads danger. The right-hand readouts show the RAG
 * score and the signed delta in mono.
 */

export interface DeltaBarProps {
  label: string;
  closed: number;
  rag: number;
  max?: number;
  className?: string;
}

const clamp = (n: number, max: number): number => Math.max(0, Math.min(max, n));
const fmt = (n: number): string => n.toFixed(2);

export function DeltaBar({
  label,
  closed,
  rag,
  max = 1,
  className,
}: DeltaBarProps): React.ReactElement {
  const c = clamp(closed, max);
  const r = clamp(rag, max);
  const base = Math.min(c, r);
  const delta = r - c;
  const gain = delta >= 0;

  const basePct = (base / max) * 100;
  const deltaPct = (Math.abs(delta) / max) * 100;
  const sign = gain ? "+" : "\u2212";

  return (
    <div
      className={className}
      style={{ display: "flex", alignItems: "center", gap: 8 }}
    >
      <span
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          width: 104,
          color: "var(--ink-2)",
          whiteSpace: "nowrap",
          overflow: "hidden",
          textOverflow: "ellipsis",
          flexShrink: 0,
        }}
        title={label}
      >
        {label}
      </span>

      <div
        style={{
          flex: 1,
          height: 16,
          background: "var(--inset)",
          borderRadius: 4,
          position: "relative",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            position: "absolute",
            left: 0,
            top: 0,
            bottom: 0,
            width: `${basePct}%`,
            background: "var(--border-strong)",
          }}
        />
        <div
          style={{
            position: "absolute",
            left: `${basePct}%`,
            top: 0,
            bottom: 0,
            width: `${deltaPct}%`,
            background: gain ? "var(--success-fill)" : "var(--danger)",
          }}
        />
      </div>

      <span
        className="data"
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          width: 40,
          textAlign: "right",
          color: "var(--ink)",
          flexShrink: 0,
        }}
      >
        {fmt(r)}
      </span>
      <span
        className="data"
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          width: 52,
          textAlign: "right",
          color: gain ? "var(--success)" : "var(--danger)",
          flexShrink: 0,
        }}
      >
        {sign}
        {fmt(Math.abs(delta))}
      </span>
    </div>
  );
}

export default DeltaBar;
