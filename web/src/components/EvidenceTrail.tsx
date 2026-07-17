import * as React from "react";
import type { EvidenceRef } from "@/lib/types";

/**
 * EvidenceTrail renders the provenance behind a hypothesis: stance chip,
 * verbatim snippet, and PMID linking to PubMed.
 */

export type Stance = "supports" | "contradicts" | "context";

export interface EvidenceTrailProps {
  evidence: EvidenceRef[];
  pubmedBaseUrl?: string;
  className?: string;
}

const STANCE_STYLE: Record<Stance, { label: string; fg: string; bg: string }> = {
  supports: { label: "supports", fg: "var(--success)", bg: "var(--success-bg)" },
  contradicts: {
    label: "contradicts",
    fg: "var(--danger)",
    bg: "var(--danger-bg)",
  },
  context: { label: "context", fg: "var(--ink-2)", bg: "var(--inset)" },
};

function asStance(s: string): Stance {
  if (s === "supports" || s === "contradicts" || s === "context") return s;
  return "context";
}

export function EvidenceTrail({
  evidence,
  pubmedBaseUrl = "https://pubmed.ncbi.nlm.nih.gov",
  className,
}: EvidenceTrailProps): React.ReactElement {
  if (evidence.length === 0) {
    return (
      <p
        className={className}
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 12,
          color: "var(--warning)",
          margin: 0,
        }}
      >
        no evidence — hypothesis withheld
      </p>
    );
  }

  return (
    <div
      className={className}
      style={{ display: "flex", flexDirection: "column", gap: 6 }}
    >
      {evidence.map((e, i) => {
        const s = STANCE_STYLE[asStance(e.stance)];
        return (
          <div
            key={`${e.pmid}-${i}`}
            style={{ display: "flex", gap: 8, alignItems: "flex-start" }}
          >
            <span
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                lineHeight: 1.6,
                color: s.fg,
                background: s.bg,
                padding: "1px 6px",
                borderRadius: 4,
                whiteSpace: "nowrap",
                flexShrink: 0,
              }}
            >
              {s.label}
            </span>
            <span style={{ fontSize: 13, lineHeight: 1.4, color: "var(--ink-2)" }}>
              {e.snippet}{" "}
              <a
                href={`${pubmedBaseUrl}/${e.pmid}/`}
                target="_blank"
                rel="noreferrer"
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: 11,
                  color: "var(--ink-3)",
                  textDecoration: "none",
                  whiteSpace: "nowrap",
                }}
              >
                PMID {e.pmid}
              </a>
            </span>
          </div>
        );
      })}
    </div>
  );
}

export default EvidenceTrail;
