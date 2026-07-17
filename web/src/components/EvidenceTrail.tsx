import type { EvidenceRef } from "@/lib/types";
import { StanceChip } from "./StanceChip";

type Props = {
  evidence: EvidenceRef[];
};

export function EvidenceTrail({ evidence }: Props) {
  if (!evidence.length) {
    return <p className="text-sm text-ink-muted">No evidence rows.</p>;
  }
  return (
    <ol className="space-y-3">
      {evidence.map((ev, i) => (
        <li
          key={`${ev.pmid}-${i}`}
          className="rounded-card border border-border bg-inset p-3"
        >
          <div className="mb-2 flex items-center gap-2">
            <StanceChip stance={ev.stance} />
            <a
              className="font-mono text-xs text-accent hover:underline"
              href={`https://pubmed.ncbi.nlm.nih.gov/${ev.pmid}/`}
              target="_blank"
              rel="noreferrer"
            >
              PMID {ev.pmid}
            </a>
          </div>
          <p className="text-sm text-ink-secondary">{ev.snippet}</p>
        </li>
      ))}
    </ol>
  );
}
