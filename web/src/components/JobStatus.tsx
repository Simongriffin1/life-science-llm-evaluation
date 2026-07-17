import type { JobStatus as JobStatusType } from "@/lib/types";

type Props = {
  job: JobStatusType | null;
  error?: string | null;
};

export function JobStatus({ job, error }: Props) {
  if (error) {
    return (
      <div className="rounded-card border border-danger/40 bg-card px-3 py-2 text-sm text-danger">
        {error}
      </div>
    );
  }
  if (!job) return null;
  const pct = Math.round((job.progress || 0) * 100);
  return (
    <div className="rounded-card border border-border bg-card px-3 py-2">
      <div className="flex items-baseline justify-between gap-2 text-sm">
        <span className="text-ink-secondary">Job</span>
        <span className="font-mono text-xs text-ink-muted">{job.job_id.slice(0, 8)}</span>
      </div>
      <div className="mt-1 flex items-baseline justify-between">
        <span className="font-medium capitalize text-ink">{job.status}</span>
        <span className="font-mono text-xs text-ink-secondary">{pct}%</span>
      </div>
      <div className="mt-2 h-1.5 w-full overflow-hidden rounded border border-border bg-inset">
        <div
          className="h-full bg-accent"
          style={{ width: `${pct}%` }}
        />
      </div>
      {job.message ? (
        <p className="mt-1 text-xs text-ink-muted">{job.message}</p>
      ) : null}
    </div>
  );
}
