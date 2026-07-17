"use client";

import { useCallback, useEffect, useState } from "react";
import { getJob } from "./api";
import type { JobStatus } from "./types";

export function useJob(jobId: string | null, pollMs = 1500) {
  const [job, setJob] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!jobId) return;
    try {
      const next = await getJob(jobId);
      setJob(next);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [jobId]);

  useEffect(() => {
    if (!jobId) {
      setJob(null);
      return;
    }
    let cancelled = false;
    const tick = async () => {
      if (cancelled) return;
      await refresh();
    };
    void tick();
    const id = window.setInterval(() => {
      void tick();
    }, pollMs);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [jobId, pollMs, refresh]);

  const done =
    job?.status === "complete" ||
    job?.status === "failed" ||
    job?.status === "completed";

  return { job, error, done: Boolean(done), refresh };
}
