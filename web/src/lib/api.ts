import type {
  HealthResponse,
  HypothesisRunResult,
  JobStatus,
  LeaderboardRow,
  PendingReview,
  RetrieveResult,
  RetrievalMode,
} from "./types";

const BASE = process.env.NEXT_PUBLIC_BIOLIT_API_BASE || "/backend";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
    cache: "no-store",
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

export async function getJob(jobId: string): Promise<JobStatus> {
  return request<JobStatus>(`/jobs/${jobId}`);
}

export async function retrieve(body: {
  query: string;
  mode: RetrievalMode;
  top_k: number;
  candidate_cap?: number;
  use_index?: boolean;
  filters?: {
    date_from?: string;
    date_to?: string;
    mesh?: string[];
    journal?: string;
  };
}): Promise<RetrieveResult> {
  return request<RetrieveResult>("/retrieve", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function enqueueEval(body: Record<string, unknown>): Promise<{
  status: string;
  job_id?: string | null;
  result?: unknown;
  estimate?: unknown;
  message?: string;
}> {
  return request("/eval", { method: "POST", body: JSON.stringify(body) });
}

export async function getLeaderboard(params?: {
  model?: string;
  dataset?: string;
  mode?: string;
}): Promise<{ rows: LeaderboardRow[]; n: number }> {
  const q = new URLSearchParams();
  if (params?.model) q.set("model", params.model);
  if (params?.dataset) q.set("dataset", params.dataset);
  if (params?.mode) q.set("mode", params.mode);
  const suffix = q.toString() ? `?${q}` : "";
  return request(`/eval/leaderboard${suffix}`);
}

export async function getEvalRun(
  runId: string,
  offset = 0,
  limit = 50,
): Promise<{
  run_id: string;
  model: string;
  mode: string;
  status: string;
  metrics: Record<string, unknown> | null;
  items: Array<Record<string, unknown>>;
  total: number;
}> {
  return request(`/eval/runs/${runId}?offset=${offset}&limit=${limit}`);
}

export function exportRunUrl(runId: string, format: "json" | "csv"): string {
  return `${BASE}/eval/runs/${runId}/export?format=${format}`;
}

export async function enqueueHypothesis(body: {
  research_goal: string;
  sync?: boolean;
  dry_run?: boolean;
  config?: Record<string, unknown>;
}): Promise<{
  status: string;
  job_id?: string | null;
  result?: HypothesisRunResult | null;
  estimate?: unknown;
  message?: string;
}> {
  return request("/hypothesize", { method: "POST", body: JSON.stringify(body) });
}

export async function getPending(runId: string): Promise<PendingReview> {
  return request(`/hypothesize/${runId}/pending`);
}

export async function postFeedback(
  runId: string,
  actions: Array<{ id: string; action: string; note?: string }>,
): Promise<unknown> {
  return request(`/hypothesize/${runId}/feedback`, {
    method: "POST",
    body: JSON.stringify({ actions }),
  });
}

export async function resumeHypothesis(
  runId: string,
  actions?: Array<{ id: string; action: string; note?: string }>,
): Promise<{ status: string; result?: HypothesisRunResult | null }> {
  return request(`/hypothesize/${runId}/resume`, {
    method: "POST",
    body: JSON.stringify(actions ? { actions } : {}),
  });
}
