export type RetrievalMode = "bm25" | "dense" | "hybrid";

export type ScoredDocument = {
  pmid: string;
  title: string | null;
  abstract: string | null;
  authors: string[];
  journal: string | null;
  pub_date: string | null;
  mesh_terms: string[];
  doi: string | null;
  rank: number;
  score: number;
  scores: Record<string, number>;
  highlight: string | null;
};

export type RetrieveResult = {
  query_id: string | null;
  query: string;
  mode: RetrievalMode;
  top_k: number;
  use_index: boolean;
  documents: ScoredDocument[];
};

export type HealthResponse = {
  status: string;
  app: string;
  db: { ok: boolean; error?: string };
  redis: { ok: boolean; error?: string };
};

export type JobStatus = {
  job_id: string;
  status: "queued" | "running" | "complete" | "failed" | string;
  progress: number;
  message?: string | null;
  result_ref?: {
    run_ids?: string[];
    run_id?: string;
    status?: string;
  } | null;
};

export type LeaderboardRow = {
  run_id: string;
  model: string;
  dataset: string;
  mode: string;
  accuracy: number | null;
  macro_f1?: number | null;
  unparsed_rate?: number | null;
  groundedness?: number | null;
  cost?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  } | null;
  rag_vs_closed_book_delta?: number | null;
  metrics?: Record<string, unknown>;
};

export type EvidenceRef = {
  pmid: string;
  snippet: string;
  stance: "supports" | "contradicts" | "context" | string;
};

export type HypothesisDraft = {
  id: string;
  statement: string;
  rationale?: string | null;
  mechanism?: string | null;
  experiment?: string | null;
  falsification?: string | null;
  elo: number;
  generation: number;
  parent_id?: string | null;
  status: string;
  evidence: EvidenceRef[];
  unvalidated_lead?: boolean;
};

export type HypothesisRunResult = {
  run_id: string;
  status: string;
  research_goal: string;
  proposals: HypothesisDraft[];
  n_hypotheses: number;
  n_matches: number;
  budget_used: Record<string, unknown>;
  retrieved_pmids: string[];
};

export type PendingReview = {
  run_id: string;
  status: string;
  research_goal: string;
  tournament_round?: number;
  evolution_round?: number;
  hypotheses: Array<{
    id: string;
    statement: string;
    elo: number;
    generation: number;
    parent_id?: string | null;
    evidence: EvidenceRef[];
  }>;
};
