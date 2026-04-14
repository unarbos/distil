import type { Teacher, SubnetConfig } from "./types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "https://api.arbos.life";

export const TEACHER: Teacher = {
  model: "Qwen/Qwen3.5-35B-A3B",
  totalParams: 35_000_000_000,
  activeParams: 3_000_000_000,
  vocabSize: 248_044,
  architecture: "qwen3_5_moe",
  maxStudentParams: 3_500_000_000,
};

export const SUBNET_CONFIG: SubnetConfig = {
  netuid: null,
  maxKlThreshold: 2.0,
  emaAlpha: 0.3,
  maxNewTokens: 512,
  maxPromptTokens: 1024,
  samplesPerEpoch: 12,
};

export interface NeuronData {
  uid: number;
  hotkey: string;
  coldkey: string;
  stake: number;
  trust: number;
  consensus: number;
  incentive: number;
  emission: number;
  dividends: number;
  is_validator: boolean;
}

export interface MetagraphResponse {
  netuid: number;
  block: number;
  n: number;
  neurons: NeuronData[];
  timestamp: number;
  error?: string;
}

export interface CommitmentsResponse {
  commitments: Record<
    string,
    { block: number; model?: string; revision?: string; raw?: string }
  >;
  count: number;
  error?: string;
}

export interface ScoresResponse {
  ema_scores: Record<string, number>;
  last_eval: {
    teacher: string;
    max_new_tokens: number;
    max_prompt_len: number;
    n_prompts: number;
    students: Record<
      string,
      {
        kl_global_avg?: number;
        kl_per_prompt?: PerPromptKL[];
        tokens_per_sec?: number | null;
        error?: string;
      }
    >;
  } | null;
  last_eval_time?: number;
  tempo_seconds?: number;
  disqualified?: Record<string, string>;
}

export interface PerPromptKL {
  kl_mean: number;
  kl_std: number;
  kl_max: number;
  kl_min: number;
  n_positions: number;
}

/** Merged miner entry for display */
export interface MinerEntry {
  uid: number;
  hotkey: string;
  model: string;
  revision: string;
  klScore: number | null;
  emaScore: number | null;
  incentive: number;
  emission: number;
  commitBlock: number;
  isWinner: boolean;
  isOnChainWinner: boolean;
  isDisqualified: boolean;
  dqReason: string | null;
  /** 95% confidence interval [lo, hi] */
  ci95: [number, number] | null;
  /** Standard error of mean KL */
  se: number | null;
  /** Per-prompt breakdown */
  perPrompt: PerPromptKL[];
  /** Total tokens evaluated */
  totalPositions: number;
}

export interface PriceResponse {
  alpha_price_tao: number;
  alpha_price_usd: number;
  tao_usd: number;
  alpha_in_pool: number;
  tao_in_pool: number;
  marketcap_tao: number;
  emission_pct: number;
  volume_tao: number;
  price_change_1h: number;
  price_change_24h: number;
  price_change_7d: number;
  miners_tao_per_day: number;
  block_number: number;
  name: string;
  symbol: string;
  timestamp: number;
  error?: string;
}

export interface TmcConfig {
  sse_price_url: string;
  sse_subnet_url: string;
  netuid: number;
}

export async function fetchTmcConfig(): Promise<TmcConfig | null> {
  return safeFetch(`${API_BASE}/api/tmc-config`);
}

async function safeFetch<T>(url: string, timeoutMs = 25000): Promise<T | null> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const res = await fetch(url, {
      next: { revalidate: 60 },
      signal: controller.signal,
    });
    clearTimeout(timer);
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export async function fetchMetagraph(): Promise<MetagraphResponse | null> {
  return safeFetch(`${API_BASE}/api/metagraph`);
}

export async function fetchCommitments(): Promise<CommitmentsResponse | null> {
  return safeFetch(`${API_BASE}/api/commitments`);
}

export async function fetchScores(): Promise<ScoresResponse | null> {
  return safeFetch(`${API_BASE}/api/scores`);
}

export async function fetchPrice(): Promise<PriceResponse | null> {
  return safeFetch(`${API_BASE}/api/price`);
}

export interface ScoreHistoryEntry {
  block: number;
  timestamp: number;
  scores: Record<string, number>;
  king_uid: number | null;
}

export async function fetchHistory(limit = 50): Promise<ScoreHistoryEntry[]> {
  const data = await safeFetch<ScoreHistoryEntry[]>(`${API_BASE}/api/history?limit=${limit}`);
  // If the API doesn't support ?limit, truncate client-side
  const entries = data ?? [];
  return entries.length > limit ? entries.slice(-limit) : entries;
}

export interface ModelInfo {
  model: string;
  author: string;
  description: string | null;
  tags: string[];
  downloads: number;
  likes: number;
  params_b: number | null;
  active_params_b: number | null;
  is_moe: boolean;
  num_experts: number | null;
  num_active_experts: number | null;
  license: string | null;
  pipeline_tag: string | null;
  base_model: string[];
  error?: string;
}

export async function fetchModelInfo(model: string): Promise<ModelInfo | null> {
  return safeFetch(`${API_BASE}/api/model-info/${model}`);
}

export async function fetchAllModelInfo(models: string[]): Promise<Record<string, ModelInfo>> {
  const results: Record<string, ModelInfo> = {};
  const unique = [...new Set(models)];
  const fetches = await Promise.allSettled(
    unique.map(async (m) => {
      const info = await fetchModelInfo(m);
      if (info && !info.error) results[m] = info;
    })
  );
  return results;
}

export interface H2hResult {
  uid?: number;
  model: string;
  kl: number;
  is_king: boolean;
  vs_king: string;
  prompts_scored?: number;
  prompts_total?: number;
  paired_prompts?: number;
  dethrone_eligible?: boolean;
  early_stopped?: boolean;
  t_test?: {
    p?: number;
    t?: number;
    n?: number;
    mean_delta?: number;
  };
}

export interface H2hLatestResponse {
  block: number;
  timestamp: number;
  king_uid: number;
  king_h2h_kl: number;
  king_global_kl: number;
  epsilon: number;
  epsilon_threshold: number;
  n_prompts: number;
  results: H2hResult[];
  king_changed: boolean;
  new_king_uid: number | null;
}

export interface H2hHistoryResponse {
  rounds: H2hLatestResponse[];
  total: number;
  page: number;
  limit: number;
  has_more: boolean;
}

export async function fetchH2hLatest(): Promise<H2hLatestResponse | null> {
  return safeFetch(`${API_BASE}/api/h2h-latest`);
}

export interface EvalProgress {
  active: boolean;
  phase?: string;
  current_student?: string;
  students_done?: number;
  students_total?: number;
  prompts_done?: number;
  prompts_total?: number;
  current_prompt?: number;
  current_kl?: number;
  current_se?: number;
  current_ci?: [number, number];
  current_best?: number;
  teacher_prompts_done?: number;
  models?: Record<string, string>;
  eval_order?: Array<{ uid: number; model: string; role: "king" | "challenger" }>;
  completed?: Array<{
    student_idx?: number;
    student_name: string;
    status: string;
    status_detail?: string;
    kl?: number;
    prompts_scored?: number;
    prompts_total?: number;
    scoring_time_s?: number;
  }>;
}

export async function fetchEvalProgress(): Promise<EvalProgress | null> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 5000);
    const res = await fetch(`${API_BASE}/api/eval-progress`, {
      cache: "no-store",
      signal: controller.signal,
    });
    clearTimeout(timer);
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

/** Build the unified miner list from chain + eval data */
export function buildMinerList(
  metagraph: MetagraphResponse | null,
  commitments: CommitmentsResponse | null,
  scores: ScoresResponse | null,
  kingUid?: number | null,
): MinerEntry[] {
  if (!metagraph || !commitments) return [];

  // Map hotkey → commitment
  const hotkeyCom = commitments.commitments;

  // Map model name → kl from last eval
  const modelKl: Record<string, number> = {};
  if (scores?.last_eval?.students) {
    for (const [name, s] of Object.entries(scores.last_eval.students)) {
      if (s.kl_global_avg != null) modelKl[name] = s.kl_global_avg;
    }
  }

  const miners: MinerEntry[] = [];
  for (const neuron of metagraph.neurons) {
    const com = hotkeyCom[neuron.hotkey];
    if (!com?.model) continue;

    const uid = neuron.uid;
    const emaStr = scores?.ema_scores?.[String(uid)];
    const ema = emaStr != null ? emaStr : null;
    // Use ema_scores as authoritative — includes disqualifications (duplicates, integrity failures)
    // Fall back to last_eval model KL only if no ema_score exists
    const rawKl = modelKl[com.model] ?? null;
    const kl = ema != null && ema > 2.0 ? null : (ema ?? rawKl);

    // Compute confidence interval from per-prompt data
    const studentData = scores?.last_eval?.students?.[com.model];
    const perPrompt: PerPromptKL[] = (studentData?.kl_per_prompt ?? []) as PerPromptKL[];
    const totalPositions = perPrompt.reduce((s, p) => s + (p.n_positions ?? 0), 0);

    let ci95: [number, number] | null = null;
    let se: number | null = null;
    if (kl != null && perPrompt.length > 1) {
      const means = perPrompt.map((p) => p.kl_mean);
      const n = means.length;
      const avg = means.reduce((a, b) => a + b, 0) / n;
      const variance = means.reduce((s, x) => s + (x - avg) ** 2, 0) / (n - 1);
      se = Math.sqrt(variance) / Math.sqrt(n);
      ci95 = [kl - 1.96 * se, kl + 1.96 * se];
    }

    // Check DQ by hotkey:block (per-commit), hotkey (legacy), or UID (legacy)
    const commitBlock = com.block;
    const dqReason = (commitBlock != null ? scores?.disqualified?.[`${neuron.hotkey}:${commitBlock}`] : null)
      ?? scores?.disqualified?.[String(uid)]
      ?? scores?.disqualified?.[neuron.hotkey]
      ?? null;
    const isDisqualified = dqReason != null || (ema != null && ema > 2.0);

    miners.push({
      uid,
      hotkey: neuron.hotkey,
      model: com.model,
      revision: com.revision ?? "main",
      klScore: kl,
      emaScore: ema,
      incentive: neuron.incentive,
      emission: neuron.emission,
      commitBlock: com.block,
      isWinner: false, // set below after sorting
      isOnChainWinner: neuron.incentive > 0.5,
      isDisqualified,
      dqReason,
      ci95,
      se,
      perPrompt,
      totalPositions,
    });
  }

  // Sort: DQ last, then by KL (lower = better), nulls before DQ
  miners.sort((a, b) => {
    if (a.isDisqualified && !b.isDisqualified) return 1;
    if (!a.isDisqualified && b.isDisqualified) return -1;
    if (a.klScore == null && b.klScore == null) return 0;
    if (a.klScore == null) return 1;
    if (b.klScore == null) return -1;
    return a.klScore - b.klScore;
  });

  // Winner = king from h2h_latest (authoritative), NOT lowest KL across different prompt sets
  if (kingUid != null) {
    const king = miners.find((m) => m.uid === kingUid);
    if (king) king.isWinner = true;
  }

  return miners;
}
