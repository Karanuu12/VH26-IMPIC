/**
 * In-memory query trace, keyed by a job id the client generates.
 *
 * A chat request is one POST that can take several seconds, and until it
 * returns the UI has nothing to say beyond a spinner. That's the worst moment
 * to be silent: the user cannot tell a slow embedding call from a hung one,
 * and on a RAG product the interesting part -- which manual was searched, how
 * many passages came back, whether the fault index answered directly -- is
 * exactly what builds trust in the answer.
 *
 * So the route reports each stage as it actually completes and the client
 * polls for them. Every step recorded here is a real event with real numbers;
 * nothing is simulated or timed to look busy. A step appears only once the
 * work it names has finished.
 *
 * Same design as ingest progress and for the same reasons: deliberately
 * in-memory, deliberately per-process, pruned on write. A trace whose server
 * restarted is not in progress anymore.
 */
export interface QueryStep {
  /** Short imperative label, e.g. "Retrieved passages". */
  label: string;
  /** The measured specifics — counts, machine names, cache hits. */
  detail?: string;
  /** ms since the trace started, for the timing column. */
  ms: number;
}

export interface QueryTrace {
  steps: QueryStep[];
  done: boolean;
  startedAt: number;
  updatedAt: number;
}

declare global {
  // eslint-disable-next-line no-var
  var __faultfinderQueryTrace: Map<string, QueryTrace> | undefined;
}

function store(): Map<string, QueryTrace> {
  if (!globalThis.__faultfinderQueryTrace) globalThis.__faultfinderQueryTrace = new Map();
  return globalThis.__faultfinderQueryTrace;
}

export function startTrace(jobId: string | undefined): void {
  if (!jobId) return;
  const now = Date.now();
  store().set(jobId, { steps: [], done: false, startedAt: now, updatedAt: now });
  pruneStale();
}

export function traceStep(jobId: string | undefined, label: string, detail?: string): void {
  if (!jobId) return;
  const trace = store().get(jobId);
  if (!trace) return;
  const now = Date.now();
  trace.steps.push({ label, detail, ms: now - trace.startedAt });
  trace.updatedAt = now;
}

export function endTrace(jobId: string | undefined): void {
  if (!jobId) return;
  const trace = store().get(jobId);
  if (!trace) return;
  trace.done = true;
  trace.updatedAt = Date.now();
}

export function getTrace(jobId: string): QueryTrace | undefined {
  return store().get(jobId);
}

/** Traces are short-lived; 10 min is generous and keeps a dev server tidy. */
function pruneStale(): void {
  const cutoff = Date.now() - 10 * 60 * 1000;
  for (const [id, t] of store()) {
    if (t.updatedAt < cutoff) store().delete(id);
  }
}
