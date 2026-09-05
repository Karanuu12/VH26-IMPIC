/**
 * Retrieval layer: hybrid query expansion, cross-manual disambiguation,
 * score filtering, and re-ranking.
 */
import type { Chunk, ScoredHit } from "./types.ts";

/**
 * Build a refined search query from the raw user message.
 * Expands error-code patterns so the vector search catches both "E101" and
 * "error 101" / "101 overheating" etc.
 */
export function expandQuery(raw: string): string[] {
  const queries = [raw];
  const codeMatch = raw.match(/\b([A-Z]+)(\s?)(\d{3,4})\b/);
  if (codeMatch) {
    const [, prefix, , num] = codeMatch;
    queries.push(`${prefix} ${num} ${prefix}${num}`);
  }
  if (raw.trim().length > 3) {
    queries.push(raw.trim());
  }
  return [...new Set(queries)];
}

/**
 * Detect whether the query mentions a specific machine model.
 * Returns the model name or null.
 */
export const MACHINE_PATTERNS: { name: string; patterns: RegExp[] }[] = [
  { name: "RoboInject-300", patterns: [/\b(roboinject[- ]?300|ri[- ]?300|injection molding|injection mold)\b/i] },
  { name: "Press-2000", patterns: [/\b(press[- ]?2000|hydraulic press)\b/i] },
  { name: "Press-2001", patterns: [/\b(press[- ]?2001|mechanical press)\b/i] },
  { name: "PowerFlex-525", patterns: [/\b(powerflex[- ]?525|pf[- ]?525|ac drive|variable frequency drive)\b/i] },
];

export function detectMachineScope(query: string): string | undefined {
  for (const m of MACHINE_PATTERNS) {
    if (m.patterns.some((p) => p.test(query))) {
      return m.name;
    }
  }
  return undefined;
}

/**
 * Exact-match pre-retrieval: scan all available chunks for query terms.
 * Injects matching chunks with score 1.0 so they bypass the score threshold.
 * This fixes exact-code / parameter-code queries that the vector search misses.
 */
export function exactMatchHits(query: string, allChunks: ScoredHit[]): ScoredHit[] {
  const terms = query.toLowerCase().match(/\b[a-z0-9][a-z0-9.-]{1,}\b/g) ?? [];
  if (terms.length === 0) return [];

  const seen = new Set<string>();
  const matches: ScoredHit[] = [];

  for (const chunk of allChunks) {
    const lower = chunk.text.toLowerCase();
    // Only match if the term appears as a token (not just substring of a word)
    const matched = terms.some((t) => {
      const escaped = t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      return new RegExp(`\\b${escaped}\\b`, "i").test(lower);
    });
    if (matched) {
      const key = `${chunk.document_id}:${chunk.page}`;
      if (!seen.has(key)) {
        seen.add(key);
        matches.push({ ...chunk, score: 1.0 });
      }
    }
  }

  return matches;
}

/**
 * Deduplicate similar hits, preferring those with the highest score per chunk.
 */
export function dedupeHits(hits: ScoredHit[]): ScoredHit[] {
  const seen = new Set<string>();
  const unique: ScoredHit[] = [];
  for (const h of hits.sort((a, b) => b.score - a.score)) {
    const key = `${h.document_id}:${h.page}:${h.section}`;
    if (!seen.has(key)) {
      seen.add(key);
      unique.push(h);
    }
  }
  return unique;
}