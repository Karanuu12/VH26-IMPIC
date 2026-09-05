/**
 * Disk-backed vector + keyword store.
 *
 * Why this exists: the old MemoryVectorStore was constructed inside the request
 * handler, so every request got a brand-new empty index and nothing could ever
 * be found. This one is a singleton that persists to a JSON file, so an ingest
 * from the browser is still there on the next query — and after a dev-server
 * restart.
 *
 * It is a stopgap by design. Qdrant Cloud replaces it for deployment (Vercel's
 * filesystem is read-only); the interface is deliberately the same shape so the
 * swap is a one-line change in the store factory.
 *
 * Hybrid search, per the Phase 7 design:
 *   - exact code index  → authoritative lookup for "F0001", "OCF"
 *   - lexical (token overlap) → rare terms, model numbers
 *   - dense (cosine)    → paraphrase and symptom language
 * fused with Reciprocal Rank Fusion so no signal can stomp the others.
 */
import { mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { writeFile, rename } from "node:fs/promises";
import { dirname } from "node:path";
import type { Chunk, FaultRecord } from "../doc/model.ts";
import { normalizeCode } from "../doc/model.ts";

export interface StoredDocument {
  documentId: string;
  title: string;
  machineId: string;
  model?: string;
  pageCount: number;
  chunkCount: number;
  faultCount: number;
  sha256: string;
  indexedAt: string;
}

interface Persisted {
  version: 2;
  documents: StoredDocument[];
  chunks: Chunk[];
  vectors: number[][];
  faults: FaultRecord[];
}

export interface ScoredChunk extends Chunk {
  score: number;
  /** Which retrievers found it — useful for debugging and for the UI. */
  matchedBy: ("code" | "lexical" | "dense")[];
}

const STOPWORDS = new Set([
  "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "in", "on", "for", "and", "or",
  "how", "what", "why", "when", "which", "do", "does", "i", "my", "it", "this", "that", "with",
  "can", "not", "you", "your", "at", "as", "by", "from", "if", "so",
]);

function tokenize(text: string): string[] {
  return (text.toLowerCase().match(/[a-z0-9][a-z0-9._-]*/g) ?? []).filter(
    (t) => t.length > 1 && !STOPWORDS.has(t),
  );
}

function cosine(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dot / denom;
}

export class LocalStore {
  private path: string;
  private documents: StoredDocument[] = [];
  private chunks: Chunk[] = [];
  private vectors: number[][] = [];
  private faults: FaultRecord[] = [];

  private saving = false;
  private pendingSave = false;

  /** codeNorm → chunk indices, for exact lookup that never depends on vector luck. */
  private codeIndex = new Map<string, Set<number>>();
  /** token → chunk indices. */
  private lexIndex = new Map<string, Set<number>>();

  constructor(path: string) {
    this.path = path;
    this.load();
  }

  // -------------------------------------------------------------------------
  // Persistence
  // -------------------------------------------------------------------------

  private load(): void {
    if (!existsSync(this.path)) return;
    try {
      const raw = JSON.parse(readFileSync(this.path, "utf8")) as Persisted;
      if (raw.version !== 2) return; // older layout — start clean rather than guess
      this.documents = raw.documents ?? [];
      this.chunks = raw.chunks ?? [];
      this.vectors = raw.vectors ?? [];
      this.faults = raw.faults ?? [];
      this.rebuildIndexes();
    } catch (err) {
      console.warn("[store] could not load index, starting empty:", String(err).slice(0, 120));
    }
  }

  /**
   * Serializing + writing the index is ~0.8s at 46MB, and writeFileSync
   * blocks Node's event loop for all of it -- which stalls every concurrent
   * chat query while an ingest is saving. Written async, via temp+rename so a
   * crash mid-write can't truncate the index, and coalesced so a burst of
   * saves does one write instead of N.
   */
  save(): void {
    this.pendingSave = true;
    if (this.saving) return;
    void this.flush();
  }

  private async flush(): Promise<void> {
    this.saving = true;
    try {
      while (this.pendingSave) {
        this.pendingSave = false;
        const payload: Persisted = {
          version: 2,
          documents: this.documents,
          chunks: this.chunks,
          vectors: this.vectors,
          faults: this.faults,
        };
        const json = JSON.stringify(payload);
        mkdirSync(dirname(this.path), { recursive: true });
        const tmp = `${this.path}.tmp`;
        await writeFile(tmp, json);
        await rename(tmp, this.path);
      }
    } catch (err) {
      console.error("[store] save failed:", String(err).slice(0, 200));
    } finally {
      this.saving = false;
    }
  }

  /** Synchronous save, for paths that must not return before it's durable. */
  saveSync(): void {
    mkdirSync(dirname(this.path), { recursive: true });
    const payload: Persisted = {
      version: 2,
      documents: this.documents,
      chunks: this.chunks,
      vectors: this.vectors,
      faults: this.faults,
    };
    writeFileSync(this.path, JSON.stringify(payload));
    this.pendingSave = false;
  }

  private rebuildIndexes(): void {
    this.codeIndex.clear();
    this.lexIndex.clear();
    this.chunks.forEach((c, i) => this.indexChunk(c, i));
  }

  private indexChunk(chunk: Chunk, i: number): void {
    for (const code of chunk.faultCodes) {
      const k = normalizeCode(code);
      if (!this.codeIndex.has(k)) this.codeIndex.set(k, new Set());
      this.codeIndex.get(k)!.add(i);
    }
    for (const tok of new Set(tokenize(chunk.text))) {
      if (!this.lexIndex.has(tok)) this.lexIndex.set(tok, new Set());
      this.lexIndex.get(tok)!.add(i);
    }
  }

  // -------------------------------------------------------------------------
  // Writing
  // -------------------------------------------------------------------------

  /** Remove a document's chunks and faults so re-ingesting doesn't duplicate. */
  deleteDocument(documentId: string): void {
    const keep: number[] = [];
    this.chunks.forEach((c, i) => {
      if (c.documentId !== documentId) keep.push(i);
    });
    if (keep.length !== this.chunks.length) {
      this.chunks = keep.map((i) => this.chunks[i]);
      this.vectors = keep.map((i) => this.vectors[i]);
      this.rebuildIndexes();
    }
    this.documents = this.documents.filter((d) => d.documentId !== documentId);
    this.faults = this.faults.filter((f) => f.provenance.documentId !== documentId);
  }

  addDocument(doc: StoredDocument, chunks: Chunk[], vectors: number[][], faults: FaultRecord[]): void {
    this.deleteDocument(doc.documentId);
    const base = this.chunks.length;
    this.chunks.push(...chunks);
    this.vectors.push(...vectors);
    this.faults.push(...faults);
    this.documents.push(doc);
    chunks.forEach((c, i) => this.indexChunk(c, base + i));
    this.save();
  }

  // -------------------------------------------------------------------------
  // Reading
  // -------------------------------------------------------------------------

  get stats() {
    return {
      documents: this.documents.length,
      chunks: this.chunks.length,
      faults: this.faults.length,
      dims: this.vectors[0]?.length ?? 0,
      machines: [...new Set(this.documents.map((d) => d.machineId))],
    };
  }

  listDocuments(): StoredDocument[] {
    return [...this.documents];
  }

  /** Every fault record for a code, across machines — this is the ambiguity signal. */
  faultsForCode(code: string): FaultRecord[] {
    const k = normalizeCode(code);
    return this.faults.filter((f) => f.codeNorm === k);
  }

  /**
   * Hybrid search with RRF fusion.
   * `machineId` filters BEFORE ranking rather than after, so a scoped query
   * cannot be crowded out by a different machine's chunks.
   */
  search(
    queryVector: number[],
    queryText: string,
    opts: { topK?: number; machineId?: string } = {},
  ): ScoredChunk[] {
    const topK = opts.topK ?? 8;
    const allowed = (i: number) => !opts.machineId || this.chunks[i].machineId === opts.machineId;
    if (this.chunks.length === 0) return [];

    const codes = (queryText.match(/\b([A-Za-z]{1,3}\d{2,5}|[A-Za-z]{2,4}F)\b/g) ?? []).map(normalizeCode);
    const codeHits: number[] = [];
    for (const c of codes) {
      for (const i of this.codeIndex.get(c) ?? []) if (allowed(i)) codeHits.push(i);
    }

    const tokens = tokenize(queryText);
    const lexScores = new Map<number, number>();
    for (const t of tokens) {
      const posting = this.lexIndex.get(t);
      if (!posting) continue;
      // Rare terms are worth more — a cheap IDF.
      const idf = Math.log(1 + this.chunks.length / posting.size);
      for (const i of posting) {
        if (allowed(i)) lexScores.set(i, (lexScores.get(i) ?? 0) + idf);
      }
    }
    const lexRanked = [...lexScores.entries()].sort((a, b) => b[1] - a[1]).map(([i]) => i);

    const denseRanked = this.chunks
      .map((_, i) => i)
      .filter(allowed)
      .map((i) => ({ i, s: cosine(this.vectors[i] ?? [], queryVector) }))
      .sort((a, b) => b.s - a.s)
      .map((x) => x.i);

    // Reciprocal Rank Fusion — no single retriever can dominate by inflating a score.
    const K = 60;
    const fused = new Map<number, { score: number; by: Set<"code" | "lexical" | "dense"> }>();
    const add = (list: number[], label: "code" | "lexical" | "dense", limit: number) => {
      list.slice(0, limit).forEach((idx, rank) => {
        const cur = fused.get(idx) ?? { score: 0, by: new Set<"code" | "lexical" | "dense">() };
        cur.score += 1 / (K + rank + 1);
        cur.by.add(label);
        fused.set(idx, cur);
      });
    };
    // Exact code matches get a rank-0 boost because they are lookups, not guesses.
    add([...new Set(codeHits)], "code", 20);
    add(lexRanked, "lexical", 50);
    add(denseRanked, "dense", 50);

    return [...fused.entries()]
      .sort((a, b) => b[1].score - a[1].score)
      .slice(0, topK)
      .map(([i, v]) => ({ ...this.chunks[i], score: v.score, matchedBy: [...v.by] }));
  }
}
