/**
 * In-memory vector store fallback for when Qdrant isn't configured.
 */
import type { Chunk, ScoredHit, VectorStore } from "./types.ts";

function cosine(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

export class MemoryVectorStore implements VectorStore {
  private points: { chunk: Chunk; vector: number[] }[] = [];

  async index(chunks: Chunk[], vectors: number[][]): Promise<void> {
    for (let i = 0; i < chunks.length; i++) {
      this.points.push({ chunk: chunks[i], vector: vectors[i] });
    }
  }

  async query(vector: number[], topK = 10): Promise<ScoredHit[]> {
    const scored = this.points
      .map((p) => ({ ...p.chunk, score: cosine(p.vector, vector) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
    return scored;
  }

  async deleteByDocument(documentId: string): Promise<void> {
    this.points = this.points.filter((p) => p.chunk.document_id !== documentId);
  }

  get count(): number {
    return this.points.length;
  }
}