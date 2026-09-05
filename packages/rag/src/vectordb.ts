/**
 * Qdrant vector store wrapper.
 * Uses @qdrant/js-client-rest to connect to Qdrant (cloud or local).
 */
import { QdrantClient } from "@qdrant/js-client-rest";
import type { Chunk, ScoredHit, VectorStore } from "./types.ts";

export interface VectorDbConfig {
  url: string;
  apiKey?: string;
  /** Qdrant collection name. */
  collection?: string;
  /** Embedding dimension to verify on first use. */
  dims?: number;
}

export class QdrantStore implements VectorStore {
  private client: QdrantClient;
  private collection: string;
  private dims: number;
  private ready = false;

  constructor(config: VectorDbConfig) {
    this.client = new QdrantClient({
      url: config.url,
      apiKey: config.apiKey,
      timeout: 30000,
      checkCompatibility: false,
    });
    this.collection = config.collection ?? "timmo_rag";
    this.dims = config.dims ?? 768;
  }

  /** Ensure collection exists (lazy) + ensure keyword index on document_id. */
  async ensure(): Promise<void> {
    if (this.ready) return;
    const exists = await this.client.collectionExists(this.collection);
    if (!exists.exists) {
      await this.client.createCollection(this.collection, {
        vectors: { size: this.dims, distance: "Cosine" },
      });
    }
    // Create keyword index on document_id for filtering (idempotent)
    try {
      await this.client.createPayloadIndex(this.collection, {
        field_name: "document_id",
        field_schema: "keyword" as any,
      });
    } catch { /* already exists — fine */ }
    this.ready = true;
  }

  /** Index chunks with their embeddings. */
  async index(chunks: Chunk[], vectors: number[][]): Promise<void> {
    if (!chunks.length) return;
    await this.ensure();
    const points = chunks.map((c, i) => ({
      id: c.id,
      vector: vectors[i] ?? new Array(this.dims).fill(0),
      payload: {
        document_id: c.document_id,
        title: c.title,
        page: c.page,
        section: c.section,
        text: c.text,
        char_count: c.char_count,
        images: c.images ?? [],
      },
    }));
    await this.client.upsert(this.collection, {
      wait: true,
      points,
    });
  }

  /** Query the collection; returns scored hits. */
  async query(vector: number[], topK = 10, filter?: Record<string, unknown>): Promise<ScoredHit[]> {
    await this.ensure();
    // Retry once on timeout (Qdrant free tier spins down after inactivity)
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const results = await this.client.query(this.collection, {
          query: vector,
          limit: topK,
          with_payload: true,
          filter: filter ?? {},
        });
        return results.points.map((p) => ({
          id: String(p.id),
          document_id: (p.payload as Record<string, unknown>).document_id as string ?? "",
          title: (p.payload as Record<string, unknown>).title as string ?? "",
          page: (p.payload as Record<string, unknown>).page as number ?? 0,
          section: (p.payload as Record<string, unknown>).section as string ?? "",
          text: (p.payload as Record<string, unknown>).text as string ?? "",
          char_count: (p.payload as Record<string, unknown>).char_count as number ?? 0,
          images: (p.payload as Record<string, unknown>).images as string[] ?? [],
          score: p.score ?? 0,
        }));
      } catch (err) {
        if (attempt === 0 && err instanceof Error && (err.message.includes("timeout") || err.message.includes("connect") || err.message.includes("econnrefused"))) {
          console.log("Qdrant retry on:", err.message.slice(0, 100));
          await new Promise((r) => setTimeout(r, 2000));
          continue;
        }
        throw err;
      }
    }
    return [];
  }

  /** Delete chunks for a specific document (re-ingestion). */
  async deleteByDocument(documentId: string): Promise<void> {
    await this.ensure();
    await this.client.delete(this.collection, {
      filter: {
        must: [{ key: "document_id", match: { value: documentId } }],
      },
    });
  }
}

export { QdrantClient };