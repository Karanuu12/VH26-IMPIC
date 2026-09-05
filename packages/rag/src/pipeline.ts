import type {
  ChatRequest,
  ChatResult,
  Chunk,
  ScoredHit,
  StreamEvent,
  VectorStore,
  CitedAnswer,
} from "./types.ts";
import { OllamaEmbeddingClient } from "./embeddings.ts";
import {
  expandQuery,
  detectMachineScope,
  dedupeHits,
  exactMatchHits,
} from "./retrieval.ts";
import type { GroqClient } from "./llm.ts";

export interface PipelineConfig {
  embedder: OllamaEmbeddingClient;
  vectorStore: VectorStore;
  llm: GroqClient;
}

export class RagPipeline {
  private embedder: OllamaEmbeddingClient;
  private vectorStore: VectorStore;
  private llm: GroqClient;

  constructor(config: PipelineConfig) {
    this.embedder = config.embedder;
    this.vectorStore = config.vectorStore;
    this.llm = config.llm;
  }

  async index(chunks: Chunk[]): Promise<void> {
    const texts = chunks.map((c) => c.text);
    const vectors = await this.embedder.embedMany(texts);
    await this.vectorStore.index(chunks, vectors);
  }

  private async retrieve(queryVector: number[], topK: number): Promise<ScoredHit[]> {
    return this.vectorStore.query(queryVector, topK);
  }

  private refineHits(
    allVecHits: ScoredHit[],
    query: string,
    machineScope?: string,
  ): { pool: ScoredHit[] } {
    const deduped = dedupeHits(allVecHits);
    const exactHits = exactMatchHits(query, deduped);
    const merged = [...exactHits, ...deduped];
    const seen = new Set<string>();
    const pool: ScoredHit[] = [];
    for (const h of merged) {
      const key = `${h.document_id}:${h.page}:${h.section}`;
      if (!seen.has(key)) {
        seen.add(key);
        pool.push(h);
      }
    }
    const scoped = machineScope
      ? pool.filter((h) =>
          h.document_id.toLowerCase().replace(/-/g, "").includes(machineScope.toLowerCase().replace(/-/g, "")),
        )
      : pool;
    return { pool: scoped.length ? scoped : pool };
  }

  async query(req: ChatRequest): Promise<ChatResult> {
    const queries = expandQuery(req.message);
    const machineScope = req.machine ?? detectMachineScope(req.message);
    const topK = Math.max(req.top_k ?? 50, 100);

    const queryVector = await this.embedder.embedQuery(queries.join("\n"));
    const allHits = await this.retrieve(queryVector, topK);
    const { pool } = this.refineHits(allHits, req.message, machineScope);
    console.log("[QUERY]", req.message, "-> scope:", machineScope, "pool:", pool.length, "docs:", [...new Set(pool.map(h => h.document_id))]);

    const accepted = pool.slice(0, 8);

    if (accepted.length === 0) {
      return {
        answer: {
          meaning: "No matching content found in the loaded manuals.",
          probable_causes: [],
          corrective_action: [],
          citations: [],
          confidence: "low",
          refusals: ["No relevant content was found."],
        },
        sources: [],
      };
    }

    let answer: CitedAnswer;
    try {
      answer = await this.llm.generateAnswer(
        req.message,
        accepted,
        req.history ?? [],
        machineScope,
      );
    } catch (err) {
      // Fallback: build answer from retrieved chunks directly (no LLM)
      console.log("[FALLBACK] LLM failed, using retrieval-only answer:", String(err).slice(0, 100));
      answer = buildFallbackAnswer(req.message, accepted);
    }

    const sourceImages = [...new Set(accepted.flatMap((s) => s.images ?? []))].filter(Boolean).slice(0, 6);

    return {
      answer: { ...answer, images: sourceImages },
      sources: accepted,
    };
  }

  async *stream(req: ChatRequest): AsyncGenerator<StreamEvent> {
    const queries = expandQuery(req.message);
    const machineScope = req.machine ?? detectMachineScope(req.message);
    const topK = Math.max(req.top_k ?? 50, 100);

    const queryVector = await this.embedder.embedQuery(queries.join("\n"));
    const allHits = await this.retrieve(queryVector, topK);
    const { pool } = this.refineHits(allHits, req.message, machineScope);

    yield { type: "context", document: machineScope ?? "any", chunks: pool };

    const accepted = pool.slice(0, 8);

    if (accepted.length === 0) {
      yield {
        type: "answer",
        answer: {
          meaning: "No matching content found.",
          probable_causes: [],
          corrective_action: [],
          citations: [],
          confidence: "low",
          refusals: ["No relevant content was found."],
        },
      };
      yield { type: "done" };
      return;
    }

    yield { type: "sources", sources: accepted };

    const answer = await this.llm.generateAnswer(
      req.message, accepted, req.history ?? [], machineScope,
    ).catch((err) => {
      console.log("[FALLBACK] stream LLM failed:", String(err).slice(0, 100));
      return buildFallbackAnswer(req.message, accepted);
    });

    const sourceImages = [...new Set(accepted.flatMap((s) => s.images ?? []))].slice(0, 6);

    yield { type: "answer", answer: { ...answer, images: sourceImages } };
    yield { type: "done", answer: { ...answer, images: sourceImages } };
  }
}

/**
 * Build a structured answer from retrieved chunks when the LLM is unavailable.
 * Extracts the best chunk, parses it for meaning/causes/steps.
 */
function buildFallbackAnswer(query: string, chunks: ScoredHit[]): CitedAnswer {
  if (!chunks.length) {
    return {
      meaning: "No matching content found in the loaded manuals.",
      probable_causes: [],
      corrective_action: [],
      citations: [],
      confidence: "low",
      refusals: ["No relevant content was found for this query."],
    };
  }

  // Pick the highest-scoring chunk
  const best = chunks.sort((a, b) => b.score - a.score)[0];
  const text = best.text;

  // Extract error code from query
  const codeMatch = query.match(/\b([A-Za-z]+)(\s?)(\d{2,4})\b/);
  const errorCode = codeMatch ? codeMatch[0] : undefined;

  // Try to extract meaning: first sentence with "mean" or the first line
  let meaning = "";
  const lines = text.split("\n").filter(Boolean);
  for (const line of lines) {
    const lower = line.toLowerCase();
    if (lower.includes("meaning") || lower.includes("e101") || lower.includes("error code")) {
      meaning = line;
      break;
    }
  }
  if (!meaning) meaning = lines.slice(0, 3).join(" ").slice(0, 200);

  // Extract probable causes: lines after "cause" keyword
  const causes: string[] = [];
  let inCauses = false;
  for (const line of lines) {
    const lower = line.toLowerCase();
    if (lower.includes("probable cause") || lower.includes("cause") || lower.includes("reason")) {
      inCauses = true;
      continue;
    }
    if (inCauses) {
      if (lower.includes("corrective") || lower.includes("action") || lower.includes("step")) break;
      const trimmed = line.replace(/^\d+[\.\)]\s*/, "").trim();
      if (trimmed && trimmed.length > 5) causes.push(trimmed);
    }
  }

  // Extract corrective action steps
  const steps: { step: number; action: string }[] = [];
  let stepNum = 0;
  for (const line of lines) {
    const match = line.match(/step\s*(\d+)|^\s*(\d+)[\.\)]/i);
    if (match) {
      stepNum++;
      steps.push({ step: stepNum, action: line.replace(/step\s*\d+[\.:]\s*/i, "").trim() });
    }
  }

  const citations = [{
    document_id: best.document_id,
    title: best.title,
    page: best.page,
    section: best.section,
  }];

  return {
    error_code: errorCode,
    meaning: meaning.slice(0, 200),
    probable_causes: causes.slice(0, 5),
    corrective_action: steps.slice(0, 6),
    citations,
    confidence: steps.length > 0 ? "medium" : "low",
    refusals: [],
  };
}