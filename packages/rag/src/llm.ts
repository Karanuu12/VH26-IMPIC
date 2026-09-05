/**
 * Groq LLM client (hosted, free tier).
 * Calls Groq's chat completions API directly with supporting context.
 */
import { z } from "zod";
import type { ChatTurn, CitedAnswer, AnswerStep, Citation, ScoredHit } from "./types.ts";

export interface LLMConfig {
  apiKey: string;
  model?: string;
  baseUrl?: string;
}

const GROQ_BASE = "https://api.groq.com/openai/v1";

const STRUCTURED_ANSWER_SCHEMA = z.object({
  error_code: z.string().optional().default(""),
  meaning: z.string().optional().default(""),
  probable_causes: z.array(z.string()).optional().default([]),
  corrective_action: z
    .array(z.object({ step: z.number(), action: z.string() }))
    .optional()
    .default([]),
  citations: z
    .array(
      z.object({
        document_id: z.string(),
        title: z.string(),
        page: z.number(),
        section: z.string(),
      }),
    )
    .optional()
    .default([]),
  confidence: z.enum(["high", "medium", "low"]).optional().default("low"),
  refusals: z.array(z.string()).optional().default([]),
});

export class GroqClient {
  private apiKey: string;
  private model: string;
  private baseUrl: string;

  constructor(config: LLMConfig) {
    this.apiKey = config.apiKey;
    this.model = config.model ?? "llama-3.3-70b-versatile";
    this.baseUrl = config.baseUrl ?? GROQ_BASE;
  }

  /**
   * Generate a structured, cited answer from retrieved context + conversation history.
   */
  async generateAnswer(
    message: string,
    context: ScoredHit[],
    history: ChatTurn[] = [],
    machineScope?: string,
  ): Promise<CitedAnswer> {
    const contextBlock = context.length
      ? context.map((h, i) =>
          `[${i}] Document: ${h.title} | Page ${h.page} | Section: ${h.section}\n${h.text}`
        ).join("\n\n")
      : "NO RELEVANT CONTEXT FOUND";

    const machineHint = machineScope
      ? `\nThe user specifically mentioned the machine: "${machineScope}". Prioritize its manual.`
      : "";

    const systemPrompt = "You read machine manuals and answer questions. The context below is from the manuals. Extract the answer and output JSON. Use the context, cite the source, keep answers short.";

    const userPrompt = "CONTEXT:\n" + contextBlock + "\n\nQUESTION: " + message + "\n\nOutput JSON: " +
      '{"error_code":"E101","meaning":"short meaning","probable_causes":["cause 1"],"corrective_action":[{"step":1,"action":"do this"}],"citations":[{"document_id":"x","title":"x","page":1,"section":"x"}],"confidence":"high","refusals":[]}';

    const messages: { role: string; content: string }[] = [
      { role: "system", content: systemPrompt },
      ...history.slice(-6).map((t) => ({ role: t.role, content: t.content })),
      { role: "user", content: userPrompt },
    ];

    const body = {
      model: this.model,
      messages,
      temperature: 0.15,
      max_tokens: 2048,
    };

    // Retry once on 429 (Groq free tier rate limit) with backoff
    let res = await fetch(`${this.baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify(body),
    });

    if (res.status === 429) {
      await new Promise((r) => setTimeout(r, 3500));
      res = await fetch(`${this.baseUrl}/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify(body),
      });
    }

    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      throw new Error(`Groq LLM error (${res.status}): ${detail.slice(0, 400)}`);
    }

    const raw = await res.json();
    const choice = raw.choices?.[0];
    if (!choice?.message?.content) {
      throw new Error("Groq returned empty response.");
    }

    const content = choice.message.content;
    const cleaned = content
      .replace(/```json\s*/gi, "")
      .replace(/```\s*/gi, "")
      .trim();
    const jsonStart = cleaned.indexOf("{");
    const jsonEnd = cleaned.lastIndexOf("}");
    let jsonStr = cleaned;
    if (jsonStart !== -1 && jsonEnd !== -1 && jsonEnd > jsonStart) {
      jsonStr = cleaned.slice(jsonStart, jsonEnd + 1);
    }
    const parsed = STRUCTURED_ANSWER_SCHEMA.parse(
      JSON.parse(jsonStr, (_key, value) => (value === null ? undefined : value)),
    );

    return {
      ...parsed,
      raw: content,
    };
  }

  /** Simple text-only Q&A (follow-up negotiation, clarifying questions). */
  async generateText(
    prompt: string,
    system = "You are a factory-troubleshooting assistant. Be concise.",
  ): Promise<string> {
    const body = {
      model: this.model,
      messages: [
        { role: "system", content: system },
        { role: "user", content: prompt },
      ],
      temperature: 0.3,
    };

    const res = await fetch(`${this.baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      throw new Error(`Groq text error (${res.status}): ${detail.slice(0, 300)}`);
    }

    const raw = await res.json();
    return raw.choices?.[0]?.message?.content ?? "";
  }
}