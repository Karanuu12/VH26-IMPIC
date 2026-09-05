/**
 * POST /api/chat — answer a troubleshooting question from the indexed manuals.
 *
 *   query → resolve machine scope (explicit → current message → conversation
 *           history) → augment the retrieval query with carried-forward
 *           context for vague follow-ups
 *         → Jina query embedding
 *         → hybrid retrieval (exact code index + lexical + dense, RRF-fused),
 *           scoped to the resolved machine when one is known
 *         → ambiguity check against the fault index (skipped once a machine
 *           is already established — that's the whole point of memory)
 *         → Groq (optional) with [S#]-tagged context AND real multi-turn
 *           conversation history, with [Figure] awareness
 *         → citations mapped from chunk ids, never written by the LLM
 *
 * Degrades on purpose: with no GROQ_API_KEY it still returns a cited,
 * evidence-backed answer built from retrieval alone, so the pipeline is
 * demonstrable before every key is in place.
 */
import { NextRequest } from "next/server";
import { getStore, getEmbedder } from "@/lib/rag-index";
import { getHallucinationSkill } from "@/lib/prompts/skill";
import { startTrace, traceStep, endTrace } from "@/lib/query-progress";
import type { ScoredChunk } from "@timmo/rag/store/local-store";
import type { FaultRecord } from "@timmo/rag/doc/model";

export const runtime = "nodejs";
export const maxDuration = 120;

interface Citation {
  document_id: string;
  title: string;
  page: number;
  section: string;
  /**
   * The retrieved passage behind this citation, so the page viewer can
   * highlight the supporting sentences rather than just opening the page.
   */
  snippet?: string;
}

interface HistoryTurn {
  role: "user" | "assistant";
  content: string;
}

function citationFor(c: ScoredChunk): Citation {
  return {
    document_id: c.documentId,
    title: c.title,
    // The printed page label is what a technician looks for, not the PDF index.
    page: Number(c.pageLabel) || c.pagePdf,
    section: c.sectionPath.join(" › "),
    // Carried so the page viewer can highlight the supporting text. Capped:
    // this rides in every chat response, and the renderer only searches the
    // first few dozen sentences anyway.
    snippet: c.text.slice(0, 1500),
  };
}

/**
 * Diagrams/figures attached to the answer, sourced ONLY from the chunks that
 * were actually cited — not the whole retrieved pool. That keeps images tied
 * to what the answer references rather than showing every diagram that
 * happened to be nearby in the manual.
 */
function imagesFor(hits: ScoredChunk[]): string[] {
  return [...new Set(hits.flatMap((h) => h.figureRefs ?? []))].filter(Boolean).slice(0, 6);
}

// ---------------------------------------------------------------------------
// Conversation memory: machine + error-code carry-forward
// ---------------------------------------------------------------------------

/**
 * ABB drives use F0001/A2001; Schneider uses OCF/SOF (3-4 letter mnemonics);
 * generic manuals use E101/b005. Deliberately conservative -- a looser
 * pattern turns ordinary words into "codes" and corrupts both ambiguity
 * detection and retrieval.
 */
const CODE_RE = /\b([A-Za-z]{1,3}\d{2,5}|[A-Za-z]{2,4}F)\b/g;

/** Turn a machine label into a regex that's tolerant of spacing/hyphenation: "ACS150" also matches "ACS 150" / "acs-150". */
function toMachinePattern(label: string): RegExp {
  const parts = label.match(/[A-Za-z]+|\d+/g) ?? [label];
  const escaped = parts.map((p) => p.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  return new RegExp(`\\b${escaped.join("[\\s-]?")}\\b`, "i");
}

interface MachineCandidate {
  machineId: string;
  label: string;
  pattern: RegExp;
}

/**
 * Built fresh from whatever's actually indexed right now (store.listDocuments()),
 * not a hardcoded list. A hardcoded machine list goes stale the moment someone
 * uploads a new manual; this one can't.
 */
function machineCandidates(store: ReturnType<typeof getStore>): MachineCandidate[] {
  return store.listDocuments().map((d) => {
    const label = d.model || d.machineId;
    return { machineId: d.machineId, label, pattern: toMachinePattern(label) };
  });
}

function detectMachineIn(text: string, candidates: MachineCandidate[]): MachineCandidate | undefined {
  return candidates.find((c) => c.pattern.test(text));
}

/**
 * Resolve which machine this turn is about, in priority order:
 *   1. explicit `machine` field (API callers that already know)
 *   2. a machine named in THIS message (the current message always wins --
 *      a technician switching machines mid-conversation must not be stuck
 *      with the old one)
 *   3. a machine established earlier in the conversation, most recent first
 *
 * This directly implements the disambiguation clues the project's own
 * problem statement calls for: "machine name, model number, conversation
 * history, document metadata" -- in that order of trust.
 */
function resolveMachineScope(
  message: string,
  history: HistoryTurn[],
  explicitMachine: string | undefined,
  candidates: MachineCandidate[],
): string | undefined {
  if (explicitMachine) return explicitMachine;

  const inMessage = detectMachineIn(message, candidates);
  if (inMessage) return inMessage.machineId;

  for (let i = history.length - 1; i >= 0; i--) {
    const found = detectMachineIn(history[i]?.content ?? "", candidates);
    if (found) return found.machineId;
  }
  return undefined;
}

/** Most recent error code mentioned anywhere in history, newest turn first. */
function lastCodeFromHistory(history: HistoryTurn[]): string | undefined {
  for (let i = history.length - 1; i >= 0; i--) {
    const m = (history[i]?.content ?? "").match(CODE_RE);
    if (m) return m[m.length - 1];
  }
  return undefined;
}

/**
 * Client-supplied history feeds directly into the LLM's message array, which
 * makes it a prompt-injection surface if trusted blindly. Coerce role to
 * exactly "user"/"assistant" (never let a client claim "system"), cap each
 * turn's length, and cap how many turns we even look at.
 */
function sanitizeHistory(raw: unknown): HistoryTurn[] {
  if (!Array.isArray(raw)) return [];
  const out: HistoryTurn[] = [];
  for (const item of raw.slice(-20)) {
    if (!item || typeof item !== "object") continue;
    const role = (item as Record<string, unknown>).role;
    const content = (item as Record<string, unknown>).content;
    if ((role !== "user" && role !== "assistant") || typeof content !== "string" || !content.trim()) continue;
    out.push({ role, content: content.slice(0, 1500) });
  }
  return out;
}

/**
 * Query-vector LRU. The Jina embedQuery round-trip is 0.46-0.56s measured --
 * about 60% of total query latency -- and a technician re-asking or refining
 * the same phrasing pays it again every time. Bounded so a long-running
 * process can't grow it without limit.
 */
const QUERY_VEC_CACHE = new Map<string, number[]>();
const QUERY_VEC_CACHE_MAX = 500;

function cachedQueryVector(key: string): number[] | undefined {
  const hit = QUERY_VEC_CACHE.get(key);
  if (hit) {
    // Refresh recency: delete + re-set moves it to the end of insertion order.
    QUERY_VEC_CACHE.delete(key);
    QUERY_VEC_CACHE.set(key, hit);
  }
  return hit;
}

function putQueryVector(key: string, vector: number[]): void {
  if (QUERY_VEC_CACHE.size >= QUERY_VEC_CACHE_MAX) {
    const oldest = QUERY_VEC_CACHE.keys().next().value;
    if (oldest !== undefined) QUERY_VEC_CACHE.delete(oldest);
  }
  QUERY_VEC_CACHE.set(key, vector);
}

/**
 * Exact fault-code lookup, answered straight from the fault index -- no
 * embedding call, no LLM call.
 *
 * A FaultRecord already holds exactly what the answer format needs (meaning,
 * causes, corrective steps, provenance), extracted deterministically at
 * ingest. Sending that to an LLM to be re-worded costs ~0.5s of embedding
 * plus a generation round-trip and can only make it LESS faithful. Answer
 * directly when the record is complete; fall through to the normal pipeline
 * when it isn't.
 */
function answerFromFaultRecord(record: FaultRecord) {
  return {
    error_code: record.codeRaw,
    meaning: record.meaning,
    probable_causes: record.causes,
    corrective_action: record.steps.map((s) => ({ step: s.n, action: s.text })),
    citations: [
      {
        document_id: record.provenance.documentId,
        title: record.provenance.title,
        page: Number(record.provenance.pageLabel) || record.provenance.pagePdf,
        section: record.provenance.sectionPath.join(" › "),
      },
    ],
    images: [] as string[],
    confidence: "high" as const,
    refusals: [] as string[],
  };
}

/**
 * Is this record clean enough to serve verbatim, skipping the LLM?
 *
 * Only table-extracted records qualify. A fault TABLE has real column
 * separation, so `meaning`/`causes`/`steps` land in the right fields --
 * that's the case the extractor's tests cover. Section-extracted records come
 * from prose and can be mushy: one measured record put all five probable
 * causes into `steps` and split the real corrective actions mid-sentence.
 * Serving that directly would be faster AND worse, so those fall through to
 * the normal retrieval+LLM path, which re-reads the underlying chunks and
 * recovers. Speed is only worth taking when it costs nothing.
 */
function isFastPathQuality(record: FaultRecord): boolean {
  if (record.extraction !== "table_row") return false;
  if (!record.meaning?.trim()) return false;
  if (record.steps.length === 0) return false;
  // A table cell that exploded into dozens of "steps" is a parse artifact,
  // not a 30-step procedure.
  if (record.steps.length > 12) return false;
  return true;
}

/** Phase 9: the same code meaning different things on different machines. */
function checkAmbiguity(records: FaultRecord[]): { ambiguous: boolean; question: string } {
  const byMachine = new Map<string, FaultRecord>();
  for (const r of records) if (!byMachine.has(r.machineId)) byMachine.set(r.machineId, r);
  if (byMachine.size < 2) return { ambiguous: false, question: "" };

  // Same code, same meaning on every machine that has it -- not actually
  // ambiguous. Cheap normalized-string comparison, no embedding call needed:
  // deterministic and fast, in keeping with "not a prompt/LLM trick."
  const normalize = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
  const distinctMeanings = new Set([...byMachine.values()].map((r) => normalize(r.meaning || "")));
  if (distinctMeanings.size <= 1) return { ambiguous: false, question: "" };

  const options = [...byMachine.values()]
    .map((r) => `**${r.model ?? r.machineId}** — ${r.meaning || "see manual"}`)
    .join("; ");
  return {
    ambiguous: true,
    question: `That code appears in more than one manual with different meanings: ${options}. Which machine are you working on?`,
  };
}

export async function POST(request: NextRequest) {
  // Hoisted so the catch below can close the trace: the body is already
  // consumed by then, so it cannot be re-read there.
  let jobId: string | undefined;
  try {
    const body = await request.json();
    const { message, machine: explicitMachine } = body as { message?: string; machine?: string };
    jobId = typeof body?.job_id === "string" ? body.job_id : undefined;
    const history = sanitizeHistory(body?.history);

    if (!message || typeof message !== "string" || !message.trim()) {
      return Response.json({ error: "message is required (string)" }, { status: 400 });
    }

    // Trace is best-effort telemetry for the UI; every call is a no-op when
    // the client didn't send a job_id, so the endpoint's contract is unchanged.
    startTrace(jobId);

    const store = getStore();
    if (store.stats.chunks === 0) {
      endTrace(jobId);
      return Response.json({
        answer: {
          meaning: "No manuals have been indexed yet.",
          probable_causes: [],
          corrective_action: [],
          citations: [],
          confidence: "low",
          refusals: ["Upload a PDF manual first — use the Upload PDF button above."],
        },
        sources: [],
      });
    }

    // --- Conversation memory: resolve machine + carried error code --------
    const candidates = machineCandidates(store);
    const resolvedMachine = resolveMachineScope(message, history, explicitMachine, candidates);
    const messageCodes = message.match(CODE_RE) ?? [];
    traceStep(
      jobId,
      "Read the question",
      [
        messageCodes.length ? `code ${messageCodes.join(", ")}` : "no explicit code",
        resolvedMachine
          ? `scoped to ${candidates.find((c) => c.machineId === resolvedMachine)?.label ?? resolvedMachine}`
          : `${candidates.length} manual${candidates.length === 1 ? "" : "s"} in scope`,
      ].join(" · "),
    );
    // Only borrow a code from history when THIS message doesn't name one --
    // an explicit code in the current message always wins.
    const carriedCode = messageCodes.length ? undefined : lastCodeFromHistory(history);

    // --- Ambiguity check runs before anything expensive --------------------
    // Skipped entirely once a machine is already known -- that's the point
    // of remembering: "Machine A shows E101" -> "and what if that doesn't
    // fix it?" must not re-ask which machine.
    if (!resolvedMachine) {
      for (const code of messageCodes) {
        const records = store.faultsForCode(code);
        const { ambiguous, question } = checkAmbiguity(records);
        traceStep(
          jobId,
          "Checked for the same code in other manuals",
          ambiguous
            ? `${code} documented differently in ${new Set(records.map((r) => r.machineId)).size} manuals — asking which`
            : `${code}: ${records.length} record${records.length === 1 ? "" : "s"}, no conflict`,
        );
        if (ambiguous) {
          endTrace(jobId);
          return Response.json({
            answer: {
              error_code: code,
              meaning: question,
              probable_causes: [],
              corrective_action: [],
              citations: records.map((r) => ({
                document_id: r.provenance.documentId,
                title: r.provenance.title,
                page: Number(r.provenance.pageLabel) || r.provenance.pagePdf,
                section: r.provenance.sectionPath.join(" › "),
              })),
              confidence: "high",
              refusals: [],
            },
            sources: [],
            ambiguous: true,
          });
        }
      }
    }

    // --- Exact fault-code fast path ---------------------------------------
    // A complete FaultRecord for a resolved machine already IS the answer, in
    // the exact shape the response format needs, extracted deterministically
    // at ingest. Skipping embedding + generation here turns a ~700ms query
    // into a few milliseconds with strictly higher fidelity -- the LLM can
    // only paraphrase what's already exact. Only taken when the record is
    // complete (has meaning AND steps); anything thinner falls through to the
    // full pipeline so we never trade an answer for speed.
    if (messageCodes.length === 1 && !history.length) {
      const candidatesForCode = store.faultsForCode(messageCodes[0]);
      const scoped = resolvedMachine
        ? candidatesForCode.filter((r) => r.machineId === resolvedMachine)
        : candidatesForCode;
      const record = scoped.length === 1 ? scoped[0] : undefined;
      if (record && isFastPathQuality(record)) {
        traceStep(
          jobId,
          "Answered from the fault index",
          `exact ${record.extraction} record — no embedding or generation needed`,
        );
        endTrace(jobId);
        return Response.json({
          answer: answerFromFaultRecord(record),
          sources: [],
          resolved_machine: record.machineId,
          fast_path: "fault-index",
        });
      }
    }

    // --- Retrieval ----------------------------------------------------------
    // A vague follow-up ("and what if that doesn't fix it?") carries no
    // retrievable terms on its own. Augmenting the SEARCH text (not the
    // question shown to the LLM) with carried-forward code/machine is what
    // makes retrieval for a follow-up actually find the right section, not
    // just the LLM's own memory of the conversation.
    const machineLabel = candidates.find((c) => c.machineId === resolvedMachine)?.label;
    const retrievalQuery = [carriedCode, machineLabel, message].filter(Boolean).join(" ");

    const embedder = getEmbedder();
    // ~0.5s of the ~0.8s query is this one round-trip; cache it.
    let queryVector = cachedQueryVector(retrievalQuery);
    if (queryVector) {
      traceStep(jobId, "Embedded the query", `cache hit — ${queryVector.length} dims, no API call`);
    } else {
      queryVector = await embedder.embedQuery(retrievalQuery);
      putQueryVector(retrievalQuery, queryVector);
      traceStep(jobId, "Embedded the query", `${queryVector.length} dims via Jina`);
    }
    const hits = store.search(queryVector, retrievalQuery, { topK: 8, machineId: resolvedMachine });
    const retrievers = [...new Set(hits.flatMap((h) => h.matchedBy))];
    traceStep(
      jobId,
      "Retrieved passages",
      hits.length
        ? `${hits.length} from ${new Set(hits.map((h) => h.title)).size} manual${
            new Set(hits.map((h) => h.title)).size === 1 ? "" : "s"
          } · matched by ${retrievers.join(" + ")}`
        : "nothing matched",
    );

    if (hits.length === 0) {
      endTrace(jobId);
      return Response.json({
        answer: {
          meaning: "Nothing in the indexed manuals matches that question.",
          probable_causes: [],
          corrective_action: [],
          citations: [],
          confidence: "low",
          refusals: [
            "No relevant content found. Try naming the machine (e.g. ACS150) or an exact error code.",
          ],
        },
        sources: [],
      });
    }

    // --- Answer ---------------------------------------------------------
    const groqKey = process.env.GROQ_API_KEY;
    const answer = groqKey
      ? await answerWithGroq(message, hits, groqKey, history)
      : answerFromRetrieval(hits);
    // Refusals are surfaced here too: an answer that fell back to raw retrieved
    // text reports "0 steps", which on its own reads like the pipeline simply
    // found nothing rather than like generation failing.
    traceStep(
      jobId,
      "Composed the answer",
      groqKey
        ? [
            `${answer.corrective_action.length} step${answer.corrective_action.length === 1 ? "" : "s"}`,
            `${answer.citations.length} citation${answer.citations.length === 1 ? "" : "s"}`,
            answer.refusals.length
              ? `${answer.refusals.length} caveat${answer.refusals.length === 1 ? "" : "s"} — see the answer`
              : null,
          ]
            .filter(Boolean)
            .join(", ")
        : "no GROQ_API_KEY — returned the retrieved text verbatim",
    );
    endTrace(jobId);

    return Response.json({
      answer,
      sources: hits.map((h) => ({
        document_id: h.documentId,
        title: h.title,
        page: Number(h.pageLabel) || h.pagePdf,
        section: h.sectionPath.join(" › "),
        text: h.text.slice(0, 1200),
        score: Number(h.score.toFixed(4)),
        matched_by: h.matchedBy,
      })),
      resolved_machine: resolvedMachine,
    });
  } catch (err) {
    console.error("/api/chat error:", err);
    const message = err instanceof Error ? err.message : "Internal server error";
    // Mark the trace finished even on failure, so the client's poll stops
    // rather than hanging on a request that is never coming back.
    traceStep(jobId, "Failed", message.slice(0, 160));
    endTrace(jobId);
    return Response.json({ error: message }, { status: 500 });
  }
}

/**
 * Fallback answer built from retrieval alone, no LLM phrasing. Used both when
 * there's no GROQ_API_KEY at all, and as answerWithGroq's own fallback when
 * the call fails or its response doesn't parse -- those are different
 * situations and get different refusal text. A refusal that misdiagnoses its
 * own cause is a small credibility problem in exactly the area this file's
 * skill prompt is trying to protect.
 */
function answerFromRetrieval(
  hits: ScoredChunk[],
  reason = "GROQ_API_KEY is not set, so this is the retrieved manual text rather than a generated answer.",
) {
  const best = hits[0];
  const body = best.text.split("\n\n").slice(1).join("\n\n").trim() || best.text;
  const cited = hits.slice(0, 4);
  return {
    meaning: body.slice(0, 600),
    probable_causes: [],
    corrective_action: [],
    citations: cited.map(citationFor),
    images: imagesFor(cited),
    confidence: "medium" as const,
    refusals: [reason],
  };
}

/** With a key: the LLM phrases the answer, sees real conversation history, but citations come from chunk ids. */
async function answerWithGroq(
  message: string,
  hits: ScoredChunk[],
  apiKey: string,
  history: HistoryTurn[] = [],
) {
  const context = hits
    .map((h, i) => `[S${i + 1}] ${h.sectionPath.join(" › ")} (page ${h.pageLabel})\n${h.text}`)
    .join("\n\n");

  const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify({
      model: process.env.GROQ_MODEL ?? "llama-3.3-70b-versatile",
      temperature: 0.15,
      max_tokens: 1600,
      messages: [
        { role: "system", content: getHallucinationSkill() },
        // Real prior turns, not a paraphrase stuffed into the user message --
        // this is what lets the model correctly read an elliptical follow-up
        // ("and what if that doesn't fix it?") as a continuation, while the
        // skill prompt above still requires every CLAIM to be re-grounded in
        // this turn's numbered excerpts, not just carried from a past turn.
        ...history.slice(-6).map((h) => ({ role: h.role, content: h.content })),
        {
          role: "user",
          content:
            `CONTEXT:\n${context}\n\nQUESTION: ${message}\n\n` +
            `Output JSON: {"error_code":"","meaning":"","probable_causes":[],` +
            `"corrective_action":[{"step":1,"action":""}],"used_sources":[1,2],` +
            `"confidence":"high|medium|low","refusals":[]}`,
        },
      ],
    }),
  });

  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    console.warn("Groq failed, falling back to retrieval:", detail.slice(0, 200));
    return answerFromRetrieval(
      hits,
      `The answer-generation model returned an error (${res.status}), so this is the retrieved manual text rather than a generated answer.`,
    );
  }

  const raw = await res.json();
  const content: string = raw.choices?.[0]?.message?.content ?? "";
  const start = content.indexOf("{");
  const end = content.lastIndexOf("}");
  if (start === -1 || end <= start) {
    return answerFromRetrieval(
      hits,
      "The answer-generation model's response wasn't valid JSON, so this is the retrieved manual text rather than a generated answer.",
    );
  }

  try {
    const parsed = JSON.parse(content.slice(start, end + 1));
    // Citations are resolved from the sources the model actually used —
    // it never gets to invent a page number.
    const used: number[] = Array.isArray(parsed.used_sources) ? parsed.used_sources : [];
    const citedHits = used.map((n: number) => hits[n - 1]).filter(Boolean);
    const fallbackHits = citedHits.length ? citedHits : hits.slice(0, 3);

    return {
      error_code: parsed.error_code || undefined,
      meaning: String(parsed.meaning ?? ""),
      probable_causes: Array.isArray(parsed.probable_causes) ? parsed.probable_causes : [],
      corrective_action: Array.isArray(parsed.corrective_action) ? parsed.corrective_action : [],
      citations: fallbackHits.map(citationFor),
      images: imagesFor(fallbackHits),
      confidence: ["high", "medium", "low"].includes(parsed.confidence) ? parsed.confidence : "low",
      refusals: Array.isArray(parsed.refusals) ? parsed.refusals : [],
    };
  } catch {
    return answerFromRetrieval(
      hits,
      "The answer-generation model's response couldn't be parsed as the expected JSON shape, so this is the retrieved manual text rather than a generated answer.",
    );
  }
}
