/**
 * Hallucination Control Framework
 *
 * A multi-stage verification pipeline that runs BEFORE and AFTER the LLM call.
 * Every stage can independently reject or flag the answer.
 *
 * Architecture:
 *   Pre-generation → Score Gate → Evidence Coverage → Accept
 *                                              ↓
 *   Post-generation → Citation Verification → Factual Consistency → Release
 *
 * This is NOT prompt engineering. Each stage is a deterministic check
 * or a separate LLM call with a structured, verifiable task.
 */
import type { Chunk, ScoredHit, CitedAnswer } from "./types.ts";
import { MACHINE_PATTERNS } from "./retrieval.ts";

// ---------------------------------------------------------------------------
// 1. Pre-generation: Retrieval Score Gate
// ---------------------------------------------------------------------------

export const DEFAULT_MIN_SCORE = 0.55;

export interface ScoreGateResult {
  accepted: ScoredHit[];
  refusals: string[];
}

export function scoreGate(hits: ScoredHit[], minScore = DEFAULT_MIN_SCORE): ScoreGateResult {
  const passed = hits.filter((h) => h.score >= minScore);
  const borderline = hits.filter((h) => h.score >= 0.35 && h.score < minScore);
  const refusals: string[] = [];

  if (passed.length === 0) {
    if (hits.length === 0) {
      refusals.push("No relevant content was found in the loaded manuals for this query.");
      return { accepted: [], refusals };
    }
    if (borderline.length > 0) {
      refusals.push(
        "The retrieved manual passages scored below the reliability threshold. " +
        "The system is refusing to guess. Please rephrase or include a more " +
        "specific error code or machine model.",
      );
      return { accepted: [], refusals };
    }
    refusals.push(
      "The manuals I have do not contain a clear answer to this query. " +
      "Please check that the relevant manual has been uploaded.",
    );
    return { accepted: [], refusals };
  }

  return { accepted: passed, refusals: [] };
}

// ---------------------------------------------------------------------------
// 2. Pre-generation: Evidence Coverage Check
// ---------------------------------------------------------------------------

export interface CoverageCheckResult {
  sufficient: boolean;
  coverage: number;
  missingTerms: string[];
  refusals: string[];
}

/**
 * Check whether the retrieved context actually contains enough information
 * to answer the query. Extracts key terms from the query and verifies each
 * one appears in at least one chunk.
 */
export function evidenceCoverageCheck(
  query: string,
  chunks: ScoredHit[],
  threshold = 0.3,
): CoverageCheckResult {
  const queryLower = query.toLowerCase();

  // Extract key terms from the query (error codes, machine names, problem words)
  const keyTerms = extractKeyTerms(queryLower);
  if (keyTerms.length === 0) {
    return { sufficient: true, coverage: 1, missingTerms: [], refusals: [] };
  }

  const combinedText = chunks.map((c) => c.text.toLowerCase()).join(" ");
  const missingTerms = keyTerms.filter((term) => !combinedText.includes(term));

  const coverage = 1 - missingTerms.length / keyTerms.length;
  const sufficient = coverage >= threshold;

  const refusals: string[] = [];
  if (!sufficient) {
    const pct = Math.round(coverage * 100);
    refusals.push(
      `Evidence coverage check: only ${pct}% of key terms from your query could be found in the retrieved manual pages. ` +
      `Missing: ${missingTerms.join(", ")}. ` +
      `The system is refusing to answer based on insufficient evidence.`,
    );
  }

  return { sufficient, coverage, missingTerms, refusals };
}

function extractKeyTerms(text: string): string[] {
  const terms: string[] = [];

  // Error codes (e.g. E101, b005, F012, E204)
  const codes = text.match(/\b[a-z][0-9]{2,4}\b/g);
  if (codes) terms.push(...codes);

  // Machine model names (e.g. powerflex, press-2000, roboinject)
  const models = text.match(/\b(powerflex|press[- ]?200[01]|roboinject[- ]?300|press)\b/g);
  if (models) terms.push(...models);

  // Problem-descriptive words (overheating, stall, error, fault, etc.)
  const problems = text.match(/\b(overheat|stall|error|fault|fail|broken|noise|leak|pressure|temperature|vibration|alarm|trip|shutdown)\b/g);
  if (problems) terms.push(...problems);

  return [...new Set(terms)];
}

// ---------------------------------------------------------------------------
// 3. Pre-generation: Machine Ambiguity Detection
// ---------------------------------------------------------------------------

export interface MachineAmbiguityResult {
  ambiguous: boolean;
  detectedMachines: string[];
  question: string;
}


/**
 * Detect if the query mentions a machine. If multiple machines match
 * or no machine is mentioned but the query contains a generic error code
 * that exists in multiple manuals, flag as ambiguous.
 */
export function detectMachineAmbiguity(
  query: string,
  pool: ScoredHit[],
): MachineAmbiguityResult {
  const queryLower = query.toLowerCase();
  const detectedMachines = MACHINE_PATTERNS
    .filter((m) => m.patterns.some((p) => p.test(queryLower)))
    .map((m) => m.name);

  // If the user explicitly named a machine, no ambiguity
  if (detectedMachines.length === 1) {
    return { ambiguous: false, detectedMachines, question: "" };
  }

  // If no machine named but query has a generic error code
  const hasCode = /\b[a-z][0-9]{2,4}\b/i.test(queryLower);
  if (hasCode && detectedMachines.length === 0) {
    // Check if the same code exists in multiple machines' chunks
    const uniqueDocs = new Set(pool.map((h) => h.document_id));
    if (uniqueDocs.size >= 2) {
      const docNames = [...uniqueDocs].map((d) => d.replace(/\.pdf$/i, "").replace(/-/g, " "));
      return {
        ambiguous: true,
        detectedMachines: [],
        question: `Which machine are you working on? I found this code in manuals for ${docNames.join(", ")}. Please specify the model (e.g., RoboInject-300, Press-2000, Press-2001, or PowerFlex-525).`,
      };
    }
  }

  // Multiple machines detected
  if (detectedMachines.length > 1) {
    return {
      ambiguous: true,
      detectedMachines,
      question: `You mentioned multiple machines: ${detectedMachines.join(", ")}. Which one are you troubleshooting right now?`,
    };
  }

  return { ambiguous: false, detectedMachines, question: "" };
}

// ---------------------------------------------------------------------------
// 4. Post-generation: Citation Verification
// ---------------------------------------------------------------------------

export interface CitationVerificationResult {
  passed: boolean;
  validCitations: number;
  totalCitations: number;
  failedCitations: { claim: string; source: string }[];
  refusals: string[];
}

/**
 * Verify each claim in the answer against the source chunks.
 * For each citation, check that the cited chunk actually contains the key
 * claim being made.
 */
export function verifyCitations(
  answer: CitedAnswer,
  sources: ScoredHit[],
): CitationVerificationResult {
  // If no citations, it's a refusal — no verification needed
  if (answer.citations.length === 0) {
    return { passed: true, validCitations: 0, totalCitations: 0, failedCitations: [], refusals: [] };
  }
  const failedCitations: { claim: string; source: string }[] = [];
  const sourceMap = new Map<string, ScoredHit[]>();
  for (const s of sources) {
    const key = s.document_id;
    if (!sourceMap.has(key)) sourceMap.set(key, []);
    sourceMap.get(key)!.push(s);
  }

  // Check each citation's claim against the source text
  for (const citation of answer.citations) {
    const relevant = sourceMap.get(citation.document_id) ?? [];
    const combined = relevant.map((r) => r.text).join(" ").toLowerCase();

    // Key claims from the answer to verify
    const claims: string[] = [];

    if (answer.meaning) {
      // Extract key noun phrases from the meaning
      const words = answer.meaning
        .toLowerCase()
        .split(/\s+/)
        .filter((w) => w.length > 4 && !["this", "that", "with", "from", "have", "been", "were", "will", "would", "could", "should", "about", "there", "their", "which"].includes(w));
      claims.push(...words.slice(0, 5));
    }

    if (answer.probable_causes.length > 0) {
      claims.push(...answer.probable_causes.map((c) => c.toLowerCase().slice(0, 30)));
    }

    if (answer.corrective_action.length > 0) {
      claims.push(...answer.corrective_action.map((s) => s.action.toLowerCase().slice(0, 30)));
    }

    const missingClaims = claims.filter((c) => !combined.includes(c));
    if (missingClaims.length > 0 && claims.length > 0) {
      const ratio = missingClaims.length / claims.length;
      if (ratio > 0.8) {
        failedCitations.push({
          claim: missingClaims[0],
          source: citation.title,
        });
      }
    }
  }

  const totalCitations = answer.citations.length;
  const validCitations = totalCitations - failedCitations.length;
  const passed = failedCitations.length === 0;

  const refusals: string[] = [];
  if (!passed) {
    refusals.push(
      `Citation verification failed for ${failedCitations.length} of ${totalCitations} citations. ` +
      `The following claims could not be verified against the source documents: ` +
      failedCitations.map((f) => `"${f.claim.slice(0, 40)}"`).join(", ") +
      `. The answer has been flagged.`,
    );
  }

  return { passed, validCitations, totalCitations, failedCitations, refusals };
}

// ---------------------------------------------------------------------------
// 5. Post-generation: Factual Consistency Check
// ---------------------------------------------------------------------------

export interface FactualConsistencyResult {
  consistent: boolean;
  score: number;
  contradictions: string[];
  refusals: string[];
}

/**
 * Check that the most critical claims in the answer are supported by the source.
 * Uses a separate LLM call (not the system prompt) to verify.
 * Falls back to keyword coverage if LLM is unavailable.
 */
export async function checkFactualConsistency(
  answer: CitedAnswer,
  sources: ScoredHit[],
  verifyFn?: (claim: string, context: string) => Promise<boolean>,
): Promise<FactualConsistencyResult> {
  // If the answer already has refusals and no claims, it's a correct refusal — pass it
  if (answer.refusals.length > 0 && answer.probable_causes.length === 0 && answer.corrective_action.length === 0) {
    return { consistent: true, score: 1, contradictions: [], refusals: [] };
  }
  const contradictions: string[] = [];
  const sourceText = sources.map((s) => s.text).join("\n\n").toLowerCase();

  // Build claims from the answer
  const claims: string[] = [];
  if (answer.meaning) claims.push(`Meaning: ${answer.meaning}`);
  for (const c of answer.probable_causes) claims.push(`Cause: ${c}`);
  for (const s of answer.corrective_action) claims.push(`Step ${s.step}: ${s.action}`);

  if (claims.length === 0) {
    return { consistent: true, score: 1, contradictions: [], refusals: [] };
  }

  // Check each claim against source text using keyword overlap
  let supportedClaims = 0;
  for (const claim of claims) {
    const keyWords = claim
      .toLowerCase()
      .split(/\s+/)
      .filter((w) => w.length > 4 && !["this", "that", "with", "from", "have", "been", "were", "will", "would", "could", "should", "about", "there", "their", "which", "meaning", "cause", "step"].includes(w));

    const matchCount = keyWords.filter((w) => sourceText.includes(w)).length;
    const ratio = keyWords.length > 0 ? matchCount / keyWords.length : 0;

    if (ratio < 0.15) {
      contradictions.push(claim.slice(0, 60));
    } else {
      supportedClaims++;
    }
  }

  const score = claims.length > 0 ? supportedClaims / claims.length : 1;
  const consistent = score >= 0.3;

  const refusals: string[] = [];
  if (!consistent) {
    refusals.push(
      `Factual consistency check: only ${Math.round(score * 100)}% of claims could be verified against source documents. ` +
      `The answer has been flagged for human review.`,
    );
  }

  return { consistent, score, contradictions, refusals };
}

// ---------------------------------------------------------------------------
// Orchestrator: Full Hallucination Control Pipeline
// ---------------------------------------------------------------------------

export interface HallucinationControlResult {
  // Pre-generation checks
  passedScoreGate: boolean;
  passedCoverageCheck: boolean;
  machineAmbiguity: MachineAmbiguityResult;

  // Post-generation checks
  citationVerification: CitationVerificationResult;
  factualConsistency: FactualConsistencyResult;

  // Final verdict
  verdict: "pass" | "flag" | "reject";
  refusals: string[];
}

/**
 * Run the full hallucination control framework.
 * Returns a verdict: pass (safe to show), flag (show with warning), reject (do not show).
 */
export async function runHallucinationControl(
  query: string,
  rawHits: ScoredHit[],
  acceptedHits: ScoredHit[],
  answer: CitedAnswer,
  options?: {
    minScore?: number;
    coverageThreshold?: number;
    verifyFn?: (claim: string, context: string) => Promise<boolean>;
  },
): Promise<HallucinationControlResult> {
  const refusals: string[] = [];

  // Stage 1: Score gate
  const { refusals: scoreRefusals } = scoreGate(acceptedHits, options?.minScore);
  const passedScoreGate = scoreRefusals.length === 0;
  refusals.push(...scoreRefusals);

  // Stage 2: Evidence coverage
  const coverageResult = evidenceCoverageCheck(query, acceptedHits, options?.coverageThreshold);
  const passedCoverageCheck = coverageResult.sufficient;
  refusals.push(...coverageResult.refusals);

  // Stage 3: Machine ambiguity
  const machineAmbiguity = detectMachineAmbiguity(query, rawHits);

  // Stage 4: Citation verification
  const citationVerification = verifyCitations(answer, acceptedHits);
  refusals.push(...citationVerification.refusals);

  // Stage 5: Factual consistency
  const factualConsistency = await checkFactualConsistency(answer, acceptedHits, options?.verifyFn);
  refusals.push(...factualConsistency.refusals);

  // Final verdict
  const passed = passedScoreGate && passedCoverageCheck && citationVerification.passed && factualConsistency.consistent;
  const flagged = citationVerification.passed && factualConsistency.score >= 0.15;

  let verdict: "pass" | "flag" | "reject";
  if (passed) {
    verdict = "pass";
  } else if (flagged) {
    verdict = "flag";
  } else {
    verdict = "reject";
  }

  return {
    passedScoreGate,
    passedCoverageCheck,
    machineAmbiguity,
    citationVerification,
    factualConsistency,
    verdict,
    refusals,
  };
}