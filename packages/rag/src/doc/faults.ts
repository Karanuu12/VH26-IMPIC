/**
 * Fault-record extraction — Phase 4.
 *
 * Turns "Error Code → Meaning → Cause → Corrective Action" into a first-class
 * record at ingest time, so query time never asks an LLM to re-derive it from
 * prose. Two deterministic extractors, no model calls:
 *
 *   table_row — for fault tables (ABB "CODE | FAULT | CAUSE | WHAT TO DO",
 *               Schneider "Error code | Name | Probable cause | Remedy")
 *   section   — for manuals that give each code its own heading
 *               ("4.1 E101 — Winding Overtemperature")
 *
 * Anything neither extractor understands still survives as an ordinary chunk.
 * This is additive; it never drops content.
 */
import type { Block, FaultRecord, StepData, Provenance, AdmonitionSeverity } from "./model.ts";
import { normalizeCode } from "./model.ts";
import { COL_PATTERNS, findColumn, foldContinuationRows, isFaultTable } from "./tables.ts";

// ---------------------------------------------------------------------------
// Shared text → steps / causes
// ---------------------------------------------------------------------------

/**
 * Split a corrective-action cell into steps.
 *
 * Real cells look like:
 *   "Check motor load.\nCheck acceleration time (parameters 2202 ...).\nCheck ambient conditions."
 * Sometimes numbered, usually not — ABB just uses one imperative per line.
 * We treat lines as steps, then fall back to sentence splitting for single-line cells.
 */
export function splitSteps(text: string): StepData[] {
  if (!text.trim()) return [];

  const lines = text
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);

  let parts: string[];
  if (lines.length > 1) {
    parts = lines;
  } else {
    // One long line: split on sentence ends, but don't split inside "(parameters 2202 ...)"
    parts = splitSentencesOutsideParens(lines[0] ?? "");
  }

  return parts
    .map((p) => p.replace(/^\s*(?:\d+[.)]|[-•*])\s*/, "").trim())
    .filter((p) => p.length > 2)
    .map((text, i) => ({ n: i + 1, text }));
}

function splitSentencesOutsideParens(s: string): string[] {
  const out: string[] = [];
  let depth = 0;
  let cur = "";
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (ch === "(") depth++;
    else if (ch === ")") depth = Math.max(0, depth - 1);
    cur += ch;
    if (depth === 0 && (ch === "." || ch === ";") && /\s|$/.test(s[i + 1] ?? " ")) {
      out.push(cur.trim());
      cur = "";
    }
  }
  if (cur.trim()) out.push(cur.trim());
  return out.length ? out : [s];
}

/** Causes split the same way, but stay as plain strings. */
export function splitCauses(text: string): string[] {
  return splitSteps(text).map((s) => s.text);
}

const SEVERITY_RE: [RegExp, AdmonitionSeverity][] = [
  [/\bdanger\b/i, "danger"],
  [/\bwarning\b/i, "warning"],
  [/\bcaution\b/i, "caution"],
  [/\bnotice\b/i, "notice"],
];

function severityOf(text: string): AdmonitionSeverity | undefined {
  for (const [re, sev] of SEVERITY_RE) if (re.test(text)) return sev;
  return undefined;
}

// ---------------------------------------------------------------------------
// Extractor 1: fault tables
// ---------------------------------------------------------------------------

export function extractFromTable(
  block: Block,
  machineId: string,
  model: string | undefined,
  documentTitle: string,
): FaultRecord[] {
  const table = block.table;
  if (!table || !isFaultTable(table.header)) return [];

  const codeCol = findColumn(table.header, COL_PATTERNS.code);
  if (codeCol === -1) return [];

  const folded = foldContinuationRows(table, codeCol);
  const nameCol = findColumn(folded.header, COL_PATTERNS.name);
  const causeCol = findColumn(folded.header, COL_PATTERNS.cause);
  const actionCol = findColumn(folded.header, COL_PATTERNS.action);

  const records: FaultRecord[] = [];

  for (const row of folded.rows) {
    const codeCell = (row[codeCol] ?? "").trim();
    if (!codeCell) continue;

    // A cell can carry several codes: "F0022, F0023" or "F0001 / F0002".
    const codes = codeCell
      .split(/[,/;]|\s+or\s+/i)
      .map((c) => c.trim())
      .filter((c) => /[A-Za-z0-9]/.test(c) && c.length <= 12);
    if (codes.length === 0) continue;

    const rawName = nameCol !== -1 ? row[nameCol] ?? "" : "";
    const rawCause = causeCol !== -1 ? row[causeCol] ?? "" : "";
    const rawAction = actionCol !== -1 ? row[actionCol] ?? "" : "";

    // ABB puts "(programmable fault function, parameters 3001 ...)" under the
    // name. Keep it — it's real content — but the first line is the meaning.
    const meaning = (rawName.split("\n")[0] || rawCause.split("\n")[0] || "").trim();
    const causes = splitCauses(rawCause);
    const steps = splitSteps(rawAction);

    const provenance: Provenance = {
      documentId: block.documentId,
      title: documentTitle,
      sectionPath: block.sectionPath,
      pagePdf: block.pagePdf,
      pageLabel: block.pageLabel,
      blockIds: [block.id],
    };

    for (const codeRaw of codes) {
      records.push({
        faultId: `${machineId}::${normalizeCode(codeRaw)}`,
        codeRaw,
        codeNorm: normalizeCode(codeRaw),
        machineId,
        model,
        meaning,
        causes,
        steps,
        severity: severityOf(`${rawName} ${rawCause}`),
        provenance,
        extraction: "table_row",
        confidence: scoreConfidence(meaning, causes, steps),
      });
    }
  }

  return records;
}

// ---------------------------------------------------------------------------
// Extractor 2: per-code sections
// ---------------------------------------------------------------------------

/** "4.1 E101 — Winding Overtemperature", "F0001 OVERCURRENT", "7.1 E101 - Drive Chain Binding" */
const SECTION_CODE_RE =
  /^(?:\d+(?:\.\d+)*\s+)?([A-Z]{1,3}\d{2,5}|[A-Z]{2,4}F)\s*(?:[—–:-]\s*|\s+)(.{2,90})$/;

const CAUSE_HEAD_RE = /^\s*(probable\s+cause|possible\s+cause|cause|reason|origin)s?\s*:?\s*$/i;
const ACTION_HEAD_RE =
  /^\s*(corrective\s+action|what\s+to\s+do|remedy|remedies|solution|action|procedure|troubleshooting)s?\s*:?\s*$/i;

/**
 * Extract a record from a heading block plus the blocks that follow it, up to
 * the next heading at the same or shallower level.
 */
export function extractFromSection(
  heading: Block,
  body: Block[],
  machineId: string,
  model: string | undefined,
  documentTitle: string,
): FaultRecord | null {
  if (heading.kind !== "heading") return null;
  const m = heading.text.trim().match(SECTION_CODE_RE);
  if (!m) return null;

  const [, codeRaw, meaning] = m;

  const causes: string[] = [];
  const steps: StepData[] = [];
  const warnings: string[] = [];
  let mode: "none" | "cause" | "action" = "none";

  for (const b of body) {
    if (b.kind === "admonition") {
      warnings.push(b.admonition?.text ?? b.text);
      continue;
    }

    const line = b.text.trim();
    if (CAUSE_HEAD_RE.test(line)) {
      mode = "cause";
      continue;
    }
    if (ACTION_HEAD_RE.test(line)) {
      mode = "action";
      continue;
    }

    // A step list under an action heading is authoritative — it's already structured.
    if (b.kind === "steps" && b.steps?.length) {
      for (const s of b.steps) steps.push({ n: steps.length + 1, text: s.text, warnings: s.warnings });
      mode = "action";
      continue;
    }

    if (mode === "cause") causes.push(...splitCauses(line));
    else if (mode === "action") {
      for (const s of splitSteps(line)) steps.push({ n: steps.length + 1, text: s.text });
    }
  }

  // Attach warnings to the first step so they can never be retrieved separately.
  if (warnings.length && steps.length) {
    steps[0] = { ...steps[0], warnings: [...(steps[0].warnings ?? []), ...warnings] };
  }

  const provenance: Provenance = {
    documentId: heading.documentId,
    title: documentTitle,
    sectionPath: heading.sectionPath,
    pagePdf: heading.pagePdf,
    pageLabel: heading.pageLabel,
    blockIds: [heading.id, ...body.map((b) => b.id)],
  };

  return {
    faultId: `${machineId}::${normalizeCode(codeRaw)}`,
    codeRaw,
    codeNorm: normalizeCode(codeRaw),
    machineId,
    model,
    // A PDF whose em-dash glyph doesn't extract cleanly (seen in ReportLab-
    // generated PDFs without a proper ToUnicode CMap) leaves a stray "?" or
    // similar junk character where the separator should have been consumed.
    // Strip a leading run of non-alphanumeric characters defensively.
    meaning: meaning.trim().replace(/^[^A-Za-z0-9]+/, "").trim(),
    causes,
    steps,
    severity: warnings.length ? severityOf(warnings.join(" ")) ?? "warning" : undefined,
    provenance,
    extraction: "section",
    confidence: scoreConfidence(meaning, causes, steps),
  };
}

/**
 * How complete is this record? Drives the confidence badge in the UI and lets
 * retrieval prefer a full record over a half-parsed one for the same code.
 */
function scoreConfidence(meaning: string, causes: string[], steps: StepData[]): number {
  let s = 0;
  if (meaning.trim().length > 2) s += 0.4;
  if (causes.length > 0) s += 0.3;
  if (steps.length > 0) s += 0.3;
  return Math.round(s * 100) / 100;
}

// ---------------------------------------------------------------------------
// Document-level driver
// ---------------------------------------------------------------------------

/** Run both extractors over a parsed document's blocks. */
export function extractFaultRecords(
  blocks: Block[],
  opts: { machineId: string; model?: string; documentTitle: string },
): FaultRecord[] {
  const { machineId, model, documentTitle } = opts;
  const out: FaultRecord[] = [];

  for (let i = 0; i < blocks.length; i++) {
    const b = blocks[i];

    if (b.kind === "table") {
      out.push(...extractFromTable(b, machineId, model, documentTitle));
      continue;
    }

    if (b.kind === "heading") {
      // Body = everything until the next heading at the same or shallower level.
      const body: Block[] = [];
      for (let j = i + 1; j < blocks.length; j++) {
        const nb = blocks[j];
        if (nb.kind === "heading" && nb.level <= b.level) break;
        body.push(nb);
      }
      const rec = extractFromSection(b, body, machineId, model, documentTitle);
      if (rec) out.push(rec);
    }
  }

  return dedupeRecords(out);
}

/**
 * The same code can be extracted twice (a summary table plus a detailed
 * section). Keep the richer record, but union the provenance so citations can
 * point at both places.
 */
export function dedupeRecords(records: FaultRecord[]): FaultRecord[] {
  const byId = new Map<string, FaultRecord>();
  for (const r of records) {
    const prev = byId.get(r.faultId);
    if (!prev) {
      byId.set(r.faultId, r);
      continue;
    }
    const winner = r.confidence > prev.confidence ? r : prev;
    const loser = winner === r ? prev : r;
    byId.set(r.faultId, {
      ...winner,
      provenance: {
        ...winner.provenance,
        blockIds: [...new Set([...winner.provenance.blockIds, ...loser.provenance.blockIds])],
      },
    });
  }
  return [...byId.values()];
}
