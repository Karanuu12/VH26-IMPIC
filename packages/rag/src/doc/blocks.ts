/**
 * Page text → typed blocks.
 *
 * The parser adapter gives us text per page (plus markdown tables, once
 * LlamaParse is wired in). This module recovers structure from it: headings,
 * markdown tables, numbered procedures, and warning boxes. Everything else
 * becomes a paragraph.
 *
 * Deliberately conservative. The original implementation called any short line
 * containing "error" or "cause" a heading, which meant body text in a fault
 * manual constantly reset the section — so sections were wrong everywhere.
 * Here a heading must be *numbered* or match a known structural label, or come
 * from the PDF outline (which is exact, and preferred when available).
 */
import type { Block, BlockKind, StepData, AdmonitionSeverity } from "./model.ts";
import { extractUnits } from "./model.ts";
import { parseMarkdownTable, isTableLine } from "./tables.ts";

export interface PageInput {
  /** 1-based physical page index. */
  page: number;
  text: string;
  /** Printed page label, when the parser recovered one. Defaults to the index. */
  pageLabel?: string;
  images?: string[];
}

export interface OutlineInput {
  title: string;
  pagePdf: number;
  level: number;
}

export interface BlockBuildOptions {
  documentId: string;
  /** PDF outline. When present it drives sectionPath exactly — no guessing. */
  outline?: OutlineInput[];
}

/** "4.1 Fault tracing", "7.1.2 Something" — numbering is the reliable signal. */
const NUMBERED_HEADING = /^(\d+(?:\.\d+){0,3})[.)]?\s+(\S.{0,88})$/;

/** Structural labels that are headings even without numbering. */
const LABEL_HEADING =
  /^(what this chapter contains|table of contents|fault tracing|diagnostics|troubleshooting|maintenance|safety|introduction|overview|probable causes?|corrective actions?|what to do|remedy|symptom)\s*:?\s*$/i;

const ADMONITION = /^(DANGER|WARNING|CAUTION|NOTICE|IMPORTANT|NOTE)\b[:!.\s-]*(.*)$/i;

const STEP_LINE = /^\s*(\d{1,2})[.)]\s+(\S.*)$/;

/**
 * "Figure 7-3 Drive dimensions", "Fig. 4-1: Wiring diagram", "Diagram 2 — Torque sequence".
 * Deliberately excludes "Table" — markdown tables are already parsed as their own
 * block kind, so a plain-text "Table N" line here would just be noise.
 */
const FIGURE_CAPTION_RE =
  /^(Fig(?:ure)?|Diagram|Schematic)\.?\s*(\d+(?:[.\-]\d+)*)\s*[:.\-–—]?\s*(.*)$/i;

function severityFrom(word: string): AdmonitionSeverity {
  const w = word.toLowerCase();
  if (w === "danger") return "danger";
  if (w === "warning") return "warning";
  if (w === "caution") return "caution";
  return "notice";
}

/**
 * Section path from the PDF outline: the deepest bookmark at or before this page.
 * Exact, and free — every manual in this corpus ships one.
 */
function outlinePathFor(outline: OutlineInput[] | undefined, page: number): string[] {
  if (!outline?.length) return [];
  const stack: string[] = [];
  for (const entry of outline) {
    if (entry.pagePdf > page) break;
    stack.length = Math.max(0, entry.level);
    stack[entry.level] = entry.title.trim();
  }
  return stack.filter(Boolean);
}

/**
 * Running headers/footers ("© Copyright 2004-2019 ABB. All rights reserved.",
 * "3HAC033453-001 Revision: AL") repeat on nearly every page. Measured on real
 * manuals they are 4-7% of ALL extracted text, which is 4-7% of the embedding
 * bill, and they pollute every chunk they land in with boilerplate that
 * competes with real content during retrieval.
 *
 * Detected by frequency rather than position: a normalized line (digits masked,
 * so page numbers and revision numbers collapse together) appearing on more
 * than `threshold` of pages is furniture. Only runs on documents long enough
 * for the signal to mean something -- on a 3-page manual a line appearing
 * twice is not boilerplate.
 */
function detectFurniture(pages: PageInput[], threshold = 0.6): Set<string> {
  const furniture = new Set<string>();
  if (pages.length < 8) return furniture;

  const counts = new Map<string, number>();
  for (const p of pages) {
    // Count each distinct line once per page; a line repeated within one page
    // is a formatting artifact, not evidence of being a running header.
    const seen = new Set<string>();
    for (const raw of (p.text ?? "").split("\n")) {
      const line = raw.trim();
      if (!line || line.length > 100) continue;
      const norm = normalizeFurniture(line);
      if (seen.has(norm)) continue;
      seen.add(norm);
      counts.set(norm, (counts.get(norm) ?? 0) + 1);
    }
  }

  const min = Math.ceil(pages.length * threshold);
  for (const [norm, n] of counts) {
    if (n >= min) furniture.add(norm);
  }
  return furniture;
}

/** Digits masked so "Page 12" and "Page 13" collapse to the same line. */
function normalizeFurniture(line: string): string {
  return line.replace(/\d+/g, "#").replace(/\s+/g, " ").trim();
}

export function buildBlocks(pages: PageInput[], opts: BlockBuildOptions): Block[] {
  const blocks: Block[] = [];
  let seq = 0;
  // Section path derived from in-page headings, used when there's no outline.
  let headingPath: string[] = [];
  const furniture = detectFurniture(pages);

  const push = (
    kind: BlockKind,
    text: string,
    page: PageInput,
    extra: Partial<Block> = {},
  ): void => {
    const trimmed = text.trim();
    if (!trimmed) return;
    const outlinePath = outlinePathFor(opts.outline, page.page);
    blocks.push({
      id: `${opts.documentId}-b${String(seq++).padStart(5, "0")}`,
      documentId: opts.documentId,
      kind,
      text: trimmed,
      pagePdf: page.page,
      pageLabel: page.pageLabel ?? String(page.page),
      sectionPath: outlinePath.length ? outlinePath : [...headingPath],
      level: extra.level ?? headingPath.length,
      textSource: "digital",
      units: extractUnits(trimmed),
      ...extra,
    });
  };

  for (const page of pages) {
    const lines = (page.text ?? "").split("\n");
    let paragraph: string[] = [];
    let tableLines: string[] = [];
    let steps: StepData[] = [];
    let stepPreamble = "";
    // Figure/diagram captions found in this page's text, in reading order —
    // paired with page.images by position below. Best-effort: PyMuPDF gives us
    // images per page with no bbox, so we can't match a caption to a specific
    // image directly, but pairing in order is right far more often than not
    // (a page rarely has more than one uncaptioned figure).
    const pageCaptions: { label: string; caption: string }[] = [];

    const flushParagraph = () => {
      if (paragraph.length) {
        push("para", paragraph.join("\n"), page);
        paragraph = [];
      }
    };

    const flushTable = () => {
      if (tableLines.length >= 2) {
        const table = parseMarkdownTable(tableLines.join("\n"));
        if (table) {
          push("table", table.markdown, page, { table });
        } else {
          push("para", tableLines.join("\n"), page);
        }
      } else if (tableLines.length) {
        paragraph.push(...tableLines);
      }
      tableLines = [];
    };

    const flushSteps = () => {
      if (steps.length) {
        push("steps", stepPreamble || "Procedure:", page, { steps: [...steps] });
        steps = [];
        stepPreamble = "";
      }
    };

    const flushAll = () => {
      flushTable();
      flushSteps();
      flushParagraph();
    };

    for (const rawLine of lines) {
      const line = rawLine.trimEnd();
      const trimmed = line.trim();

      if (!trimmed) {
        flushTable();
        flushSteps();
        flushParagraph();
        continue;
      }

      // Running header/footer -- drop before it reaches any block.
      if (furniture.size && furniture.has(normalizeFurniture(trimmed))) continue;

      // Record figure/diagram captions as we pass them, but don't consume the
      // line — it still flows into paragraph/heading handling below like any
      // other text.
      if (trimmed.length <= 140) {
        const fig = trimmed.match(FIGURE_CAPTION_RE);
        if (fig) {
          pageCaptions.push({ label: `${fig[1]} ${fig[2]}`, caption: fig[3]?.trim() ?? "" });
        }
      }

      // Markdown table rows accumulate until the run ends.
      if (isTableLine(trimmed) && trimmed.split("|").length >= 3) {
        flushSteps();
        flushParagraph();
        tableLines.push(trimmed);
        continue;
      }
      if (tableLines.length) flushTable();

      // Headings
      const numbered = trimmed.match(NUMBERED_HEADING);
      // A bare single-level number ("3.", "4)") also matches a numbered LIST
      // ITEM under "Probable causes:" / "Corrective action:" -- "3. Relief
      // valve stuck partially open..." is prose, not a heading, and treating
      // it as one both corrupts the section hierarchy for everything after
      // it AND steals the line from the step-list detection below, which is
      // where it actually belongs. A real heading here is short and doesn't
      // end mid-sentence; a list item is a full, period-terminated sentence.
      const looksLikeListItem =
        !!numbered && /[.!?]$/.test(numbered[2]) && numbered[2].length > 40;
      if (numbered && !looksLikeListItem && trimmed.length <= 90) {
        flushAll();
        const level = numbered[1].split(".").length - 1;
        headingPath = [...headingPath.slice(0, level), trimmed];
        push("heading", trimmed, page, { level });
        continue;
      }
      if (LABEL_HEADING.test(trimmed)) {
        flushAll();
        // Nest under the CURRENT deepest heading ("3.1 E101 -- Low Hydraulic
        // Pressure"), not just the top-level ancestor ("3. ERROR CODE
        // REFERENCE"). Truncating to depth 1 here was discarding exactly the
        // code/section context that makes "Corrective action:" retrievable
        // by the error code it belongs to -- without it, a "what's the fix"
        // query scores this chunk no better than unrelated page content,
        // because neither its breadcrumb nor its own text mentions the code.
        headingPath = [...headingPath, trimmed].slice(-6);
        push("heading", trimmed, page, { level: Math.max(0, headingPath.length - 1) });
        continue;
      }

      // Warning / caution boxes
      const adm = trimmed.match(ADMONITION);
      if (adm) {
        flushAll();
        const severity = severityFrom(adm[1]);
        const body = adm[2]?.trim() || adm[1];
        push("admonition", body, page, { admonition: { severity, text: body } });
        continue;
      }

      // Numbered procedure steps
      const step = trimmed.match(STEP_LINE);
      if (step) {
        flushTable();
        if (!steps.length) {
          // The line before the first step is the procedure's preamble.
          stepPreamble = paragraph.length ? paragraph[paragraph.length - 1] : "";
          if (paragraph.length) {
            paragraph.pop();
            flushParagraph();
          }
        }
        steps.push({ n: Number(step[1]), text: step[2].trim() });
        continue;
      }
      if (steps.length) {
        // A non-step line ends the procedure, unless it's an obvious continuation.
        if (/^[a-z(]/.test(trimmed) && steps.length) {
          steps[steps.length - 1].text += ` ${trimmed}`;
          continue;
        }
        flushSteps();
      }

      paragraph.push(trimmed);
    }

    flushAll();

    // Figures: one block per page image, paired in order with any figure/diagram
    // captions found on the page. A captioned figure is retrievable by its
    // caption text (dense + lexical search can't index pixels); an uncaptioned
    // one still carries its page and section, which is enough for it to surface
    // alongside the surrounding text that cites it.
    (page.images ?? []).forEach((href, i) => {
      const cap = pageCaptions[i];
      push("figure", cap ? `[Figure] ${cap.label}: ${cap.caption}` : `[Figure p${page.page}-${i + 1}]`, page, {
        figure: {
          figureId: `${opts.documentId}-p${page.page}-f${i + 1}`,
          href,
          label: cap?.label,
          caption: cap?.caption,
        },
      });
    });
  }

  return blocks;
}
