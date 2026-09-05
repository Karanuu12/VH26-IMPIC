/**
 * Markdown table → structure.
 *
 * Hosted parsers (LlamaParse, Unstructured, Mistral OCR) all emit tables as
 * GitHub-flavoured markdown. We keep the markdown for the LLM but also recover
 * header + rows, because the fault extractor needs real cells: the whole point
 * is that "CAUSE" and "WHAT TO DO" stay in different columns.
 */
import type { TableData } from "./model.ts";

const SEPARATOR_RE = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/;

/** Split one markdown table row into cells, honouring escaped pipes. */
function splitRow(line: string): string[] {
  const cells: string[] = [];
  let cur = "";
  let escaped = false;
  const trimmed = line.trim().replace(/^\|/, "").replace(/\|$/, "");
  for (const ch of trimmed) {
    if (escaped) {
      cur += ch;
      escaped = false;
    } else if (ch === "\\") {
      escaped = true;
    } else if (ch === "|") {
      cells.push(cur);
      cur = "";
    } else {
      cur += ch;
    }
  }
  cells.push(cur);
  return cells.map(normalizeCell);
}

/**
 * Collapse whitespace and undo the line-break encodings parsers use inside cells.
 * A multi-line "WHAT TO DO" cell arrives as "<br>"-joined or "\n"-joined text;
 * we keep the newlines as real newlines so step splitting still works later.
 */
function normalizeCell(s: string): string {
  return s
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/[ \t]+/g, " ")
    .replace(/\n{2,}/g, "\n")
    .trim();
}

export function isTableLine(line: string): boolean {
  return line.includes("|") && line.trim().length > 0;
}

/**
 * Parse a markdown table block. Returns null when the text isn't a table.
 *
 * Tolerates: missing leading/trailing pipes, ragged row lengths, and a missing
 * separator row (some parsers omit it) — in which case row 0 is still treated as
 * the header if it looks like one.
 */
export function parseMarkdownTable(markdown: string): TableData | null {
  const lines = markdown
    .split("\n")
    .map((l) => l.trimEnd())
    .filter((l) => l.trim().length > 0);
  if (lines.length < 2) return null;

  const tableLines = lines.filter(isTableLine);
  if (tableLines.length < 2) return null;

  const sepIndex = tableLines.findIndex((l) => SEPARATOR_RE.test(l));
  let header: string[];
  let bodyLines: string[];

  if (sepIndex === 1) {
    header = splitRow(tableLines[0]);
    bodyLines = tableLines.slice(2);
  } else if (sepIndex === -1) {
    header = splitRow(tableLines[0]);
    bodyLines = tableLines.slice(1);
  } else {
    // Separator in an unexpected place — treat everything before it as header junk.
    header = splitRow(tableLines[sepIndex - 1] ?? tableLines[0]);
    bodyLines = tableLines.slice(sepIndex + 1);
  }

  const width = header.length;
  if (width < 2) return null;

  const rows = bodyLines
    .filter((l) => !SEPARATOR_RE.test(l))
    .map((l) => {
      const cells = splitRow(l);
      // Pad short rows, merge overflow into the last column rather than dropping it.
      if (cells.length < width) {
        return [...cells, ...Array(width - cells.length).fill("")];
      }
      if (cells.length > width) {
        return [...cells.slice(0, width - 1), cells.slice(width - 1).join(" | ")];
      }
      return cells;
    })
    .filter((r) => r.some((c) => c.length > 0));

  if (rows.length === 0) return null;

  return { header, rows, markdown: renderMarkdown(header, rows) };
}

/** Re-render canonical markdown so what we embed matches what we parsed. */
export function renderMarkdown(header: string[], rows: string[][]): string {
  const esc = (s: string) => s.replace(/\|/g, "\\|").replace(/\n/g, "<br>");
  const out = [
    `| ${header.map(esc).join(" | ")} |`,
    `| ${header.map(() => "---").join(" | ")} |`,
    ...rows.map((r) => `| ${r.map(esc).join(" | ")} |`),
  ];
  return out.join("\n");
}

/**
 * Column lookup by fuzzy header name.
 * ABB uses CODE/FAULT/CAUSE/WHAT TO DO; Schneider uses
 * "Error code"/"Name"/"Probable Cause"/"Remedy". Both must hit.
 */
export function findColumn(header: string[], candidates: RegExp[]): number {
  for (const re of candidates) {
    const i = header.findIndex((h) => re.test(h));
    if (i !== -1) return i;
  }
  return -1;
}

export const COL_PATTERNS = {
  code: [/^\s*(code|error\s*code|fault\s*code|alarm\s*code|no\.?|number)\s*$/i, /\bcode\b/i],
  name: [/^\s*(fault|alarm|name|error|description|message|display)\s*$/i, /\b(fault|alarm|name)\b/i],
  cause: [/\b(cause|reason|origin|probable\s*cause)\b/i],
  action: [
    /\b(what\s*to\s*do|remedy|corrective|action|solution|fix|check|clearing)\b/i,
  ],
};

/**
 * Does this table look like a fault/error-code table?
 * Requires a code-ish column plus at least one of cause/action — that pairing is
 * what makes a row convertible into a FaultRecord.
 */
export function isFaultTable(header: string[]): boolean {
  const code = findColumn(header, COL_PATTERNS.code);
  if (code === -1) return false;
  const cause = findColumn(header, COL_PATTERNS.cause);
  const action = findColumn(header, COL_PATTERNS.action);
  const name = findColumn(header, COL_PATTERNS.name);
  return cause !== -1 || action !== -1 || name !== -1;
}

/**
 * Stitch a table that continues onto the next page.
 *
 * Vendor manuals repeat the header row after a page break (ABB does this in the
 * fault-tracing chapter). If two consecutive table blocks share a header, they
 * are one logical table and must not become two chunks — splitting them is how
 * you end up citing a page that doesn't contain the row you quoted.
 */
export function sameTable(a: TableData, b: TableData): boolean {
  if (a.header.length !== b.header.length) return false;
  return a.header.every((h, i) => h.toLowerCase() === b.header[i].toLowerCase());
}

export function mergeTables(a: TableData, b: TableData, pages: number[]): TableData {
  const rows = [...a.rows, ...b.rows];
  return {
    header: a.header,
    rows,
    markdown: renderMarkdown(a.header, rows),
    spansPages: [...new Set([...(a.spansPages ?? []), ...pages])].sort((x, y) => x - y),
  };
}

/**
 * Rows whose code cell is empty are continuations of the row above — a wrapped
 * cell that the parser emitted as its own row. Fold them back up, otherwise the
 * fault extractor produces a record with no code and loses the text.
 */
export function foldContinuationRows(table: TableData, codeCol: number): TableData {
  const rows: string[][] = [];
  for (const row of table.rows) {
    const code = (row[codeCol] ?? "").trim();
    if (code === "" && rows.length > 0) {
      const prev = rows[rows.length - 1];
      for (let i = 0; i < row.length; i++) {
        if (row[i]) prev[i] = prev[i] ? `${prev[i]}\n${row[i]}` : row[i];
      }
    } else {
      rows.push([...row]);
    }
  }
  return { ...table, rows, markdown: renderMarkdown(table.header, rows) };
}
