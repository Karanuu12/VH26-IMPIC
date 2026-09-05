/**
 * Keeps the uploaded PDF alongside the index so a citation can be verified
 * against the actual page.
 *
 * Ingest previously read the file into a Buffer, sent it to the parser and
 * dropped it, which meant a citation was only ever a claim: "page 412" with
 * nothing behind it. Retaining the bytes is what lets the UI render the real
 * page next to the answer -- the difference between asking someone to trust
 * the citation and letting them check it.
 *
 * Stored as files rather than inside index.json deliberately: the index is
 * already 28MB and is parsed in full on every cold start, and a manual is tens
 * of megabytes. Pages are rendered on demand instead of at ingest, so a
 * 460-page manual costs one file, not 460 images.
 *
 * Same stopgap status as LocalStore: this is a local-disk implementation
 * behind a narrow interface, so object storage is a swap of these three
 * functions when the app is deployed.
 */
import { mkdirSync, readFileSync, writeFileSync, existsSync, rmSync } from "node:fs";
import { join } from "node:path";

const ROOT = join(process.cwd(), "..", "..", ".data", "pdfs");

/** Document ids come from filenames, so they must never escape the store. */
function safePath(documentId: string): string | undefined {
  if (!documentId || documentId.includes("/") || documentId.includes("\\") || documentId.includes("..")) {
    return undefined;
  }
  return join(ROOT, `${documentId}.pdf`);
}

export function savePdf(documentId: string, bytes: Buffer): void {
  const path = safePath(documentId);
  if (!path) return;
  mkdirSync(ROOT, { recursive: true });
  writeFileSync(path, bytes);
}

export function readPdf(documentId: string): Buffer | undefined {
  const path = safePath(documentId);
  if (!path || !existsSync(path)) return undefined;
  return readFileSync(path);
}

export function deletePdf(documentId: string): void {
  const path = safePath(documentId);
  if (!path || !existsSync(path)) return;
  rmSync(path, { force: true });
}

export function hasPdf(documentId: string): boolean {
  const path = safePath(documentId);
  return !!path && existsSync(path);
}
