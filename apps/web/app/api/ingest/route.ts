/**
 * POST /api/ingest — upload a PDF manual and index it.
 *
 * Flow:  PDF → parser → typed blocks → structure-aware chunks
 *                                   → fault records (code/meaning/cause/action)
 *                                   → Jina embeddings → local index
 *
 * The parser is currently the local FastAPI doc-processor. It is isolated behind
 * `parseDocument()` so swapping in LlamaParse (which returns markdown tables,
 * and therefore lights up the fault-table extractor) is a change to one function.
 */
import { NextRequest } from "next/server";
import { createHash } from "node:crypto";

import { buildBlocks } from "@timmo/rag/doc/blocks";
import { chunkBlocks } from "@timmo/rag/doc/chunker";
import { extractFaultRecords } from "@timmo/rag/doc/faults";
import type { PageInput, OutlineInput } from "@timmo/rag/doc/blocks";
import { getStore, getEmbedder } from "@/lib/rag-index";
import { setProgress } from "@/lib/ingest-progress";
import { getEmbedCache } from "@/lib/rag-index";
import { embedWithCache } from "@timmo/rag/store/embed-cache";
import { savePdf } from "@/lib/pdf-store";

export const runtime = "nodejs";
// A 170-page manual takes minutes to parse + embed; don't let the platform cut it off.
export const maxDuration = 800;

const PYTHON_URL = process.env.DOC_PROCESSOR_URL ?? "http://127.0.0.1:8080";

/** Known machines, so chunks can be filtered by machine rather than by filename substring. */
const MACHINE_RULES: { id: string; model: string; manufacturer: string; test: RegExp }[] = [
  { id: "abb-acs150", model: "ACS150", manufacturer: "ABB", test: /acs\s*-?150/i },
  { id: "abb-acs350", model: "ACS350", manufacturer: "ABB", test: /acs\s*-?350/i },
  { id: "abb-irb4600", model: "IRB 4600", manufacturer: "ABB", test: /irb\s*-?4600|3hac033453/i },
  { id: "schneider-atv320", model: "ATV320", manufacturer: "Schneider", test: /atv\s*-?320/i },
  { id: "schneider-atv28", model: "ATV28", manufacturer: "Schneider", test: /atv\s*-?28/i },
  { id: "roboinject-300", model: "RoboInject-300", manufacturer: "Synthetic", test: /roboinject/i },
  { id: "press-2000", model: "Press-2000", manufacturer: "Synthetic", test: /press-?2000/i },
  { id: "press-2001", model: "Press-2001", manufacturer: "Synthetic", test: /press-?2001/i },
  { id: "powerflex-525", model: "PowerFlex-525", manufacturer: "Rockwell", test: /powerflex/i },
];

function detectMachine(filename: string, firstPages: string) {
  // Filename first: it's a deliberate, reliable signal. Page text is not --
  // a manual can legitimately cross-reference a DIFFERENT machine's error
  // code in its own text (e.g. "not the same as E101 on the X-300"), which
  // would otherwise mis-tag the whole document as that other machine and
  // silently defeat cross-document ambiguity detection (two machines that
  // both reference each other end up sharing one machineId, so neither
  // looks ambiguous against the other anymore).
  for (const rule of MACHINE_RULES) {
    if (rule.test.test(filename)) return rule;
  }
  for (const rule of MACHINE_RULES) {
    if (rule.test.test(firstPages.slice(0, 4000))) return rule;
  }
  const slug = filename
    .replace(/\.pdf$/i, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 40);
  return { id: slug || "unknown", model: filename.replace(/\.pdf$/i, ""), manufacturer: "", test: /$^/ };
}

/** Parser adapter. Swap the body for LlamaParse when the key is available. */
async function parseDocument(
  file: File,
  documentId: string,
  useOcr: boolean = false,
): Promise<{ title: string; pages: PageInput[]; outline: OutlineInput[] }> {
  const form = new FormData();
  form.set("file", file);
  form.set("document_id", documentId);
  form.set("include_images", "true");
  form.set("use_ocr", useOcr ? "true" : "false");

  const res = await fetch(`${PYTHON_URL}/parse`, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`Parser failed (${res.status}): ${detail.slice(0, 300)}`);
  }

  const data = await res.json();
  const pages: PageInput[] = (data.pages ?? []).map((p: Record<string, unknown>) => ({
    page: Number(p.page),
    text: String(p.text ?? ""),
    images: Array.isArray(p.images) ? (p.images as string[]) : [],
  }));
  // The PDF's own bookmark tree — exact section titles and pages. Preferred over
  // inferring headings from the text, which produced section labels like
  // "4 = 380…480 V AC" on ABB manuals.
  const outline: OutlineInput[] = (data.outline ?? []).map((o: Record<string, unknown>) => ({
    title: String(o.title ?? ""),
    pagePdf: Number(o.page_pdf ?? 0),
    level: Number(o.level ?? 0),
  }));
  return { title: String(data.title ?? file.name), pages, outline };
}

export async function POST(request: NextRequest) {
  const started = Date.now();
  let jobIdForError = "";
  try {
    const form = await request.formData();
    const file = form.get("file");
    if (!file || !(file instanceof File)) {
      return Response.json({ error: "file is required (multipart form field 'file')" }, { status: 400 });
    }
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      return Response.json({ error: "Only PDF files are supported." }, { status: 400 });
    }

    const bytes = Buffer.from(await file.arrayBuffer());
    const sha256 = createHash("sha256").update(bytes).digest("hex");
    const documentId =
      (form.get("document_id") as string) || file.name.replace(/\.pdf$/i, "").replace(/\s+/g, "-");
    const useOcr = form.get("use_ocr") === "true";
    // Client-supplied id so the browser can poll real progress while this
    // single long request is in flight.
    const jobId = (form.get("job_id") as string) || "";
    jobIdForError = jobId;

    // Idempotency: the exact same file, already indexed, is a no-op.
    // Re-uploading a manual to retry something used to redo minutes of work.
    const existing = getStore().listDocuments().find((d) => d.sha256 === sha256);
    if (existing) {
      setProgress(jobId, "done", 100, "Already indexed");
      return Response.json({
        document_id: existing.documentId,
        title: existing.title,
        machine_id: existing.machineId,
        model: existing.model,
        pages: existing.pageCount,
        chunks: existing.chunkCount,
        faults: existing.faultCount,
        fault_codes: [],
        dims: 0,
        low_text_pages: [],
        unchanged: true,
        elapsed_ms: Date.now() - started,
        indexed_at: existing.indexedAt,
      });
    }

    // 1. Parse
    setProgress(jobId, "parsing", 5, "Extracting text and diagrams…");
    const { title, pages, outline } = await parseDocument(file, documentId, useOcr);
    if (!pages.length) {
      return Response.json({ error: "No extractable text found in the PDF." }, { status: 422 });
    }

    // 2. Identify the machine so retrieval can filter on it
    const machine = detectMachine(file.name, pages.slice(0, 3).map((p) => p.text).join("\n"));

    // OCR was applied during parsing if requested; low-text pages after OCR
    // are genuinely blank/scanned pages that couldn't be read.
    const lowTextPages = pages.filter((p) => p.text.trim().length < 40).map((p) => p.page);

    // 3. Blocks → chunks → fault records
    setProgress(jobId, "chunking", 20, `${pages.length} pages parsed`);
    const blocks = buildBlocks(pages, { documentId, outline });
    const chunks = chunkBlocks(blocks, {
      machineId: machine.id,
      machineLabel: machine.model,
      model: machine.model,
      documentTitle: title,
    });
    const faults = extractFaultRecords(blocks, {
      machineId: machine.id,
      model: machine.model,
      documentTitle: title,
    });

    if (!chunks.length) {
      return Response.json({ error: "Parsed the PDF but produced no chunks." }, { status: 422 });
    }

    // 4. Embed (batched — one request per 64 chunks, not one per chunk)
    const embedder = getEmbedder();
    setProgress(jobId, "embedding", 25, `0/${chunks.length} chunks embedded`);
    // Cache + duplicate collapsing before anything hits the API. Embedding is
    // ~90-95% of ingest wall time and is rate-limited per minute, so the only
    // real lever is embedding fewer tokens. Both savings are lossless.
    const { vectors, cacheHits, embedded } = await embedWithCache(
      chunks.map((c) => c.text),
      getEmbedCache(embedder.dims),
      (inputs, onProgress) => embedder.embedMany(inputs, onProgress),
      // Real progress across the slowest phase by far -- 25% to 90%.
      (done, total) =>
        setProgress(jobId, "embedding", 25 + (done / total) * 65, `${done}/${total} chunks embedded`),
    );

    // 5. Index
    setProgress(jobId, "indexing", 92, "Writing to index…");
    const store = getStore();
    store.addDocument(
      {
        documentId,
        title,
        machineId: machine.id,
        model: machine.model,
        pageCount: pages.length,
        chunkCount: chunks.length,
        faultCount: faults.length,
        sha256,
        indexedAt: new Date().toISOString(),
      },
      chunks,
      vectors,
      faults,
    );

    // Keep the source PDF so a citation can be rendered and checked against
    // the real page. Written after indexing succeeds, so a failed ingest never
    // leaves an orphaned file behind.
    savePdf(documentId, bytes);

    setProgress(jobId, "done", 100, "Indexed");

    return Response.json({
      document_id: documentId,
      title,
      machine_id: machine.id,
      model: machine.model,
      pages: pages.length,
      chunks: chunks.length,
      faults: faults.length,
      fault_codes: [...new Set(faults.map((f) => f.codeRaw))].slice(0, 40),
      dims: vectors[0]?.length ?? 0,
      cache_hits: cacheHits,
      embedded_chunks: embedded,
      // Pages with near-zero extractable text -- likely scanned/image-only.
      // No OCR pipeline yet, so content on these pages was not indexed.
      low_text_pages: lowTextPages,
      elapsed_ms: Date.now() - started,
      indexed_at: new Date().toISOString(),
    });
  } catch (err) {
    console.error("/api/ingest error:", err);
    const message = err instanceof Error ? err.message : "Internal server error";
    setProgress(jobIdForError, "error", 100, message.slice(0, 200));
    return Response.json({ error: message, elapsed_ms: Date.now() - started }, { status: 500 });
  }
}
