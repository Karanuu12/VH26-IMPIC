/**
 * GET /api/page?doc=<document_id>&page=<n> — the cited page, rendered as PNG.
 *
 * This is what turns a citation from a claim into something checkable: the
 * answer says "page 412", and this serves page 412 of the actual manual so the
 * reader can see the sentence for themselves.
 *
 * Rendered on demand via the Python service (PyMuPDF), then cached in memory:
 * people re-open the same handful of cited pages, and re-rendering a page of a
 * 460-page manual on every click is wasteful. The cache is small and bounded
 * because these are ~200KB PNGs.
 */
import { NextRequest } from "next/server";
import { createHash } from "node:crypto";
import { readPdf } from "@/lib/pdf-store";

export const dynamic = "force-dynamic";

const PYTHON_URL = process.env.DOC_PROCESSOR_URL ?? "http://127.0.0.1:8080";

const CACHE_MAX = 24;
declare global {
  // eslint-disable-next-line no-var
  var __faultfinderPageCache: Map<string, Buffer> | undefined;
}
function cache(): Map<string, Buffer> {
  if (!globalThis.__faultfinderPageCache) globalThis.__faultfinderPageCache = new Map();
  return globalThis.__faultfinderPageCache;
}

export async function GET(request: NextRequest) {
  const documentId = request.nextUrl.searchParams.get("doc");
  const pageRaw = request.nextUrl.searchParams.get("page");
  if (!documentId || !pageRaw) {
    return Response.json({ error: "doc and page query params required" }, { status: 400 });
  }
  const page = Number(pageRaw);
  if (!Number.isInteger(page) || page < 1) {
    return Response.json({ error: "page must be a positive integer" }, { status: 400 });
  }
  // The passage to mark on the page. Part of the cache key: the same page
  // cited by two different answers highlights different sentences.
  const highlight = (request.nextUrl.searchParams.get("q") ?? "").slice(0, 1500);
  const cacheKey = `${documentId}:${page}:${
    highlight ? createHash("sha1").update(highlight).digest("hex").slice(0, 12) : "plain"
  }`;
  const cached = cache().get(cacheKey);
  if (cached) {
    return new Response(new Uint8Array(cached), {
      headers: { "Content-Type": "image/png", "Cache-Control": "private, max-age=3600" },
    });
  }

  const pdf = readPdf(documentId);
  if (!pdf) {
    // Manuals indexed before page retention existed have no stored PDF. That
    // is a re-upload away from working, so say so rather than 500.
    return Response.json(
      { error: "No stored PDF for this manual. Re-upload it to enable page preview." },
      { status: 404 },
    );
  }

  const form = new FormData();
  form.set("file", new Blob([new Uint8Array(pdf)], { type: "application/pdf" }), `${documentId}.pdf`);
  form.set("page", String(page));
  if (highlight) form.set("highlight", highlight);

  let res: Response;
  try {
    res = await fetch(`${PYTHON_URL}/page-image`, { method: "POST", body: form });
  } catch {
    return Response.json(
      { error: `Document processor unreachable at ${PYTHON_URL}. Is it running?` },
      { status: 502 },
    );
  }
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    return Response.json({ error: `Could not render page ${page}: ${detail.slice(0, 200)}` }, { status: res.status });
  }

  const png = Buffer.from(await res.arrayBuffer());
  const store = cache();
  if (store.size >= CACHE_MAX) {
    const oldest = store.keys().next().value;
    if (oldest !== undefined) store.delete(oldest);
  }
  store.set(cacheKey, png);

  return new Response(new Uint8Array(png), {
    headers: { "Content-Type": "image/png", "Cache-Control": "private, max-age=3600" },
  });
}
