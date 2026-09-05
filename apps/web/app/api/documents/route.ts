/** DELETE /api/documents?id=<document_id> — remove one manual and its chunks/faults. */
import { NextRequest } from "next/server";
import { getStore } from "@/lib/rag-index";
import { deletePdf } from "@/lib/pdf-store";

export const dynamic = "force-dynamic";

export async function DELETE(request: NextRequest) {
  const id = request.nextUrl.searchParams.get("id");
  if (!id) return Response.json({ error: "id query param required" }, { status: 400 });

  const store = getStore();
  const existed = store.listDocuments().some((d) => d.documentId === id);
  if (!existed) return Response.json({ error: "document not found" }, { status: 404 });

  store.deleteDocument(id);
  store.save();
  // Delete the retained source PDF too — leaving it behind would keep the
  // user's document on disk after they asked for it to be removed.
  deletePdf(id);
  return Response.json({ deleted: id });
}
