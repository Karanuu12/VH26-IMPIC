/** GET /api/chat/progress?id=<job_id> — real per-stage trace for a running query. */
import { NextRequest } from "next/server";
import { getTrace } from "@/lib/query-progress";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  const id = request.nextUrl.searchParams.get("id");
  if (!id) return Response.json({ error: "id query param required" }, { status: 400 });
  const trace = getTrace(id);
  // An unknown id is not an error: the client can poll before the POST has
  // been handled, and a trace that has already been pruned is simply gone.
  if (!trace) return Response.json({ steps: [], done: false });
  return Response.json({ steps: trace.steps, done: trace.done });
}
