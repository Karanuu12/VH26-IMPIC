"""Timmo document processor.

Handles the heavy document-processing stage of the RAG pipeline:
PDF text extraction, page/section metadata, and heading-aware chunking.
Exposed as a small FastAPI service that Next.js calls during ingestion.
"""
from __future__ import annotations

import re
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import anyio

from .chunking import chunk_pages
from .pdf import extract_outline, extract_pdf_pages

# Page rendering needs PyMuPDF directly; the flag mirrors pdf.py so the
# endpoint can fail with a clear message instead of a NameError.
try:
    import pymupdf  # type: ignore[import-not-found]

    _MUPDF = True
except Exception:  # pragma: no cover
    _MUPDF = False
    pymupdf = None  # type: ignore[assignment]

app = FastAPI(
    title="Timmo Document Processor",
    version="0.1.0",
    description="PDF parsing, OCR, and chunking micro-service.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class Page(BaseModel):
    page: int = Field(ge=1)
    section: str = ""
    text: str = ""
    images: list[str] = Field(default_factory=list)


class OutlineEntry(BaseModel):
    title: str
    page_pdf: int
    level: int = 0


class ParseResponse(BaseModel):
    document_id: str
    title: str = ""
    total_pages: int = 0
    pages: list[Page] = []
    outline: list[OutlineEntry] = []


class ChunkRequest(BaseModel):
    document_id: str
    title: str = ""
    pages: list[Page]
    max_chars: int = Field(default=1800, ge=200, le=8000)
    overlap: int = Field(default=120, ge=0, le=800)


class Chunk(BaseModel):
    id: str
    document_id: str
    title: str
    page: int
    section: str
    text: str
    char_count: int = 0
    images: list[str] = Field(default_factory=list)


class ChunkResponse(BaseModel):
    chunks: list[Chunk]


class HealthResponse(BaseModel):
    status: str
    service: str
    ready: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="document-processor", ready=True)


@app.post("/parse", response_model=ParseResponse)
async def parse_pdf(
    file: UploadFile = File(...),
    document_id: str = Form(...),
    include_images: bool = Form(False),
    use_ocr: bool = Form(False),
) -> ParseResponse:
    """Extract text + page/section metadata from an uploaded PDF.

    `include_images` defaults to False on purpose: images come back as base64
    JPEGs, and on a 170+ page manual that turns a few megabytes of text into a
    several-hundred-megabyte JSON response. Callers that actually want figures
    opt in per request.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        pages = extract_pdf_pages(raw, include_images=include_images, use_ocr=use_ocr)
    except Exception as exc:  # noqa: BLE001 - surface any parse failure to caller
        raise HTTPException(status_code=422, detail=f"Could not parse PDF: {exc}") from exc

    if not pages:
        raise HTTPException(status_code=422, detail="No extractable text found in PDF.")

    return ParseResponse(
        document_id=document_id,
        title=file.filename,
        total_pages=len(pages),
        pages=pages,
        outline=extract_outline(raw),
    )


def _highlight_candidates(passage: str, limit: int = 40) -> list[str]:
    """Break a retrieved passage into strings worth searching for on the page.

    The passage almost never matches the page verbatim: the chunker prefixes a
    breadcrumb, joins lines, and normalises whitespace, none of which exist in
    the PDF's own text layer. Searching sentence by sentence recovers most of
    it anyway -- whatever is found gets highlighted, whatever isn't simply
    doesn't, which degrades to "no highlight" rather than to a wrong one.

    Fragments are filtered on two axes rather than length alone. A pure length
    floor was too blunt: it threw away real headings like "OCF Overcurrent"
    (15 chars) so a chunk containing only a heading highlighted nothing, while
    still admitting nothing useful. Requiring at least two words drops the
    single-word labels that would match half the page ("Remedy", "Note") and
    keeps the short-but-specific ones.
    """
    parts: list[str] = []
    for line in passage.replace("\r", "\n").split("\n"):
        for sentence in re.split(r"(?<=[.;:!?])\s+", line):
            # Markdown table pipes and heading hashes are chunker artefacts.
            s = " ".join(sentence.split()).strip(" -•*|#")
            if len(s) >= 12 and len(s.split()) >= 2:
                parts.append(s)
    # Longest first: the most specific strings are the least likely to
    # false-positive, and the cap keeps a big chunk from costing 200 searches.
    parts.sort(key=len, reverse=True)
    return parts[:limit]


@app.post("/page-image")
async def page_image(
    file: UploadFile = File(...),
    page: int = Form(...),
    dpi: int = Form(110),
    highlight: str = Form(""),
) -> Response:
    """Render one page of a PDF to PNG, so a citation can be checked against
    the actual manual rather than taken on trust.

    When `highlight` is given (the retrieved passage the answer was built
    from), the text is located on the page and marked, so the reader sees the
    exact sentences behind the claim instead of hunting a dense A4 page for
    them. Highlighting is best-effort by design: a passage that cannot be
    located renders a clean page rather than a misleading mark.

    Rendered on demand rather than at ingest: a 460-page manual would otherwise
    cost 460 images to store for the handful of pages anyone ever cites.

    `page` is 1-based, matching what the citation displays. dpi is capped
    because this is a screen preview, not a print job -- 110dpi is legible on a
    laptop and keeps a page under ~200KB.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")
    if not _MUPDF:
        raise HTTPException(status_code=501, detail="PyMuPDF is required to render pages.")

    dpi = max(40, min(dpi, 200))
    try:
        with pymupdf.open(stream=raw, filetype="pdf") as doc:
            if page < 1 or page > doc.page_count:
                raise HTTPException(
                    status_code=404,
                    detail=f"Page {page} out of range (document has {doc.page_count}).",
                )
            pdf_page = doc.load_page(page - 1)

            if highlight.strip():
                seen: set[tuple[int, int, int, int]] = set()
                quads: list[Any] = []
                for needle in _highlight_candidates(highlight):
                    try:
                        found = pdf_page.search_for(needle, quads=True)
                    except Exception:  # noqa: BLE001 - a bad needle must not fail the render
                        continue
                    for quad in found:
                        # Sentences overlap after splitting; dedupe by rounded
                        # rect so the same line isn't highlighted repeatedly
                        # (stacked annots render visibly darker).
                        r = quad.rect
                        key = (round(r.x0), round(r.y0), round(r.x1), round(r.y1))
                        if key in seen:
                            continue
                        seen.add(key)
                        quads.append(quad)
                if quads:
                    annot = pdf_page.add_highlight_annot(quads)
                    annot.set_colors(stroke=(1.0, 0.86, 0.35))
                    annot.update()

            pixmap = pdf_page.get_pixmap(dpi=dpi)
            png = pixmap.tobytes("png")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - surface any render failure to caller
        raise HTTPException(status_code=422, detail=f"Could not render page: {exc}") from exc

    return Response(content=png, media_type="image/png")


@app.post("/chunk", response_model=ChunkResponse)
async def chunk(request: ChunkRequest) -> ChunkResponse:
    """Split a parsed document into semantic, heading-aware chunks."""
    with anyio.fail_after(15):
        return ChunkResponse(chunks=chunk_pages(request, max_chars=request.max_chars, overlap=request.overlap))


@app.post("/process", response_model=ChunkResponse)
async def process_pdf(
    file: UploadFile = File(...),
    document_id: str = Form(...),
    max_chars: int = Form(1800),
    overlap: int = Form(120),
    use_ocr: bool = Form(False),
) -> ChunkResponse:
    """One-call convenience: parse then chunk a PDF."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    raw = await file.read()
    try:
        pages = extract_pdf_pages(raw, use_ocr=use_ocr)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Could not parse PDF: {exc}") from exc

    if not pages:
        raise HTTPException(status_code=422, detail="No extractable text found in PDF.")

    parsed = ParseResponse(
        document_id=document_id,
        title=file.filename,
        total_pages=len(pages),
        pages=pages,
    )
    req = ChunkRequest(
        document_id=document_id,
        title=file.filename,
        pages=pages,
        max_chars=max_chars,
        overlap=overlap,
    )
    with anyio.fail_after(30):
        return ChunkResponse(chunks=chunk_pages(req, max_chars=max_chars, overlap=overlap))


def build_app() -> FastAPI:
    return app


# Re-export for FX/tooling that introspects via `fitz`/PdfReader.
__all__: list[str] = ["app", "build_app", "Page", "Chunk", "ParseResponse", "ChunkResponse"]