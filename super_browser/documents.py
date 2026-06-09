"""Document pipeline: normalize to PDF, VLM page extraction, and indexing helpers."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Self

from loguru import logger
from pydantic import BaseModel, Field, model_validator

from .llm_env import gemini_models_ordered, shared_gemini_client, vlm_index_dpi_scale, vlm_index_max_pages
from .llm_retry import (
    generate_content_with_retry,
    vlm_batch_sleep_seconds,
    vlm_page_batch_size,
)

# --- Format detection & PDF normalization ---

PDF_SUFFIXES = frozenset({".pdf"})
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif"})
TEXT_SUFFIXES = frozenset({".md", ".txt", ".html", ".htm", ".csv", ".json", ".xml", ".log", ".rst"})
OFFICE_SUFFIXES = frozenset(
    {
        ".doc",
        ".docx",
        ".ppt",
        ".pptx",
        ".pps",
        ".ppsx",
        ".odt",
        ".ods",
        ".odp",
        ".rtf",
        ".xls",
        ".xlsx",
    }
)
INDEXABLE_SUFFIXES = PDF_SUFFIXES | IMAGE_SUFFIXES | TEXT_SUFFIXES | OFFICE_SUFFIXES

_CONTENT_TYPE_SUFFIX = {
    "application/pdf": ".pdf",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "text/html": ".html",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/msword": ".doc",
    "application/vnd.ms-powerpoint": ".ppt",
}


def suffix_for_path(path: str) -> str:
    return Path(str(path)).suffix.lower()


def is_indexable_document(path: str | Path) -> bool:
    return suffix_for_path(str(path)) in INDEXABLE_SUFFIXES


def suffix_from_content_type(content_type: str) -> str:
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct in _CONTENT_TYPE_SUFFIX:
        return _CONTENT_TYPE_SUFFIX[ct]
    if ct.startswith("text/"):
        return ".txt"
    if ct.startswith("image/"):
        ext = ct.split("/", 1)[-1]
        return f".{ext}" if ext else ".png"
    return ""


def _find_libreoffice() -> str | None:
    for cmd in ("libreoffice", "soffice", "/usr/bin/libreoffice", "/usr/bin/soffice"):
        if shutil.which(cmd):
            return cmd
    return None


def _strip_html(raw: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw)
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</p\s*>", "\n\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _text_for_pdf(raw: bytes, suffix: str) -> str:
    text = raw.decode("utf-8", errors="replace")
    if suffix in {".html", ".htm"}:
        return _strip_html(text)
    return text


def _paginate_plain_text(body: str, *, chars_per_page: int = 3500) -> list[str]:
    pages: list[str] = []
    start = 0
    n = len(body)
    while start < n:
        end = min(n, start + chars_per_page)
        if end < n:
            split = body.rfind("\n", start, end)
            if split <= start:
                split = body.rfind(" ", start, end)
            if split > start:
                end = split
        chunk = body[start:end].strip()
        if chunk:
            pages.append(chunk)
        start = max(start + 1, end) if end <= start else end
    return pages or [body.strip()] if body.strip() else []


def _text_bytes_to_pdf(raw: bytes, suffix: str, *, title: str) -> bytes:
    import fitz  # pymupdf

    body = _text_for_pdf(raw, suffix)
    if not body.strip():
        raise ValueError(f"Empty text document ({title})")

    doc = fitz.open()
    rect = fitz.Rect(36, 36, 559, 806)
    try:
        for chunk in _paginate_plain_text(body):
            page = doc.new_page(width=595, height=842)
            page.insert_textbox(
                rect,
                chunk,
                fontsize=10,
                fontname="helv",
                align=fitz.TEXT_ALIGN_LEFT,
            )
        return doc.tobytes()
    finally:
        doc.close()


def _image_bytes_to_pdf(raw: bytes, suffix: str) -> bytes:
    import fitz  # pymupdf

    ft = suffix.lstrip(".") or "png"
    img_doc = fitz.open(stream=raw, filetype=ft)
    try:
        return img_doc.convert_to_pdf()
    finally:
        img_doc.close()


def _libreoffice_to_pdf(src_name: str, raw: bytes) -> bytes:
    cmd = _find_libreoffice()
    if not cmd:
        raise RuntimeError(
            f"Cannot convert {Path(src_name).suffix} to PDF — install LibreOffice "
            "(``libreoffice`` or ``soffice`` on PATH) for Office documents."
        )
    with tempfile.TemporaryDirectory(prefix="cog-rag-convert-") as td:
        td_path = Path(td)
        in_file = td_path / Path(src_name).name
        in_file.write_bytes(raw)
        proc = subprocess.run(
            [cmd, "--headless", "--convert-to", "pdf", "--outdir", str(td_path), str(in_file)],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"LibreOffice conversion failed for {src_name}: {err or proc.returncode}")

        pdf_path = td_path / f"{in_file.stem}.pdf"
        if not pdf_path.is_file():
            pdfs = sorted(td_path.glob("*.pdf"))
            if not pdfs:
                raise RuntimeError(f"LibreOffice produced no PDF for {src_name}")
            pdf_path = pdfs[0]
        return pdf_path.read_bytes()


def normalize_to_pdf(path: str, raw: bytes) -> bytes:
    """Convert any supported document bytes to a PDF suitable for page rasterization."""
    suffix = suffix_for_path(path)
    if suffix in PDF_SUFFIXES:
        return raw
    if suffix in IMAGE_SUFFIXES:
        return _image_bytes_to_pdf(raw, suffix)
    if suffix in TEXT_SUFFIXES:
        return _text_bytes_to_pdf(raw, suffix, title=path)
    if suffix in OFFICE_SUFFIXES:
        return _libreoffice_to_pdf(path, raw)
    raise ValueError(
        f"Unsupported document type '{suffix or '(no extension)'}' for {path}. "
        f"Supported: {', '.join(sorted(INDEXABLE_SUFFIXES))}"
    )


def normalize_artifact_to_pdf(artifact_id: str, raw: bytes, content_type: str = "") -> tuple[bytes, str]:
    """Return ``(pdf_bytes, logical_path)`` for an ``art:`` blob."""
    suffix = suffix_from_content_type(content_type)
    logical = f"{artifact_id}{suffix or '.bin'}"
    if suffix and is_indexable_document(logical):
        return normalize_to_pdf(logical, raw), logical
    try:
        text = raw.decode("utf-8")
        if text.strip():
            return _text_bytes_to_pdf(raw, ".txt", title=artifact_id), f"{artifact_id}.txt"
    except UnicodeDecodeError:
        pass
    raise ValueError(
        f"Artifact '{artifact_id}' is not a supported document type "
        f"(content_type={content_type!r}). Index PDF, Office, image, or text artifacts."
    )


# --- VLM page extraction ---


class PageExtract(BaseModel):
    page_number: int
    page_total: int
    text: str
    mime_type: str = "image/png"


class DocumentExtract(BaseModel):
    path: str
    extraction: str = "vlm"
    pages: list[PageExtract] = Field(default_factory=list)
    page_map: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _fill_page_map(self) -> Self:
        if not self.page_map and self.pages:
            self.page_map = {f"page_{p.page_number}": p.text for p in self.pages}
        return self


def citation_label(path: str, page_number: int, page_total: int) -> str:
    return f"{path} p.{page_number}/{page_total}"


def vlm_page_extraction_prompt(*, page_number: int, page_total: int, path: str) -> str:
    """Structured prompt for high-fidelity page extraction including visual form elements."""
    return f"""You are indexing **page {page_number} of {page_total}** from `{path}` for a searchable knowledge base with accurate citations.

Extract **all** semantic content from this page image. Use clean markdown. Do not skip small print, headers, footers, watermarks, or slide numbers.

## Required coverage

### 1. Text & structure
- Transcribe headings, paragraphs, captions, footnotes, equations (LaTeX), URLs, labels, and code blocks faithfully.
- Preserve hierarchy (`#`, `##`, bullets, numbered lists).

### 2. Tables
- Render as markdown tables with headers.
- Note merged cells, units, and footnotes.

### 3. Checkboxes, tick marks, and form fields
- Record selection state explicitly on its own line or inline:
  - `[x]` or `[✓]` — checked / ticked / selected / filled
  - `[ ]` — unchecked / empty
  - `[/]` — partial / indeterminate when visible
  - For radio groups: note which option is selected.
  - For text fields: quote visible placeholder or entered value.

### 4. Figures, photos, plots, and diagrams
For **each** visual (chart, graph, photo, architecture diagram, screenshot, icon strip):
```
### Figure: <caption or best-guess title>
- **Type:** bar chart | line plot | scatter | pie | diagram | photo | …
- **Summary:** 2–4 sentences on what it shows and why it matters.
- **Labels/legend:** axes, units, series names, categories.
- **Key data:** trends, comparisons, peaks, outliers, numeric callouts visible on the chart.
- **Takeaway:** the main claim or relationship the figure supports.
```
If no caption exists, infer a short descriptive title from content.

### 5. Slide decks
- Capture slide title, bullet hierarchy, diagram text, and speaker notes if visible.

## Rules
- **Do not invent** content not visible on the page.
- Mark illegible regions as `[illegible]`.
- Output **only** the page extraction — no preamble, no "Here is…", no meta commentary about the task."""


def _pdf_page_images(raw: bytes, *, max_pages: int, dpi_scale: float | None = None) -> list[tuple[bytes, str]]:
    import fitz  # pymupdf

    scale = dpi_scale if dpi_scale is not None else vlm_index_dpi_scale()
    doc = fitz.open(stream=raw, filetype="pdf")
    try:
        n = min(len(doc), max_pages)
        out: list[tuple[bytes, str]] = []
        matrix = fitz.Matrix(scale, scale)
        for i in range(n):
            pix = doc.load_page(i).get_pixmap(matrix=matrix, alpha=False)
            out.append((pix.tobytes("png"), "image/png"))
        return out
    finally:
        doc.close()


def _vlm_extract_page(image_bytes: bytes, mime_type: str, *, page_number: int, page_total: int, path: str) -> str:
    if shared_gemini_client() is None:
        raise RuntimeError("Gemini client unavailable — set GEMINI_API_KEY for VLM document indexing.")

    from google.genai import types

    prompt = vlm_page_extraction_prompt(page_number=page_number, page_total=page_total, path=path)
    models = gemini_models_ordered()
    if not models:
        raise RuntimeError("No GEMINI_MODEL configured for VLM extraction.")

    last_err: Exception | None = None
    for model_id in models:
        try:
            response = generate_content_with_retry(
                model=model_id,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    prompt,
                ],
                config=types.GenerateContentConfig(temperature=0.1),
                label=f"vlm-index:p{page_number}",
            )
            text = (response.text or "").strip()
            if text:
                return text
            raise RuntimeError("empty VLM response")
        except Exception as e:
            last_err = e
            logger.warning(f"[vlm-index] page {page_number} model {model_id} failed: {e}")
    raise RuntimeError(f"VLM page extraction failed for {path} p.{page_number}: {last_err}")


def _extract_page_safe(
    i: int,
    img_bytes: bytes,
    mime: str,
    *,
    total: int,
    path: str,
) -> PageExtract:
    """Extract one page; on failure return a placeholder so indexing continues."""
    try:
        logger.info(f"[vlm-index] extracting {path} page {i}/{total}")
        text = _vlm_extract_page(img_bytes, mime, page_number=i, page_total=total, path=path)
        return PageExtract(page_number=i, page_total=total, text=text, mime_type=mime)
    except Exception as e:
        logger.error(f"[vlm-index] page {i}/{total} failed after retries: {e}")
        placeholder = f"[VLM extraction failed for page {i}/{total}: {e}]"
        return PageExtract(page_number=i, page_total=total, text=placeholder, mime_type=mime)


def _extract_pages_in_batches(
    path: str,
    page_images: list[tuple[bytes, str]],
    *,
    batch_size: int | None = None,
    batch_sleep: float | None = None,
) -> list[PageExtract]:
    """Process VLM pages in batches (default 10) with sleep between batches."""
    total = len(page_images)
    if total == 0:
        return []

    size = batch_size if batch_size is not None else vlm_page_batch_size()
    pause = batch_sleep if batch_sleep is not None else vlm_batch_sleep_seconds()
    pages: list[PageExtract] = []

    for batch_start in range(0, total, size):
        batch = page_images[batch_start : batch_start + size]
        batch_end = batch_start + len(batch)
        logger.info(f"[vlm-index] batch {batch_start + 1}-{batch_end}/{total} for {path}")

        workers = min(len(batch), size)
        if workers <= 1:
            for offset, (img_bytes, mime) in enumerate(batch):
                pages.append(
                    _extract_page_safe(
                        batch_start + offset + 1,
                        img_bytes,
                        mime,
                        total=total,
                        path=path,
                    )
                )
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(
                        _extract_page_safe,
                        batch_start + offset + 1,
                        img_bytes,
                        mime,
                        total=total,
                        path=path,
                    )
                    for offset, (img_bytes, mime) in enumerate(batch)
                ]
                for fut in as_completed(futures):
                    pages.append(fut.result())

        if batch_end < total and pause > 0:
            logger.debug(f"[vlm-index] sleeping {pause:.1f}s before next batch")
            time.sleep(pause)

    pages.sort(key=lambda p: p.page_number)
    return pages


def extract_document_vlm(
    path: str,
    raw: bytes,
    *,
    max_pages: int | None = None,
    page_workers: int | None = None,
) -> DocumentExtract:
    """Any supported document → PDF → page images → Gemini VLM per page (batched, retry-safe)."""
    if not is_indexable_document(path):
        raise ValueError(
            f"Unsupported document '{path}'. Supported suffixes: {', '.join(sorted(INDEXABLE_SUFFIXES))}"
        )

    limit = max_pages if max_pages is not None else vlm_index_max_pages()
    logger.info(f"[vlm-index] normalizing {path} to PDF")
    pdf_bytes = normalize_to_pdf(path, raw)
    page_images = _pdf_page_images(pdf_bytes, max_pages=limit)

    if not page_images:
        return DocumentExtract(path=path, extraction="vlm", pages=[], page_map={})

    batch_size = page_workers if page_workers is not None else vlm_page_batch_size()
    pages = _extract_pages_in_batches(path, page_images, batch_size=batch_size)
    page_map = {f"page_{p.page_number}": p.text for p in pages}
    return DocumentExtract(path=path, extraction="vlm", pages=pages, page_map=page_map)


def extract_artifact_vlm(artifact_id: str, raw: bytes, content_type: str = "", *, max_pages: int | None = None) -> DocumentExtract:
    """VLM pipeline for ``art:`` blobs (uses content-type to pick converter)."""
    pdf_bytes, logical_path = normalize_artifact_to_pdf(artifact_id, raw, content_type)
    limit = max_pages if max_pages is not None else vlm_index_max_pages()
    page_images = _pdf_page_images(pdf_bytes, max_pages=limit)
    if not page_images:
        return DocumentExtract(path=logical_path, extraction="vlm", pages=[], page_map={})

    pages = _extract_pages_in_batches(logical_path, page_images)
    page_map = {f"page_{p.page_number}": p.text for p in pages}
    return DocumentExtract(path=logical_path, extraction="vlm", pages=pages, page_map=page_map)
