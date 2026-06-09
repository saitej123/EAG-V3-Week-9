"""Document indexing — unified VLM page pipeline for all supported formats.

Chunking is **heuristic**: ``_chunk_text`` uses a fixed sliding window at
arbitrary word boundaries. Semantic chunking replaces this pattern in a future release.
See ``docs/DEFERRALS.md`` and ``super_browser/catalog.py``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from loguru import logger

from .documents import (
    INDEXABLE_SUFFIXES,
    DocumentExtract,
    PageExtract,
    TEXT_SUFFIXES,
    citation_label,
    extract_artifact_vlm,
    extract_document_vlm,
    is_indexable_document,
    suffix_for_path,
)
from .llm_env import vlm_page_chunk_size
from .paths import SANDBOX

SANDBOX.mkdir(exist_ok=True)

# Heuristic sliding window — semantic chunking planned for a future release.
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 80


def _safe(path: str) -> Path:
    p = (SANDBOX / path).resolve()
    base = SANDBOX.resolve()
    if p != base and base not in p.parents:
        raise ValueError(f"Path '{path}' escapes the sandbox")
    return p


def resolve_fast_text_sidecar(path: str) -> str | None:
    """Return a sandbox-relative ``.md`` sidecar for fast text indexing (skip VLM on PDFs)."""
    p = (path or "").strip().replace("\\", "/")
    if not p or p.startswith("art:") or p.lower().endswith((".md", ".txt")):
        return None

    candidates: list[str] = []
    if p.lower().endswith(".pdf"):
        base = p[:-4]
        candidates.append(f"{base}.md")
        stem = Path(base).name
        if re.match(r"^(\d+\.\d+)v\d+$", stem):
            arxiv_id = re.match(r"^(\d+\.\d+)v\d+$", stem).group(1)  # type: ignore[union-attr]
            parent = str(Path(base).parent).replace("\\", "/")
            prefix = f"{parent}/" if parent and parent != "." else ""
            candidates.append(f"{prefix}{arxiv_id}.md")
            candidates.append(f"research_papers/{arxiv_id}.md")

    seen: set[str] = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        try:
            if _safe(c).is_file():
                return c
        except ValueError:
            continue
    return None


def related_index_paths(path: str) -> set[str]:
    """Paths that share one logical index (PDF ↔ markdown sidecar)."""
    p = (path or "").strip().replace("\\", "/")
    out = {p}
    sidecar = resolve_fast_text_sidecar(p)
    if sidecar:
        out.add(sidecar)
    if p.lower().endswith(".md"):
        pdf = f"{p[:-3]}.pdf"
        try:
            if _safe(pdf).is_file():
                out.add(pdf)
        except ValueError:
            pass
    return out


def paths_share_index(a: str, b: str) -> bool:
    """True when two sandbox paths refer to the same indexed document."""
    a = (a or "").strip().replace("\\", "/")
    b = (b or "").strip().replace("\\", "/")
    if not a or not b:
        return False
    if a == b:
        return True
    related = related_index_paths(a)
    return b in related or a in related_index_paths(b)


def resolve_index_storage_path(requested: str) -> tuple[str, str]:
    """Return ``(storage_path, read_path)`` for chunk metadata vs text/VLM source."""
    requested = (requested or "").strip().replace("\\", "/")
    sidecar = resolve_fast_text_sidecar(requested)
    if sidecar and requested.lower().endswith(".pdf"):
        return requested, sidecar
    if requested.lower().endswith(".md"):
        pdf = f"{requested[:-3]}.pdf"
        try:
            if _safe(pdf).is_file():
                return pdf, requested
        except ValueError:
            pass
    return requested, requested


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunk_size = max(1, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size - 1))
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        next_start = end - overlap
        start = next_start if next_start > start else end
    return chunks


def _read_document_text(path: str) -> tuple[str, str]:
    """UTF-8 text read — used only when ``use_vlm=False`` opt-out."""
    p = path.strip()
    if p.startswith("art:"):
        from .artifact_store import ArtifactStore

        store = ArtifactStore()
        if not store.exists(p):
            raise ValueError(f"Artifact '{p}' not found")
        raw = store.get_bytes(p)
        if not raw:
            raise ValueError(f"Artifact '{p}' is empty or unreadable")
        return raw.decode("utf-8", errors="replace"), "artifact"
    p_obj = _safe(p)
    if not p_obj.is_file():
        raise ValueError(f"File '{path}' does not exist in the sandbox")
    return p_obj.read_text(encoding="utf-8"), "sandbox"


def _page_source_tag(source_kind: str, path: str, page_number: int, page_total: int, sub: int, sub_total: int) -> str:
    cite = citation_label(path, page_number, page_total)
    if sub_total <= 1:
        return f"[{source_kind}:{cite}]"
    return f"[{source_kind}:{cite} part {sub}/{sub_total}]"


def _index_page_chunks(
    *,
    path: str,
    source_kind: str,
    page: PageExtract,
    chunk_size: int,
    overlap: int,
) -> int:
    from .memory import add_fact

    body = (page.text or "").strip()
    if not body:
        return 0

    pieces = _chunk_text(body, chunk_size, overlap) if len(body) > chunk_size else [body]
    total = len(pieces)
    cite = citation_label(path, page.page_number, page.page_total)

    for i, chunk in enumerate(pieces):
        tag = _page_source_tag(source_kind, path, page.page_number, page.page_total, i + 1, total)
        descriptor = f"{tag} {chunk[:200]}"
        add_fact(
            descriptor,
            value={
                "path": path,
                "source_kind": source_kind,
                "extraction": "vlm",
                "page_number": page.page_number,
                "page_total": page.page_total,
                "citation": cite,
                "chunk_index": i,
                "chunk_total": total,
                "text": chunk,
            },
            keywords=[w for w in re.split(r"\W+", path.lower()) if w][:4]
            + [f"page{page.page_number}"]
            + [w for w in re.split(r"\W+", chunk.lower()) if len(w) > 2][:8],
            source=f"mcp:{source_kind}:vlm",
            run_id="mcp-index",
        )
    return total


def _index_text_chunks(
    *,
    path: str,
    source_kind: str,
    text: str,
    chunk_size: int,
    overlap: int,
) -> int:
    from .memory import add_fact

    chunks = _chunk_text(text, chunk_size, overlap)
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        tag = f"[{source_kind}:{path} chunk {i + 1}/{total}]"
        descriptor = f"{tag} {chunk[:200]}"
        add_fact(
            descriptor,
            value={
                "path": path,
                "source_kind": source_kind,
                "extraction": "text",
                "chunk_index": i,
                "chunk_total": total,
                "text": chunk,
            },
            keywords=[w for w in re.split(r"\W+", path.lower()) if w][:6]
            + [w for w in re.split(r"\W+", chunk.lower()) if len(w) > 2][:8],
            source=f"mcp:{source_kind}",
            run_id="mcp-index",
        )
    return total


def _index_vlm_extract(extracted: DocumentExtract, *, source_kind: str, chunk_size: int, overlap: int) -> dict[str, Any]:
    page_chunk = vlm_page_chunk_size()
    path = extracted.path
    if not extracted.pages:
        logger.warning(f"[index] VLM extraction produced no pages for {path}")
        return {"path": path, "extraction": "vlm", "chunks_indexed": 0, "pages_indexed": 0, "page_map": {}}

    total_chunks = 0
    for page in extracted.pages:
        total_chunks += _index_page_chunks(
            path=path,
            source_kind=source_kind,
            page=page,
            chunk_size=page_chunk,
            overlap=min(overlap, page_chunk // 5),
        )
    return {
        "path": path,
        "extraction": "vlm",
        "pages_indexed": len(extracted.pages),
        "chunks_indexed": total_chunks,
        "page_map": extracted.page_map,
    }


def extract_document(path: str, *, use_vlm: bool | None = None) -> DocumentExtract | None:
    """Run unified VLM extraction (default). Returns None only when ``use_vlm=False``."""
    if use_vlm is False:
        return None

    if path.strip().startswith("art:"):
        from .artifact_store import ArtifactStore

        store = ArtifactStore()
        if not store.exists(path):
            raise ValueError(f"Artifact '{path}' not found")
        raw = store.get_bytes(path)
        meta = store.get_meta(path)
        ct = meta.content_type if meta else ""
        return extract_artifact_vlm(path, raw, ct)

    p_obj = _safe(path)
    if not p_obj.is_file():
        raise ValueError(f"File '{path}' does not exist in the sandbox")
    if not is_indexable_document(path):
        raise ValueError(
            f"Unsupported file '{path}'. Supported: {', '.join(sorted(INDEXABLE_SUFFIXES))}"
        )
    return extract_document_vlm(path, p_obj.read_bytes())


def index_document_path(
    path: str,
    chunk_size: int = 400,
    overlap: int = 80,
    *,
    use_vlm: bool | None = None,
) -> dict[str, Any]:
    """Index a sandbox file or artifact into searchable Memory facts."""
    requested = (path or "").strip().replace("\\", "/")
    storage_path, read_path = resolve_index_storage_path(requested)

    try:
        from .catalog import remove_document_index

        remove_document_index(requested)
    except Exception as e:
        logger.warning(f"[index] could not clear prior chunks for {requested}: {e}")

    # Native text formats: read UTF-8 directly — never rasterize or call VLM.
    if suffix_for_path(read_path) in TEXT_SUFFIXES:
        text, source_kind = _read_document_text(read_path)
        n = _index_text_chunks(
            path=storage_path,
            source_kind=source_kind,
            text=text,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        return {
            "path": storage_path,
            "requested_path": requested,
            "read_path": read_path,
            "extraction": "text",
            "chunks_indexed": n,
            "pages_indexed": 0,
        }

    # PDF without sidecar: VLM page pipeline unless explicitly disabled.
    if use_vlm is False:
        raise ValueError(
            f"Cannot text-index '{read_path}' without a .md sidecar — remove use_vlm=false or add a sidecar."
        )

    extracted = extract_document(read_path, use_vlm=True)
    if extracted is not None:
        if storage_path != extracted.path:
            extracted = extracted.model_copy(update={"path": storage_path})
        source_kind = "artifact" if storage_path.strip().startswith("art:") else "sandbox"
        return _index_vlm_extract(extracted, source_kind=source_kind, chunk_size=chunk_size, overlap=overlap)

    text, source_kind = _read_document_text(read_path)
    n = _index_text_chunks(
        path=storage_path, source_kind=source_kind, text=text, chunk_size=chunk_size, overlap=overlap
    )
    return {
        "path": storage_path,
        "requested_path": requested,
        "read_path": read_path,
        "extraction": "text",
        "chunks_indexed": n,
        "pages_indexed": 0,
    }


def index_directory(path: str = "rag_corpus", chunk_size: int = 400, overlap: int = 80) -> dict:
    """Bulk-index all supported documents under a sandbox directory (recursive, VLM pipeline)."""
    base = _safe(path)
    if not base.is_dir():
        raise ValueError(f"Directory '{path}' does not exist in the sandbox")

    files: set[Path] = set()
    for suffix in INDEXABLE_SUFFIXES:
        files.update(base.rglob(f"*{suffix}"))
    files_sorted = sorted(files)

    if not files_sorted:
        return {"directory": path, "files_indexed": 0, "chunks_indexed": 0, "files": []}

    files_to_index = [
        f
        for f in files_sorted
        if not (f.suffix.lower() == ".md" and f.with_suffix(".pdf").is_file())
    ]

    rel_root = SANDBOX.resolve()
    per_file: list[dict] = []
    total_chunks = 0
    for f in files_to_index:
        rel = str(f.resolve().relative_to(rel_root)).replace("\\", "/")
        result = index_document_path(rel, chunk_size=chunk_size, overlap=overlap)
        n = int(result.get("chunks_indexed", 0))
        total_chunks += n
        per_file.append(
            {
                "path": rel,
                "chunks_indexed": n,
                "pages_indexed": result.get("pages_indexed", 0),
                "extraction": result.get("extraction", "vlm"),
            }
        )
    return {
        "directory": path,
        "files_indexed": len(per_file),
        "chunks_indexed": total_chunks,
        "files": per_file,
    }
