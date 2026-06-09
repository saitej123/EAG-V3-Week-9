"""Async wrappers for document indexing — keeps the FastAPI event loop responsive (SSE logs)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from loguru import logger

from .documents import INDEXABLE_SUFFIXES
from .indexing import index_directory, index_document_path
from .llm_retry import index_file_sleep_seconds, llm_retry_max_attempts, llm_retry_sleep_seconds


async def index_document_async(
    path: str,
    *,
    chunk_size: int = 400,
    overlap: int = 80,
    use_vlm: bool | None = None,
) -> dict[str, Any]:
    """Run ``index_document_path`` in a worker thread (non-blocking for SSE/UI)."""
    return await asyncio.to_thread(
        index_document_path,
        path,
        chunk_size=chunk_size,
        overlap=overlap,
        use_vlm=use_vlm,
    )


async def index_document_with_retry_async(
    path: str,
    *,
    use_vlm: bool | None = None,
    max_attempts: int | None = None,
) -> dict[str, Any]:
    """Index one file with sync retry + backoff (does not abort the whole bulk job)."""
    attempts = max_attempts if max_attempts is not None else llm_retry_max_attempts()
    sleep_sec = llm_retry_sleep_seconds()
    last: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            return await index_document_async(path, use_vlm=use_vlm)
        except Exception as e:
            last = e
            if attempt >= attempts:
                break
            wait = sleep_sec * attempt
            logger.warning(f"[index] {path} attempt {attempt}/{attempts} failed: {e} — retry in {wait:.1f}s")
            await asyncio.sleep(wait)

    assert last is not None
    raise last


async def index_directory_async(
    path: str = "rag_corpus",
    *,
    chunk_size: int = 400,
    overlap: int = 80,
) -> dict[str, Any]:
    return await asyncio.to_thread(index_directory, path, chunk_size, overlap)


async def _index_one_file(
    rel: str,
    *,
    use_vlm: bool | None,
    log_prefix: str,
    index: int,
    total: int,
) -> dict[str, Any]:
    logger.info(f"{log_prefix} Indexing {index}/{total}: {rel}")
    try:
        row = await index_document_with_retry_async(rel, use_vlm=use_vlm)
        chunks = int(row.get("chunks_indexed", 0))
        pause = index_file_sleep_seconds()
        if pause > 0 and index < total:
            await asyncio.sleep(pause)
        return {"path": rel, "chunks_indexed": chunks, "status": "ok", **row}
    except Exception as e:
        logger.error(f"{log_prefix} Failed {rel} after retries: {e}")
        return {"path": rel, "chunks_indexed": 0, "status": "error", "error": str(e)}


async def bulk_index_sidecars_async(
    corpus_path: str,
    corpus_dir: Path,
    *,
    use_vlm: bool | None = False,
    log_prefix: str = "[UI]",
) -> dict[str, Any]:
    """Index ``*.md`` sidecars; failures are logged and skipped (bulk continues)."""
    md_files = sorted(corpus_dir.glob("*.md"))
    if not md_files:
        return {"directory": corpus_path, "files_indexed": 0, "chunks_indexed": 0, "files": [], "errors": 0}

    per_file: list[dict] = []
    total_chunks = 0
    errors = 0
    n = len(md_files)

    for i, f in enumerate(md_files, start=1):
        rel = f"{corpus_path}/{f.name}"
        row = await _index_one_file(rel, use_vlm=use_vlm, log_prefix=log_prefix, index=i, total=n)
        if row.get("status") == "error":
            errors += 1
        else:
            total_chunks += int(row.get("chunks_indexed", 0))
        per_file.append(row)

    ok_files = sum(1 for r in per_file if r.get("status") != "error")
    return {
        "directory": corpus_path,
        "files_indexed": ok_files,
        "chunks_indexed": total_chunks,
        "files": per_file,
        "errors": errors,
    }


async def bulk_index_tree_async(
    directory: str,
    corpus_dir: Path,
    *,
    use_vlm: bool | None = False,
    log_prefix: str = "[UI]",
) -> dict[str, Any]:
    """Index all supported files under a directory; continue after per-file failures."""
    if use_vlm:
        return await index_directory_async(directory)

    files: list[Path] = []
    for suffix in INDEXABLE_SUFFIXES:
        files.extend(corpus_dir.rglob(f"*{suffix}"))
    files_sorted = sorted(set(files))
    files_to_index = [
        f
        for f in files_sorted
        if not (f.suffix.lower() == ".md" and f.with_suffix(".pdf").is_file())
    ]
    if not files_to_index:
        return {"directory": directory, "files_indexed": 0, "chunks_indexed": 0, "files": [], "errors": 0}

    sandbox = corpus_dir.parent.resolve()
    per_file: list[dict] = []
    total_chunks = 0
    errors = 0
    n = len(files_to_index)

    for i, f in enumerate(files_to_index, start=1):
        rel = str(f.resolve().relative_to(sandbox)).replace("\\", "/")
        row = await _index_one_file(rel, use_vlm=use_vlm, log_prefix=log_prefix, index=i, total=n)
        if row.get("status") == "error":
            errors += 1
        else:
            total_chunks += int(row.get("chunks_indexed", 0))
        per_file.append(row)

    ok_files = sum(1 for r in per_file if r.get("status") != "error")
    return {
        "directory": directory,
        "files_indexed": ok_files,
        "chunks_indexed": total_chunks,
        "files": per_file,
        "errors": errors,
    }
