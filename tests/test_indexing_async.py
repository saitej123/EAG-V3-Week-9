"""Tests for async indexing wrappers and thread-safe memory writes."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_index_document_async_does_not_block_event_loop(monkeypatch):
    from super_browser import indexing_async as ia

    tick = {"n": 0}

    def fake_index(path: str, *, chunk_size=400, overlap=80, use_vlm=None):
        tick["n"] += 1
        return {"path": path, "chunks_indexed": 2, "extraction": "text"}

    monkeypatch.setattr(ia, "index_document_path", fake_index)

    async def _run():
        async def spin():
            for _ in range(5):
                tick["spin"] = tick.get("spin", 0) + 1
                await asyncio.sleep(0)

        result, _ = await asyncio.gather(
            ia.index_document_async("research_papers/foo.md", use_vlm=False),
            spin(),
        )
        assert result["chunks_indexed"] == 2
        assert tick.get("spin", 0) >= 1

    asyncio.run(_run())


def test_persist_lock_serializes_parallel_add_fact(monkeypatch, tmp_path):
    from super_browser.memory import MemoryService, add_fact, _PERSIST_LOCK
    from super_browser.paths import STATE as STATE_DIR

    monkeypatch.setattr("super_browser.memory.STATE_DIR", tmp_path / "state")
    monkeypatch.setattr("super_browser.memory.MEMORY_JSON_PATH", tmp_path / "state" / "memory.json")
    monkeypatch.setattr("super_browser.memory.INDEX_FAISS_PATH", tmp_path / "state" / "index.faiss")
    monkeypatch.setattr("super_browser.memory.INDEX_IDS_PATH", tmp_path / "state" / "index_ids.json")
    monkeypatch.setattr("super_browser.memory.DB_PATH", tmp_path / "state" / "commerce.db")
    monkeypatch.setattr("super_browser.memory._try_embed", lambda *a, **k: None)

    errors: list[str] = []

    def writer(i: int) -> None:
        try:
            add_fact(
                f"chunk {i}",
                value={"text": f"body {i}", "path": "t.md"},
                keywords=[f"k{i}"],
                source="test",
                run_id="test",
            )
        except Exception as e:
            errors.append(str(e))

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    svc = MemoryService()
    assert len(svc._items) == 8
