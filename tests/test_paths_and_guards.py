"""Sandbox path validation and heuristic follow-up after index."""

from __future__ import annotations

import pytest

from super_browser.paths import SANDBOX, resolve_sandbox_subdir
from super_browser.schemas import Goal, MemoryItem
from super_browser.search_providers import heuristic_tool_call


def test_resolve_sandbox_subdir_rejects_traversal():
    with pytest.raises(ValueError, match="Invalid"):
        resolve_sandbox_subdir("../")
    with pytest.raises(ValueError, match="Invalid"):
        resolve_sandbox_subdir("papers/../../etc")


def test_resolve_sandbox_subdir_accepts_papers():
    if not (SANDBOX / "papers").is_dir():
        pytest.skip("sandbox/papers not present")
    resolved = resolve_sandbox_subdir("papers")
    assert resolved.resolve() == (SANDBOX / "papers").resolve()


def test_heuristic_after_index_uses_search_knowledge():
    goal = Goal(id="g1", text="Read and index papers/attention.md", done=False)
    history = [
        {
            "kind": "action",
            "tool": "index_document",
            "arguments": {"path": "papers/attention.md"},
        }
    ]
    tc = heuristic_tool_call(
        goal=goal,
        user_query=(
            "Index the file papers/attention.md and tell me what the three key "
            "contributions of the Transformer architecture are according to this paper."
        ),
        hits=[],
        history=history,
    )
    assert tc is not None
    assert tc.name == "search_knowledge"


def test_paths_share_index_pdf_sidecar(tmp_path, monkeypatch):
    from super_browser import indexing

    rp = tmp_path / "research_papers"
    rp.mkdir(parents=True)
    (rp / "2605.23904.pdf").write_bytes(b"%PDF")
    (rp / "2605.23904.md").write_text("sidecar", encoding="utf-8")
    monkeypatch.setattr(indexing, "SANDBOX", tmp_path)

    assert indexing.paths_share_index(
        "research_papers/2605.23904.pdf", "research_papers/2605.23904.md"
    )


def test_catalog_stats_pdf_alias(tmp_path, monkeypatch):
    from super_browser import catalog, indexing, paths
    from super_browser.catalog import IndexedDocumentRecord, _index_record_for_path

    rp = tmp_path / "research_papers"
    rp.mkdir(parents=True)
    (rp / "2605.23904.pdf").write_bytes(b"%PDF")
    (rp / "2605.23904.md").write_text("sidecar", encoding="utf-8")
    monkeypatch.setattr(paths, "SANDBOX", tmp_path)
    monkeypatch.setattr(indexing, "SANDBOX", tmp_path)
    monkeypatch.setattr(catalog, "SANDBOX", tmp_path)

    rec = IndexedDocumentRecord(
        path="research_papers/2605.23904.pdf",
        chunk_count=6,
        page_count=None,
        extraction="text",
        preview="sidecar",
        citation_sample=None,
        first_indexed_at=None,
        last_indexed_at=None,
    )
    index_map = {rec.path: rec}
    assert _index_record_for_path("research_papers/2605.23904.md", index_map) is rec
    assert _index_record_for_path("research_papers/2605.23904.pdf", index_map) is rec

    legacy = IndexedDocumentRecord(
        path="research_papers/2605.23904.md",
        chunk_count=3,
        page_count=None,
        extraction="text",
        preview="legacy",
        citation_sample=None,
        first_indexed_at=None,
        last_indexed_at=None,
    )
    legacy_map = {legacy.path: legacy}
    assert _index_record_for_path("research_papers/2605.23904.pdf", legacy_map) is legacy


def test_enrich_index_document_keeps_pdf_path_with_sidecar(tmp_path, monkeypatch):
    from super_browser import indexing, paths
    from super_browser.schemas import Goal, ToolCall
    from super_browser.search_providers import enrich_tool_call

    rp = tmp_path / "research_papers"
    rp.mkdir(parents=True)
    (rp / "2605.23904.pdf").write_bytes(b"%PDF")
    (rp / "2605.23904.md").write_text("sidecar", encoding="utf-8")
    monkeypatch.setattr(paths, "SANDBOX", tmp_path)
    monkeypatch.setattr(indexing, "SANDBOX", tmp_path)

    goal = Goal(id="g", text="Index PDF", done=False)
    tc = enrich_tool_call(
        ToolCall(name="index_document", arguments={"path": "research_papers/2605.23904.pdf"}),
        goal=goal,
        user_query="Index research_papers/2605.23904.pdf",
    )
    assert tc.arguments["path"] == "research_papers/2605.23904.pdf"
    assert tc.arguments.get("use_vlm") is False


def test_heuristic_pdf_indexed_via_storage_path(tmp_path, monkeypatch):
    from super_browser import indexing, paths

    rp = tmp_path / "research_papers"
    rp.mkdir(parents=True)
    (rp / "2605.23904.pdf").write_bytes(b"%PDF")
    (rp / "2605.23904.md").write_text("sidecar", encoding="utf-8")
    monkeypatch.setattr(paths, "SANDBOX", tmp_path)
    monkeypatch.setattr(indexing, "SANDBOX", tmp_path)

    goal = Goal(id="g3", text="Answer from PDF", done=False)
    hits = [
        MemoryItem(
            id="m1",
            kind="fact",
            descriptor="[sandbox:research_papers/2605.23904.pdf chunk 1/1]",
            value={"path": "research_papers/2605.23904.pdf", "text": "skill memory"},
        )
    ]
    tc = heuristic_tool_call(
        goal=goal,
        user_query="Index research_papers/2605.23904.pdf and summarize skill memory.",
        hits=hits,
        history=[],
    )
    assert tc is not None
    assert tc.name == "search_knowledge"


def test_heuristic_after_index_hits_uses_search_knowledge():
    goal = Goal(id="g2", text="Answer from indexed content", done=False)
    hits = [
        MemoryItem(
            id="m1",
            kind="fact",
            descriptor="[sandbox:papers/attention.md chunk 1/3]",
            value={"path": "papers/attention.md", "text": "self-attention"},
        )
    ]
    tc = heuristic_tool_call(
        goal=goal,
        user_query="Index papers/attention.md and summarize Transformer contributions.",
        hits=hits,
        history=[],
    )
    assert tc is not None
    assert tc.name == "search_knowledge"
