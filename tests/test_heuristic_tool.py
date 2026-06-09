"""Heuristic tool selection when Decision LLM fails."""

from __future__ import annotations

from super_browser.schemas import Goal, MemoryItem
from super_browser.search_providers import extract_sandbox_paths, heuristic_tool_call


def test_extract_sandbox_paths():
    text = "Index the file papers/attention.md and explain the Transformer."
    assert extract_sandbox_paths(text) == ["papers/attention.md"]


def test_heuristic_index_document_for_query_e():
    goal = Goal(id="g1", text="Read and extract papers/attention.md", done=False)
    tc = heuristic_tool_call(
        goal=goal,
        user_query=(
            "Index the file papers/attention.md and tell me what the three key "
            "contributions of the Transformer architecture are according to this paper."
        ),
        hits=[],
        history=[],
    )
    assert tc is not None
    assert tc.name == "index_document"
    assert tc.arguments["path"] == "papers/attention.md"
    assert tc.arguments.get("use_vlm") is False


def test_heuristic_index_document_even_with_unrelated_hits():
    """Query E must index even when memory returns unrelated hits (e.g. Mom's birthday)."""
    goal = Goal(id="g1b", text="Read and index papers/attention.md", done=False)
    hits = [
        MemoryItem(
            id="m0",
            kind="fact",
            descriptor="Mom's birthday and reminder preferences",
            value={"text": "15 May 2026"},
        )
    ]
    tc = heuristic_tool_call(
        goal=goal,
        user_query=(
            "Index the file papers/attention.md and tell me what the three key "
            "contributions of the Transformer architecture are according to this paper."
        ),
        hits=hits,
        history=[],
    )
    assert tc is not None
    assert tc.name == "index_document"
    assert tc.arguments["path"] == "papers/attention.md"


def test_enrich_index_document_fills_missing_path():
    from super_browser.schemas import ToolCall
    from super_browser.search_providers import enrich_tool_call

    goal = Goal(id="g1c", text="Index papers/attention.md", done=False)
    tc = enrich_tool_call(
        ToolCall(name="index_document", arguments={}),
        goal=goal,
        user_query="Index the file papers/attention.md and summarize it.",
    )
    assert tc.arguments["path"] == "papers/attention.md"
    assert tc.arguments.get("use_vlm") is False


def test_heuristic_search_knowledge_after_index():
    goal = Goal(id="g2", text="Answer from indexed content", done=False)
    hits = [
        MemoryItem(
            id="m1",
            kind="fact",
            descriptor="[sandbox:papers/attention.md chunk 1/3] scaled dot-product",
            value={"path": "papers/attention.md", "text": "self-attention"},
        )
    ]
    history = [
        {
            "kind": "action",
            "tool": "index_document",
            "arguments": {"path": "papers/attention.md"},
        }
    ]
    tc = heuristic_tool_call(
        goal=goal,
        user_query="What are the Transformer contributions according to this paper?",
        hits=hits,
        history=history,
    )
    assert tc is not None
    assert tc.name == "search_knowledge"


def test_heuristic_bulk_index_directory():
    goal = Goal(id="g3", text="Index papers", done=False)
    tc = heuristic_tool_call(
        goal=goal,
        user_query="Index every .md file under papers/. Confirm how many chunks were indexed.",
        hits=[],
        history=[],
    )
    assert tc is not None
    assert tc.name == "index_directory"
    assert tc.arguments["path"] == "papers"


def test_heuristic_none_for_generic_web_query():
    goal = Goal(id="g4", text="Find Tokyo activities", done=False)
    tc = heuristic_tool_call(
        goal=goal,
        user_query="Find 3 family-friendly things to do in Tokyo this weekend.",
        hits=[],
        history=[],
    )
    assert tc is None
