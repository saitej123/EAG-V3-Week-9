"""Unit tests for structured iteration log helpers (Queries A, C, D)."""

from __future__ import annotations

import json

import pytest

import super_browser.agent as loop
from super_browser.action import format_artifact_size, summarize_tool_result
from super_browser.schemas import Goal, MemoryItem, Observation


def _goal(text: str) -> Goal:
    return Goal(id="g-test", text=text, done=False)


def _obs(*goals: Goal) -> Observation:
    return Observation(goals=list(goals))


_QUERY_D = (
    "Search for 'Python asyncio best practices', read the top 3 results, "
    "and give me a short numbered list of the advice they agree on."
)
_TOKYO_B = (
    "Find 3 family-friendly things to do in Tokyo this weekend. "
    "Check Saturday's weather forecast there and tell me which one is most appropriate."
)


@pytest.mark.parametrize(
    "text,expect_log",
    [
        ("Fetch the Wikipedia page for Claude Shannon", True),
        ("Extract birth date, death date, and three contributions", True),
        ("Search for Python asyncio best practices", True),
        ("Fetch the first search result URL", True),
        ("Synthesise common advice from sources", True),
    ],
)
def test_should_log_perception(text: str, expect_log: bool) -> None:
    obs = _obs(_goal(text))
    assert loop._should_log_perception(_goal(text), obs, _TOKYO_B) is expect_log


def test_format_artifact_size_large_wikipedia() -> None:
    assert format_artifact_size(263_065) == "263065 bytes"


def test_format_artifact_size_query_d_kb() -> None:
    assert format_artifact_size(45_000) == "44KB"


def test_fetch_url_action_summary_uses_packed_descriptor() -> None:
    packed = (
        "[artifact art:abc123, 263065 bytes] preview: "
        "'Claude Shannon From Wikipedia the free encyclopedia'"
    )
    out = summarize_tool_result("fetch_url", {}, packed, "art:abc123", artifact_bytes=263_065)
    assert out.startswith("[artifact art:abc123, 263065 bytes]")
    assert "preview:" in out


def test_create_file_action_summary() -> None:
    body = json.dumps({"ok": True, "path": "reminders/mom_birthday_2026.txt", "size_bytes": 120})
    assert summarize_tool_result("create_file", {}, body, None) == "ok"


def test_list_dir_single_file_summary() -> None:
    body = json.dumps([{"name": "mom_birthday_2026.txt", "type": "file", "size_bytes": 120}])
    assert summarize_tool_result("list_dir", {"path": "reminders/"}, body, None) == "[file: mom_birthday_2026.txt]"


def test_memory_read_helper_exists() -> None:
    fact = MemoryItem(
        id="m1",
        kind="fact",
        keywords=["mom", "birthday"],
        descriptor="Mom birthday",
        value={"text": "Mom's birthday is on 15 May 2026"},
    )
    assert loop._log_memory_read is not None
    assert fact.kind == "fact"


def test_web_search_action_summary() -> None:
    body = json.dumps(
        [
            {"title": "A", "url": "https://a.example", "snippet": "a"},
            {"title": "B", "url": "https://b.example", "snippet": "b"},
            {"title": "C", "url": "https://c.example", "snippet": "c"},
        ]
    )
    assert summarize_tool_result("web_search", {}, body, None) == "[3 URLs in descriptors]"


def test_multi_source_skips_perception_variants() -> None:
    for text in (
        "Search for 'Python asyncio best practices'",
        "Fetch top 3 results from search",
        "Read the first search result",
        "Fetch result 2 from descriptors",
    ):
        assert loop._should_log_perception(_goal(text), _obs(_goal(text)), _QUERY_D) is False


def test_all_done_always_logs_perception() -> None:
    g1 = Goal(id="g1", text="Fetch page", done=True)
    g2 = Goal(id="g2", text="Extract facts", done=True)
    obs = _obs(g1, g2)
    assert obs.all_done()
    assert loop._should_log_perception(None, obs, _TOKYO_B) is True
