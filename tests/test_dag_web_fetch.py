"""DAG web fetch guards — URL detection, planner coercion, tool enrichment."""

from __future__ import annotations

from super_browser.dag_schemas import NodeSpec
from super_browser.flow import coerce_planner_successors
from super_browser.search_providers import enrich_tool_call, extract_http_urls
from super_browser.schemas import Goal, ToolCall
from super_browser.skills import (
    SkillRegistry,
    _auto_tool_for_web_skill,
    _dag_enrich_tool_call,
    explicit_url_fetch_mode,
    fetch_url_succeeded,
)


def test_extract_http_urls():
    q = "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me facts."
    assert extract_http_urls(q) == ["https://en.wikipedia.org/wiki/Claude_Shannon"]


def test_coerce_planner_formatter_only_url_query():
    q = "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his birth date."
    out = coerce_planner_successors(
        q,
        [NodeSpec(skill="formatter", inputs=["USER_QUERY"], metadata={"label": "out"})],
    )
    skills = [s.skill for s in out]
    assert skills == ["researcher", "distiller", "formatter"]


def test_coerce_planner_leaves_researcher_plan():
    q = "Fetch https://example.com/page"
    original = [
        NodeSpec(skill="researcher", inputs=["USER_QUERY"], metadata={"label": "r1"}),
        NodeSpec(skill="formatter", inputs=["n:r1"], metadata={"label": "out"}),
    ]
    assert coerce_planner_successors(q, original) == original


def test_enrich_fetch_url_from_query_text():
    reg = SkillRegistry()
    skill = reg.get("researcher")
    tc = ToolCall(name="fetch_url", arguments={})
    enriched = _dag_enrich_tool_call(
        tc,
        user_query="Fetch https://en.wikipedia.org/wiki/Claude_Shannon",
        sub_query="Shannon page",
    )
    assert enriched.arguments["url"].startswith("https://")


def test_auto_tool_fetch_url_for_researcher():
    reg = SkillRegistry()
    skill = reg.get("researcher")
    tc = _auto_tool_for_web_skill(
        skill,
        user_query="Fetch https://en.wikipedia.org/wiki/Claude_Shannon",
        sub_query="",
    )
    assert tc is not None
    assert tc.name == "fetch_url"
    assert "wikipedia" in tc.arguments["url"]


def test_explicit_url_fetch_mode_researcher():
    q = "Fetch https://en.wikipedia.org/wiki/Claude_Shannon"
    assert explicit_url_fetch_mode("researcher", q, "")
    assert not explicit_url_fetch_mode("distiller", q, "")


def test_fetch_url_succeeded_with_artifact():
    assert fetch_url_succeeded("[artifact x]", "art-1", body="short")


def test_auto_tool_skips_repeat_fetch_when_requested():
    skill = SkillRegistry().get("researcher")
    assert _auto_tool_for_web_skill(
        skill,
        user_query="Fetch https://en.wikipedia.org/wiki/Claude_Shannon",
        sub_query="",
        skip_fetch=True,
    ) is None


def test_enrich_web_search_query():
    tc = enrich_tool_call(
        ToolCall(name="web_search", arguments={}),
        goal=Goal(id="g1", text="population of Tokyo"),
        user_query="population of Tokyo",
    )
    assert tc.arguments.get("query")
