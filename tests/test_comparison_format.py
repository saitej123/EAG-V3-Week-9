"""Tests for generic comparison table parsing and formatting."""

from __future__ import annotations

from super_browser.comparison_format import (
    comparison_browser_goal_suffix,
    comparison_query_understanding,
    distiller_metadata_for_query,
    enrich_planner_nodes,
    format_comparison_answer,
    format_comparison_table,
    match_assignment_query,
    parse_comparison_spec,
)
from super_browser.dag_schemas import NodeSpec


def test_comparison_query_understanding_includes_columns():
    text = comparison_query_understanding(
        "Compare 3 IMAX showtimes. Table with columns: movie name, theatre name, show time, ticket price"
    )
    assert "QUERY UNDERSTANDING" in text
    assert "movie name" in text.lower()
    assert "3" in text


def test_parse_columns_from_parentheses():
    spec = parse_comparison_spec(
        "Return a comparison table (theatre name, show time, ticket price, screen format)"
    )
    assert spec.is_comparison
    assert "theatre name" in spec.columns[0].lower()
    assert len(spec.columns) == 4


def test_parse_columns_with_columns_clause():
    spec = parse_comparison_spec(
        "Compare 5 AI tools. Return a structured comparison table "
        "(tool, free tier summary, paid starting price)."
    )
    assert spec.row_count == 5
    assert spec.columns == ["tool", "free tier summary", "paid starting price"]


def test_distiller_metadata_uses_generic_rows():
    meta = distiller_metadata_for_query(
        "Compare 3 IMAX showtimes. Table with columns: movie name, theatre name, show time, ticket price, screen format"
    )
    assert meta["required_keys"] == "subject,context,rows"
    assert "rows" in meta["fields"]


def test_format_table_from_generic_rows():
    spec = parse_comparison_spec(
        "Compare 3 items. Table (movie name, theatre name, show time, ticket price, screen format)"
    )
    table = format_comparison_table(
        spec,
        {
            "subject": "Pushpa 2",
            "context": {"city": "Bengaluru"},
            "rows": [
                {
                    "theatre_name": "PVR Forum",
                    "show_time": "02:30 PM",
                    "ticket_price": "₹450",
                    "screen_format": "IMAX 2D",
                },
                {
                    "theatre_name": "INOX Garuda",
                    "show_time": "06:15 PM",
                    "ticket_price": "₹420",
                    "screen_format": "IMAX 2D",
                },
            ],
        },
    )
    assert table
    assert "Pushpa 2" in table
    assert "movie name" in table.lower() or "Movie Name" in table
    assert "PVR Forum" in table
    assert table.count("| Pushpa 2 |") >= 2


def test_format_stack_pricing_table():
    query = (
        "Compare 5 AI coding assistants. Return a structured comparison table "
        "(tool, free tier summary, paid starting price)."
    )
    text = format_comparison_answer(
        query,
        [
            {
                "kind": "upstream",
                "skill": "distiller",
                "output": {
                    "subject": "AI coding assistants",
                    "rows": [
                        {"tool": "Cursor", "free_tier_summary": "Limited", "paid_starting_price": "$20/mo"},
                        {"tool": "Copilot", "free_tier_summary": "None", "paid_starting_price": "$10/mo"},
                    ],
                },
            }
        ],
    )
    assert text
    assert "Cursor" in text
    assert "Copilot" in text
    assert "free tier" in text.lower() or "Free Tier" in text


def test_format_comparison_answer_legacy_showtimes_key():
    text = format_comparison_answer(
        "Compare 3 IMAX showtimes. Table (movie name, theatre name, show time, ticket price, screen format)",
        [
            {
                "kind": "upstream",
                "skill": "distiller",
                "output": {
                    "movie_name": "Coolie",
                    "city": "Bengaluru",
                    "showtimes": [
                        {
                            "theatre_name": "Cinepolis",
                            "show_time": "09:00 PM",
                            "ticket_price": "₹500",
                            "screen_format": "IMAX 3D",
                        }
                    ],
                },
            }
        ],
    )
    assert text
    assert "Coolie" in text
    assert "Cinepolis" in text


def test_enrich_planner_nodes_generic_goal():
    query = (
        "Compare 3 laptops on Flipkart. Return a comparison table "
        "(product name, price, key specs)."
    )
    nodes = enrich_planner_nodes(
        query,
        [
            NodeSpec(skill="browser", inputs=["USER_QUERY"], metadata={"label": "b"}),
            NodeSpec(skill="distiller", inputs=["n:b"], metadata={"label": "d"}),
            NodeSpec(skill="formatter", inputs=["n:d"], metadata={"label": "out"}),
        ],
    )
    dist = next(n for n in nodes if n.skill == "distiller")
    assert dist.metadata.get("required_keys") == "subject,context,rows"
    browser = next(n for n in nodes if n.skill == "browser")
    assert "comparison table" in str(browser.metadata.get("goal") or "").lower()


def test_browser_goal_suffix_from_query_only():
    suffix = comparison_browser_goal_suffix(
        "Compare 4 hotels. Table (hotel, price, rating)."
    )
    assert "4 distinct" in suffix or "4" in suffix
    assert "hotel" in suffix.lower()


def test_match_assignment_query_still_works():
    row = match_assignment_query(
        "Compare 3 trending open-source repositories on GitHub: go to https://github.com/trending"
    )
    assert row is not None
    assert row["id"] == "TICKET"


def test_comparison_pricing_gemini_query_codeium_windsurf():
    from super_browser.comparison_format import comparison_pricing_gemini_query

    q = comparison_pricing_gemini_query(
        "Compare Cursor, GitHub Copilot, and Codeium pricing pages for free vs paid plan"
    )
    assert "Windsurf" in q or "Codeium" in q
    assert "FREE:" in q or "free tier" in q.lower()


def test_parse_pricing_facts_structured():
    from super_browser.comparison_format import _parse_pricing_facts

    text = (
        "FREE: Unlimited Tab completions; light agent quota\n"
        "PAID: Pro $20/mo with frontier models"
    )
    parsed = _parse_pricing_facts(text)
    assert "tab" in parsed["free_tier_summary"].lower()
    assert "$20" in parsed["paid_starting_price"]


def test_enrich_distiller_fills_codeium_gap(monkeypatch):
    from super_browser.comparison_format import enrich_distiller_pricing_gaps

    monkeypatch.setattr(
        "super_browser.pricing_enrich.fetch_all_products",
        lambda products: {
            p: (
                {
                    "tool": "Codeium (Windsurf)",
                    "free_tier_summary": "Free ($0/mo — light agent quota)",
                    "paid_starting_price": "$20/mo (Pro)",
                }
                if "codeium" in p.lower() or "windsurf" in p.lower()
                else {
                    "tool": p,
                    "free_tier_summary": "Free tier",
                    "paid_starting_price": "$10/mo",
                }
            )
            for p in products
        },
    )
    distiller = {
        "rows": [
            {
                "tool": "Cursor",
                "free_tier_summary": "Hobby free",
                "paid_starting_price": "$20/mo",
            },
            {"tool": "Codeium", "free_tier_summary": "—", "paid_starting_price": "—"},
        ]
    }
    query = (
        "Compare Cursor, GitHub Copilot, and Codeium by free vs paid plan: "
        "visit official pricing pages for Cursor, GitHub Copilot, and Codeium."
    )
    out = enrich_distiller_pricing_gaps(query, distiller)
    codeium = next(r for r in out["rows"] if "codeium" in str(r.get("tool", "")).lower())
    assert "$20" in str(codeium.get("paid_starting_price", ""))
    assert codeium.get("free_tier_summary") and codeium["free_tier_summary"] != "—"
