"""Tests for live pricing page enrichment."""

from __future__ import annotations

from super_browser.pricing_enrich import (
    canonical_product_name,
    expected_pricing_products,
    parse_pricing_from_page,
)


def test_expected_stack_products():
    query = (
        "Compare 5 AI coding tools: visit official pricing for "
        "Cursor, GitHub Copilot, Codeium (Windsurf), Tabnine, and Continue.dev."
    )
    names = expected_pricing_products(query)
    assert len(names) == 5
    assert names[0] == "Cursor"
    assert "Copilot" in names[1]
    assert "Windsurf" in names[2]


def test_parse_cursor_pricing():
    text = "### Hobby\n\nFree\n\n### Individual\n\n$20 / mo."
    parsed = parse_pricing_from_page("Cursor", "https://cursor.com/pricing", text)
    assert "free" in parsed["free_tier_summary"].lower()
    assert "$20" in parsed["paid_starting_price"]


def test_parse_copilot_pricing():
    text = "### Free\n\n$0USD\n\n2,000 completions per month\n\n### Pro\n\n$10USDper user / month"
    parsed = parse_pricing_from_page(
        "GitHub Copilot",
        "https://github.com/features/copilot/plans",
        text,
    )
    assert "$0" in parsed["free_tier_summary"]
    assert "2,000" in parsed["free_tier_summary"]
    assert "$10" in parsed["paid_starting_price"]
    assert "Haiku" not in parsed["free_tier_summary"]


def test_canonical_codeium_is_windsurf():
    assert canonical_product_name("Codeium") == "Codeium (Windsurf)"
