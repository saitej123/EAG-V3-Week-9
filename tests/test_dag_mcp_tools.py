"""Unit tests for DAG assignment MCP tools."""

from __future__ import annotations

import pytest

from super_browser.mcp_server import count_syllables, safe_calculate, validate_json_keys


def test_validate_json_keys_pass():
    out = validate_json_keys('{"author":"Ada","title":"Notes","year":1843}', "author,title,year")
    assert out["valid"] is True
    assert out["missing"] == []


def test_validate_json_keys_fail_missing():
    out = validate_json_keys('{"author":"Ada","title":"Notes"}', "author,title,year")
    assert out["valid"] is False
    assert "year" in out["missing"]


def test_count_syllables_lines():
    out = count_syllables("hello world\nfoo bar baz")
    assert out["lines"] == [3, 3]
    assert out["total"] == 6


def test_safe_calculate_integer_power():
    out = safe_calculate("(17 * 23 - 4) ** 2 + 1000")
    assert out["value"] == 150769.0


def test_safe_calculate_expression():
    out = safe_calculate("(987654321 ** 0) + ((17 * 23 + 41) / 7)")
    assert out["value"] == pytest.approx(62.714285714285715)


def test_count_syllables_prosody_query_lines():
    """Part 5 PROS demo — per-line totals and winner."""
    lines = {
        "A": "The orchestrator runs parallel waves",
        "B": "Each researcher fetches population data independently",
        "C": "Critic validates JSON keys before formatting",
    }
    totals = {k: count_syllables(v)["total"] for k, v in lines.items()}
    assert totals == {"A": 11, "B": 17, "C": 13}
    assert max(totals, key=totals.get) == "B"
