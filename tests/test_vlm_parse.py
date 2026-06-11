"""Tests for resilient VLM/LLM action parsing."""

from __future__ import annotations

from super_browser.browser.vlm_parse import parse_action_json


def test_parse_action_json_empty_never_raises():
    assert parse_action_json("") == {"action": "noop"}
    assert parse_action_json("   ") == {"action": "noop"}


def test_parse_action_json_prose_comparison_table():
    raw = "| Laptop | Price |\n|--------|-------|\n| X | ₹45000 |"
    data = parse_action_json(raw)
    assert data["action"] == "done"
    assert "Laptop" in data["answer"]


def test_parse_action_json_salvages_mark_from_prose():
    data = parse_action_json('I would click mark 4 next')
    assert data.get("mark") == 4


def test_parse_action_json_click_coord_from_text():
    data = parse_action_json('{"action":"click_coord","x": 120, "y": 340}')
    assert data["action"] == "click_coord"
    assert data["x"] == 120.0
