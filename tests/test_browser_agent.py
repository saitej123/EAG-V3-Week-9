"""Tests for browser-use-inspired indexed agent helpers."""

from __future__ import annotations

import asyncio

from super_browser.browser.driver import agent_turn_prompt, normalize_actions
from super_browser.browser.indexed_dom import build_indexed_interactive_state


def test_agent_turn_prompt_includes_indices():
    prompt = agent_turn_prompt(
        goal="Compare pricing",
        state="[1]<button> Monthly\n[2]<a> Pricing",
        url="https://example.com",
        turn=1,
        element_count=2,
    )
    assert "click_index" in prompt
    assert "[1]" in prompt
    assert "browser-use" in prompt.lower() or "index" in prompt.lower()


def test_normalize_click_index_action():
    raw = {"action": "click_index", "index": 4}
    actions = normalize_actions(raw)
    assert len(actions) == 1
    assert actions[0]["index"] == 4


def test_build_indexed_state_from_page():
    class FakePage:
        url = "https://example.com/pricing"

        async def evaluate(self, _script):
            return [
                {
                    "tag": "button",
                    "role": "button",
                    "label": "Monthly billing",
                    "href": "",
                    "box": {"x": 10, "y": 20, "width": 80, "height": 30},
                },
                {
                    "tag": "a",
                    "role": "link",
                    "label": "Pro plan $20",
                    "href": "/pro",
                    "box": {"x": 5, "y": 100, "width": 120, "height": 24},
                },
            ]

    async def _run():
        return await build_indexed_interactive_state(FakePage())

    text, selector_map = asyncio.run(_run())
    assert "[1]" in text
    assert "[2]" in text
    assert 1 in selector_map
    assert "Monthly" in selector_map[1]["label"]
