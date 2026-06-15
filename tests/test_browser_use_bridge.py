"""Tests for browser-use bridge."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from super_browser.browser.browser_use_bridge import (
    _action_notes_from_history,
    browser_use_should_try,
    try_browser_use_task,
)


def test_browser_use_disabled_by_default(monkeypatch):
    monkeypatch.delenv("BROWSER_USE_ENABLED", raising=False)
    assert browser_use_should_try() is False


def test_browser_use_can_be_enabled(monkeypatch):
    monkeypatch.setenv("BROWSER_USE_ENABLED", "1")
    assert browser_use_should_try() is True


def test_action_notes_from_history():
    class FakeHistory:
        def action_names(self):
            return ["click", "scroll", "type"]

    notes = _action_notes_from_history(FakeHistory())
    assert notes[0] == "browser_use:agent"
    assert len(notes) == 4


def test_try_browser_use_returns_none_when_import_fails(monkeypatch):
    monkeypatch.setenv("BROWSER_USE_ENABLED", "1")

    async def _run():
        with patch(
            "super_browser.browser.browser_use_bridge.Agent",
            side_effect=ImportError("no browser_use"),
            create=True,
        ):
            return await try_browser_use_task(task="x", url="https://example.com")

    with patch.dict("sys.modules", {"browser_use": None}):
        result = asyncio.run(try_browser_use_task(task="x", url="https://example.com"))
    assert result is None
