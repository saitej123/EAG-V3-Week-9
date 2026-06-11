"""Tests for BrowserOS MCP bridge helpers."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

from super_browser.browser.browser_backend import browser_backend, browseros_enabled
from super_browser.browser.browseros_bridge import try_browseros_task
from super_browser.browser.browseros_mcp import (
    _parse_mcp_response,
    _unwrap_tool_result,
    parse_snapshot_index_map,
)


def test_parse_mcp_response_json():
    payload = {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}}
    assert _parse_mcp_response(json.dumps(payload)) == payload


def test_parse_mcp_response_sse():
    text = 'event: message\ndata: {"jsonrpc":"2.0","id":2,"result":{}}\n\n'
    assert _parse_mcp_response(text)["id"] == 2


def test_unwrap_tool_result_text_blocks():
    result = {
        "content": [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "World"},
        ]
    }
    assert _unwrap_tool_result(result) == "Hello\nWorld"


def test_parse_snapshot_index_map():
    snap = """
[1] uid=btn_search button "Search"
[2] uid=link_pricing link "Pricing"
"""
    mapping = parse_snapshot_index_map(snap)
    assert mapping[1] == "btn_search"
    assert mapping[2] == "link_pricing"


def test_try_browseros_task_returns_content(monkeypatch):
    monkeypatch.setenv("BROWSER_BACKEND", "browseros")

    class FakeClient:
        def __init__(self, _url=None):
            pass

        async def ping(self):
            return True

        async def call_tool(self, name, arguments=None):
            if name == "navigate_page":
                return "ok"
            if name == "get_page_content":
                return "# Pricing\n\nPro plan $20/month\n" * 5
            return ""

    async def _run():
        with patch("super_browser.browser.browseros_bridge.BrowserOsMcpClient", FakeClient):
            return await try_browseros_task(task="Get pricing", url="https://example.com", llm=None)

    result = asyncio.run(_run())
    assert result is not None
    assert result["path"] == "agent"
    assert "Pricing" in result["content"]
    assert any("browseros:" in n for n in result["transcript"])


def test_try_browseros_unavailable_returns_none(monkeypatch):
    monkeypatch.setenv("BROWSER_BACKEND", "browseros")

    class FakeClient:
        def __init__(self, _url=None):
            pass

        async def ping(self):
            return False

    async def _run():
        with patch("super_browser.browser.browseros_bridge.BrowserOsMcpClient", FakeClient):
            return await try_browseros_task(task="x", url="https://example.com", llm=None)

    result = asyncio.run(_run())
    assert result is None


def test_browseros_disabled_by_default(monkeypatch):
    monkeypatch.delenv("BROWSER_BACKEND", raising=False)
    monkeypatch.delenv("BROWSEROS_ENABLED", raising=False)
    monkeypatch.delenv("BROWSEROS_MCP_URL", raising=False)
    monkeypatch.delenv("BROWSEROS_CDP_URL", raising=False)
    assert browser_backend() == "chromium"
    assert browseros_enabled() is False
