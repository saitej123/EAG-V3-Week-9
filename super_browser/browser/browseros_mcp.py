"""Minimal HTTP MCP client for BrowserOS (chrome://browseros/mcp)."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

import httpx
from loguru import logger

from .browser_backend import browseros_mcp_url

_SESSION_HEADER = "mcp-session-id"
_RPC_VERSION = "2024-11-05"


class BrowserOsMcpError(RuntimeError):
    pass


class BrowserOsMcpClient:
    """Streamable-HTTP MCP client for BrowserOS automation tools."""

    def __init__(self, base_url: str | None = None, *, timeout: float = 45.0) -> None:
        self.base_url = (base_url or browseros_mcp_url()).rstrip("/")
        self.timeout = timeout
        self._session_id: str | None = None
        self._req_id = 0
        self._initialized = False

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers[_SESSION_HEADER] = self._session_id

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(self.base_url, json=payload, headers=headers)
            resp.raise_for_status()
            sid = resp.headers.get(_SESSION_HEADER) or resp.headers.get("Mcp-Session-Id")
            if sid:
                self._session_id = sid
            return _parse_mcp_response(resp.text)

    async def initialize(self) -> None:
        if self._initialized:
            return
        init = await self._post(
            {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": _RPC_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "super-browser-agent", "version": "1.0"},
                },
            }
        )
        if init.get("error"):
            raise BrowserOsMcpError(str(init["error"]))
        await self._post(
            {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            }
        )
        self._initialized = True

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        await self.initialize()
        result = await self._post(
            {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments or {}},
            }
        )
        if result.get("error"):
            raise BrowserOsMcpError(str(result["error"]))
        return _unwrap_tool_result(result.get("result"))

    async def ping(self) -> bool:
        try:
            await self.initialize()
            return True
        except Exception as e:
            logger.debug(f"[browseros] MCP ping failed: {e}")
            return False


def _parse_mcp_response(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    if text.startswith("{"):
        return json.loads(text)
    # SSE: event: message\ndata: {...}
    for line in text.splitlines():
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload:
                return json.loads(payload)
    raise BrowserOsMcpError(f"Unparseable MCP response: {text[:240]}")


def _unwrap_tool_result(result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    content = result.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        if parts:
            return "\n".join(parts).strip()
    if "structuredContent" in result:
        return result["structuredContent"]
    return result


def parse_snapshot_index_map(snapshot: str) -> dict[int, str]:
    """Build index → element id map from BrowserOS snapshot text."""
    mapping: dict[int, str] = {}
    for line in (snapshot or "").splitlines():
        bracket = re.search(r"\[(\d+)\]", line)
        uid = re.search(r"(?:uid|elementId|id)[=:\s]+[\"']?([A-Za-z0-9_-]+)", line, re.I)
        if bracket and uid:
            mapping[int(bracket.group(1))] = uid.group(1)
            continue
        if bracket:
            # Fallback: use bracket index as string id (some servers accept numeric refs)
            mapping[int(bracket.group(1))] = bracket.group(1)
    if not mapping:
        idx = 1
        for line in (snapshot or "").splitlines():
            uid = re.search(r"(?:uid|elementId|ref)[=:\s]+[\"']?([A-Za-z0-9_-]+)", line, re.I)
            if uid:
                mapping[idx] = uid.group(1)
                idx += 1
    return mapping
