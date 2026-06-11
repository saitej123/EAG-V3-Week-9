"""Browser backend selection — bundled Chromium, Chrome channel, or BrowserOS."""

from __future__ import annotations

import os

_VALID_BACKENDS = frozenset({"chromium", "chrome", "browseros"})


def browser_backend() -> str:
    """Return ``chromium`` (default), ``chrome``, or ``browseros``."""
    raw = os.environ.get("BROWSER_BACKEND", "chromium").strip().lower()
    if raw in _VALID_BACKENDS:
        return raw
    return "chromium"


def browseros_mcp_url() -> str:
    return (
        os.environ.get("BROWSEROS_MCP_URL", "").strip()
        or "http://127.0.0.1:9239/mcp"
    )


def browseros_cdp_url() -> str:
    return os.environ.get("BROWSEROS_CDP_URL", "").strip()


def browseros_enabled() -> bool:
    """True when BrowserOS MCP or CDP integration is requested."""
    if os.environ.get("BROWSEROS_ENABLED", "").strip().lower() in {"1", "true", "yes"}:
        return True
    if browser_backend() == "browseros":
        return True
    if os.environ.get("BROWSEROS_MCP_URL", "").strip():
        return True
    if browseros_cdp_url():
        return True
    return False


def use_browseros_cdp() -> bool:
    """Connect Playwright to a running BrowserOS/Chromium via CDP."""
    if browseros_cdp_url():
        return True
    return browser_backend() == "browseros" and bool(os.environ.get("BROWSEROS_USE_CDP", "").strip())
