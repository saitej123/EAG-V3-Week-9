"""Layer 1 — httpx + trafilatura static extract (0 LLM cost)."""

from __future__ import annotations

import re
import time
from typing import Any
from urllib.parse import urlparse

import httpx
import trafilatura

from ..search_providers import HTTP_FETCH_TIMEOUT_SEC, _HTTP_HEADERS
from .dom import detect_gateway_block

_MIN_CHARS = 200


def goal_keywords(goal: str) -> list[str]:
    """Tokenize goal text into keywords for usefulness checks."""
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "from",
        "find",
        "get",
        "fetch",
        "read",
        "tell",
        "me",
        "what",
        "which",
        "how",
        "is",
        "are",
        "page",
        "website",
        "url",
        "http",
        "https",
        "www",
        "com",
    }
    words = re.findall(r"[a-z0-9]{3,}", (goal or "").lower())
    return [w for w in words if w not in stop][:12]


def content_is_useful(content: str, goal: str) -> bool:
    """True when extracted text is long enough and matches at least one goal keyword."""
    text = (content or "").strip()
    if len(text) < _MIN_CHARS:
        return False
    keys = goal_keywords(goal)
    if not keys:
        return True
    low = text.lower()
    return any(k in low for k in keys)


async def layer_extract(url: str, goal: str) -> dict[str, Any] | None:
    """Try static extract; return payload or None to escalate."""
    started = time.time()
    try:
        async with httpx.AsyncClient(
            timeout=HTTP_FETCH_TIMEOUT_SEC,
            headers={
                **_HTTP_HEADERS,
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
            },
            follow_redirects=True,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text
    except Exception:
        return None

    if detect_gateway_block(html):
        # Static httpx fetch only — escalate to Playwright layers (a11y/vision).
        return None

    content = trafilatura.extract(
        html,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
    )
    if not content or not content_is_useful(content, goal):
        return None

    host = urlparse(url).netloc
    return {
        "path": "extract",
        "url": url,
        "host": host,
        "content": content.strip(),
        "content_type": "text/markdown",
        "length_chars": len(content),
        "elapsed_s": round(time.time() - started, 2),
        "llm_calls": 0,
    }
