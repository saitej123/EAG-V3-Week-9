"""Layer 1b — Playwright-rendered static extract (0 LLM, JS-aware)."""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlparse

from loguru import logger

from .urls import initial_browser_url
from .extract import content_is_useful
from .navigation import (
    dismiss_cookie_banners,
    live_page_blocked,
    navigate_robust,
    page_body_fallback,
    page_markdown_extract,
)
from .playwright_ctx import browser_page


async def layer_render(
    url: str,
    goal: str,
    *,
    page=None,
) -> dict[str, Any] | None:
    """Extract from JS-rendered DOM — same ``path=extract``, escalates on failure."""
    started = time.time()
    target = initial_browser_url(url, goal)

    async def _on_page(pg) -> dict[str, Any] | None:
        try:
            if page is None:
                await navigate_robust(pg, target)
            await dismiss_cookie_banners(pg)
            if await live_page_blocked(pg):
                logger.info("[browser] render: live captcha wall — escalating to interaction layers")
                return None

            content = await page_markdown_extract(pg, goal)
            if not content:
                body = await page_body_fallback(pg, min_chars=200)
                if body and content_is_useful(body, goal):
                    content = body

            if not content:
                return None

            host = urlparse(pg.url).netloc
            return {
                "path": "extract",
                "url": pg.url,
                "host": host,
                "content": content,
                "content_type": "text/markdown",
                "length_chars": len(content),
                "elapsed_s": round(time.time() - started, 2),
                "llm_calls": 0,
                "transcript": ["render:playwright"],
            }
        except Exception as e:
            logger.warning(f"[browser] render layer error: {e}")
            return None

    try:
        if page is not None:
            return await _on_page(page)

        async with browser_page() as (_pw, _browser, pg):
            return await _on_page(pg)
    except Exception as e:
        logger.warning(f"[browser] render session error: {e}")
        return None
