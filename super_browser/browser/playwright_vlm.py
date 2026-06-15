"""Playwright + VLM fast path — one screenshot, one vision call (before full SoM loop)."""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from ..vision_api import vision_analyze, vision_extract_prompt
from .ledger import apply_cost_fields
from .navigation import dismiss_cookie_banners, navigate_robust, wait_for_page_ready
from .playwright_ctx import browser_page

_MIN_CHARS = 40


async def layer_playwright_vlm(url: str, goal: str, *, page=None) -> dict[str, Any] | None:
    """Screenshot the live Playwright page and ask Gemini to read it — never raises."""
    started = time.time()

    async def _extract(pg) -> dict[str, Any] | None:
        await wait_for_page_ready(pg)
        await dismiss_cookie_banners(pg)
        try:
            screenshot = await pg.screenshot(type="png", full_page=False)
        except Exception as e:
            logger.warning(f"[browser] playwright_vlm screenshot failed: {e}")
            return None

        try:
            result = vision_analyze(
                image_bytes=screenshot,
                prompt=vision_extract_prompt(goal=goal, url=pg.url),
                label="browser-playwright-vlm",
                max_tokens=2048,
            )
        except RuntimeError as e:
            logger.warning(f"[browser] playwright_vlm unavailable: {e}")
            return None

        text = str(result.get("text") or "").strip()
        if len(text) < _MIN_CHARS:
            return None

        return apply_cost_fields(
            {
                "path": "vision",
                "url": pg.url,
                "content": text[:12000],
                "content_type": "text/plain",
                "turns": 1,
                "transcript": ["vlm_live:extract", "vision_turn:1"],
                "elapsed_s": round(time.time() - started, 2),
                "llm_calls": 1,
                "input_tokens": int(result.get("input_tokens") or 0),
                "output_tokens": int(result.get("output_tokens") or 0),
                "mode": "playwright_vlm",
            }
        )

    try:
        if page is not None:
            return await _extract(page)
        async with browser_page() as (_pw, _browser, pg):
            await navigate_robust(pg, url)
            return await _extract(pg)
    except Exception as e:
        logger.warning(f"[browser] playwright_vlm layer error: {e}")
        return None
