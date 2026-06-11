"""Browser-use-style indexed agent loop (click by index, scroll, navigate)."""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from .agent_loop import run_indexed_agent_loop
from .ledger import apply_cost_fields
from .navigation import navigate_robust, page_body_fallback, page_live_extract
from .playwright_ctx import browser_page


async def layer_agent(url: str, goal: str, llm: Any, *, page=None) -> dict[str, Any] | None:
    """Indexed element agent — browser-use pattern before a11y/vision."""
    started = time.time()

    async def _on_page(pg) -> dict[str, Any] | None:
        try:
            if page is None:
                await navigate_robust(pg, url)

            result = await run_indexed_agent_loop(pg, goal, llm)
            if result and result.done and result.answer:
                return apply_cost_fields(
                    {
                        "path": "agent",
                        "url": pg.url,
                        "content": result.answer,
                        "content_type": "text/plain",
                        "turns": result.turns,
                        "transcript": result.notes,
                        "elapsed_s": round(time.time() - started, 2),
                        "llm_calls": result.llm_calls,
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                    }
                )

            live = await page_live_extract(pg, goal)
            if live and len(live) >= 400:
                return {
                    "path": "agent",
                    "url": pg.url,
                    "content": live,
                    "content_type": "text/plain",
                    "transcript": ["agent:fallback_live_extract"],
                    "elapsed_s": round(time.time() - started, 2),
                    "llm_calls": 0,
                }

            body = await page_body_fallback(pg, min_chars=400)
            if body:
                return {
                    "path": "agent",
                    "url": pg.url,
                    "content": body,
                    "content_type": "text/plain",
                    "transcript": ["agent:fallback_body"],
                    "elapsed_s": round(time.time() - started, 2),
                    "llm_calls": 0,
                }
        except Exception as e:
            logger.warning(f"[browser] agent layer error: {e}")
        return None

    try:
        if page is not None:
            return await _on_page(page)
        async with browser_page() as (_pw, _browser, pg):
            return await _on_page(pg)
    except Exception as e:
        logger.warning(f"[browser] agent session error: {e}")
        return None
