"""Optional bridge to the browser-use package when installed.

Set ``BROWSER_USE_ENABLED=1`` to try browser-use Agent before the local cascade.
Requires: ``pip install "browser-use[core]"`` and a configured LLM API key.
"""

from __future__ import annotations

import os
from typing import Any

from loguru import logger


def browser_use_enabled() -> bool:
    return os.environ.get("BROWSER_USE_ENABLED", "").strip().lower() in {"1", "true", "yes"}


async def try_browser_use_task(*, task: str, url: str) -> dict[str, Any] | None:
    """Run browser-use Agent if available; return cascade-shaped payload or None."""
    if not browser_use_enabled():
        return None
    try:
        from browser_use import Agent, Browser, ChatBrowserUse  # type: ignore[import-untyped]
    except ImportError:
        logger.info("[browser] browser-use not installed — using local indexed agent layer")
        return None

    model = os.environ.get("BROWSER_USE_MODEL", "bu-3")
    headless = os.environ.get("BROWSER_USE_HEADLESS", "true").lower() != "false"
    full_task = f"{task.strip()}\nStart at: {url}" if url else task

    try:
        browser = Browser(headless=headless)
        agent = Agent(task=full_task, llm=ChatBrowserUse(model=model), browser=browser)
        history = await agent.run(max_steps=25)
        result = ""
        if hasattr(history, "final_result"):
            result = str(history.final_result() or "")
        elif hasattr(history, "final_answer"):
            result = str(history.final_answer() or "")
        if len(result.strip()) < 80:
            return None
        return {
            "path": "agent",
            "url": url,
            "content": result.strip()[:48000],
            "content_type": "text/plain",
            "transcript": ["browser_use:agent"],
            "llm_calls": 1,
        }
    except Exception as e:
        logger.warning(f"[browser] browser-use bridge failed: {e}")
        return None
