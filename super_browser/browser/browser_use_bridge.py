"""Optional bridge to the browser-use package (https://github.com/browser-use/browser-use).

Tries browser-use Agent before the local Playwright cascade when the package is installed.
Disable with ``BROWSER_USE_ENABLED=0``.

Install: ``uv sync --extra browser-use`` or ``pip install browser-use``
"""

from __future__ import annotations

import os
from typing import Any

from loguru import logger


def browser_use_should_try() -> bool:
    """True unless explicitly disabled via env."""
    flag = os.environ.get("BROWSER_USE_ENABLED", "0").strip().lower()
    return flag not in {"0", "false", "no", "off"}


def browser_use_enabled() -> bool:
    """Back-compat alias."""
    return browser_use_should_try()


def _action_notes_from_history(history: Any) -> list[str]:
    notes = ["browser_use:agent"]
    try:
        if hasattr(history, "action_names"):
            names = history.action_names() or []
            for i, name in enumerate(names, 1):
                notes.append(f"browser_use:action:{i}:{name}")
        elif hasattr(history, "history") and history.history:
            for i in range(len(history.history)):
                notes.append(f"browser_use:step:{i + 1}")
        elif hasattr(history, "number_of_steps"):
            count = int(history.number_of_steps() or 0)
            for i in range(count):
                notes.append(f"browser_use:step:{i + 1}")
    except Exception:
        pass
    return notes


async def try_browser_use_task(*, task: str, url: str, llm: Any | None = None) -> dict[str, Any] | None:
    """Run browser-use Agent if available; return cascade-shaped payload or None."""
    if not browser_use_should_try():
        return None
    try:
        from browser_use import Agent, Browser, ChatBrowserUse  # type: ignore[import-untyped]
    except ImportError:
        logger.debug("[browser] browser-use not installed — using local Playwright cascade")
        return None

    model = os.environ.get("BROWSER_USE_MODEL", "bu-3")
    headless = os.environ.get("BROWSER_USE_HEADLESS", "true").lower() != "false"
    max_steps = int(os.environ.get("BROWSER_USE_MAX_STEPS", "25") or 25)
    full_task = f"{task.strip()}\nStart at: {url}" if url else task

    try:
        browser = Browser(headless=headless)
        agent = Agent(task=full_task, llm=ChatBrowserUse(model=model), browser=browser)
        history = await agent.run(max_steps=max_steps)
        result = ""
        if hasattr(history, "final_result"):
            result = str(history.final_result() or "")
        elif hasattr(history, "final_answer"):
            result = str(history.final_answer() or "")
        if len(result.strip()) < 80:
            return None
        notes = _action_notes_from_history(history)
        return {
            "path": "agent",
            "url": url,
            "content": result.strip()[:48000],
            "content_type": "text/plain",
            "transcript": notes,
            "turns": max(1, len(notes) - 1),
            "llm_calls": max(1, len(notes) - 1),
        }
    except Exception as e:
        logger.warning(f"[browser] browser-use bridge failed: {e}")
        return None
