"""Browser skill public API — extract → a11y → vision cascade."""

from __future__ import annotations

from typing import Any

from loguru import logger

from ..dag_schemas import BrowserErrorCode, BrowserOutput
from .output import classify_browser_error, to_browser_output
from .validation import (
    action_count as _action_count,
    comparison_content_ready as _comparison_content_ready,
    layer_succeeded as _layer_succeeded,
)

__all__ = [
    "run_browser",
    "run_browser_cascade",
    "to_browser_output",
    "classify_browser_error",
    "_action_count",
    "_comparison_content_ready",
    "_layer_succeeded",
]


async def run_browser_cascade(
    url: str,
    goal: str,
    *,
    llm: Any,
    force_path: str | None = None,
    min_browser_actions: int = 0,
    all_urls: list[str] | None = None,
    session_id: str = "",
    node_id: str = "",
) -> tuple[BrowserOutput, BrowserErrorCode | None]:
    """Run extract → render → a11y → vision; never raises."""
    from .drivers.cascade import BrowserSkill
    from .page_capture import browser_capture_session

    try:
        async with browser_capture_session(session_id, node_id):
            skill = BrowserSkill(llm=llm, session_id=session_id, node_id=node_id)
            raw, err_code = await skill.run(
                url.strip(),
                (goal or url).strip(),
                force_path=force_path,
                min_browser_actions=min_browser_actions,
                all_urls=all_urls,
            )
    except Exception as e:
        logger.error(f"[browser] cascade fatal (contained): {e}")
        failed = {
            "path": "failed",
            "url": url,
            "content": None,
            "error": str(e)[:500],
            "total_elapsed_s": 0.0,
        }
        return to_browser_output(url=url, goal=goal or url, raw=failed), "extraction_failed"

    out = to_browser_output(url=url, goal=goal or url, raw=raw)
    if err_code and not out.content and out.path != "gateway_blocked":
        return out, err_code
    if out.path == "failed" or (not out.content and out.path != "gateway_blocked"):
        return out, err_code or classify_browser_error(raw)
    return out, None


async def layer_extract(url: str, goal: str):
    from .extract import layer_extract as _layer

    return await _layer(url, goal)


async def layer_render(url: str, goal: str, *, page=None):
    from .playwright_render import layer_render as _layer

    return await _layer(url, goal, page=page)


async def layer_a11y(url: str, goal: str, llm, *, page=None):
    from .drivers.cascade import BrowserSkill

    skill = BrowserSkill(llm=llm)
    raw, _ = await skill.run(url, goal, force_path="a11y", min_browser_actions=0)
    return raw


async def layer_vision(url: str, goal: str, *, page=None):
    from .drivers.cascade import BrowserSkill

    skill = BrowserSkill(llm=None)
    raw, _ = await skill.run(url, goal, force_path="vision", min_browser_actions=0)
    return raw


run_browser = run_browser_cascade
