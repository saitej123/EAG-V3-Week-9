"""Layer 3 — Playwright + set-of-marks + VLM (coordinate fallback for canvas-only pages)."""

from __future__ import annotations

import re
import time
from typing import Any

from loguru import logger

from ..llm_retry import loads_json_lenient
from ..vision_api import vision_analyze
from .dom import collect_clickables, page_device_pixel_ratio
from .highlight import dedupe_clickables, draw_marks
from .ledger import apply_cost_fields
from .playwright_ctx import browser_page

_MAX_TURNS = 8


def _vision_som_prompt(*, goal: str, url: str, turn: int, marks: list[dict[str, Any]]) -> str:
    legend = "\n".join(
        f"[{it['mark']}] {it.get('label') or it.get('tag')}" for it in marks[:40]
    )
    return f"""You see a screenshot with numbered boxes on clickable elements.

GOAL: {goal}
URL: {url}
TURN: {turn}

MARK LEGEND:
{legend}

Respond with JSON only:
{{"action": "click"|"done", "mark": <number or null>, "answer": "<final answer when done>"}}
"""


def _vision_coord_prompt(*, goal: str, url: str, turn: int, width: int, height: int) -> str:
    return f"""You see a screenshot with NO numbered marks — likely a canvas-only page.

GOAL: {goal}
URL: {url}
TURN: {turn}
VIEWPORT: {width}x{height} CSS pixels (origin top-left)

Respond with JSON only:
{{"action": "click_coord"|"done", "x": <css x>, "y": <css y>, "answer": "<when done>"}}

Use click_coord to click a visible target (e.g. coloured shape). Do not guess — pick coordinates you can see.
"""


async def _click_mark(page, items: list[dict[str, Any]], mark: int) -> None:
    target = next((it for it in items if int(it.get("mark") or 0) == mark), None)
    if not target:
        raise ValueError(f"unknown mark {mark}")
    box = target["box"]
    x = box["x"] + box["width"] / 2
    y = box["y"] + box["height"] / 2
    await page.mouse.click(x, y)
    await page.wait_for_timeout(900)


async def _click_coord(page, x: float, y: float) -> None:
    await page.mouse.click(x, y)
    await page.wait_for_timeout(900)


async def layer_vision(url: str, goal: str) -> dict[str, Any] | None:
    """Screenshot + set-of-marks + VLM; coordinate mode when no clickables."""
    started = time.time()
    vision_calls = 0
    input_tokens = 0
    output_tokens = 0
    transcript: list[str] = []

    async with browser_page() as (_pw, _browser, page):
        await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        await page.wait_for_timeout(1000)
        viewport = page.viewport_size or {"width": 1280, "height": 900}
        vw = int(viewport.get("width") or 1280)
        vh = int(viewport.get("height") or 900)

        for turn in range(1, _MAX_TURNS + 1):
            raw_clickables = await collect_clickables(page)
            marks = dedupe_clickables(raw_clickables)
            screenshot = await page.screenshot(type="png", full_page=False)

            if marks:
                dpr = await page_device_pixel_ratio(page)
                image_bytes = draw_marks(screenshot, marks, device_pixel_ratio=dpr)
                prompt = _vision_som_prompt(goal=goal, url=page.url, turn=turn, marks=marks)
            else:
                image_bytes = screenshot
                prompt = _vision_coord_prompt(goal=goal, url=page.url, turn=turn, width=vw, height=vh)

            result = vision_analyze(
                image_bytes=image_bytes,
                prompt=prompt,
                label=f"browser-vision:t{turn}",
            )
            vision_calls += 1
            input_tokens += int(result.get("input_tokens") or 0)
            output_tokens += int(result.get("output_tokens") or 0)

            raw = str(result.get("text") or "")
            data = loads_json_lenient(raw)
            if not isinstance(data, dict):
                m = re.search(r"\b(\d{1,2})\b", raw)
                data = {"action": "click", "mark": int(m.group(1)) if m else None}

            kind = (data.get("action") or "").lower()
            if kind == "done":
                answer = (data.get("answer") or "").strip()
                if answer:
                    return apply_cost_fields(
                        {
                            "path": "vision",
                            "url": page.url,
                            "content": answer,
                            "content_type": "text/plain",
                            "turns": turn,
                            "transcript": transcript,
                            "elapsed_s": round(time.time() - started, 2),
                            "llm_calls": vision_calls,
                            "vision_tokens": {"input": input_tokens, "output": output_tokens},
                            "mode": "som" if marks else "coordinate",
                        }
                    )

            if kind == "click_coord":
                try:
                    x = float(data.get("x"))
                    y = float(data.get("y"))
                except (TypeError, ValueError):
                    transcript.append("coord_invalid")
                    break
                await _click_coord(page, x, y)
                transcript.append(f"click_coord:{x:.0f},{y:.0f}")
                if turn >= 1 and marks == []:
                    return apply_cost_fields(
                        {
                            "path": "vision",
                            "url": page.url,
                            "content": f"Clicked at ({x:.0f}, {y:.0f}) for goal: {goal}",
                            "content_type": "text/plain",
                            "turns": turn,
                            "transcript": transcript,
                            "elapsed_s": round(time.time() - started, 2),
                            "llm_calls": vision_calls,
                            "vision_tokens": {"input": input_tokens, "output": output_tokens},
                            "mode": "coordinate",
                        }
                    )
                continue

            mark = data.get("mark")
            try:
                mark_i = int(mark) if mark is not None else 0
            except (TypeError, ValueError):
                mark_i = 0
            if mark_i <= 0:
                logger.info("[browser] vision returned no mark — stopping")
                break
            try:
                await _click_mark(page, marks, mark_i)
                transcript.append(f"click_mark:{mark_i}")
            except Exception as e:
                transcript.append(f"click_failed:{e}")
                break

    return None
