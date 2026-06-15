"""Layer 3 — Playwright + set-of-marks + VLM (coordinate fallback for canvas-only pages)."""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from ..vision_api import vision_analyze, vision_extract_prompt
from .dom import collect_clickables, page_device_pixel_ratio
from .highlight import dedupe_clickables, draw_marks
from .ledger import apply_cost_fields
from .navigation import navigate_robust, page_body_fallback
from .playwright_ctx import browser_page
from .vlm_parse import parse_action_json

_MAX_TURNS = 8
_MIN_DONE_CHARS = 20
_MIN_EXTRACT_CHARS = 30


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

Prefer JSON: {{"action":"click"|"done","mark":<number>,"answer":"<when done>"}}
If the page already shows enough data for a comparison table in the goal, reply with markdown (include all requested columns and rows) or JSON done+answer.
If the page already shows product names/prices for the goal, reply with plain markdown or JSON done+answer.
"""


def _vision_coord_prompt(*, goal: str, url: str, turn: int, width: int, height: int) -> str:
    return f"""You see a screenshot with NO numbered marks.

GOAL: {goal}
URL: {url}
TURN: {turn}
VIEWPORT: {width}x{height} CSS pixels (origin top-left)

Prefer JSON: {{"action":"click_coord"|"done","x":<css x>,"y":<css y>,"answer":"<when done>"}}
Or list visible products/prices as plain text or a markdown table when the goal is already visible.
"""


async def _click_mark(page, items: list[dict[str, Any]], mark: int) -> bool:
    target = next((it for it in items if int(it.get("mark") or 0) == mark), None)
    if not target:
        return False
    box = target["box"]
    x = box["x"] + box["width"] / 2
    y = box["y"] + box["height"] / 2
    await page.mouse.click(x, y)
    await page.wait_for_timeout(900)
    return True


async def _click_coord(page, x: float, y: float) -> None:
    await page.mouse.click(x, y)
    await page.wait_for_timeout(900)


def _vision_success_payload(
    *,
    pg,
    goal: str,
    started: float,
    turn: int,
    transcript: list[str],
    vision_calls: int,
    input_tokens: int,
    output_tokens: int,
    content: str,
    mode: str,
) -> dict[str, Any]:
    return apply_cost_fields(
        {
            "path": "vision",
            "url": pg.url,
            "content": content.strip(),
            "content_type": "text/plain",
            "turns": turn,
            "transcript": list(transcript),
            "elapsed_s": round(time.time() - started, 2),
            "llm_calls": vision_calls,
            "vision_tokens": {"input": input_tokens, "output": output_tokens},
            "mode": mode,
        }
    )


def _combine_vlm_notes(notes: list[str]) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for bit in notes:
        text = (bit or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        parts.append(text)
    return "\n\n".join(parts).strip()


async def _vision_extract_fallback(
    pg,
    *,
    goal: str,
    screenshot: bytes,
    started: float,
    turn: int,
    transcript: list[str],
    vision_calls: int,
    input_tokens: int,
    output_tokens: int,
    mode: str,
    prior_text: str,
) -> dict[str, Any] | None:
    """Final screenshot → VLM readout when the action loop did not finish."""
    combined = prior_text.strip()
    if len(combined) >= _MIN_EXTRACT_CHARS:
        transcript.append("vision:combined_vlm")
        return _vision_success_payload(
            pg=pg,
            goal=goal,
            started=started,
            turn=turn,
            transcript=transcript,
            vision_calls=vision_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            content=combined,
            mode=mode,
        )

    try:
        result = vision_analyze(
            image_bytes=screenshot,
            prompt=vision_extract_prompt(goal=goal, url=pg.url),
            label="browser-vision:extract",
            max_tokens=1024,
        )
    except RuntimeError as e:
        logger.warning(f"[browser] vision extract fallback unavailable: {e}")
        result = None

    text = ""
    if result:
        vision_calls += 1
        input_tokens += int(result.get("input_tokens") or 0)
        output_tokens += int(result.get("output_tokens") or 0)
        text = str(result.get("text") or "").strip()

    if text:
        combined = _combine_vlm_notes([combined, text]) if combined else text
    if len(combined) < _MIN_EXTRACT_CHARS:
        body = await page_body_fallback(pg, min_chars=100)
        if body:
            combined = _combine_vlm_notes([combined, body]) if combined else body

    if len(combined) < _MIN_EXTRACT_CHARS:
        return None

    transcript.append("vision:extract_fallback")
    return _vision_success_payload(
        pg=pg,
        goal=goal,
        started=started,
        turn=turn,
        transcript=transcript,
        vision_calls=vision_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        content=combined[:12000],
        mode=mode,
    )


async def layer_vision(url: str, goal: str, *, page=None) -> dict[str, Any] | None:
    """Screenshot + set-of-marks + VLM; never raises — prose and extract fallbacks."""
    started = time.time()
    vision_calls = 0
    input_tokens = 0
    output_tokens = 0
    transcript: list[str] = []
    vlm_notes: list[str] = []
    last_screenshot: bytes | None = None
    last_mode = "som"

    async def _run_loop(pg) -> dict[str, Any] | None:
        nonlocal vision_calls, input_tokens, output_tokens, last_screenshot, last_mode
        viewport = pg.viewport_size or {"width": 1280, "height": 900}
        vw = int(viewport.get("width") or 1280)
        vh = int(viewport.get("height") or 900)

        for turn in range(1, _MAX_TURNS + 1):
            raw_clickables = await collect_clickables(pg)
            marks = dedupe_clickables(raw_clickables)
            screenshot = await pg.screenshot(type="png", full_page=False)
            last_screenshot = screenshot
            from .page_capture import capture_png_bytes

            capture_png_bytes(screenshot, turn=turn, note=f"vision_turn:{turn}", action="vision")
            mode = "som" if marks else "coordinate"
            last_mode = mode

            if marks:
                dpr = await page_device_pixel_ratio(pg)
                try:
                    image_bytes = draw_marks(screenshot, marks, device_pixel_ratio=dpr)
                except Exception as e:
                    logger.warning(f"[browser] draw_marks failed turn {turn}: {e}")
                    image_bytes = screenshot
                    mode = "coordinate"
                    prompt = _vision_coord_prompt(goal=goal, url=pg.url, turn=turn, width=vw, height=vh)
                else:
                    prompt = _vision_som_prompt(goal=goal, url=pg.url, turn=turn, marks=marks)
            else:
                image_bytes = screenshot
                prompt = _vision_coord_prompt(goal=goal, url=pg.url, turn=turn, width=vw, height=vh)

            try:
                result = vision_analyze(
                    image_bytes=image_bytes,
                    prompt=prompt,
                    label=f"browser-vision:t{turn}",
                    max_tokens=2048,
                )
            except RuntimeError as e:
                logger.warning(f"[browser] vision unavailable: {e}")
                break

            vision_calls += 1
            transcript.append(f"vision_turn:{turn}")
            input_tokens += int(result.get("input_tokens") or 0)
            output_tokens += int(result.get("output_tokens") or 0)

            raw = str(result.get("text") or "").strip()
            if raw:
                vlm_notes.append(raw)
            data = parse_action_json(raw, allow_prose_done=True)
            if data.get("action") == "noop" and raw:
                transcript.append(f"vlm_raw:t{turn}")

            kind = (data.get("action") or "").lower()
            if kind == "done":
                answer = (data.get("answer") or raw).strip()
                if len(answer) >= _MIN_DONE_CHARS:
                    transcript.append("vision:done_answer")
                    return _vision_success_payload(
                        pg=pg,
                        goal=goal,
                        started=started,
                        turn=turn,
                        transcript=transcript,
                        vision_calls=vision_calls,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        content=answer,
                        mode=mode,
                    )

            if kind == "click_coord":
                try:
                    x = float(data.get("x"))
                    y = float(data.get("y"))
                except (TypeError, ValueError):
                    transcript.append("coord_invalid")
                    continue
                await _click_coord(pg, x, y)
                transcript.append(f"click_coord:{x:.0f},{y:.0f}")
                continue

            mark = data.get("mark")
            try:
                mark_i = int(mark) if mark is not None else 0
            except (TypeError, ValueError):
                mark_i = 0
            if mark_i <= 0:
                continue
            try:
                clicked = await _click_mark(pg, marks, mark_i)
                if clicked:
                    transcript.append(f"click_mark:{mark_i}")
                else:
                    transcript.append(f"click_failed:unknown_mark:{mark_i}")
            except Exception as e:
                transcript.append(f"click_failed:{e}")
                continue

        prior = _combine_vlm_notes(vlm_notes)
        if last_screenshot and vision_calls > 0:
            extracted = await _vision_extract_fallback(
                pg,
                goal=goal,
                screenshot=last_screenshot,
                started=started,
                turn=max(vision_calls, _MAX_TURNS),
                transcript=transcript,
                vision_calls=vision_calls,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                mode=last_mode,
                prior_text=prior,
            )
            if extracted:
                return extracted

        if vision_calls >= 3 and prior:
            transcript.append("vision:accumulated_vlm")
            return _vision_success_payload(
                pg=pg,
                goal=goal,
                started=started,
                turn=vision_calls,
                transcript=transcript,
                vision_calls=vision_calls,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                content=prior[:12000],
                mode=last_mode,
            )

        body = await page_body_fallback(pg, min_chars=100)
        if body and vision_calls >= 1:
            transcript.append("fallback:body_text")
            return _vision_success_payload(
                pg=pg,
                goal=goal,
                started=started,
                turn=vision_calls or 1,
                transcript=transcript,
                vision_calls=vision_calls,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                content=body[:12000],
                mode=last_mode,
            )

        return None

    try:
        if page is not None:
            return await _run_loop(page)
        async with browser_page() as (_pw, _browser, pg):
            await navigate_robust(pg, url)
            return await _run_loop(pg)
    except Exception as e:
        logger.warning(f"[browser] vision layer error: {e}")
        return None
