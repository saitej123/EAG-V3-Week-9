"""Browser turn driver — a11y loop with dropdown-as-fence rules."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

MAX_ACTIONS_PER_TURN = 2


def is_dropdown_trigger(name: str) -> bool:
    """Dropdown triggers must be the only action in a turn (popover not in DOM yet)."""
    label = (name or "").strip()
    if not label:
        return False
    if label.startswith("Sort:"):
        return True
    if label.endswith("▾") or label.endswith(":"):
        return True
    return False


def normalize_actions(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Accept single action or an actions array from the LLM."""
    if not isinstance(raw, dict):
        return []
    if raw.get("action"):
        return [raw]
    actions = raw.get("actions")
    if isinstance(actions, list):
        out: list[dict[str, Any]] = []
        for row in actions:
            if isinstance(row, dict) and row.get("action"):
                out.append(row)
        return out
    return []


def fence_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Max 2 actions per turn; dropdown trigger must be solo."""
    if not actions:
        return []
    fenced: list[dict[str, Any]] = []
    for action in actions:
        if len(fenced) >= MAX_ACTIONS_PER_TURN:
            break
        target = str(action.get("target") or action.get("name") or "").strip()
        fenced.append(action)
        if is_dropdown_trigger(target):
            break
    return fenced


def action_target(action: dict[str, Any]) -> str:
    return str(action.get("target") or action.get("name") or "").strip()


@dataclass
class TurnResult:
    notes: list[str] = field(default_factory=list)
    done: bool = False
    answer: str = ""
    turns: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


def a11y_turn_prompt(*, goal: str, tree: str, url: str, turn: int) -> str:
    return f"""You are a browser agent reading an accessibility tree. Plan the next turn.

GOAL: {goal}
URL: {url}
TURN: {turn}

ACCESSIBILITY TREE (fresh at turn start — popover options appear after triggers are clicked):
{tree[:10000]}

Respond with JSON only.

Done:
{{"action": "done", "answer": "<final answer>"}}

One or two actions (never click a dropdown trigger AND its option in the same turn):
{{"actions": [
  {{"action": "click", "target": "<visible label from tree>"}},
  {{"action": "type", "target": "<field label>", "text": "<value>"}}
]}}

Optional canvas drag:
{{"action": "drag", "from_x": 100, "from_y": 200, "to_x": 300, "to_y": 400}}

Rules:
- Read the tree at turn start; after a click opens a menu/popover, STOP — next turn sees new options.
- Dropdown triggers (names ending ▾ or :, or starting Sort:) must be the ONLY action that turn.
- Max 2 actions per turn overall.
- Prefer done/extract when the tree already answers the goal (e.g. top model names visible).
- Do not guess success on an empty tree.
"""


def agent_turn_prompt(*, goal: str, state: str, url: str, turn: int, element_count: int) -> str:
    """browser-use-style indexed element prompt (click by [index])."""
    return f"""You are a browser automation agent (browser-use pattern). Plan the next step.

GOAL: {goal}
URL: {url}
TURN: {turn}
INTERACTIVE ELEMENTS: {element_count}

{state[:12000]}

Respond with JSON only. Prefer clicking by index — most reliable.

Done when goal is visible on page:
{{"action": "done", "answer": "<markdown table or extracted facts from LIVE page only>"}}

Click indexed element (preferred):
{{"action": "click_index", "index": 5}}

Navigate:
{{"action": "go_to_url", "url": "https://example.com/pricing"}}

Scroll to reveal more content:
{{"action": "scroll", "direction": "down"}}

Wait for dynamic content:
{{"action": "wait", "seconds": 2}}

Label-based fallback (same turn max 2 actions):
{{"actions": [
  {{"action": "click_index", "index": 3}},
  {{"action": "scroll", "direction": "down"}}
]}}

Rules:
- Use ONLY indices from the list above — do not invent element numbers.
- After opening a menu/tab, STOP; next turn gets a fresh element list.
- For pricing/comparison goals: expand billing tabs, scroll to plans, then done with live prices.
- Never answer from memory — only text visible on the current page.
- Max 2 actions per turn.
"""


async def _click_index(page, action: dict[str, Any]) -> str:
    selector_map = action.get("_selector_map") or {}
    try:
        idx = int(action.get("index") or action.get("element_index") or 0)
    except (TypeError, ValueError):
        return "click_index_invalid"
    item = selector_map.get(idx)
    if not item:
        return f"click_index_missing:{idx}"
    box = item.get("box") or {}
    try:
        x = float(box.get("x", 0)) + float(box.get("width", 0)) / 2
        y = float(box.get("y", 0)) + float(box.get("height", 0)) / 2
    except (TypeError, ValueError):
        return f"click_index_bad_box:{idx}"
    await page.mouse.click(x, y)
    await page.wait_for_timeout(900)
    label = str(item.get("label") or "")[:40]
    return f"click_index:{idx}:{label}"


async def execute_action(page, action: dict[str, Any], *, turn: int | None = None) -> str:
    """Run one fenced action on the live page."""
    from .page_capture import capture_page_state

    kind = (action.get("action") or "").lower()
    target = action_target(action)
    text = str(action.get("text") or "").strip()

    async def _finish(note: str) -> str:
        if note and kind not in {"done", "extract"}:
            await capture_page_state(page, turn=turn, note=note, action=kind or "state")
        return note

    if kind in {"done", "extract"}:
        return (action.get("answer") or text or "done").strip()

    if kind in {"click_index", "click"} and action.get("index") is not None:
        return await _finish(await _click_index(page, action))

    if kind == "scroll":
        direction = str(action.get("direction") or "down").lower()
        delta = 700 if direction != "up" else -700
        try:
            await page.mouse.wheel(0, delta)
            await page.wait_for_timeout(600)
            return await _finish(f"scroll:{direction}")
        except Exception as e:
            return await _finish(f"scroll_failed:{e}")

    if kind == "wait":
        try:
            secs = min(float(action.get("seconds") or 1), 5.0)
        except (TypeError, ValueError):
            secs = 1.0
        await page.wait_for_timeout(int(secs * 1000))
        return await _finish(f"wait:{secs}s")

    if kind in {"go_to_url", "navigate", "open_url"}:
        target_url = str(action.get("url") or text or "").strip()
        if not target_url.startswith("http"):
            return await _finish("go_to_url_invalid")
        try:
            from .navigation import navigate_robust

            await navigate_robust(page, target_url)
            return await _finish(f"go_to_url:{target_url[:60]}")
        except Exception as e:
            return await _finish(f"go_to_url_failed:{e}")

    if kind == "click" and target:
        aria_snippet = target[:40].replace("\\", "\\\\").replace('"', '\\"')
        for factory in (
            lambda: page.get_by_label(re.compile(re.escape(target[:60]), re.I)),
            lambda: page.get_by_role("button", name=re.compile(re.escape(target[:60]), re.I)),
            lambda: page.get_by_role("menuitem", name=re.compile(re.escape(target[:60]), re.I)),
            lambda: page.get_by_role("option", name=re.compile(re.escape(target[:60]), re.I)),
            lambda: page.get_by_role("link", name=re.compile(re.escape(target[:60]), re.I)),
            lambda: page.get_by_role("checkbox", name=re.compile(re.escape(target[:60]), re.I)),
            lambda: page.get_by_text(target[:80], exact=False),
            lambda: page.locator(f'[aria-label*="{aria_snippet}"]'),
        ):
            loc = factory()
            if await loc.count() > 0:
                await loc.first.click(timeout=10000)
                await page.wait_for_timeout(900)
                return await _finish(f"clicked:{target[:60]}")
        return await _finish(f"click_not_found:{target[:60]}")

    if kind == "type" and target:
        for factory in (
            lambda: page.get_by_label(target[:60]),
            lambda: page.get_by_role("textbox", name=re.compile(re.escape(target[:40]), re.I)),
            lambda: page.locator(f'input[placeholder*="{target[:30]}"]'),
            lambda: page.locator("input[type='search']").first,
        ):
            loc = factory()
            if await loc.count() > 0:
                await loc.first.fill(text, timeout=8000)
                await page.wait_for_timeout(400)
                return await _finish(f"typed:{target[:40]}")
        return await _finish(f"type_not_found:{target[:40]}")

    if kind == "drag":
        fx = float(action.get("from_x", 0))
        fy = float(action.get("from_y", 0))
        tx = float(action.get("to_x", fx))
        ty = float(action.get("to_y", fy))
        await page.mouse.move(fx, fy)
        await page.mouse.down()
        await page.mouse.move(tx, ty)
        await page.mouse.up()
        await page.wait_for_timeout(500)
        return await _finish(f"drag:{fx},{fy}->{tx},{ty}")

    return await _finish(f"unsupported_action:{kind or 'unknown'}")


async def extract_huggingface_top_models(page, *, limit: int = 3) -> str:
    """Read visible model card titles/links after HF filters are applied."""
    script = """
    (limit) => {
      const out = [];
      const seen = new Set();
      for (const a of document.querySelectorAll('a[href*="/models/"]')) {
        const href = a.getAttribute('href') || '';
        if (!href.includes('/models/') || href.endsWith('/models')) continue;
        const title = (a.innerText || a.getAttribute('aria-label') || '').trim().split('\\n')[0];
        if (!title || title.length < 2) continue;
        const key = href.split('?')[0];
        if (seen.has(key)) continue;
        seen.add(key);
        out.push({ title, href: key });
        if (out.length >= limit) break;
      }
      return out;
    }
    """
    rows = await page.evaluate(script, limit)
    if not isinstance(rows, list) or not rows:
        body = await page.locator("main").inner_text(timeout=8000)
        return body.strip()[:8000]
    lines = [f"{i + 1}. {r.get('title')} ({r.get('href')})" for i, r in enumerate(rows)]
    return "Top models (by current filters/sort):\n" + "\n".join(lines)
