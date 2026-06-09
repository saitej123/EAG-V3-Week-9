"""Layer 2b — Playwright accessibility tree + cheap text LLM judgment."""

from __future__ import annotations

import re
import time
from typing import Any
from urllib.parse import urlparse

from loguru import logger

from ..llm_retry import loads_json_lenient
from .dom import collect_clickables
from .ledger import apply_cost_fields
from .driver import (
    TurnResult,
    a11y_turn_prompt,
    execute_action,
    extract_huggingface_top_models,
    fence_actions,
    normalize_actions,
)
from .playwright_ctx import browser_page

_MAX_TURNS = 8
_MAX_TREE_CHARS = 12000
_HF_HOSTS = frozenset({"huggingface.co", "www.huggingface.co"})


def _flatten_a11y(node: dict[str, Any] | None, depth: int = 0) -> list[str]:
    if not node or depth > 12:
        return []
    lines: list[str] = []
    role = node.get("role") or ""
    name = (node.get("name") or "").strip()
    value = (node.get("value") or "").strip()
    if role or name:
        bit = f"{'  ' * depth}[{role}] {name}"
        if value and value != name:
            bit += f" = {value[:80]}"
        lines.append(bit)
    for child in node.get("children") or []:
        if isinstance(child, dict):
            lines.extend(_flatten_a11y(child, depth + 1))
    return lines


async def a11y_snapshot(page) -> str:
    """Compact accessibility summary — snapshot API with aria fallback."""
    tree = None
    try:
        snap_fn = getattr(getattr(page, "accessibility", None), "snapshot", None)
        if callable(snap_fn):
            tree = await snap_fn()
    except Exception:
        tree = None
    if not tree:
        try:
            aria = await page.locator("body").aria_snapshot()
            if isinstance(aria, str) and aria.strip():
                return aria.strip()[:_MAX_TREE_CHARS]
        except Exception:
            pass
        return ""
    lines = _flatten_a11y(tree if isinstance(tree, dict) else None)
    return "\n".join(lines).strip()[:_MAX_TREE_CHARS]


def _initial_url(url: str, goal: str) -> str:
    host = urlparse(url).netloc.lower()
    goal_l = goal.lower()
    if host in _HF_HOSTS or "huggingface.co" in url.lower():
        if "/models" not in url:
            return "https://huggingface.co/models"
    if "huggingface" in goal_l and "models" in goal_l and "/models" not in url:
        return "https://huggingface.co/models"
    return url


def _hf_goal(goal: str) -> bool:
    g = goal.lower()
    return "huggingface" in g or ("model" in g and any(k in g for k in ("filter", "sort", "likes", "transformers")))


def _tree_too_empty(tree: str) -> bool:
    lines = [ln for ln in tree.splitlines() if ln.strip()]
    return len(tree.strip()) < 40 or len(lines) < 3


def _refuse_empty_done(raw: dict[str, Any], tree: str, goal: str) -> bool:
    """Block done(success) on empty tree when goal says do not guess."""
    if not _tree_too_empty(tree):
        return False
    if "do not guess" not in goal.lower() and "don't guess" not in goal.lower():
        return False
    kind = (raw.get("action") or "").lower()
    if kind == "done" and raw.get("success") is False:
        return True
    answer = str(raw.get("answer") or "")
    return kind == "done" and not answer.strip()


async def run_a11y_loop(page, goal: str, llm: Any) -> TurnResult | None:
    """Turn loop: fresh a11y summary → LLM → fenced actions → execute."""
    started_notes: list[str] = []
    llm_calls = 0
    input_tokens = 0
    output_tokens = 0

    for turn in range(1, _MAX_TURNS + 1):
        tree = await a11y_snapshot(page)
        if _tree_too_empty(tree):
            clickables = await collect_clickables(page)
            if len(clickables) < 3:
                if "do not guess" in goal.lower() or "don't guess" in goal.lower():
                    return None
                break

        prompt = a11y_turn_prompt(goal=goal, tree=tree, url=page.url, turn=turn)
        raw_text = llm.chat(agent="browser", prompt=prompt, temperature=0.2, max_tokens=512)
        llm_calls += 1
        # Rough token estimate when gateway usage metadata is unavailable (~prompt + ~120 out)
        input_tokens += max(len(prompt) // 4, 1)
        output_tokens += max(len(str(raw_text)) // 4, 40)
        raw = loads_json_lenient(raw_text)
        if not isinstance(raw, dict):
            logger.info("[browser] a11y LLM returned non-JSON — escalating")
            return None

        if _refuse_empty_done(raw, tree, goal):
            logger.info("[browser] a11y refused empty-tree done — escalating to vision")
            return None

        if (raw.get("action") or "").lower() in {"done", "extract"}:
            answer = str(raw.get("answer") or "").strip()
            if answer:
                return TurnResult(
                    notes=started_notes,
                    done=True,
                    answer=answer,
                    turns=turn,
                    llm_calls=llm_calls,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

        actions = fence_actions(normalize_actions(raw))
        if not actions:
            continue

        notes: list[str] = []
        for action in actions:
            kind = (action.get("action") or "").lower()
            if kind in {"done", "extract"}:
                answer = str(action.get("answer") or "").strip()
                if answer:
                    return TurnResult(
                        notes=started_notes + notes,
                        done=True,
                        answer=answer,
                        turns=turn,
                        llm_calls=llm_calls,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
            try:
                notes.append(await execute_action(page, action))
            except Exception as e:
                notes.append(f"action_failed:{e}")
                break
        started_notes.extend(notes)

        # HF: auto-extract when filters appear applied and cards are visible
        if _hf_goal(goal) and "pipeline_tag=" in page.url and "sort=likes" in page.url:
            answer = await extract_huggingface_top_models(page, limit=3)
            if answer:
                return TurnResult(
                    notes=started_notes,
                    done=True,
                    answer=answer,
                    turns=turn,
                    llm_calls=llm_calls,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

    return None


async def layer_a11y(url: str, goal: str, llm: Any) -> dict[str, Any] | None:
    """Navigate with Playwright; use a11y tree + text LLM for actions."""
    started = time.time()
    url = _initial_url(url, goal)

    async with browser_page() as (_pw, _browser, page):
        await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        await page.wait_for_timeout(1200)

        result = await run_a11y_loop(page, goal, llm)
        if result and result.done and result.answer:
            return apply_cost_fields(
                {
                    "path": "a11y",
                    "url": page.url,
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

        body = await page.locator("body").inner_text(timeout=8000)
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        if len(body) >= 400 and not _tree_too_empty(await a11y_snapshot(page)):
            return {
                "path": "a11y",
                "url": page.url,
                "content": body[:12000],
                "content_type": "text/plain",
                "transcript": [],
                "elapsed_s": round(time.time() - started, 2),
                "llm_calls": 0,
            }
    return None
