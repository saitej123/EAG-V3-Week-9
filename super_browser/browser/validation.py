"""Layer success gates and action counting for the browser cascade."""

from __future__ import annotations

import re
from typing import Any


def action_count(result: dict[str, Any]) -> int:
    """Count logged browser interactions in layer output."""
    actions = result.get("actions")
    if isinstance(actions, list) and actions:
        # Count actual actions inside turns if it's the nested format
        if len(actions) > 0 and "turn" in actions[0]:
            count = 0
            for turn_data in actions:
                for a in turn_data.get("actions", []):
                    if a.get("type") in {"click", "type", "scroll", "key", "drag"}:
                        count += 1
            if count > 0:
                return count
        return len(actions)
    transcript = result.get("transcript") or []
    if not isinstance(transcript, list):
        return 0
    count = 0
    for item in transcript:
        note = item if isinstance(item, str) else str((item or {}).get("note") or "")
        if not note or note.startswith("render:"):
            continue
        if note in {"fallback:body_text"}:
            continue
        if note.startswith("scroll:multi_page"):
            continue
        if note.startswith(("click_index:", "scroll:", "go_to_url:", "wait:")):
            count += 1
            continue
        if note.startswith(("browser_use:action:", "browser_use:step:")):
            count += 1
            continue
        if note.startswith(("click_", "clicked:", "typed:", "opened:", "action_failed:", "click_not_found:")):
            count += 1
            continue
        if note.startswith(("vision_turn:", "click_mark:", "click_coord:", "vlm_raw:", "vlm_live:")):
            count += 1
            continue
        if note.startswith(("a11y_turn:", "vision_turn:")):
            count += 1
            continue
        if note.startswith("vision:"):
            count += 1
            continue
    return count


def comparison_content_ready(content: Any, goal: str, *, min_browser_actions: int) -> bool:
    """True when extracted text looks like a finished comparison (not a portal homepage)."""
    if min_browser_actions <= 0:
        return True
    text = str(content or "").strip()
    if not text:
        return False

    from ..comparison_format import parse_comparison_spec

    spec = parse_comparison_spec(goal)
    row_count = spec.row_count if spec.is_comparison else 3

    table_lines = [
        ln
        for ln in text.splitlines()
        if "|" in ln and not re.match(r"^\s*\|?\s*[-:| ]+\s*\|?\s*$", ln.strip())
    ]
    if len(table_lines) >= row_count + 1:
        return True

    price_hits = len(re.findall(r"₹[\d,]+|\$[\d,]+|(?:Rs\.?|INR)\s*[\d,]+", text, re.I))
    page_sections = len(re.findall(r"^## Page \d+:", text, re.M))
    if page_sections >= row_count and len(text) >= 900:
        return True

    if needs_interactive_listing(goal, ""):
        spec_hits = len(
            re.findall(
                r"\b(?:GB|TB|RAM|Intel|Ryzen|Core i[3579]|SSD|Windows|NVIDIA|GeForce)\b",
                text,
                re.I,
            )
        )
        rating_hits = len(re.findall(r"\b[1-5](?:\.\d)?\s*(?:★|star|/5|out of 5)\b", text, re.I))
        numeric_ratings = len(re.findall(r"\b[3-5]\.\d\b", text))
        repo_hits = len(re.findall(r"\b(?:stars?|forks?|issues?|pull requests?|commits?)\b", text, re.I))
        lang_hits = len(re.findall(r"\b(?:Python|JavaScript|TypeScript|Go|Rust|C\+\+|Java|Ruby|PHP|C#|HTML|CSS|Shell|C)\b", text, re.I))
        
        if price_hits >= row_count and spec_hits >= row_count:
            return True
        if price_hits >= row_count and max(rating_hits, numeric_ratings) >= row_count:
            return True
        if repo_hits >= row_count and lang_hits >= row_count:
            return True
        if "github" in goal.lower() and repo_hits >= row_count:
            return True
        return False

    if price_hits >= row_count:
        return True

    return False


def content_rich_enough(content: Any, *, min_browser_actions: int, goal: str = "") -> bool:
    text = str(content or "").strip()
    if not text:
        return False
    if "|" in text and text.count("|") >= 4:
        return True
    if min_browser_actions <= 0:
        return len(text) >= 400
    if comparison_content_ready(text, goal, min_browser_actions=min_browser_actions):
        return True
    if "|" in text:
        from ..comparison_format import parse_comparison_spec

        spec = parse_comparison_spec(goal)
        table_lines = [
            ln
            for ln in text.splitlines()
            if "|" in ln and not re.match(r"^\s*\|?\s*[-:| ]+\s*\|?\s*$", ln.strip())
        ]
        if len(table_lines) >= spec.row_count + 1:
            return True
    return False


def needs_interactive_listing(goal: str, url: str) -> bool:
    blob = f"{goal}\n{url}".lower()
    return any(
        token in blob
        for token in (
            "flipkart",
            "amazon.in",
            "amazon.com",
            "search laptop",
            "filter price",
            "open three",
            "product page",
            "sort by rating",
            "urbanpro",
            "github.com/trending",
            "trending open-source",
            "repository pages",
        )
    )


def layer_succeeded(result: dict[str, Any] | None, *, min_browser_actions: int = 0, goal: str = "") -> bool:
    if not result:
        return False
    if result.get("error_code") == "gateway_blocked" or result.get("gateway_blocked"):
        return False
    path = str(result.get("path") or "")
    if path not in {"extract", "deterministic", "agent", "a11y", "vision"} or not result.get("content"):
        return False
    if min_browser_actions > 0:
        page_url = str(result.get("url") or result.get("final_url") or "")
        content_ok = comparison_content_ready(
            result.get("content"), goal, min_browser_actions=min_browser_actions
        ) or content_rich_enough(result.get("content"), min_browser_actions=min_browser_actions, goal=goal)
        if needs_interactive_listing(goal, page_url):
            return content_ok
        if content_ok:
            return True
        return action_count(result) >= min_browser_actions and len(str(result.get("content") or "")) >= 12
    return True
