"""Parse LLM/VLM browser responses — never raise; salvage prose and coordinates."""

from __future__ import annotations

import json
import re
from typing import Any

from ..llm_retry import loads_json_lenient


def parse_action_json(raw: str, *, allow_prose_done: bool = True) -> dict[str, Any]:
    """Turn model output into an action dict. Safe on empty or non-JSON replies."""
    text = (raw or "").strip()
    if not text:
        return {"action": "noop"}

    try:
        data = loads_json_lenient(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    low = text.lower()
    if re.search(r"\|\s*[^|]+\s*\|", text):
        return {"action": "done", "answer": text}

    if allow_prose_done and len(text) >= 80 and not text.startswith("{"):
        if any(k in low for k in ("compare", "price", "laptop", "model", "product", "₹", "rs.", "table")):
            return {"action": "done", "answer": text}
        if re.search(r"\|\s*[^|]+\s*\|", text):
            return {"action": "done", "answer": text}

    if re.search(r'\baction["\']?\s*[:=]\s*["\']?done', low):
        ans = _extract_answer_field(text) or text
        return {"action": "done", "answer": ans}

    mark_m = re.search(r'\bmark["\']?\s*[:=]\s*(\d{1,2})\b', text, re.I)
    if mark_m:
        return {"action": "click", "mark": int(mark_m.group(1))}

    bracket_m = re.search(r"\[(\d{1,2})\]", text)
    if bracket_m:
        return {"action": "click", "mark": int(bracket_m.group(1))}

    lone_m = re.search(r"\bclick(?:\s+mark)?\s*(\d{1,2})\b", low)
    if lone_m:
        return {"action": "click", "mark": int(lone_m.group(1))}

    coord_m = re.search(
        r'x["\']?\s*[:=]\s*(\d+(?:\.\d+)?)\D+ y["\']?\s*[:=]\s*(\d+(?:\.\d+)?)',
        text,
        re.I | re.S,
    )
    if coord_m:
        return {
            "action": "click_coord",
            "x": float(coord_m.group(1)),
            "y": float(coord_m.group(2)),
        }

    if allow_prose_done and len(text) >= 120:
        return {"action": "done", "answer": text}

    return {"action": "noop", "raw": text[:500]}


def _extract_answer_field(text: str) -> str:
    m = re.search(r'"answer"\s*:\s*"((?:\\.|[^"\\])*)"', text, re.S)
    if m:
        try:
            return json.loads(f'"{m.group(1)}"')
        except json.JSONDecodeError:
            return m.group(1).replace('\\"', '"')
    return ""
