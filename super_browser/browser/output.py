"""Build BrowserOutput from raw cascade dicts and classify errors."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from ..dag_schemas import BrowserErrorCode, BrowserOutput
from .ledger import apply_cost_fields, estimate_cost_usd

_LAYER_PATHS = frozenset({"extract", "deterministic", "agent", "a11y", "vision", "gateway_blocked", "failed"})


def _content_text(content: Any) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, indent=2)
    return str(content)


def _actions_from_transcript(transcript: list[str] | None) -> list[dict[str, Any]]:
    return [{"note": note} for note in (transcript or []) if note]


def to_browser_output(*, url: str, goal: str, raw: dict[str, Any]) -> BrowserOutput:
    from .page_capture import merge_capture_into

    try:
        raw = apply_cost_fields(merge_capture_into(dict(raw)))
    except Exception:
        raw = merge_capture_into(dict(raw))
    path = str(raw.get("path") or "failed")
    if path not in _LAYER_PATHS:
        path = "failed"
    inp = int(raw.get("input_tokens") or 0)
    out = int(raw.get("output_tokens") or 0)
    page_logs = raw.get("page_state_logs")
    if not isinstance(page_logs, list) or not page_logs:
        page_logs = _actions_from_transcript(raw.get("transcript"))
    actions = raw.get("actions")
    if not isinstance(actions, list) or not actions:
        actions = page_logs if page_logs else _actions_from_transcript(raw.get("transcript"))
    payload = dict(
        url=url,
        goal=goal,
        path=path,
        turns=int(raw.get("turns") or 0),
        content=_content_text(raw.get("content")),
        actions=actions,
        page_state_logs=page_logs if isinstance(page_logs, list) else [],
        final_url=str(raw.get("url") or raw.get("final_url") or url),
        elapsed_s=raw.get("elapsed_s") or raw.get("total_elapsed_s"),
        llm_calls=int(raw.get("llm_calls") or 0),
        input_tokens=inp,
        output_tokens=out,
        cost_usd=float(
            raw.get("cost_usd")
            if raw.get("cost_usd") is not None
            else estimate_cost_usd(input_tokens=inp, output_tokens=out)
        ),
    )
    try:
        return BrowserOutput(**payload)
    except Exception as e:
        logger.warning(f"[browser] BrowserOutput validation failed ({e}); coercing to failed")
        payload["path"] = "failed"
        return BrowserOutput(**payload)


def classify_browser_error(raw: dict[str, Any], *, last_layer: str | None = None) -> BrowserErrorCode:
    code = raw.get("error_code")
    if code in {
        "gateway_blocked",
        "extraction_failed",
        "interaction_failed",
        "timeout",
        "vlm_unavailable",
    }:
        return code  # type: ignore[return-value]
    if raw.get("gateway_blocked"):
        return "gateway_blocked"
    if last_layer == "vision" or raw.get("vlm_error"):
        return "vlm_unavailable"
    if last_layer in {"a11y", "agent", "vision", "deterministic"}:
        return "interaction_failed"
    return "extraction_failed"
