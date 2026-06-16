"""Shared turn fencing rules for browser action JSON (used by tests and replay parsing)."""

from __future__ import annotations

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
