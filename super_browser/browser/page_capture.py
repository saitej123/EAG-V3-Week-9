"""Persist Playwright viewport screenshots for browser replay section 5."""

from __future__ import annotations

import re
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from loguru import logger


def _sessions_root() -> Path:
    from ..persistence import SESSIONS_DIR

    return SESSIONS_DIR

_active: ContextVar["PageCapture | None"] = ContextVar("browser_page_capture", default=None)

_SAFE = re.compile(r"[^a-zA-Z0-9._/-]+")


class PageCapture:
    """Write PNGs under ``state/sessions/<sid>/browser_screenshots/<node>/``."""

    def __init__(self, session_id: str, node_id: str) -> None:
        self.session_id = session_id.strip()
        self.node_id = node_id.strip() or "browser"
        safe_node = self.node_id.replace(":", "_")
        self.root = _sessions_root() / self.session_id / "browser_screenshots" / safe_node
        self._seq = 0
        self.logs: list[dict[str, Any]] = []

    def _rel_path(self, filename: str) -> str:
        safe_node = self.node_id.replace(":", "_")
        return f"browser_screenshots/{safe_node}/{filename}"

    async def log_from_page(
        self,
        page,
        *,
        turn: int | None,
        note: str,
        action: str = "",
    ) -> dict[str, Any]:
        self._seq += 1
        filename = f"{self._seq:03d}.png"
        entry: dict[str, Any] = {
            "turn": turn,
            "note": note,
            "action": action or _infer_action(note),
        }
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            out = self.root / filename
            await page.screenshot(type="png", full_page=False, path=str(out))
            entry["screenshot"] = self._rel_path(filename)
        except Exception as e:
            logger.debug(f"[browser] screenshot capture failed: {e}")
            entry["screenshot_error"] = str(e)[:160]
        self.logs.append(entry)
        return entry

    def log_bytes(
        self,
        png_bytes: bytes,
        *,
        turn: int | None,
        note: str,
        action: str = "",
    ) -> dict[str, Any]:
        self._seq += 1
        filename = f"{self._seq:03d}.png"
        entry: dict[str, Any] = {
            "turn": turn,
            "note": note,
            "action": action or _infer_action(note),
        }
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / filename).write_bytes(png_bytes)
            entry["screenshot"] = self._rel_path(filename)
        except Exception as e:
            entry["screenshot_error"] = str(e)[:160]
        self.logs.append(entry)
        return entry


def _infer_action(note: str) -> str:
    low = (note or "").lower()
    if low.startswith("clicked:"):
        return "click"
    if low.startswith("click_"):
        return "click"
    if low.startswith("scroll:"):
        return "scroll"
    if low.startswith("go_to_url:") or low.startswith("opened:"):
        return "navigate"
    if low.startswith("vision_turn:"):
        return "vision"
    if low.startswith("initial"):
        return "navigate"
    return "state"


def get_capture() -> PageCapture | None:
    return _active.get()


async def capture_page_state(
    page,
    *,
    turn: int | None = None,
    note: str,
    action: str = "",
) -> None:
    cap = get_capture()
    if cap and page is not None:
        await cap.log_from_page(page, turn=turn, note=note, action=action)


def capture_png_bytes(
    png_bytes: bytes,
    *,
    turn: int | None = None,
    note: str,
    action: str = "",
) -> None:
    cap = get_capture()
    if cap and png_bytes:
        cap.log_bytes(png_bytes, turn=turn, note=note, action=action)


def merge_capture_into(raw: dict[str, Any]) -> dict[str, Any]:
    """Attach accumulated page_state_logs to a layer result dict."""
    cap = get_capture()
    if not cap or not cap.logs:
        return raw
    raw = dict(raw)
    raw["page_state_logs"] = list(cap.logs)
    return raw


@asynccontextmanager
async def browser_capture_session(session_id: str, node_id: str):
    if session_id and node_id:
        cap = PageCapture(session_id, node_id)
        token = _active.set(cap)
        try:
            yield cap
        finally:
            _active.reset(token)
    else:
        yield None


def resolve_screenshot_path(session_id: str, rel_path: str) -> Path | None:
    """Resolve a session-relative screenshot path safely for HTTP serving."""
    sid = (session_id or "").strip()
    rel = (rel_path or "").strip().replace("\\", "/")
    if not sid or not rel or ".." in rel.split("/"):
        return None
    if not rel.startswith("browser_screenshots/"):
        return None
    if _SAFE.search(rel):
        return None
    full = (_sessions_root() / sid / rel).resolve()
    root = (_sessions_root() / sid).resolve()
    if not str(full).startswith(str(root)):
        return None
    return full if full.is_file() else None
