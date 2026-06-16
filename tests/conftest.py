"""Pytest configuration — ensure repo root is on sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_REF_MARKER = ROOT / "state" / "sessions" / "dag_COMP_ref" / "graph.json"


@pytest.fixture(scope="session", autouse=True)
def _ensure_playwright_runtime_ready() -> None:
    """API tests must not 503 when Chromium is not installed in CI."""
    try:
        import app as app_mod

        app_mod._runtime_env_detail = lambda **kwargs: None  # type: ignore[method-assign]
    except Exception:
        pass


@pytest.fixture(scope="session", autouse=True)
def _ensure_browser_reference_sessions() -> None:
    """Seed dag_*_ref demos when missing (e.g. after scripts/clean.py)."""
    if _REF_MARKER.is_file():
        return
    from scripts.browser.seed_browser_sessions import seed_browser_reference_sessions

    seed_browser_reference_sessions()
