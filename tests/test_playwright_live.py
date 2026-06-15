"""Live Playwright smoke tests (skipped when Chromium unavailable)."""

from __future__ import annotations

import asyncio

import pytest

from super_browser.browser.playwright_ctx import (
    async_playwright_chromium_status,
    browser_session,
    playwright_chromium_status,
)


def _require_chromium() -> None:
    ready, err = playwright_chromium_status(refresh=True)
    if not ready:
        pytest.skip(err or "Playwright Chromium not installed")


def test_playwright_status_safe_inside_asyncio_loop():
    """Status probe must not use sync Playwright on the event-loop thread."""

    async def _probe() -> tuple[bool, str | None]:
        return playwright_chromium_status(probe_launch=True, refresh=True)

    ready, err = asyncio.run(_probe())
    if not ready:
        pytest.skip(err or "Chromium not available")
    assert err is None


def test_async_status_probe_launch():
    _require_chromium()

    async def _run() -> None:
        ready, err = await async_playwright_chromium_status(probe_launch=True)
        assert ready is True
        assert err is None

    asyncio.run(_run())


def test_browser_session_navigates_example_com():
    _require_chromium()

    async def _run() -> None:
        async with browser_session() as page:
            await page.goto("https://example.com", wait_until="domcontentloaded", timeout=45000)
            title = await page.title()
            assert "Example" in title

    asyncio.run(_run())
