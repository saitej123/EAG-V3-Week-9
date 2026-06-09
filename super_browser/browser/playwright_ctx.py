"""Shared Playwright browser context with polite default anti-detection."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncIterator

if TYPE_CHECKING:
    from playwright.async_api import Browser, Page, Playwright

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_WEBDRIVER_PATCH = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
"""


@asynccontextmanager
async def browser_page(*, headless: bool = True) -> AsyncIterator[tuple["Playwright", "Browser", "Page"]]:
    """Yield (playwright, browser, page) with stealth-ish defaults."""
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=headless,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(
            user_agent=_USER_AGENT,
            java_script_enabled=True,
            viewport={"width": 1280, "height": 900},
        )
        await context.add_init_script(_WEBDRIVER_PATCH)
        page = await context.new_page()
        try:
            yield pw, browser, page
        finally:
            await context.close()
            await browser.close()
