"""Shared Playwright browser context with polite default anti-detection."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncIterator

from loguru import logger

from .browser_backend import browser_backend, browseros_cdp_url, use_browseros_cdp

if TYPE_CHECKING:
    from playwright.async_api import Browser, BrowserContext, Page, Playwright

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_WEBDRIVER_PATCH = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
window.chrome = { runtime: {} };
"""
_LAUNCH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-dev-shm-usage",
    "--no-sandbox",
    "--disable-infobars",
]

PLAYWRIGHT_INSTALL_HINT = (
    "Playwright Chromium is not installed. "
    "From the project root run: .venv/bin/python -m playwright install chromium "
    "(or restart with ./scripts/serve.sh which installs it automatically)."
)

_STATUS_CACHE: tuple[bool, str | None] | None = None


def is_playwright_browser_missing_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "executable doesn't exist" in text
        or "playwright install" in text
        or "browser has been closed" in text and "chromium" in text
    )


def playwright_chromium_status(*, probe_launch: bool = False, refresh: bool = False) -> tuple[bool, str | None]:
    """Return (ready, error_message). Does not raise.

    Default check verifies the Chromium binary path exists (fast).
    Pass ``probe_launch=True`` to actually launch the browser (serve.sh verification).
    """
    global _STATUS_CACHE
    if _STATUS_CACHE is not None and not probe_launch and not refresh:
        return _STATUS_CACHE

    try:
        from pathlib import Path

        from playwright.sync_api import sync_playwright
    except ImportError as e:
        result = (False, f"playwright package missing: {e}")
        if not probe_launch:
            _STATUS_CACHE = result
        return result

    try:
        with sync_playwright() as pw:
            exe = pw.chromium.executable_path
            if not exe or not Path(exe).is_file():
                result = (False, PLAYWRIGHT_INSTALL_HINT)
            elif probe_launch:
                browser = pw.chromium.launch(headless=True, args=_LAUNCH_ARGS)
                browser.close()
                result = (True, None)
            else:
                result = (True, None)
    except Exception as e:
        if is_playwright_browser_missing_error(e):
            result = (False, PLAYWRIGHT_INSTALL_HINT)
        else:
            result = (False, str(e))

    if not probe_launch or result[0]:
        _STATUS_CACHE = result
    return result


async def _launch_browser(pw: "Playwright", *, headless: bool = True) -> tuple["Browser", bool]:
    """Return (browser, external). External browsers must not be closed by us."""
    cdp = browseros_cdp_url() if use_browseros_cdp() else ""
    if cdp:
        try:
            browser = await pw.chromium.connect_over_cdp(cdp)
            logger.info(f"[browser] connected to BrowserOS/CDP at {cdp}")
            return browser, True
        except Exception as e:
            logger.warning(f"[browser] CDP connect failed ({cdp}): {e} — falling back to launch")

    backend = browser_backend()
    try:
        if backend == "chrome":
            browser = await pw.chromium.launch(
                channel="chrome",
                headless=headless,
                args=_LAUNCH_ARGS,
            )
        else:
            browser = await pw.chromium.launch(headless=headless, args=_LAUNCH_ARGS)
        return browser, False
    except Exception as e:
        if is_playwright_browser_missing_error(e):
            raise RuntimeError(PLAYWRIGHT_INSTALL_HINT) from e
        raise


async def _new_context(browser: "Browser") -> "BrowserContext":
    context = await browser.new_context(
        user_agent=_USER_AGENT,
        java_script_enabled=True,
        viewport={"width": 1280, "height": 900},
        locale="en-US",
        timezone_id="America/New_York",
        color_scheme="light",
    )
    await context.add_init_script(_WEBDRIVER_PATCH)
    return context


async def _open_page(browser: "Browser", *, external: bool) -> tuple["BrowserContext", "Page"]:
    if external and browser.contexts:
        context = browser.contexts[0]
        page = context.pages[0] if context.pages else await context.new_page()
    else:
        context = await _new_context(browser)
        page = await context.new_page()
    page.on("dialog", lambda dialog: dialog.accept())
    return context, page


@asynccontextmanager
async def browser_page(*, headless: bool = True) -> AsyncIterator[tuple["Playwright", "Browser", "Page"]]:
    """Yield (playwright, browser, page) with stealth-ish defaults."""
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser, external = await _launch_browser(pw, headless=headless)
        context, page = await _open_page(browser, external=external)
        try:
            yield pw, browser, page
        finally:
            if not external:
                await context.close()
                await browser.close()


@asynccontextmanager
async def browser_session(*, headless: bool = True) -> AsyncIterator["Page"]:
    """Single Chromium session for the full Playwright cascade (cookies + DOM state preserved)."""
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser, external = await _launch_browser(pw, headless=headless)
        context, page = await _open_page(browser, external=external)
        try:
            yield page
        finally:
            if not external:
                await context.close()
                await browser.close()
