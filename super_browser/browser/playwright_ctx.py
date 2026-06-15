"""Shared Playwright browser context with polite default anti-detection."""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

from loguru import logger

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

PLAYWRIGHT_DEPS_HINT = (
    "Playwright Chromium failed to launch (often missing Linux libraries on WSL). "
    "Try: sudo .venv/bin/python -m playwright install-deps chromium "
    "then .venv/bin/python -m playwright install chromium"
)

PLAYWRIGHT_PROXY_HINT = (
    "Playwright navigation failed (net::ERR_TUNNEL_CONNECTION_FAILED). "
    "A broken HTTP(S)_PROXY is often the cause on WSL/VPN. "
    "Try: export BROWSER_IGNORE_PROXY=1 then restart the server."
)

_STATUS_CACHE: tuple[bool, str | None] | None = None
_STATUS_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pw-status")


def _ignore_browser_proxy() -> bool:
    return os.environ.get("BROWSER_IGNORE_PROXY", "").strip().lower() in {"1", "true", "yes", "on"}


def _launch_env() -> dict[str, str] | None:
    """Strip proxy env vars when BROWSER_IGNORE_PROXY=1 (fixes ERR_TUNNEL_CONNECTION_FAILED)."""
    if not _ignore_browser_proxy():
        return None
    env = dict(os.environ)
    for key in list(env):
        if key.lower() in {"http_proxy", "https_proxy", "all_proxy"}:
            env[key] = ""
    return env


def is_playwright_proxy_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "err_tunnel_connection_failed" in text or "tunnel connection failed" in text


def is_playwright_browser_missing_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "executable doesn't exist" in text
        or "playwright install" in text
        or "browser has been closed" in text
        and "chromium" in text
    )


def _status_result(*, ready: bool, err: str | None, probe_launch: bool) -> tuple[bool, str | None]:
    global _STATUS_CACHE
    result = (ready, err)
    if not probe_launch or ready:
        _STATUS_CACHE = result
    return result


def _playwright_chromium_status_sync(*, probe_launch: bool) -> tuple[bool, str | None]:
    """Sync probe — must not run on the asyncio event loop thread."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        return False, f"playwright package missing: {e}"

    try:
        with sync_playwright() as pw:
            exe = pw.chromium.executable_path
            if not exe or not Path(exe).is_file():
                return False, PLAYWRIGHT_INSTALL_HINT
            if not probe_launch:
                return True, None
            browser = pw.chromium.launch(headless=True, args=_LAUNCH_ARGS, env=_launch_env())
            browser.close()
            return True, None
    except Exception as e:
        if is_playwright_browser_missing_error(e):
            return False, PLAYWRIGHT_INSTALL_HINT
        low = str(e).lower()
        if probe_launch and any(k in low for k in ("lib", "shared object", "cannot open", "failed to launch")):
            return False, f"{PLAYWRIGHT_DEPS_HINT} ({e})"
        return False, str(e)


def playwright_chromium_status(*, probe_launch: bool = False, refresh: bool = False) -> tuple[bool, str | None]:
    """Return (ready, error_message). Does not raise.

    Safe to call from sync code or from inside a running asyncio loop (uses a worker thread).
    """
    global _STATUS_CACHE
    if _STATUS_CACHE is not None and not probe_launch and not refresh:
        return _STATUS_CACHE

    in_loop = False
    try:
        asyncio.get_running_loop()
        in_loop = True
    except RuntimeError:
        in_loop = False

    if in_loop:
        future = _STATUS_POOL.submit(_playwright_chromium_status_sync, probe_launch=probe_launch)
        ready, err = future.result(timeout=90)
    else:
        ready, err = _playwright_chromium_status_sync(probe_launch=probe_launch)

    return _status_result(ready=ready, err=err, probe_launch=probe_launch)


async def async_playwright_chromium_status(*, probe_launch: bool = False) -> tuple[bool, str | None]:
    """Async-friendly Chromium readiness check (real launch probe when requested)."""
    if probe_launch:
        try:
            from playwright.async_api import async_playwright

            async with async_playwright() as pw:
                exe = pw.chromium.executable_path
                if not exe or not Path(exe).is_file():
                    return _status_result(ready=False, err=PLAYWRIGHT_INSTALL_HINT, probe_launch=True)
                browser = await pw.chromium.launch(headless=True, args=_LAUNCH_ARGS, env=_launch_env())
                await browser.close()
            return _status_result(ready=True, err=None, probe_launch=True)
        except Exception as e:
            if is_playwright_browser_missing_error(e):
                return _status_result(ready=False, err=PLAYWRIGHT_INSTALL_HINT, probe_launch=True)
            low = str(e).lower()
            if any(k in low for k in ("lib", "shared object", "cannot open", "failed to launch")):
                return _status_result(
                    ready=False,
                    err=f"{PLAYWRIGHT_DEPS_HINT} ({e})",
                    probe_launch=True,
                )
            return _status_result(ready=False, err=str(e), probe_launch=True)
    return await asyncio.to_thread(playwright_chromium_status, probe_launch=False, refresh=True)


async def _launch_browser(pw: "Playwright", *, headless: bool = True) -> "Browser":
    backend = os.environ.get("BROWSER_BACKEND", "chromium").strip().lower()
    env = _launch_env()
    try:
        if backend == "chrome":
            try:
                return await pw.chromium.launch(
                    channel="chrome",
                    headless=headless,
                    args=_LAUNCH_ARGS,
                    env=env,
                )
            except Exception as e:
                logger.warning(f"[browser] Chrome channel unavailable ({e}); using bundled Chromium")
        return await pw.chromium.launch(headless=headless, args=_LAUNCH_ARGS, env=env)
    except Exception as e:
        if is_playwright_browser_missing_error(e):
            raise RuntimeError(PLAYWRIGHT_INSTALL_HINT) from e
        low = str(e).lower()
        if any(k in low for k in ("lib", "shared object", "cannot open", "failed to launch")):
            raise RuntimeError(f"{PLAYWRIGHT_DEPS_HINT} ({e})") from e
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


@asynccontextmanager
async def browser_page(*, headless: bool = True) -> AsyncIterator[tuple["Playwright", "Browser", "Page"]]:
    """Yield (playwright, browser, page) with stealth-ish defaults."""
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await _launch_browser(pw, headless=headless)
        context = await _new_context(browser)
        page = await context.new_page()
        page.on("dialog", lambda dialog: dialog.accept())
        try:
            yield pw, browser, page
        finally:
            await context.close()
            await browser.close()


@asynccontextmanager
async def browser_session(*, headless: bool = True) -> AsyncIterator["Page"]:
    """Single Chromium session for the full Playwright cascade (cookies + DOM state preserved)."""
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await _launch_browser(pw, headless=headless)
        context = await _new_context(browser)
        page = await context.new_page()
        page.on("dialog", lambda dialog: dialog.accept())
        try:
            yield page
        finally:
            await context.close()
            await browser.close()
