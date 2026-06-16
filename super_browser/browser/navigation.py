"""Robust Playwright navigation — retries, fallbacks, never raise to caller."""

from __future__ import annotations

import asyncio
import re
from typing import Any
from urllib.parse import urlparse

import trafilatura
from loguru import logger

from .gateway import detect_gateway_block, detect_live_gateway_block
from .extract import content_is_useful
from .playwright_ctx import PLAYWRIGHT_PROXY_HINT, is_playwright_proxy_error

_NAV_TIMEOUT_MS = 45_000
_RETRY_DELAYS_S = (0.0, 1.5, 3.0)
_WAIT_UNTIL = ("domcontentloaded", "load", "networkidle")
_COOKIE_LABELS = ("Accept all", "Accept", "I agree", "Got it", "Allow all", "OK")


async def wait_for_page_ready(page) -> None:
    """Let SPAs hydrate — scroll + short settle after load."""
    for state in ("domcontentloaded", "networkidle"):
        try:
            await page.wait_for_load_state(state, timeout=12_000 if state == "networkidle" else 20_000)
        except Exception:
            pass
    await page.wait_for_timeout(900)
    try:
        await page.evaluate(
            """async () => {
              const h = Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);
              window.scrollTo(0, Math.min(h * 0.4, 1200));
              await new Promise(r => setTimeout(r, 400));
              window.scrollTo(0, 0);
            }"""
        )
    except Exception:
        pass
    await page.wait_for_timeout(400)


async def dismiss_cookie_banners(page) -> None:
    """Click common consent buttons so pricing text is visible."""
    for label in _COOKIE_LABELS:
        try:
            btn = page.get_by_role("button", name=re.compile(re.escape(label), re.I))
            if await btn.count() > 0:
                await btn.first.click(timeout=2500)
                await page.wait_for_timeout(500)
                return
        except Exception:
            continue


async def page_pricing_snippets(page) -> str:
    """Collect visible pricing/plan fragments from the live DOM."""
    script = """
    () => {
      const out = [];
      const seen = new Set();
      const push = (t) => {
        const s = (t || '').replace(/\\s+/g, ' ').trim();
        if (s.length < 4 || s.length > 600 || seen.has(s)) return;
        seen.add(s);
        out.push(s);
      };
      const selectors = [
        '[class*="price" i]', '[class*="pricing" i]', '[class*="plan" i]',
        '[class*="tier" i]', '[data-testid*="price" i]', '[data-testid*="plan" i]',
        'h1', 'h2', 'h3', '[role="heading"]',
      ];
      for (const sel of selectors) {
        for (const el of document.querySelectorAll(sel)) {
          push(el.innerText);
        }
      }
      return out.slice(0, 40).join('\\n');
    }
    """
    try:
        raw = await page.evaluate(script)
        return str(raw or "").strip()
    except Exception:
        return ""


async def page_live_extract(page, goal: str) -> str | None:
    """Fresh visible text from the rendered page (not httpx cache)."""
    await wait_for_page_ready(page)
    snippets = await page_pricing_snippets(page)
    body = await page_body_fallback(page, min_chars=120)
    parts: list[str] = [f"LIVE_URL: {page.url}"]
    if snippets:
        parts.append("PRICING_SNIPPETS:\n" + snippets)
    if body:
        parts.append("VISIBLE_TEXT:\n" + body)
    merged = "\n\n".join(parts).strip()
    if len(merged) < 80:
        return None
    if not content_is_useful(merged, goal) and len(merged) < 400:
        return merged if len(merged) >= 120 else None
    return merged[:14000]


async def navigate_robust(page, url: str) -> str:
    """Load ``url`` with retries; returns final URL (may differ after redirects)."""
    last_err: Exception | None = None
    for attempt, delay in enumerate(_RETRY_DELAYS_S):
        if delay:
            await asyncio.sleep(delay)
        for wait_until in _WAIT_UNTIL:
            try:
                await page.goto(
                    url,
                    wait_until=wait_until,
                    timeout=_NAV_TIMEOUT_MS,
                )
                await page.wait_for_timeout(800 if wait_until != "networkidle" else 400)
                return page.url
            except Exception as e:
                last_err = e
                logger.debug(f"[browser] goto attempt {attempt + 1} wait={wait_until}: {e}")
                continue
    logger.warning(f"[browser] navigate_robust failed for {url!r}: {last_err}")
    if last_err and is_playwright_proxy_error(last_err):
        logger.error(f"[browser] {PLAYWRIGHT_PROXY_HINT}")
    return page.url or url


async def html_to_useful_markdown(html: str, goal: str) -> str | None:
    if detect_gateway_block(html):
        return None
    content = trafilatura.extract(
        html,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
    )
    if not content or not content_is_useful(content, goal):
        return None
    return content.strip()


async def page_markdown_extract(page, goal: str) -> str | None:
    """Rendered DOM → prefer live visible text; trafilatura as fallback."""
    live = await page_live_extract(page, goal)
    if live and len(live) >= 200:
        return live
    try:
        html = await page.content()
    except Exception as e:
        logger.debug(f"[browser] page.content failed: {e}")
        return live
    md = await html_to_useful_markdown(html, goal)
    if md and live:
        return f"{live}\n\n---\n\n{md}"[:14000]
    return md or live


async def page_body_fallback(page, *, min_chars: int = 400) -> str | None:
    """Plain innerText when structured extract fails — last-resort readable content."""
    for selector in ("main", "article", "[role='main']", "body"):
        try:
            loc = page.locator(selector).first
            if await loc.count() == 0:
                continue
            text = await loc.inner_text(timeout=8000)
            text = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
            if len(text) >= min_chars:
                return text[:12000]
        except Exception:
            continue
    return None


async def live_page_blocked(page) -> bool:
    try:
        if await detect_live_gateway_block(page):
            return True
        html = await page.content()
        return detect_gateway_block(html)
    except Exception:
        return False
