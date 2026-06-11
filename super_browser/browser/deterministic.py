"""Layer 2a — Playwright + pinned CSS selectors (0 LLM cost)."""

from __future__ import annotations

import re
import time
from typing import Any
from urllib.parse import quote_plus, urlparse

from loguru import logger

from .a11y import _initial_url
from .navigation import navigate_robust
from .playwright_ctx import browser_page


_AMAZON_HOSTS = frozenset({"amazon.com", "www.amazon.com", "amazon.in", "www.amazon.in"})


def _host(url: str) -> str:
    return urlparse(url).netloc.lower()


def _search_query_from_goal(goal: str) -> str | None:
    m = re.search(r'search(?:\s+for|\s+query)?\s+["\']?([^"\']+)["\']?', goal, re.I)
    if m:
        return m.group(1).strip()[:120]
    m = re.search(r"for\s+(.+?)(?:\s+and\s+|\s+then\s+|\.|$)", goal, re.I)
    if m:
        return m.group(1).strip()[:120]
    return None


async def _amazon_product_extract(page, goal: str) -> dict[str, Any]:
    query = _search_query_from_goal(goal) or "laptop"
    search_url = f"https://www.amazon.com/s?k={quote_plus(query)}"
    await navigate_robust(page, search_url)

    link = page.locator(
        'div.s-main-slot div[data-component-type="s-search-result"] h2 a.a-link-normal'
    ).first
    await link.click(timeout=15000)
    await page.wait_for_load_state("domcontentloaded", timeout=45000)
    await page.wait_for_timeout(1000)

    title = await page.locator("#productTitle").inner_text(timeout=8000)
    price = ""
    for sel in ("span.a-price span.a-offscreen", "#priceblock_ourprice", ".a-price .a-offscreen"):
        loc = page.locator(sel).first
        if await loc.count() > 0:
            price = (await loc.inner_text()).strip()
            if price:
                break
    brand = ""
    bl = page.locator("#bylineInfo").first
    if await bl.count() > 0:
        brand = (await bl.inner_text()).strip()
    description = ""
    for sel in ("#feature-bullets ul", "#productDescription"):
        loc = page.locator(sel).first
        if await loc.count() > 0:
            description = (await loc.inner_text()).strip()[:4000]
            if description:
                break

    return {
        "title": title.strip(),
        "price": price,
        "brand": brand,
        "description": description,
        "product_url": page.url,
        "search_query": query,
    }


async def layer_deterministic(url: str, goal: str, *, page=None) -> dict[str, Any] | None:
    """Run a hand-written selector workflow when the host is known."""
    host = _host(url)
    if host not in _AMAZON_HOSTS and "amazon." not in host:
        return None

    started = time.time()

    async def _on_page(pg) -> dict[str, Any] | None:
        try:
            if "amazon." in host:
                data = await _amazon_product_extract(pg, goal)
            else:
                return None
        except Exception as e:
            logger.info(f"[browser] deterministic failed for {host}: {e}")
            return None

        if not data.get("title"):
            return None

        return {
            "path": "deterministic",
            "url": pg.url,
            "host": host,
            "content": data,
            "content_type": "application/json",
            "elapsed_s": round(time.time() - started, 2),
            "llm_calls": 0,
        }

    if page is not None:
        return await _on_page(page)

    async with browser_page() as (_pw, _browser, pg):
        await navigate_robust(pg, _initial_url(url, goal))
        return await _on_page(pg)
