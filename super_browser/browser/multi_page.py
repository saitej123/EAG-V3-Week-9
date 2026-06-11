"""Visit multiple URLs in one Playwright session and merge live page text."""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlparse

from loguru import logger

from .navigation import (
    dismiss_cookie_banners,
    live_page_blocked,
    navigate_robust,
    page_live_extract,
)


async def crawl_urls_live(page, urls: list[str], goal: str) -> dict[str, Any] | None:
    """Navigate each URL; return merged live extract with per-page sections."""
    if len(urls) <= 1:
        return None

    started = time.time()
    transcript: list[str] = []
    sections: list[str] = []
    hosts: list[str] = []

    for idx, url in enumerate(urls[:8], start=1):
        try:
            final = await navigate_robust(page, url)
            transcript.append(f"opened:{urlparse(final).netloc or url[:40]}")
            if await live_page_blocked(page):
                sections.append(f"## Page {idx}: {final}\n\n(bot wall / captcha — could not read live pricing)")
                continue
            await dismiss_cookie_banners(page)
            try:
                await page.mouse.wheel(0, 600)
                await page.wait_for_timeout(400)
            except Exception:
                pass
            text = await page_live_extract(page, goal)
            if not text or len(text) < 80:
                sections.append(f"## Page {idx}: {final}\n\n(no readable live content extracted)")
                continue
            host = urlparse(final).netloc
            hosts.append(host)
            sections.append(
                f"## Page {idx}: {host}\n"
                f"URL: {final}\n\n"
                f"{text.strip()}"
            )
        except Exception as e:
            logger.warning(f"[browser] multi-page crawl failed for {url}: {e}")
            transcript.append(f"open_failed:{url[:50]}")
            sections.append(f"## Page {idx}: {url}\n\n(fetch error: {e})")

    body = "\n\n---\n\n".join(sections).strip()
    if len(body) < 200:
        return None

    return {
        "path": "extract",
        "url": page.url,
        "final_url": page.url,
        "host": ", ".join(dict.fromkeys(hosts)),
        "content": body[:48000],
        "content_type": "text/markdown",
        "length_chars": len(body),
        "elapsed_s": round(time.time() - started, 2),
        "llm_calls": 0,
        "transcript": transcript,
        "pages_visited": len(urls[:8]),
    }
