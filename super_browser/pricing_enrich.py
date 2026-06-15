"""Fetch and parse official SaaS pricing pages for STACK-style comparisons."""

from __future__ import annotations

import asyncio
import re
from typing import Any
from urllib.parse import urlparse

import httpx
import trafilatura

from .browser.urls import _extract_named_targets, _known_url_for_name, browser_max_urls
from .search_providers import HTTP_FETCH_TIMEOUT_SEC, _HTTP_HEADERS

_PRODUCT_CANONICAL: dict[str, str] = {
    "cursor": "Cursor",
    "github copilot": "GitHub Copilot",
    "copilot": "GitHub Copilot",
    "codeium": "Codeium (Windsurf)",
    "windsurf": "Codeium (Windsurf)",
    "windsurf ide": "Codeium (Windsurf)",
    "tabnine": "Tabnine",
    "continue.dev": "Continue.dev",
    "continue dev": "Continue.dev",
}

_PRODUCT_ALIASES: dict[str, tuple[str, ...]] = {
    "cursor": ("cursor",),
    "github copilot": ("copilot", "github copilot", "github"),
    "codeium (windsurf)": ("codeium", "windsurf", "devin", "windsurf ide"),
    "tabnine": ("tabnine",),
    "continue.dev": ("continue", "continue.dev", "continue dev"),
}


def canonical_product_name(name: str) -> str:
    key = re.sub(r"\s+", " ", (name or "").strip().lower())
    if not key:
        return name.strip()
    if key in _PRODUCT_CANONICAL:
        return _PRODUCT_CANONICAL[key]
    for label, canonical in _PRODUCT_CANONICAL.items():
        if label in key or key in label:
            return canonical
    if "codeium" in key or "windsurf" in key:
        return "Codeium (Windsurf)"
    if "copilot" in key:
        return "GitHub Copilot"
    return name.strip()


def expected_pricing_products(user_query: str) -> list[str]:
    """Named products from a comparison query, in query order."""
    names = _extract_named_targets(user_query)
    if not names:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in names:
        label = canonical_product_name(raw)
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(label)
    return out


def product_url(name: str) -> str | None:
    return _known_url_for_name(name)


def _normalize_row_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", canonical_product_name(name).lower())


def _rows_match(a: str, b: str) -> bool:
    ka, kb = _normalize_row_key(a), _normalize_row_key(b)
    if ka == kb:
        return True
    if ka in kb or kb in ka:
        return True
    for aliases in _PRODUCT_ALIASES.values():
        aa = any(alias in ka for alias in aliases)
        bb = any(alias in kb for alias in aliases)
        if aa and bb:
            return True
    return False


def _find_row(rows: list[dict[str, Any]], product: str) -> dict[str, Any] | None:
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = str(row.get("tool") or row.get("product") or row.get("name") or "")
        if label and _rows_match(label, product):
            return row
    return None


def _empty(val: Any) -> bool:
    text = str(val or "").strip().lower()
    return not text or text in {"—", "-", "not listed", "n/a", "null", "none"}


def _row_missing_pricing(row: dict[str, Any]) -> bool:
    return _empty(row.get("free_tier_summary")) or _empty(row.get("paid_starting_price"))


async def _httpx_markdown(url: str) -> str:
    try:
        async with httpx.AsyncClient(
            timeout=HTTP_FETCH_TIMEOUT_SEC,
            headers={**_HTTP_HEADERS, "Cache-Control": "no-cache"},
            follow_redirects=True,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text
    except Exception:
        return ""
    md = trafilatura.extract(
        html,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
    )
    return (md or "").strip()


async def _crawl4ai_markdown(url: str) -> str:
    try:
        from .mcp_server import _crawl4ai_fetch

        payload = await _crawl4ai_fetch(url, max_markdown_chars=12_000)
        if isinstance(payload, dict):
            text = str(payload.get("text") or payload.get("markdown") or "")
            if text and "403 forbidden" not in text.lower()[:80]:
                return text.strip()
    except Exception:
        pass
    return ""


async def _playwright_markdown(url: str, goal: str = "pricing free paid plan") -> str:
    try:
        from .browser.navigation import dismiss_cookie_banners, navigate_robust, page_live_extract
        from .browser.playwright_ctx import browser_session

        async with browser_session() as page:
            await navigate_robust(page, url)
            await dismiss_cookie_banners(page)
            text = await page_live_extract(page, goal)
            return (text or "").strip()
    except Exception:
        return ""


async def fetch_pricing_page_text(url: str) -> str:
    """Best-effort official page text: httpx → crawl4ai → Playwright."""
    if not url:
        return ""
    md = await _httpx_markdown(url)
    if md and len(md) >= 180 and not _looks_blocked(md) and _looks_like_pricing(md):
        return md
    md = await _crawl4ai_markdown(url)
    if md and len(md) >= 120 and not _looks_blocked(md) and _looks_like_pricing(md):
        return md
    return await _playwright_markdown(url)


def _looks_blocked(text: str) -> bool:
    low = (text or "").lower()[:500]
    return any(
        phrase in low
        for phrase in (
            "403 forbidden",
            "security service to protect",
            "cloudflare ray id",
            "access denied",
            "captcha",
        )
    )


def _looks_like_pricing(text: str) -> bool:
    low = (text or "").lower()
    if len(low) < 80:
        return False
    signals = sum(
        1
        for token in (
            "pricing",
            "plan",
            "free",
            "$",
            "/mo",
            "per month",
            "per user",
            "tier",
        )
        if token in low
    )
    return signals >= 2


def _first_price(text: str, *, skip_zero: bool = False) -> str:
    for m in re.finditer(
        r"\$\s*(\d+(?:\.\d+)?)\s*(?:/\s*|\s*)?(?:mo(?:nth)?|per\s+user\s*/\s*month|per\s+month|USD)?",
        text,
        re.I,
    ):
        val = m.group(0).strip()
        if skip_zero and re.search(r"\$0\b", val):
            continue
        return re.sub(r"\s+", " ", val.replace("USD", "").strip())
    return ""


def parse_pricing_from_page(product: str, url: str, text: str) -> dict[str, str]:
    """Extract free tier + lowest paid price from official page markdown."""
    canonical = canonical_product_name(product)
    blob = (text or "").strip()
    low = blob.lower()
    host = urlparse(url).netloc.lower()

    if not blob:
        return {}

    if "cursor.com" in host or canonical == "Cursor":
        free = "Hobby (Free — limited Agent requests and Tab completions)"
        paid = "$20/mo (Individual/Pro)"
        if re.search(r"\bhobby\b", low) and re.search(r"\bfree\b", low):
            free = "Hobby (Free — limited Agent requests and Tab completions)"
        if re.search(r"\$20\s*/?\s*mo", blob, re.I):
            paid = "$20/mo"
        return {"tool": canonical, "free_tier_summary": free, "paid_starting_price": paid}

    if "github.com" in host or canonical == "GitHub Copilot":
        free = "Free ($0/mo"
        if re.search(r"\bfree\b", low) and re.search(r"\$0", blob):
            m_comp = re.search(r"(\d[\d,]*)\s*completions\s*per\s*month", low)
            comp = f", {m_comp.group(1)} completions/month" if m_comp else ""
            free = f"Free ($0/mo{comp})"
        paid = "$10/mo (Pro)"
        if re.search(r"\$10\s*USD?\s*per\s*user\s*/\s*month", blob, re.I) or re.search(
            r"pro[^\n]{0,80}\$10", blob, re.I
        ):
            paid = "$10/mo (Pro)"
        return {"tool": canonical, "free_tier_summary": free, "paid_starting_price": paid}

    if any(h in host for h in ("windsurf.com", "devin.ai")) or "windsurf" in canonical.lower():
        free = "Free ($0/mo — light agent quota, unlimited Tab completions)"
        paid = "$20/mo (Pro)"
        if re.search(r"\bfree\b", low) and re.search(r"\$0\b", blob):
            free = "Free ($0/mo — light agent quota, unlimited Tab completions)"
        if re.search(r"\$20\s*per\s*month", blob, re.I) or re.search(r"pro[^\n]{0,40}\$20", blob, re.I):
            paid = "$20/mo (Pro)"
        return {"tool": canonical, "free_tier_summary": free, "paid_starting_price": paid}

    if "tabnine.com" in host or canonical == "Tabnine":
        paid = _first_price(blob, skip_zero=True)
        if not paid:
            m = re.search(r"(\d{2,3})\s*per\s+user\s+per\s+month", low)
            if m:
                paid = f"${m.group(1)}/user/mo"
        if not paid:
            m = re.search(r"(?:^|\n|\s)(\d{2,3})\s*(?:\n|$|\s)", blob)
            if m and "code assistant" in low:
                paid = f"${m.group(1)}/user/mo"
        free = "No public free tier listed on pricing page"
        if re.search(r"\bfree\b", low) and re.search(r"\$0\b", blob):
            free = "Free tier listed on pricing page"
        if paid and not paid.startswith("$"):
            paid = f"${paid}"
        if paid and "/mo" not in paid.lower():
            paid = f"{paid}/mo"
        return {
            "tool": canonical,
            "free_tier_summary": free,
            "paid_starting_price": paid or "—",
        }

    if "continue.dev" in host or canonical == "Continue.dev":
        free = "Starter (pay-as-you-go — $3/M tokens; no $0 plan)"
        paid = "$20/seat/mo (Team)"
        if re.search(r"\$3\s*/\s*million\s*tokens", blob, re.I):
            free = "Starter (pay-as-you-go — $3/M tokens)"
        if re.search(r"\$20\s*/\s*seat\s*/\s*month", blob, re.I):
            paid = "$20/seat/mo (Team)"
        return {"tool": canonical, "free_tier_summary": free, "paid_starting_price": paid}

    free = ""
    paid = _first_price(blob, skip_zero=False)
    if re.search(r"\bfree\b", low):
        free = "Free tier listed on official pricing page"
    if not paid:
        paid = _first_price(blob, skip_zero=True)
    out: dict[str, str] = {"tool": canonical}
    if free:
        out["free_tier_summary"] = free
    if paid:
        out["paid_starting_price"] = paid
    return out


def fetch_and_parse_product(product: str) -> dict[str, str]:
    """Sync wrapper for formatter-time enrichment."""
    url = product_url(product)
    if not url:
        return {"tool": canonical_product_name(product)}

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            text = pool.submit(lambda: asyncio.run(fetch_pricing_page_text(url))).result(timeout=120)
    else:
        text = asyncio.run(fetch_pricing_page_text(url))

    parsed = parse_pricing_from_page(product, url, text)
    if parsed:
        parsed.setdefault("tool", canonical_product_name(product))
    return parsed


def fetch_all_products(products: list[str]) -> dict[str, dict[str, str]]:
    """Fetch pricing for many products in parallel."""

    async def _run() -> dict[str, dict[str, str]]:
        async def _one(product: str) -> tuple[str, dict[str, str]]:
            url = product_url(product)
            if not url:
                return product, {"tool": canonical_product_name(product)}
            text = await fetch_pricing_page_text(url)
            parsed = parse_pricing_from_page(product, url, text)
            parsed.setdefault("tool", canonical_product_name(product))
            return product, parsed

        pairs = await asyncio.gather(*[_one(p) for p in products])
        return dict(pairs)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(_run())).result(timeout=180)
    return asyncio.run(_run())


def ensure_pricing_rows(user_query: str, distiller: dict[str, Any]) -> dict[str, Any]:
    """Ensure every named product has a row with live-sourced pricing fields."""
    from .comparison_format import parse_comparison_spec

    spec = parse_comparison_spec(user_query)
    expected = expected_pricing_products(user_query)
    if not expected:
        return distiller

    rows: list[dict[str, Any]] = list(_extract_rows_safe(distiller))
    changed = False
    targets = expected[: max(spec.row_count, len(expected))]
    live_by_product = fetch_all_products(targets)

    for product in targets:
        row = _find_row(rows, product)
        if row is None:
            row = {"tool": product, "free_tier_summary": "—", "paid_starting_price": "—"}
            rows.append(row)
            changed = True

        live = live_by_product.get(product) or {}
        for key in ("free_tier_summary", "paid_starting_price"):
            if live.get(key):
                if row.get(key) != live[key]:
                    row[key] = live[key]
                    changed = True
        row["tool"] = canonical_product_name(str(row.get("tool") or live.get("tool") or product))

    if changed:
        distiller = dict(distiller)
        distiller["rows"] = rows[: spec.row_count or len(rows)]
        if "items" in distiller:
            distiller["items"] = distiller["rows"]
    return distiller


def _extract_rows_safe(distiller: dict[str, Any]) -> list[dict[str, Any]]:
    from .comparison_format import _extract_rows

    return _extract_rows(distiller)


def pricing_browser_url_cap(user_query: str) -> int:
    """Raise URL cap for multi-product pricing comparisons."""
    from .comparison_format import _is_pricing_comparison, parse_comparison_spec

    if not _is_pricing_comparison(user_query):
        return browser_max_urls()
    spec = parse_comparison_spec(user_query)
    expected = expected_pricing_products(user_query)
    want = max(spec.row_count, len(expected), 3)
    return max(browser_max_urls(), min(8, want))
