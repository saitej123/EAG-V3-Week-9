"""Resolve one or many browser target URLs from goal + query text."""

from __future__ import annotations

import re

from ..search_providers import extract_http_urls

# Official pricing / plans pages for common assignment comparisons (live fetch, not LLM memory).
_KNOWN_PRICING_URLS: dict[str, str] = {
    "cursor": "https://cursor.com/pricing",
    "github copilot": "https://github.com/features/copilot/plans",
    "copilot": "https://github.com/features/copilot/plans",
    "codeium": "https://codeium.com/pricing",
    "windsurf": "https://codeium.com/pricing",
    "tabnine": "https://www.tabnine.com/pricing",
    "continue.dev": "https://continue.dev/pricing",
    "continue dev": "https://continue.dev/pricing",
    "continue": "https://continue.dev/pricing",
    "hugging face": "https://huggingface.co/models",
    "flipkart": "https://www.flipkart.com/",
    "bookmyshow": "https://in.bookmyshow.com/",
}

_MULTI_SITE_HINTS = (
    "pricing pages for",
    "visit official",
    "across the sites",
    "each site",
    "five institute",
    "five ",
    "compare 5",
    "compare five",
    "multiple sites",
    "different theatre",
    "open five",
)


def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def _known_url_for_name(name: str) -> str | None:
    key = _normalize_name(name)
    if not key:
        return None
    if key in _KNOWN_PRICING_URLS:
        return _KNOWN_PRICING_URLS[key]
    for label, url in _KNOWN_PRICING_URLS.items():
        if label in key or key in label:
            return url
    return None


def _extract_named_targets(text: str) -> list[str]:
    """Pull comma/and-separated product or site names from comparison goals."""
    blob = (text or "").strip()
    if not blob:
        return []

    patterns = (
        r"(?:pricing pages for|visit official pricing pages for|compare)\s+(.+?)(?:\.|:\s|on each|return|use the browser|$)",
        r"(?:for)\s+([A-Za-z0-9][\w\s.\-]+(?:,\s*[\w\s.\-]+)+)",
    )
    for pat in patterns:
        m = re.search(pat, blob, re.I | re.S)
        if not m:
            continue
        chunk = m.group(1).strip()
        chunk = re.split(r"\s+by\s+", chunk, maxsplit=1, flags=re.I)[0]
        parts = re.split(r",\s*|\s+and\s+", chunk)
        names = [p.strip(" .") for p in parts if p.strip(" .")]
        if len(names) >= 2:
            return names
    return []


def looks_multi_site(text: str) -> bool:
    low = (text or "").lower()
    if len(_extract_named_targets(text)) >= 2:
        return True
    return any(h in low for h in _MULTI_SITE_HINTS)


def resolve_browser_urls(primary: str, goal: str, query: str = "") -> list[str]:
    """Merge explicit http(s) links with inferred official URLs for named products/sites."""
    combined = "\n".join(x for x in (primary, goal, query) if x)
    seen: set[str] = set()
    out: list[str] = []

    def _add(url: str) -> None:
        u = (url or "").strip().rstrip(".,;)")
        if not u or u in seen:
            return
        seen.add(u)
        out.append(u)

    for url in extract_http_urls(combined):
        _add(url)
    if primary.startswith("http"):
        _add(primary)

    if looks_multi_site(combined):
        for name in _extract_named_targets(combined):
            known = _known_url_for_name(name)
            if known:
                _add(known)

    if not out and primary:
        _add(primary if primary.startswith("http") else f"https://{primary.lstrip('/')}")
    return out
