"""Resolve one or many browser target URLs from goal + query text."""

from __future__ import annotations

import re
from urllib.parse import urlparse

from ..search_providers import extract_http_urls


def canonical_browser_url(url: str) -> str:
    """Normalize http(s) URLs so trailing slashes and trivial variants dedupe."""
    u = (url or "").strip().rstrip(".,;)")
    if not u:
        return ""
    parsed = urlparse(u)
    if parsed.scheme and parsed.netloc:
        path = parsed.path.rstrip("/") or ""
        return f"{parsed.scheme}://{parsed.netloc}{path}"
    return u.rstrip("/")

# Official pricing / plans pages for common assignment comparisons (live fetch, not LLM memory).
_KNOWN_PRICING_URLS: dict[str, str] = {
    "cursor": "https://cursor.com/pricing",
    "github copilot": "https://github.com/features/copilot/plans",
    "copilot": "https://github.com/features/copilot/plans",
    "codeium": "https://windsurf.com/pricing",
    "windsurf": "https://windsurf.com/pricing",
    "windsurf ide": "https://windsurf.com/pricing",
    "devin": "https://devin.ai/pricing",
    "tabnine": "https://www.tabnine.com/pricing",
    "continue.dev": "https://continue.dev/pricing",
    "continue dev": "https://continue.dev/pricing",
    "hugging face": "https://huggingface.co/models",
    "flipkart": "https://www.flipkart.com/",
    "github trending": "https://github.com/trending",
    "github.com/trending": "https://github.com/trending",
}

# Listing / course-search portals for assignment queries without explicit URLs.
_LISTING_PORTALS: dict[str, str] = {
    "cnc bangalore": "https://www.urbanpro.com/bangalore/cnc-programming-training",
    "vmc bangalore": "https://www.urbanpro.com/bangalore/cnc-programming-training",
    "cnc vmc bangalore": "https://www.urbanpro.com/bangalore/cnc-programming-training",
    "training bangalore": "https://www.urbanpro.com/bangalore/it-training",
}

_BLOCKED_BROWSER_HOSTS = frozenset(
    {
        "google.com",
        "www.google.com",
        "bing.com",
        "www.bing.com",
        "duckduckgo.com",
        "www.duckduckgo.com",
    }
)

_MULTI_SITE_HINTS = (
    "pricing pages for",
    "visit official",
    "across the sites",
    "each site",
    "three institute",
    "three ",
    "compare 3",
    "compare three",
    "multiple sites",
    "different theatre",
    "open three",
)


def browser_max_urls() -> int:
    import os

    try:
        return max(1, min(8, int(os.environ.get("BROWSER_MAX_URLS", "3"))))
    except (TypeError, ValueError):
        return 3


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
        r"(?:pricing pages for|visit official pricing(?: pages)? for|compare)\s+(.+?)(?:\.|:\s|on each|return|use the browser|$)",
        r"(?:for)\s+([A-Za-z0-9][\w\s.\-()]+(?:,\s*[\w\s.\-()]+)+)",
    )
    for pat in patterns:
        m = re.search(pat, blob, re.I | re.S)
        if not m:
            continue
        chunk = m.group(1).strip()
        chunk = re.split(r"\s+by\s+", chunk, maxsplit=1, flags=re.I)[0]
        chunk = re.sub(r"\([^)]*\)", "", chunk)
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


def _host_of(url: str) -> str:
    try:
        from urllib.parse import urlparse

        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def is_blocked_browser_portal(url: str) -> bool:
    """Generic search homepages are poor browser targets for comparison tasks."""
    host = _host_of(url)
    if not host:
        return False
    bare = host[4:] if host.startswith("www.") else host
    return host in _BLOCKED_BROWSER_HOSTS or bare in _BLOCKED_BROWSER_HOSTS


def resolve_listing_portal_url(text: str) -> str | None:
    """Pick a course/training listings site when the query has no explicit URL."""
    low = (text or "").lower()
    if not any(
        k in low
        for k in (
            "training institute",
            "training institutes",
            "course search",
            "cnc",
            "vmc",
            "institute",
        )
    ):
        return None
    city = "bangalore" if any(c in low for c in ("bangalore", "bengaluru")) else ""
    if city and ("cnc" in low or "vmc" in low):
        return _LISTING_PORTALS["cnc bangalore"]
    if city and "training" in low:
        return _LISTING_PORTALS["training bangalore"]
    if "cnc" in low or "vmc" in low:
        return "https://www.urbanpro.com/cnc-programming-training"
    return None


def resolve_browser_urls(primary: str, goal: str, query: str = "") -> list[str]:
    """Merge explicit http(s) links with inferred official URLs for named products/sites."""
    combined = "\n".join(x for x in (primary, goal, query) if x)
    seen: set[str] = set()
    out: list[str] = []

    def _add(url: str) -> None:
        u = canonical_browser_url(url)
        if not u or u in seen:
            return
        if is_blocked_browser_portal(u):
            return
        seen.add(u)
        out.append(u)

    portal = resolve_listing_portal_url(combined)
    if portal:
        _add(portal)

    for url in extract_http_urls(combined):
        _add(url)
    if primary.startswith("http"):
        _add(primary)

    if looks_multi_site(combined):
        for name in _extract_named_targets(combined):
            known = _known_url_for_name(name)
            if known:
                _add(known)

    if not out and portal:
        _add(portal)
    elif not out and primary:
        candidate = primary if primary.startswith("http") else f"https://{primary.lstrip('/')}"
        if not is_blocked_browser_portal(candidate):
            _add(candidate)
        elif portal:
            _add(portal)

    if not out and resolve_listing_portal_url(combined):
        _add(resolve_listing_portal_url(combined) or "")

    try:
        from ..pricing_enrich import pricing_browser_url_cap

        cap = pricing_browser_url_cap(combined)
    except Exception:
        cap = browser_max_urls()
    if len(out) > cap:
        out = out[:cap]
    return out
