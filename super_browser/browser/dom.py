"""DOM helpers — clickable enumeration and gateway-block detection."""

from __future__ import annotations

import re
from typing import Any

_GATEWAY_PATTERNS = (
    r"confirm you are human",
    r"verify you are human",
    r"captcha",
    r"access denied",
    r"rate limit",
    r"please enable javascript",
    r"unusual traffic",
)

_CLICKABLE_JS = """
() => {
  const selectors = [
    'a[href]',
    'button',
    'input:not([type="hidden"])',
    'select',
    'textarea',
    '[role="button"]',
    '[role="link"]',
    '[role="tab"]',
    '[onclick]',
    'summary',
  ];
  const seen = new Set();
  const out = [];
  for (const sel of selectors) {
    for (const el of document.querySelectorAll(sel)) {
      if (!(el instanceof HTMLElement)) continue;
      const rect = el.getBoundingClientRect();
      if (rect.width < 4 || rect.height < 4) continue;
      const style = window.getComputedStyle(el);
      if (style.visibility === 'hidden' || style.display === 'none') continue;
      const key = `${Math.round(rect.x)}:${Math.round(rect.y)}:${Math.round(rect.width)}:${Math.round(rect.height)}`;
      if (seen.has(key)) continue;
      seen.add(key);
      const label = (el.getAttribute('aria-label')
        || el.innerText
        || el.getAttribute('title')
        || el.getAttribute('placeholder')
        || el.tagName).trim().slice(0, 120);
      out.push({
        tag: el.tagName.toLowerCase(),
        label,
        box: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
      });
      if (out.length >= 80) return out;
    }
  }
  return out;
}
"""


def detect_gateway_block(text: str) -> bool:
    """True when page content looks like CAPTCHA / login / rate-limit wall."""
    low = (text or "").lower()
    return any(re.search(pat, low) for pat in _GATEWAY_PATTERNS)


async def page_device_pixel_ratio(page) -> float:
    try:
        dpr = await page.evaluate("() => window.devicePixelRatio || 1")
        return float(dpr) if dpr else 1.0
    except Exception:
        return 1.0


async def collect_clickables(page) -> list[dict[str, Any]]:
    raw = await page.evaluate(_CLICKABLE_JS)
    return raw if isinstance(raw, list) else []
