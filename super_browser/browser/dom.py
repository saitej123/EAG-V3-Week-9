"""DOM helpers — clickable enumeration and gateway-block detection."""

from __future__ import annotations

import re
from typing import Any

_GATEWAY_PATTERNS = (
    r"confirm you are human",
    r"verify you are human",
    r"complete the captcha",
    r"recaptcha",
    r"hcaptcha",
    r"cf-turnstile",
    r"access denied",
    r"unusual traffic from your (computer|network|ip)",
    r"please enable javascript to continue",
    r"checking if the site connection is secure",
)


async def detect_live_gateway_block(page) -> bool:
    """True when the live page shows captcha / bot-wall widgets (Playwright-only)."""
    script = """
    () => {
      const sel = [
        'iframe[src*="recaptcha"]',
        'iframe[src*="hcaptcha"]',
        'iframe[title*="captcha" i]',
        '#cf-turnstile',
        '.g-recaptcha',
        '[data-sitekey]',
      ];
      for (const s of sel) {
        if (document.querySelector(s)) return true;
      }
      const t = (document.body && document.body.innerText || '').toLowerCase();
      if (t.includes('verify you are human') || t.includes('confirm you are human')) return true;
      return false;
    }
    """
    try:
        blocked = await page.evaluate(script)
        return bool(blocked)
    except Exception:
        return False

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
    from .highlight import normalize_box

    raw = await page.evaluate(_CLICKABLE_JS)
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        box = normalize_box(item.get("box"))
        if not box:
            continue
        cleaned = dict(item)
        cleaned["box"] = box
        out.append(cleaned)
    return out
