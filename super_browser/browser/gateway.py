"""Captcha / bot-wall detection for static HTML and live Playwright pages."""

from __future__ import annotations

import re

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


def detect_gateway_block(text: str) -> bool:
    """True when page content looks like CAPTCHA / login / rate-limit wall."""
    low = (text or "").lower()
    return any(re.search(pat, low) for pat in _GATEWAY_PATTERNS)
