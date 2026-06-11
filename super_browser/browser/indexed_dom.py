"""Indexed interactive DOM state — inspired by browser-use element indexing."""

from __future__ import annotations

from typing import Any

from .highlight import normalize_box

_MAX_ELEMENTS = 120

_INDEXED_JS = """
() => {
  const isVisible = (el) => {
    if (!(el instanceof HTMLElement)) return false;
    const style = window.getComputedStyle(el);
    if (style.visibility === 'hidden' || style.display === 'none' || style.opacity === '0') return false;
    const rect = el.getBoundingClientRect();
    return rect.width >= 4 && rect.height >= 4 && rect.bottom > 0 && rect.right > 0
      && rect.top < (window.innerHeight || 900) + 200
      && rect.left < (window.innerWidth || 1280) + 200;
  };
  const labelOf = (el) => {
    return (el.getAttribute('aria-label')
      || el.getAttribute('title')
      || el.getAttribute('placeholder')
      || el.innerText
      || el.getAttribute('value')
      || el.tagName).trim().replace(/\\s+/g, ' ').slice(0, 120);
  };
  const roleOf = (el) => el.getAttribute('role') || el.tagName.toLowerCase();
  const selectors = [
    'a[href]', 'button', 'input:not([type="hidden"])', 'select', 'textarea',
    '[role="button"]', '[role="link"]', '[role="tab"]', '[role="menuitem"]',
    '[role="option"]', '[role="checkbox"]', '[onclick]', 'summary',
  ];
  const seen = new Set();
  const out = [];
  for (const sel of selectors) {
    for (const el of document.querySelectorAll(sel)) {
      if (!isVisible(el)) continue;
      const rect = el.getBoundingClientRect();
      const key = `${Math.round(rect.x)}:${Math.round(rect.y)}:${Math.round(rect.width)}:${Math.round(rect.height)}:${labelOf(el)}`;
      if (seen.has(key)) continue;
      seen.add(key);
      out.push({
        tag: el.tagName.toLowerCase(),
        role: roleOf(el),
        label: labelOf(el),
        href: el.getAttribute('href') || '',
        box: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
      });
      if (out.length >= LIMIT) return out;
    }
  }
  return out;
}
""".replace("LIMIT", str(_MAX_ELEMENTS))


async def build_indexed_interactive_state(page) -> tuple[str, dict[int, dict[str, Any]]]:
    """Return (LLM text legend, index→element map) like browser-use ``state``."""
    try:
        raw = await page.evaluate(_INDEXED_JS)
    except Exception:
        return "", {}
    if not isinstance(raw, list):
        return "", {}

    lines: list[str] = []
    selector_map: dict[int, dict[str, Any]] = {}
    for idx, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        box = normalize_box(item.get("box"))
        if not box:
            continue
        label = str(item.get("label") or "").strip()
        role = str(item.get("role") or item.get("tag") or "element")
        href = str(item.get("href") or "").strip()
        extra = f" href={href[:60]}" if href else ""
        lines.append(f"[{idx}]<{role}> {label}{extra}")
        selector_map[idx] = {**item, "box": box, "index": idx}

    header = f"URL: {page.url}\nInteractive elements ({len(selector_map)}):\n"
    return header + "\n".join(lines), selector_map
