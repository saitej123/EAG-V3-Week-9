"""Set-of-marks helpers — dedupe overlapping clickables and draw numbered boxes."""

from __future__ import annotations

import io
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def _area(box: dict[str, float]) -> float:
    return max(0.0, box["width"]) * max(0.0, box["height"])


def _contains(outer: dict[str, float], inner: dict[str, float], *, margin: float = 2.0) -> bool:
    return (
        inner["x"] >= outer["x"] - margin
        and inner["y"] >= outer["y"] - margin
        and inner["x"] + inner["width"] <= outer["x"] + outer["width"] + margin
        and inner["y"] + inner["height"] <= outer["y"] + outer["height"] + margin
    )


def _overlap_ratio(a: dict[str, float], b: dict[str, float]) -> float:
    x1 = max(a["x"], b["x"])
    y1 = max(a["y"], b["y"])
    x2 = min(a["x"] + a["width"], b["x"] + b["width"])
    y2 = min(a["y"] + a["height"], b["y"] + b["height"])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    smaller = min(_area(a), _area(b))
    return inter / smaller if smaller > 0 else 0.0


def dedupe_clickables(items: list[dict[str, Any]], *, overlap_threshold: float = 0.85) -> list[dict[str, Any]]:
    """Remove nested/overlapping boxes; keep the larger outer control."""
    if len(items) <= 1:
        return items

    sorted_items = sorted(items, key=lambda it: _area(it["box"]), reverse=True)
    kept: list[dict[str, Any]] = []
    for candidate in sorted_items:
        box = candidate["box"]
        drop = False
        for existing in kept:
            ex_box = existing["box"]
            if _contains(ex_box, box) or _overlap_ratio(ex_box, box) >= overlap_threshold:
                drop = True
                break
        if not drop:
            kept.append(candidate)

    for i, item in enumerate(kept, start=1):
        item["mark"] = i
    return kept


def draw_marks(
    screenshot_bytes: bytes,
    items: list[dict[str, Any]],
    *,
    device_pixel_ratio: float = 1.0,
) -> bytes:
    """Draw numbered boxes on a screenshot; scale CSS coords by DPR."""
    img = Image.open(io.BytesIO(screenshot_bytes)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    dpr = device_pixel_ratio if device_pixel_ratio > 0 else 1.0

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(12, int(14 * dpr)))
    except OSError:
        font = ImageFont.load_default()

    for item in items:
        box = item["box"]
        mark = int(item.get("mark") or 0)
        x = int(box["x"] * dpr)
        y = int(box["y"] * dpr)
        w = int(box["width"] * dpr)
        h = int(box["height"] * dpr)
        draw.rectangle([x, y, x + w, y + h], outline=(255, 64, 64, 230), width=max(2, int(2 * dpr)))
        label = f"[{mark}]"
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        pad = max(2, int(2 * dpr))
        draw.rectangle([x, max(0, y - th - pad * 2), x + tw + pad * 2, y], fill=(255, 64, 64, 220))
        draw.text((x + pad, max(0, y - th - pad)), label, fill=(255, 255, 255, 255), font=font)

    out = Image.alpha_composite(img, overlay).convert("RGB")
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()
