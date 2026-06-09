"""Cost ledger helpers — token totals and USD estimate for BrowserOutput."""

from __future__ import annotations

import os
from typing import Any


def aggregate_tokens(raw: dict[str, Any]) -> tuple[int, int]:
    """Sum input/output tokens from layer payloads (vision_tokens, a11y tokens, etc.)."""
    inp = int(raw.get("input_tokens") or 0)
    out = int(raw.get("output_tokens") or 0)
    vt = raw.get("vision_tokens")
    if isinstance(vt, dict):
        inp += int(vt.get("input") or 0)
        out += int(vt.get("output") or 0)
    return inp, out


def estimate_cost_usd(*, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD from token counts. Free-tier Flash-Lite runs observed at $0.00."""
    if input_tokens <= 0 and output_tokens <= 0:
        return 0.0
    in_rate = float(os.environ.get("BROWSER_COST_INPUT_PER_MTOK", "0") or 0)
    out_rate = float(os.environ.get("BROWSER_COST_OUTPUT_PER_MTOK", "0") or 0)
    if in_rate == 0.0 and out_rate == 0.0:
        return 0.0
    return (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate


def apply_cost_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """Attach input_tokens, output_tokens, cost_usd to a layer result dict."""
    inp, out = aggregate_tokens(raw)
    raw["input_tokens"] = inp
    raw["output_tokens"] = out
    raw["cost_usd"] = round(estimate_cost_usd(input_tokens=inp, output_tokens=out), 4)
    return raw
