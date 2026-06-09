"""Browser skill entry — four-layer cascade returning BrowserOutput."""

from __future__ import annotations

import json
import time
from typing import Any

from loguru import logger

from ..dag_schemas import BrowserErrorCode, BrowserOutput
from .a11y import layer_a11y
from .ledger import apply_cost_fields, estimate_cost_usd
from .deterministic import layer_deterministic
from .extract import layer_extract
from .vision import layer_vision


def _content_text(content: Any) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, indent=2)
    return str(content)


def _actions_from_transcript(transcript: list[str] | None) -> list[dict[str, Any]]:
    return [{"note": note} for note in (transcript or []) if note]


def to_browser_output(*, url: str, goal: str, raw: dict[str, Any]) -> BrowserOutput:
    raw = apply_cost_fields(dict(raw))
    path = raw.get("path") or "failed"
    if path not in ("extract", "deterministic", "a11y", "vision", "gateway_blocked", "failed"):
        path = "failed"
    inp = int(raw.get("input_tokens") or 0)
    out = int(raw.get("output_tokens") or 0)
    return BrowserOutput(
        url=url,
        goal=goal,
        path=path,  # type: ignore[arg-type]
        turns=int(raw.get("turns") or 0),
        content=_content_text(raw.get("content")),
        actions=_actions_from_transcript(raw.get("transcript")),
        final_url=str(raw.get("url") or raw.get("final_url") or url),
        elapsed_s=raw.get("elapsed_s") or raw.get("total_elapsed_s"),
        llm_calls=int(raw.get("llm_calls") or 0),
        input_tokens=inp,
        output_tokens=out,
        cost_usd=float(raw.get("cost_usd") if raw.get("cost_usd") is not None else estimate_cost_usd(input_tokens=inp, output_tokens=out)),
    )


def classify_browser_error(raw: dict[str, Any], *, last_layer: str | None = None) -> BrowserErrorCode:
    code = raw.get("error_code")
    if code in {
        "gateway_blocked",
        "extraction_failed",
        "interaction_failed",
        "timeout",
        "vlm_unavailable",
    }:
        return code  # type: ignore[return-value]
    if raw.get("gateway_blocked"):
        return "gateway_blocked"
    if last_layer == "vision" or raw.get("vlm_error"):
        return "vlm_unavailable"
    if last_layer in {"a11y", "vision", "deterministic"}:
        return "interaction_failed"
    return "extraction_failed"


async def run_browser_cascade(
    url: str,
    goal: str,
    *,
    llm: Any,
    force_path: str | None = None,
) -> tuple[BrowserOutput, BrowserErrorCode | None]:
    """Run extract → deterministic → a11y → vision; return output and optional error_code.

    Pass ``force_path`` (extract | deterministic | a11y | vision) to skip escalation —
    opt-in metadata for debugging or when the caller already knows the required layer.
    """
    started = time.time()
    goal = (goal or url).strip()
    url = url.strip()
    last_layer: str | None = None
    forced = (force_path or "").strip().lower()
    if forced not in {"extract", "deterministic", "a11y", "vision"}:
        forced = ""

    logger.info(f"[browser] cascade start url={url!r}" + (f" force_path={forced}" if forced else ""))

    async def _finish(result: dict[str, Any] | None, layer: str) -> tuple[BrowserOutput, BrowserErrorCode | None] | None:
        nonlocal last_layer
        last_layer = layer
        if not result:
            return None
        if result.get("error_code") == "gateway_blocked" or result.get("gateway_blocked"):
            result["total_elapsed_s"] = round(time.time() - started, 2)
            out = to_browser_output(url=url, goal=goal, raw=result)
            logger.info(f"[browser] gateway_blocked wall={out.elapsed_s}s cost=${out.cost_usd:.2f}")
            return out, "gateway_blocked"
        result["total_elapsed_s"] = round(time.time() - started, 2)
        out = to_browser_output(url=url, goal=goal, raw=result)
        logger.info(
            f"[browser] path={out.path} turns={out.turns} llm_calls={out.llm_calls} "
            f"tokens={out.input_tokens}/{out.output_tokens} cost=${out.cost_usd:.2f} wall={out.elapsed_s}s"
        )
        return out, None

    layers: tuple[str, ...] = (forced,) if forced else ("extract", "deterministic", "a11y", "vision")

    for layer in layers:
        if layer == "extract":
            finished = await _finish(await layer_extract(url, goal), "extract")
            if finished:
                if finished[1] == "gateway_blocked" or finished[0].path == "extract":
                    return finished
            continue

        if layer == "deterministic":
            finished = await _finish(await layer_deterministic(url, goal), "deterministic")
            if finished and finished[0].path == "deterministic":
                return finished
            continue

        if layer == "a11y":
            finished = await _finish(await layer_a11y(url, goal, llm), "a11y")
            if finished and finished[0].path == "a11y":
                return finished
            continue

        if layer == "vision":
            try:
                finished = await _finish(await layer_vision(url, goal), "vision")
                if finished:
                    return finished
            except RuntimeError as e:
                logger.warning(f"[browser] vision unavailable: {e}")
                failed = {
                    "path": "failed",
                    "url": url,
                    "content": None,
                    "vlm_error": str(e),
                    "total_elapsed_s": round(time.time() - started, 2),
                }
                return to_browser_output(url=url, goal=goal, raw=failed), "vlm_unavailable"

    logger.warning(f"[browser] cascade exhausted for {url}")
    failed = {
        "path": "failed",
        "url": url,
        "content": None,
        "error": "All browser layers failed to extract useful content.",
        "total_elapsed_s": round(time.time() - started, 2),
    }
    return to_browser_output(url=url, goal=goal, raw=failed), classify_browser_error(failed, last_layer=last_layer)


# Back-compat alias
run_browser = run_browser_cascade
