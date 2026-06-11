"""Browser skill entry — resilient multi-layer cascade returning BrowserOutput."""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from ..dag_schemas import BrowserErrorCode, BrowserOutput
from .agent_layer import layer_agent
from .a11y import _initial_url, layer_a11y
from .deterministic import layer_deterministic
from .extract import layer_extract
from .ledger import apply_cost_fields, estimate_cost_usd
from .navigation import live_page_blocked, navigate_robust
from .playwright_ctx import PLAYWRIGHT_INSTALL_HINT, browser_session, is_playwright_browser_missing_error
from .playwright_render import layer_render
from .vision import layer_vision

_LAYER_PATHS = frozenset({"extract", "deterministic", "agent", "a11y", "vision", "gateway_blocked", "failed"})


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
    try:
        raw = apply_cost_fields(dict(raw))
    except Exception:
        raw = dict(raw)
    path = raw.get("path") or "failed"
    if path not in _LAYER_PATHS:
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
        cost_usd=float(
            raw.get("cost_usd")
            if raw.get("cost_usd") is not None
            else estimate_cost_usd(input_tokens=inp, output_tokens=out)
        ),
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
    if last_layer in {"a11y", "agent", "vision", "deterministic"}:
        return "interaction_failed"
    return "extraction_failed"


async def _safe_layer(label: str, coro: Awaitable[dict[str, Any] | None]) -> dict[str, Any] | None:
    """Run one layer; log and return None on any failure — never raise."""
    try:
        return await coro
    except Exception as e:
        logger.warning(f"[browser] {label} layer error: {e}")
        return None


def _action_count(result: dict[str, Any]) -> int:
    """Count logged browser interactions (clicks, types, vision turns) in layer output."""
    actions = result.get("actions")
    if isinstance(actions, list) and actions:
        return len(actions)
    transcript = result.get("transcript") or []
    if not isinstance(transcript, list):
        return 0
    count = 0
    for item in transcript:
        note = item if isinstance(item, str) else str((item or {}).get("note") or "")
        if not note or note.startswith("render:"):
            continue
        if note in {"fallback:body_text"}:
            continue
        if note.startswith(("click_index:", "scroll:", "go_to_url:", "wait:")):
            count += 1
            continue
        if note.startswith(("click_", "clicked:", "typed:", "opened:", "action_failed:", "click_not_found:")):
            count += 1
            continue
        if note.startswith(("vision_turn:", "click_mark:", "click_coord:", "vlm_raw:")):
            count += 1
            continue
        if note.startswith("vision:"):
            count += 1
            continue
    return count


def _layer_succeeded(result: dict[str, Any] | None, *, min_browser_actions: int = 0) -> bool:
    if not result:
        return False
    if result.get("error_code") == "gateway_blocked" or result.get("gateway_blocked"):
        return False
    path = str(result.get("path") or "")
    if path not in {"extract", "deterministic", "agent", "a11y", "vision"} or not result.get("content"):
        return False
    if min_browser_actions > 0 and _action_count(result) < min_browser_actions:
        return False
    return True


def _richer_result(current: dict[str, Any] | None, candidate: dict[str, Any]) -> bool:
    if not candidate.get("content"):
        return False
    if current is None:
        return True
    cur_len = len(str(current.get("content") or ""))
    new_len = len(str(candidate.get("content") or ""))
    if new_len != cur_len:
        return new_len > cur_len
    return _action_count(candidate) > _action_count(current)


async def run_browser_cascade(
    url: str,
    goal: str,
    *,
    llm: Any,
    force_path: str | None = None,
    min_browser_actions: int = 0,
    all_urls: list[str] | None = None,
) -> tuple[BrowserOutput, BrowserErrorCode | None]:
    """Run extract → render → deterministic → a11y → vision; never raises."""
    try:
        return await _run_browser_cascade_impl(
            url,
            goal,
            llm=llm,
            force_path=force_path,
            min_browser_actions=min_browser_actions,
            all_urls=all_urls,
        )
    except Exception as e:
        logger.error(f"[browser] cascade fatal (contained): {e}")
        failed = {
            "path": "failed",
            "url": url,
            "content": None,
            "error": str(e)[:500],
            "total_elapsed_s": 0.0,
        }
        return to_browser_output(url=url, goal=goal or url, raw=failed), "extraction_failed"


async def _run_browser_cascade_impl(
    url: str,
    goal: str,
    *,
    llm: Any,
    force_path: str | None = None,
    min_browser_actions: int = 0,
    all_urls: list[str] | None = None,
) -> tuple[BrowserOutput, BrowserErrorCode | None]:
    started = time.time()
    goal = (goal or url).strip()
    url = url.strip()
    targets: list[str] = []
    seen: set[str] = set()
    for candidate in list(all_urls or []) + [url]:
        u = (candidate or "").strip().rstrip(".,;)")
        if u and u not in seen:
            seen.add(u)
            targets.append(u)
    if not targets:
        targets = [url]

    forced = (force_path or "").strip().lower()
    if forced not in {"extract", "deterministic", "agent", "a11y", "vision"}:
        forced = ""

    from .browser_use_bridge import browser_use_enabled, try_browser_use_task
    from .browseros_bridge import try_browseros_task
    from .browser_backend import browseros_enabled

    if not forced and browseros_enabled():
        browseros = await _safe_layer(
            "browseros",
            try_browseros_task(task=goal, url=targets[0], llm=llm),
        )
        finished = await _finish(browseros, "agent")
        if finished:
            return finished

    if not forced and browser_use_enabled():
        bridge = await _safe_layer(
            "browser_use",
            try_browser_use_task(task=goal, url=targets[0]),
        )
        finished = await _finish(bridge, "agent")
        if finished:
            return finished

    last_layer: str | None = None
    best_partial: dict[str, Any] | None = None
    try:
        min_actions = max(0, int(min_browser_actions or 0))
    except (TypeError, ValueError):
        min_actions = 0

    logger.info(
        f"[browser] cascade start url={url!r}"
        + (f" force_path={forced}" if forced else "")
        + (f" min_actions={min_actions}" if min_actions else "")
    )

    def _failed_payload(*, error: str, vlm_error: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": "failed",
            "url": url,
            "content": None,
            "error": error,
            "total_elapsed_s": round(time.time() - started, 2),
        }
        if vlm_error:
            payload["vlm_error"] = vlm_error
        if best_partial and best_partial.get("content"):
            payload["content"] = best_partial.get("content")
            payload["transcript"] = best_partial.get("transcript")
            payload["partial_path"] = best_partial.get("path")
        return payload

    def _remember_partial(result: dict[str, Any] | None) -> None:
        nonlocal best_partial
        if not isinstance(result, dict) or not result.get("content"):
            return
        if _richer_result(best_partial, result):
            best_partial = dict(result)

    async def _finish(
        result: dict[str, Any] | None,
        layer: str,
    ) -> tuple[BrowserOutput, BrowserErrorCode | None] | None:
        nonlocal last_layer, best_partial
        last_layer = layer
        if not result:
            return None
        if result.get("error_code") == "gateway_blocked" or result.get("gateway_blocked"):
            result["total_elapsed_s"] = round(time.time() - started, 2)
            out = to_browser_output(url=url, goal=goal, raw=result)
            logger.info(f"[browser] gateway_blocked wall={out.elapsed_s}s cost=${out.cost_usd:.2f}")
            return out, "gateway_blocked"
        if result.get("content"):
            _remember_partial(result)
        if result.get("content") and not _layer_succeeded(result, min_browser_actions=min_actions):
            logged = _action_count(result)
            if min_actions and logged < min_actions:
                logger.info(
                    f"[browser] {layer} has content but only {logged} action(s) "
                    f"(need ≥{min_actions}) — escalating"
                )
            return None
        if not _layer_succeeded(result, min_browser_actions=min_actions):
            return None
        result["total_elapsed_s"] = round(time.time() - started, 2)
        out = to_browser_output(url=url, goal=goal, raw=result)
        logger.info(
            f"[browser] path={out.path} turns={out.turns} llm_calls={out.llm_calls} "
            f"tokens={out.input_tokens}/{out.output_tokens} cost=${out.cost_usd:.2f} wall={out.elapsed_s}s"
        )
        return out, None

    async def _run_playwright_layers(
        layers: list[tuple[str, Callable[[Any], Awaitable[dict[str, Any] | None]]]],
        *,
        shared_page: Any | None = None,
    ) -> tuple[BrowserOutput, BrowserErrorCode | None] | None:
        async def _try_layers(page) -> tuple[BrowserOutput, BrowserErrorCode | None] | None:
            for layer_name, layer_fn in layers:
                finished = await _finish(await _safe_layer(layer_name, layer_fn(page)), layer_name)
                if finished:
                    return finished
            return None

        if shared_page is not None:
            return await _try_layers(shared_page)

        try:
            async with browser_session() as page:
                await navigate_robust(page, _initial_url(url, goal))
                return await _try_layers(page)
        except Exception as e:
            hint = PLAYWRIGHT_INSTALL_HINT if is_playwright_browser_missing_error(e) else str(e)
            logger.error(f"[browser] playwright session failed: {hint}")
            return None

    def _exhausted() -> tuple[BrowserOutput, BrowserErrorCode | None]:
        if best_partial and _layer_succeeded(best_partial, min_browser_actions=min_actions):
            best_partial["total_elapsed_s"] = round(time.time() - started, 2)
            logger.info(f"[browser] accepting best partial result from {best_partial.get('path')}")
            return to_browser_output(url=url, goal=goal, raw=best_partial), None
        logger.warning(f"[browser] cascade exhausted for {url}")
        err = "All browser layers failed to extract useful content."
        if min_actions:
            err = f"All browser layers failed; comparison task requires ≥{min_actions} visible actions."
        failed = _failed_payload(error=err)
        return to_browser_output(url=url, goal=goal, raw=failed), classify_browser_error(
            failed, last_layer=last_layer
        )

    if forced == "extract":
        finished = await _finish(await _safe_layer("extract", layer_extract(url, goal)), "extract")
        return finished or (
            to_browser_output(url=url, goal=goal, raw=_failed_payload(error="Static extract failed.")),
            "extraction_failed",
        )

    if forced == "deterministic":
        finished = await _finish(
            await _safe_layer("deterministic", layer_deterministic(url, goal)),
            "deterministic",
        )
        return finished or _exhausted()

    if forced == "agent":
        finished = await _finish(
            await _safe_layer("agent", layer_agent(url, goal, llm)),
            "agent",
        )
        return finished or _exhausted()

    if forced == "a11y":
        finished = await _finish(
            await _safe_layer("a11y", layer_a11y(url, goal, llm)),
            "a11y",
        )
        return finished or _exhausted()

    if forced == "vision":
        finished = await _finish(
            await _safe_layer("vision", layer_vision(url, goal)),
            "vision",
        )
        return finished or _exhausted()

    if min_actions <= 0:
        finished = await _finish(await _safe_layer("extract", layer_extract(url, goal)), "extract")
        if finished:
            return finished
    else:
        logger.info(
            f"[browser] comparison task requires ≥{min_actions} Playwright actions — "
            "skipping static-only extract"
        )

    playwright_layers: list[tuple[str, Callable[[Any], Awaitable[dict[str, Any] | None]]]] = [
        ("extract", lambda pg: layer_render(url, goal, page=pg)),
        ("agent", lambda pg: layer_agent(url, goal, llm, page=pg)),
        ("deterministic", lambda pg: layer_deterministic(url, goal, page=pg)),
        ("a11y", lambda pg: layer_a11y(url, goal, llm, page=pg)),
        ("vision", lambda pg: layer_vision(url, goal, page=pg)),
    ]

    try:
        async with browser_session() as page:
            from .multi_page import crawl_urls_live

            if len(targets) > 1:
                multi = await _safe_layer("multi_page", crawl_urls_live(page, targets, goal))
                finished = await _finish(multi, "extract")
                if finished:
                    return finished

            await navigate_robust(page, _initial_url(targets[0], goal))
            finished = await _run_playwright_layers(playwright_layers, shared_page=page)
            if finished:
                return finished

            try:
                blocked = await live_page_blocked(page)
            except Exception:
                blocked = False
            if blocked:
                blocked_payload = {
                    "path": "gateway_blocked",
                    "url": page.url,
                    "content": None,
                    "gateway_blocked": True,
                    "total_elapsed_s": round(time.time() - started, 2),
                }
                out = to_browser_output(url=url, goal=goal, raw=blocked_payload)
                logger.info(f"[browser] gateway_blocked (live wall) wall={out.elapsed_s}s")
                return out, "gateway_blocked"
    except Exception as e:
        hint = PLAYWRIGHT_INSTALL_HINT if is_playwright_browser_missing_error(e) else str(e)
        logger.error(f"[browser] playwright cascade failed: {hint}")

    return _exhausted()


# Back-compat alias
run_browser = run_browser_cascade
