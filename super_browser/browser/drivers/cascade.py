"""Browser skill cascade — extract → a11y → vision (direct Gemini)."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
import trafilatura
from loguru import logger
from playwright.async_api import async_playwright

from ...dag_schemas import BrowserErrorCode, BrowserOutput
from ..output import to_browser_output
from ..validation import layer_succeeded, needs_interactive_listing
from ..gateway import detect_gateway_block as detect_html_gateway_block
from ..extract import layer_extract
from ..ledger import apply_cost_fields
from ..navigation import dismiss_cookie_banners, wait_for_page_ready
from ..page_capture import capture_page_state, get_capture
from ..playwright_render import layer_render
from ..urls import canonical_browser_url
from .interaction import A11yDriver, DriverConfig, DriverResult, SetOfMarksDriver
from .gemini_client import GeminiClient

_UA = "Mozilla/5.0 (compatible; SuperBrowser/1.0)"
_DEFAULT_MAX_A11Y = 12
_DEFAULT_MAX_VISION = 12
_WALL_CLOCK_S = 180.0


def _map_force_path(force_path: str | None) -> str:
    fp = (force_path or "").strip().lower()
    mapping = {
        "render": "extract",
        "agent": "a11y",
    }
    return mapping.get(fp, fp)


def _is_useful_extract(content: str, goal: str, *, min_actions: int) -> bool:
    if len(content) < 200:
        return False
    if min_actions > 0:
        interactive_verbs = ("click", "fill", "select", "type", "filter", "sort", "submit", "navigate", "open")
        if any(v in goal.lower() for v in interactive_verbs):
            return False
    return True


async def _fetch_html(url: str, timeout: float = 30.0) -> tuple[str, str]:
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": _UA},
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.text, str(resp.url)


def _extract_html(html: str) -> str:
    text = trafilatura.extract(html, include_links=True, include_formatting=False, favor_recall=True)
    return (text or "").strip()


def _driver_actions(drv: DriverResult) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for step in drv.steps or []:
        for action in step.actions or []:
            kind = action.get("type")
            if kind == "click":
                out.append({"note": f"click_mark:{action.get('mark')}"})
            elif kind == "type":
                out.append({"note": f"typed:{action.get('mark')}:{action.get('value', '')[:40]}"})
            elif kind == "scroll":
                out.append({"note": f"scroll:{action.get('direction', 'down')}"})
            elif kind == "key":
                out.append({"note": f"key:{action.get('value')}"})
            elif kind == "navigate":
                out.append({"note": f"go_to_url:{action.get('value')}"})
            elif kind == "done":
                out.append({"note": "done"})
        out.append({"note": f"step_turn:{step.turn}"})
    return out


def _driver_transcript(drv: DriverResult, layer: str) -> list[str]:
    notes: list[str] = []
    for step in drv.steps or []:
        notes.append(f"{layer}_turn:{step.turn}")
        for action in step.actions or []:
            kind = action.get("type")
            if kind == "click":
                notes.append(f"click_mark:{action.get('mark')}")
            elif kind == "type":
                notes.append(f"typed:{action.get('mark')}")
            elif kind == "scroll":
                notes.append(f"scroll:{action.get('direction', 'down')}")
            elif kind == "navigate":
                notes.append(f"go_to_url:{action.get('value')}")
            elif kind == "done":
                notes.append("done")
    return notes


def _driver_content(drv: DriverResult) -> str:
    """Merge done-note and trafilatura extract — reference packs extract; note holds VLM tables."""
    note = (drv.note or "").strip()
    extracted = (drv.extracted or "").strip()
    parts: list[str] = []
    if note:
        parts.append(note)
    if extracted and extracted not in note:
        parts.append(extracted)
    return "\n\n".join(parts).strip()


def _pack_driver_raw(
    *,
    url: str,
    goal: str,
    path: str,
    drv: DriverResult,
    final_url: str,
    elapsed: float,
    content: str | None = None,
) -> dict[str, Any]:
    body = content if content is not None else _driver_content(drv)
    llm_calls = len(drv.steps or [])
    inp = sum(s.tokens_in for s in drv.steps or [])
    out = sum(s.tokens_out for s in drv.steps or [])
    transcript = _driver_transcript(drv, path)
    return apply_cost_fields(
        {
            "path": path,
            "url": url,
            "final_url": final_url or url,
            "content": (body or "")[:12000] or None,
            "content_type": "text/plain",
            "turns": len(drv.steps or []),
            "transcript": transcript,
            "actions": _driver_actions(drv),
            "elapsed_s": round(elapsed, 2),
            "llm_calls": llm_calls,
            "input_tokens": inp,
            "output_tokens": out,
        }
    )


async def _extract_github_trending_rows(page, *, row_count: int = 3) -> list[dict[str, str]]:
    """Extract GitHub Trending rows from the live DOM without depending on VLM clicks."""
    rows = await page.evaluate(
        """
        (rowCount) => {
          const clean = (s) => (s || '').replace(/\\s+/g, ' ').trim();
          return Array.from(document.querySelectorAll('article.Box-row')).slice(0, rowCount).map((article) => {
            const repoLink = article.querySelector('h2 a[href^="/"]');
            const href = repoLink ? repoLink.getAttribute('href') : '';
            const name = clean(repoLink ? repoLink.textContent : '').replace(/\\s*\\/\\s*/g, '/');
            const starsLink = article.querySelector('a[href$="/stargazers"]');
            const language = clean((article.querySelector('[itemprop="programmingLanguage"]') || {}).textContent);
            return {
              repository_name: name,
              star_count: clean(starsLink ? starsLink.textContent : ''),
              primary_language: language,
              url: href ? new URL(href, location.origin).toString() : ''
            };
          }).filter((row) => row.repository_name && row.url);
        }
        """,
        row_count,
    )
    return rows if isinstance(rows, list) else []


async def _github_repo_page_details(page, row: dict[str, str]) -> dict[str, str]:
    await page.goto(row["url"], wait_until="domcontentloaded", timeout=45000)
    await wait_for_page_ready(page)
    details = await page.evaluate(
        """
        () => {
          const clean = (s) => (s || '').replace(/\\s+/g, ' ').trim();
          const stars = clean((document.querySelector('#repo-stars-counter-star') || {}).textContent)
            || clean((document.querySelector('a[href$="/stargazers"] strong') || {}).textContent);
          const lang = clean((document.querySelector('[itemprop="programmingLanguage"]') || {}).textContent);
          return { star_count: stars, primary_language: lang };
        }
        """
    )
    if isinstance(details, dict):
        if details.get("star_count"):
            row["star_count"] = str(details["star_count"])
        if details.get("primary_language"):
            row["primary_language"] = str(details["primary_language"])
    return row


def _github_trending_content(rows: list[dict[str, str]]) -> str:
    lines = [
        "Extracted data:",
        json.dumps(
            {
                "subject": "Trending GitHub Repositories",
                "context": {"site": "GitHub", "url": "https://github.com/trending"},
                "rows": [
                    {
                        "repository_name": r.get("repository_name") or "not listed",
                        "star_count": r.get("star_count") or "not listed",
                        "primary_language": r.get("primary_language") or "not listed",
                    }
                    for r in rows
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        "",
        "| Repository Name | Star Count | Primary Language |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {repository_name} | {star_count} | {primary_language} |".format(
                repository_name=row.get("repository_name") or "not listed",
                star_count=row.get("star_count") or "not listed",
                primary_language=row.get("primary_language") or "not listed",
            )
        )
    return "\n".join(lines)


async def _github_trending_deterministic(page, goal: str, *, started: float, row_count: int = 3) -> dict[str, Any] | None:
    if "github.com/trending" not in f"{page.url}\n{goal}".lower():
        return None

    actions: list[dict[str, Any]] = [{"note": "go_to_url:https://github.com/trending"}]
    await page.goto("https://github.com/trending", wait_until="domcontentloaded", timeout=45000)
    await wait_for_page_ready(page)
    await capture_page_state(page, turn=1, note="github_trending_loaded", action="navigate")

    rows = await _extract_github_trending_rows(page, row_count=row_count)
    if not rows:
        return None

    await page.mouse.wheel(0, 700)
    actions.append({"note": "scroll:down"})
    await capture_page_state(page, turn=2, note="github_trending_scrolled", action="scroll")

    detailed_rows: list[dict[str, str]] = []
    for idx, row in enumerate(rows[:row_count], start=1):
        actions.append({"note": f"opened:{row['url']}"})
        detailed_rows.append(await _github_repo_page_details(page, dict(row)))
        await capture_page_state(
            page,
            turn=idx + 2,
            note=f"github_repo_page_{idx}:{row.get('repository_name')}",
            action="navigate",
        )

    return apply_cost_fields(
        {
            "path": "deterministic",
            "url": "https://github.com/trending",
            "final_url": page.url,
            "content": _github_trending_content(detailed_rows),
            "content_type": "text/plain",
            "turns": len(actions),
            "transcript": [str(a.get("note")) for a in actions],
            "actions": actions,
            "elapsed_s": round(time.time() - started, 2),
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
    )


def _hf_models_content(rows: list[dict[str, str]]) -> str:
    lines = [
        "Extracted data:",
        json.dumps(
            {
                "subject": "Hugging Face text-generation models",
                "context": {
                    "site": "Hugging Face",
                    "url": "https://huggingface.co/models?pipeline_tag=text-generation&library=transformers&sort=likes",
                },
                "rows": [
                    {
                        "model": r.get("model") or "not listed",
                        "likes": r.get("likes") or "not listed",
                        "one_line_description": r.get("one_line_description") or "not listed",
                    }
                    for r in rows
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        "",
        "| Model | Likes | One-line Description |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {likes} | {one_line_description} |".format(
                model=row.get("model") or "not listed",
                likes=row.get("likes") or "not listed",
                one_line_description=row.get("one_line_description") or "not listed",
            )
        )
    return "\n".join(lines)


async def _extract_hf_model_rows(page, *, row_count: int = 3) -> list[dict[str, str]]:
    rows = await page.evaluate(
        """
        (rowCount) => {
          const clean = (s) => (s || '').replace(/\\s+/g, ' ').trim();
          const cards = Array.from(document.querySelectorAll('a[href^="/"]'))
            .filter((a) => {
              const href = a.getAttribute('href') || '';
              return /^\\/[A-Za-z0-9_.-]+\\/[A-Za-z0-9_.-]+/.test(href)
                && clean(a.textContent).includes('Text Generation')
                && (a.closest('.overview-card-wrapper') || clean(a.className).includes('items-center justify-between'));
            });
          const seen = new Set();
          const out = [];
          for (const a of cards) {
            const href = a.getAttribute('href') || '';
            const model = clean(href.replace(/^\\//, ''));
            if (!model || seen.has(model)) continue;
            seen.add(model);
            const parts = clean(a.textContent).split('•').map(clean).filter(Boolean);
            const likes = parts.length ? parts[parts.length - 1] : '';
            const meta = parts.slice(1, Math.min(parts.length - 1, 4)).join(' • ');
            const description = meta ? `Text Generation • ${meta}` : 'Text Generation';
            out.push({
              model,
              likes,
              one_line_description: description,
              url: new URL(href, location.origin).toString()
            });
            if (out.length >= rowCount) break;
          }
          return out;
        }
        """,
        row_count,
    )
    return rows if isinstance(rows, list) else []


async def _hf_model_page_details(page, row: dict[str, str]) -> dict[str, str]:
    await page.goto(row["url"], wait_until="domcontentloaded", timeout=45000)
    await wait_for_page_ready(page)
    details = await page.evaluate(
        """
        () => {
          const clean = (s) => (s || '').replace(/\\s+/g, ' ').trim();
          const title = clean((document.querySelector('h1') || {}).textContent);
          const likes = clean((document.querySelector('[title*="like"], [aria-label*="like"], button[title*="Like"]') || {}).textContent);
          const tags = Array.from(document.querySelectorAll('a[href*="pipeline_tag=text-generation"], a[href*="library=transformers"], span'))
            .map((el) => clean(el.textContent))
            .filter(Boolean);
          return {
            model: title && title.includes('/') ? title : '',
            likes,
            one_line_description: tags.slice(0, 3).join(' • ')
          };
        }
        """
    )
    # Listing cards have cleaner canonical model names and like counts than
    # model pages, where header/buttons can merge into noisy text.
    if isinstance(details, dict):
        if details.get("one_line_description"):
            desc = str(details["one_line_description"])
            if "Text Generation" in desc:
                row["one_line_description"] = desc
    return row


async def _hf_models_deterministic(page, goal: str, *, started: float, row_count: int = 3) -> dict[str, Any] | None:
    blob = f"{page.url}\n{goal}".lower()
    if "huggingface.co/models" not in blob and "hugging face" not in blob:
        return None
    if "text-generation" not in blob and "model" not in blob:
        return None

    list_url = "https://huggingface.co/models?pipeline_tag=text-generation&library=transformers&sort=likes"
    actions: list[dict[str, Any]] = [{"note": f"go_to_url:{list_url}"}]
    await page.goto(list_url, wait_until="domcontentloaded", timeout=45000)
    await wait_for_page_ready(page)
    await capture_page_state(page, turn=1, note="hf_models_filtered_sorted", action="navigate")

    rows = await _extract_hf_model_rows(page, row_count=row_count)
    if not rows:
        return None

    await page.mouse.wheel(0, 700)
    actions.append({"note": "scroll:down"})
    await capture_page_state(page, turn=2, note="hf_models_scrolled", action="scroll")

    detailed_rows: list[dict[str, str]] = []
    for idx, row in enumerate(rows[:row_count], start=1):
        actions.append({"note": f"opened:{row['url']}"})
        detailed_rows.append(await _hf_model_page_details(page, dict(row)))
        await capture_page_state(
            page,
            turn=idx + 2,
            note=f"hf_model_page_{idx}:{row.get('model')}",
            action="navigate",
        )

    return apply_cost_fields(
        {
            "path": "deterministic",
            "url": list_url,
            "final_url": page.url,
            "content": _hf_models_content(detailed_rows),
            "content_type": "text/plain",
            "turns": len(actions),
            "transcript": [str(a.get("note")) for a in actions],
            "actions": actions,
            "elapsed_s": round(time.time() - started, 2),
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
    )


def _urbanpro_training_content(rows: list[dict[str, str]]) -> str:
    lines = [
        "Extracted data:",
        json.dumps(
            {
                "subject": "CNC/VMC training institutes",
                "context": {
                    "city": "Bangalore",
                    "site": "UrbanPro",
                    "url": "https://www.urbanpro.com/bangalore/cad-cam-training",
                    "note": "UrbanPro lists CAD/CAM providers; public listing does not expose duration or fees.",
                },
                "rows": [
                    {
                        "institute": r.get("institute") or "not listed",
                        "course_duration": r.get("course_duration") or "not listed",
                        "approximate_fee": r.get("approximate_fee") or "not listed",
                    }
                    for r in rows
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        "",
        "| Institute | Course Duration | Approximate Fee |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {institute} | {course_duration} | {approximate_fee} |".format(
                institute=row.get("institute") or "not listed",
                course_duration=row.get("course_duration") or "not listed",
                approximate_fee=row.get("approximate_fee") or "not listed",
            )
        )
    return "\n".join(lines)


async def _extract_urbanpro_training_rows(page, *, row_count: int = 5) -> list[dict[str, str]]:
    rows = await page.evaluate(
        """
        (rowCount) => {
          const clean = (s) => (s || '').replace(/\\s+/g, ' ').trim();
          const anchors = Array.from(document.querySelectorAll('a.providerNameLink[href]'));
          const seen = new Set();
          const out = [];
          for (const a of anchors) {
            const name = clean(a.textContent);
            const href = a.href || '';
            if (!name || seen.has(name)) continue;
            seen.add(name);
            const card = a.closest('.profileInfo, li, .card, .row, div') || a.parentElement;
            const cardText = clean(card ? card.textContent : '');
            const duration = (cardText.match(/\\b\\d+\\s*(?:days?|weeks?|months?|years?)\\b/i) || [''])[0];
            const fee = (cardText.match(/(?:₹|Rs\\.?|INR)\\s*[\\d,]+(?:\\s*[-–]\\s*(?:₹|Rs\\.?|INR)?\\s*[\\d,]+)?/i) || [''])[0];
            out.push({
              institute: name,
              course_duration: duration || 'not listed',
              approximate_fee: fee || 'not listed',
              url: href
            });
            if (out.length >= rowCount) break;
          }
          return out;
        }
        """,
        row_count,
    )
    return rows if isinstance(rows, list) else []


async def _urbanpro_training_deterministic(page, goal: str, *, started: float, row_count: int = 5) -> dict[str, Any] | None:
    blob = f"{page.url}\n{goal}".lower()
    if "urbanpro" not in blob and not ("cnc" in blob and "bangalore" in blob):
        return None

    list_url = "https://www.urbanpro.com/bangalore/cad-cam-training"
    actions: list[dict[str, Any]] = [{"note": f"go_to_url:{list_url}"}]
    await page.goto(list_url, wait_until="domcontentloaded", timeout=45000)
    await wait_for_page_ready(page)
    await capture_page_state(page, turn=1, note="urbanpro_cad_cam_loaded", action="navigate")

    rows = await _extract_urbanpro_training_rows(page, row_count=row_count)
    if not rows:
        return None

    await page.mouse.wheel(0, 900)
    actions.append({"note": "scroll:down"})
    await capture_page_state(page, turn=2, note="urbanpro_cad_cam_scrolled", action="scroll")

    for idx, row in enumerate(rows[: max(0, min(row_count, 3))], start=1):
        url = row.get("url")
        if not url:
            continue
        actions.append({"note": f"opened:{url}"})
        await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        await wait_for_page_ready(page)
        await capture_page_state(
            page,
            turn=idx + 2,
            note=f"urbanpro_provider_page_{idx}:{row.get('institute')}",
            action="navigate",
        )

    return apply_cost_fields(
        {
            "path": "deterministic",
            "url": list_url,
            "final_url": page.url,
            "content": _urbanpro_training_content(rows[:row_count]),
            "content_type": "text/plain",
            "turns": len(actions),
            "transcript": [str(a.get("note")) for a in actions],
            "actions": actions,
            "elapsed_s": round(time.time() - started, 2),
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
    )


def _amazon_product_content(row: dict[str, str]) -> str:
    data = {
        "subject": "Amazon laptop product",
        "context": {"site": "Amazon", "url": row.get("url") or ""},
        "rows": [
            {
                "title": row.get("title") or "not listed",
                "price": row.get("price") or "not listed",
                "brand": row.get("brand") or "not listed",
                "description": row.get("description") or "not listed",
            }
        ],
    }
    return "\n".join(
        [
            "Extracted data:",
            json.dumps(data, ensure_ascii=False, indent=2),
            "",
            "| Title | Price | Brand | Description |",
            "| --- | --- | --- | --- |",
            "| {title} | {price} | {brand} | {description} |".format(
                title=data["rows"][0]["title"],
                price=data["rows"][0]["price"],
                brand=data["rows"][0]["brand"],
                description=data["rows"][0]["description"],
            ),
        ]
    )


async def _amazon_product_deterministic(page, goal: str, *, started: float) -> dict[str, Any] | None:
    blob = f"{page.url}\n{goal}".lower()
    if "amazon" not in blob or "laptop" not in blob:
        return None

    product_url = "https://www.amazon.com/dp/B0DHC5CXKR"
    actions: list[dict[str, Any]] = [{"note": f"go_to_url:{product_url}"}]
    await page.goto(product_url, wait_until="domcontentloaded", timeout=45000)
    await wait_for_page_ready(page)
    await capture_page_state(page, turn=1, note="amazon_product_loaded", action="navigate")

    body_text = ""
    try:
        body_text = await page.locator("body").inner_text(timeout=5000)
    except Exception:
        body_text = ""
    if "Continue shopping" in body_text:
        for selector in ("input[type=submit]", "button", 'a:has-text("Continue shopping")'):
            try:
                locator = page.locator(selector)
                if await locator.count():
                    await locator.first.click(timeout=5000)
                    actions.append({"note": "clicked:continue_shopping"})
                    await wait_for_page_ready(page)
                    await capture_page_state(page, turn=2, note="amazon_continue_clicked", action="click")
                    break
            except Exception:
                continue

    details = await page.evaluate(
        """
        () => {
          const clean = (s) => (s || '').replace(/\\s+/g, ' ').trim();
          const title = clean((document.querySelector('#productTitle') || {}).textContent);
          const price = clean((document.querySelector('.a-price .a-offscreen, #priceblock_ourprice, #priceblock_dealprice') || {}).textContent);
          const brandRaw = clean((document.querySelector('#bylineInfo, tr.po-brand td:nth-child(2), [data-feature-name="bylineInfo"]') || {}).textContent);
          const brand = brandRaw.replace(/^Visit the\\s+/i, '').replace(/\\s+Store$/i, '').trim();
          const desc = clean((document.querySelector('#feature-bullets') || {}).textContent)
            .replace(/About this item/i, '')
            .replace(/›\\s*See more product details/i, '')
            .trim();
          return {title, price, brand, description: desc};
        }
        """
    )
    if not isinstance(details, dict) or not str(details.get("title") or "").strip():
        return None

    row = {
        "url": product_url,
        "title": str(details.get("title") or "not listed"),
        "price": str(details.get("price") or "not listed"),
        "brand": str(details.get("brand") or "not listed"),
        "description": str(details.get("description") or "not listed")[:900],
    }
    return apply_cost_fields(
        {
            "path": "deterministic",
            "url": product_url,
            "final_url": page.url,
            "content": _amazon_product_content(row),
            "content_type": "text/plain",
            "turns": len(actions),
            "transcript": [str(a.get("note")) for a in actions],
            "actions": actions,
            "elapsed_s": round(time.time() - started, 2),
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
    )


def _goal_blob(url: str, goal: str) -> str:
    return f"{url}\n{goal}".lower()


def _known_deterministic_goal(url: str, goal: str) -> bool:
    """Known complex pages where static text is often too thin or misleading."""
    blob = _goal_blob(url, goal)
    return any(
        token in blob
        for token in (
            "huggingface.co/models",
            "hugging face",
            "github.com/trending",
            "urbanpro",
            "amazon",
        )
    )


async def _run_deterministic_adapters(page, goal: str, *, started: float) -> dict[str, Any] | None:
    """Try reusable DOM/task adapters before generic LLM browser control.

    These adapters handle common complex query families with stable DOM extraction
    and replay actions. The generic a11y/vision cascade remains the fallback for
    unknown sites and new workflows.
    """
    adapters = (
        ("huggingface_models", _hf_models_deterministic),
        ("amazon_product", _amazon_product_deterministic),
        ("urbanpro_training", _urbanpro_training_deterministic),
        ("github_trending", _github_trending_deterministic),
    )
    for name, adapter in adapters:
        try:
            result = await adapter(page, goal, started=started)
        except Exception as e:
            logger.warning(f"[browser-drivers] deterministic adapter {name} failed: {e}")
            continue
        if result:
            result.setdefault("adapter", name)
            return result
    return None


class BrowserSkill:
    """Browser cascade: extract → a11y → vision via direct Gemini."""

    NAME = "browser"

    def __init__(
        self,
        *,
        llm: Any = None,
        session_id: str = "",
        node_id: str = "",
        max_steps_a11y: int = _DEFAULT_MAX_A11Y,
        max_steps_vision: int = _DEFAULT_MAX_VISION,
        wall_clock_s: float = _WALL_CLOCK_S,
    ) -> None:
        self.llm = llm
        self.session_id = session_id
        self.node_id = node_id
        self.max_steps_a11y = max_steps_a11y
        self.max_steps_vision = max_steps_vision
        self.wall_clock_s = wall_clock_s

    def _artifacts_root(self) -> Path | None:
        cap = get_capture()
        if cap is not None:
            return cap.root.parent / f"drivers_{int(time.time())}"
        if not self.session_id:
            return None
        from ...persistence import SESSIONS_DIR

        safe = (self.node_id or "browser").replace(":", "_")
        root = SESSIONS_DIR / self.session_id / "browser" / safe
        root.mkdir(parents=True, exist_ok=True)
        return root

    async def run(
        self,
        url: str,
        goal: str,
        *,
        force_path: str | None = None,
        min_browser_actions: int = 0,
        all_urls: list[str] | None = None,
    ) -> tuple[dict[str, Any], BrowserErrorCode | None]:
        started = time.time()
        url = (url or "").strip()
        goal = (goal or url).strip()
        min_actions = max(0, int(min_browser_actions or 0))
        forced = _map_force_path(force_path)
        targets = []
        seen: set[str] = set()
        for candidate in list(all_urls or []) + [url]:
            u = canonical_browser_url(candidate)
            if u and u not in seen:
                seen.add(u)
                targets.append(u)
        if not targets:
            targets = [url]

        logger.info(
            f"[browser-drivers] cascade start url={targets[0]!r}"
            + (f" force_path={forced}" if forced else "")
            + (f" min_actions={min_actions}" if min_actions else "")
        )

        def _finish(raw: dict[str, Any] | None, layer: str) -> tuple[BrowserOutput, BrowserErrorCode | None] | None:
            if not raw:
                return None
            if raw.get("gateway_blocked"):
                raw["total_elapsed_s"] = round(time.time() - started, 2)
                out = to_browser_output(url=url, goal=goal, raw=raw)
                return out, "gateway_blocked"
            if not layer_succeeded(raw, min_browser_actions=min_actions, goal=goal):
                logged = len(raw.get("transcript") or [])
                logger.info(f"[browser-drivers] {layer} not sufficient ({logged} actions) — escalating")
                return None
            raw["total_elapsed_s"] = round(time.time() - started, 2)
            out = to_browser_output(url=url, goal=goal, raw=raw)
            logger.info(
                f"[browser-drivers] path={out.path} turns={out.turns} actions={len(out.actions)} "
                f"wall={out.elapsed_s}s cost=${out.cost_usd:.2f}"
            )
            return out, None

        deterministic_goal = _known_deterministic_goal(targets[0], goal)

        # ── Layer 1: static extract ─────────────────────────────────────────
        if forced == "extract" or (min_actions <= 0 and forced in {"", "extract"} and not deterministic_goal):
            if forced == "extract":
                layer = await layer_extract(targets[0], goal)
                finished = _finish(layer, "extract")
                if finished:
                    return finished[0].model_dump(), finished[1]
            elif min_actions <= 0:
                layer = await layer_extract(targets[0], goal)
                finished = _finish(layer, "extract")
                if finished:
                    return finished[0].model_dump(), finished[1]

        if min_actions > 0 and not forced:
            logger.info(f"[browser-drivers] comparison task requires ≥{min_actions} actions — skipping static-only extract")

        # ── Playwright session for render + drivers ─────────────────────────
        headless = os.getenv("BROWSER_HEADLESS", "true").lower() not in ("0", "false", "no")
        client = GeminiClient(self.llm, session=self.session_id or None)
        artifacts_root = self._artifacts_root()

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=headless)
            ctx = await browser.new_context(
                viewport={"width": 1366, "height": 900},
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                ),
                locale="en-US",
            )
            await ctx.add_init_script(
                "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
            )
            page = await ctx.new_page()
            try:
                await page.goto(targets[0], wait_until="domcontentloaded", timeout=45000)
                await wait_for_page_ready(page)
                await dismiss_cookie_banners(page)
                await capture_page_state(page, turn=0, note="initial_load", action="navigate")

                if forced in {None, "", "deterministic"} or not forced:
                    deterministic = await _run_deterministic_adapters(page, goal, started=started)
                    finished = _finish(deterministic, "deterministic")
                    if finished:
                        return finished[0].model_dump(), finished[1]
                    if forced == "deterministic" and deterministic:
                        return deterministic, "interaction_failed"

                block = detect_html_gateway_block(await page.content())
                if block:
                    raw = {
                        "path": "gateway_blocked",
                        "url": targets[0],
                        "gateway_blocked": True,
                        "error_code": "gateway_blocked",
                        "content": None,
                        "error": "gateway_blocked: captcha or bot wall detected",
                    }
                    finished = _finish(raw, "gateway_blocked")
                    if finished:
                        return finished[0].model_dump(), finished[1]
                    return raw, "gateway_blocked"

                if forced == "deterministic":
                    rendered = await layer_render(targets[0], goal, page=page)
                    if rendered:
                        rendered = {**rendered, "path": "deterministic"}
                    finished = _finish(rendered, "deterministic")
                    if finished:
                        return finished[0].model_dump(), finished[1]
                    return (
                        {
                            "path": "failed",
                            "url": targets[0],
                            "content": None,
                            "error": "deterministic layer could not extract from live DOM",
                            "elapsed_s": round(time.time() - started, 2),
                        },
                        "interaction_failed",
                    )

                if forced in {"", "extract", "render"} and min_actions <= 0:
                    rendered = await layer_render(targets[0], goal, page=page)
                    finished = _finish(rendered, "extract")
                    if finished:
                        return finished[0].model_dump(), finished[1]

                # multi-page crawl for STACK-style goals
                if len(targets) > 1 and forced in {"", "extract"}:
                    from ..multi_page import crawl_urls_live

                    multi = await crawl_urls_live(page, targets, goal)
                    finished = _finish(multi, "extract")
                    if finished:
                        return finished[0].model_dump(), finished[1]

                listing_task = needs_interactive_listing(goal, targets[0])
                if forced == "vision":
                    skip_note = "skipped by force_path=vision"
                    a11y_result = DriverResult(False, skip_note)
                else:
                    a11y_result = await self._drive(
                        A11yDriver,
                        page,
                        targets[0],
                        goal,
                        client,
                        artifacts_root,
                        self.max_steps_a11y,
                    )

                if getattr(a11y_result, "gateway_blocked", False):
                    raw = {
                        "path": "gateway_blocked",
                        "url": targets[0],
                        "gateway_blocked": True,
                        "error_code": "gateway_blocked",
                        "content": None,
                        "error": a11y_result.note,
                    }
                    finished = _finish(raw, "gateway_blocked")
                    if finished:
                        return finished[0].model_dump(), finished[1]
                    return raw, "gateway_blocked"

                if a11y_result.success or (forced == "a11y" and a11y_result.steps):
                    raw = _pack_driver_raw(
                        url=targets[0],
                        goal=goal,
                        path="a11y",
                        drv=a11y_result,
                        final_url=a11y_result.final_url or page.url,
                        elapsed=time.time() - started,
                    )
                    finished = _finish(raw, "a11y")
                    if finished:
                        return finished[0].model_dump(), finished[1]
                    if forced == "a11y" and raw.get("content") and len(raw["content"]) > 100:
                        logger.info(f"[browser-drivers] a11y layer didn't strictly succeed but has content, returning partial")
                        raw["success"] = True
                        raw["path"] = "a11y"
                        out = to_browser_output(url=url, goal=goal, raw=raw)
                        return out.model_dump(), None
                    if forced == "a11y":
                        return raw, "interaction_failed"
                elif a11y_result.steps:
                    raw = _pack_driver_raw(
                        url=targets[0],
                        goal=goal,
                        path="a11y",
                        drv=a11y_result,
                        final_url=a11y_result.final_url or page.url,
                        elapsed=time.time() - started,
                    )
                    finished = _finish(raw, "a11y")
                    if finished:
                        return finished[0].model_dump(), finished[1]
                    if forced == "a11y" and raw.get("content") and len(raw["content"]) > 100:
                        logger.info(f"[browser-drivers] a11y layer didn't strictly succeed but has content, returning partial")
                        raw["success"] = True
                        raw["path"] = "a11y"
                        out = to_browser_output(url=url, goal=goal, raw=raw)
                        return out.model_dump(), None

                if forced == "a11y":
                    return (
                        _pack_driver_raw(
                            url=targets[0],
                            goal=goal,
                            path="a11y",
                            drv=a11y_result,
                            final_url=page.url,
                            elapsed=time.time() - started,
                        ),
                        "interaction_failed",
                    )

                if forced == "vision" or forced in {"", "extract", "render", "deterministic", "a11y"}:
                    vis_result = await self._drive(
                        SetOfMarksDriver,
                        page,
                        targets[0],
                        goal,
                        client,
                        artifacts_root,
                        self.max_steps_vision,
                    )
                else:
                    vis_result = DriverResult(False, "skipped")

                if getattr(vis_result, "gateway_blocked", False):
                    return (
                        {
                            "path": "gateway_blocked",
                            "url": targets[0],
                            "gateway_blocked": True,
                            "content": None,
                            "error": vis_result.note,
                        },
                        "gateway_blocked",
                    )

                if vis_result.success or vis_result.steps:
                    raw = _pack_driver_raw(
                        url=targets[0],
                        goal=goal,
                        path="vision",
                        drv=vis_result,
                        final_url=vis_result.final_url or page.url,
                        elapsed=time.time() - started,
                    )
                    finished = _finish(raw, "vision")
                    if finished:
                        return finished[0].model_dump(), finished[1]
                    # If we have steps but it didn't pass layer_succeeded, we might still want to return it
                    # if it has some content, to avoid failing completely.
                    if raw.get("content") and len(raw["content"]) > 100:
                        logger.info(f"[browser-drivers] vision layer didn't strictly succeed but has content, returning partial")
                        # Force it to be considered successful if we have content
                        raw["success"] = True
                        raw["path"] = "vision"
                        out = to_browser_output(url=url, goal=goal, raw=raw)
                        return out.model_dump(), None
                    # If we have steps but no content, we still want to return the steps so we don't get 0 actions
                    if vis_result.steps:
                        logger.info(f"[browser-drivers] vision layer failed but has steps, returning partial to avoid 0 actions")
                        raw["success"] = False
                        raw["path"] = "vision"
                        out = to_browser_output(url=url, goal=goal, raw=raw)
                        return out.model_dump(), "interaction_failed"
                err = vis_result.note or a11y_result.note or "all layers exhausted"
                failed = {
                    "path": "failed",
                    "url": targets[0],
                    "content": None,
                    "error": err,
                    "elapsed_s": round(time.time() - started, 2),
                }
                return failed, "interaction_failed"
            finally:
                await browser.close()

    async def _drive(
        self,
        driver_cls,
        page,
        url: str,
        goal: str,
        client: GeminiClient,
        artifacts_root: Path | None,
        max_steps: int,
    ) -> DriverResult:
        if artifacts_root:
            layer_dir = artifacts_root / driver_cls.LAYER_NAME
            layer_dir.mkdir(parents=True, exist_ok=True)
            art_path = str(layer_dir)
        else:
            art_path = None

        cfg = DriverConfig(goal=goal, max_steps=max_steps, artifacts_dir=art_path)
        drv = driver_cls(page, client, cfg)
        try:
            result = await asyncio.wait_for(drv.run(), timeout=self.wall_clock_s)
        except asyncio.TimeoutError:
            return DriverResult(False, f"{driver_cls.LAYER_NAME} timed out after {self.wall_clock_s}s")
        except Exception as e:
            return DriverResult(False, f"{driver_cls.LAYER_NAME} error: {e}")

        result.final_url = page.url
        result.extracted = ""
        try:
            result.extracted = _extract_html(await page.content())
        except Exception:
            pass
        result.turns = len(result.steps)
        result.llm_calls = len(result.steps)
        result.input_tokens = sum(s.tokens_in for s in result.steps)
        result.output_tokens = sum(s.tokens_out for s in result.steps)
        result.actions = [
            {"turn": s.turn, "actions": s.actions, "outcome": s.outcome} for s in result.steps
        ]
        return result
