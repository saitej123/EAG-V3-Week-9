"""
Shared web search, fetch fallbacks, and tool-argument enrichment.

Search order: Tavily → crawl4ai → Gemini live search → DuckDuckGo (SEARCH_PIPELINE).
Used by mcp_server (MCP tools), action.py (direct fallback when MCP fails),
and decision.py (auto-fill empty tool args from user query + goal).
Never raises — always returns structured dicts/lists the agent can consume.
"""

from __future__ import annotations

import asyncio
import json
import re
from html import unescape
from typing import Any, Awaitable, Callable
from urllib.parse import quote_plus

import httpx
from duckduckgo_search import DDGS

from .llm_env import (
    gemini_api_key,
    gemini_models_ordered,
    mcp_tool_timeout_seconds,
    shared_gemini_client,
    tavily_api_key,
)
from .schemas import Goal, ToolCall

# Single source of truth for web_search provider order (also used in prompts/docs).
SEARCH_PIPELINE = ("tavily", "crawl4ai", "gemini_live_search", "duckduckgo")
SEARCH_PIPELINE_LABEL = "Tavily → crawl4ai → Gemini live search → DuckDuckGo"

SearchProviderFn = Callable[[str, int], Awaitable[list[dict[str, str]]]]

SEARCH_TIMEOUT_SEC = 18.0
HTTP_FETCH_TIMEOUT_SEC = 20.0
CRAWL4AI_SEARCH_TIMEOUT_SEC = 25.0
_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _norm_hit(title: str, url: str, snippet: str) -> dict[str, str]:
    return {
        "title": (title or "").strip(),
        "url": (url or "").strip(),
        "snippet": (snippet or "").strip(),
    }


def tavily_search(query: str, max_results: int) -> list[dict[str, str]]:
    key = tavily_api_key()
    if not key:
        return []
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=key)
        resp = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
            include_answer=False,
        )
        return [
            _norm_hit(r.get("title", ""), r.get("url", ""), r.get("content", ""))
            for r in resp.get("results", [])
            if r.get("url")
        ]
    except Exception:
        return []


def ddg_search(query: str, max_results: int) -> list[dict[str, str]]:
    hits: list[dict] = []
    try:
        with DDGS(timeout=15) as ddgs:
            for backend in ("auto", "html", "lite"):
                try:
                    hits = list(ddgs.text(query, max_results=max_results, backend=backend))
                except Exception:
                    hits = []
                if hits:
                    break
    except Exception:
        return []
    return [
        _norm_hit(h.get("title", ""), h.get("href", ""), h.get("body", ""))
        for h in hits
        if h.get("href")
    ]


def _parse_ddg_html(html: str, max_results: int) -> list[dict[str, str]]:
    """Extract DDG HTML SERP hits from raw HTML."""
    out: list[dict[str, str]] = []
    for block in re.findall(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</',
        html or "",
        flags=re.DOTALL | re.IGNORECASE,
    ):
        link, title_raw, snippet_raw = block
        title = unescape(re.sub(r"<[^>]+>", "", title_raw)).strip()
        snippet = unescape(re.sub(r"<[^>]+>", "", snippet_raw)).strip()
        if link and title:
            out.append(_norm_hit(title, link, snippet))
        if len(out) >= max_results:
            break
    return out


def ddg_html_fallback(query: str, max_results: int) -> list[dict[str, str]]:
    """Last-resort httpx HTML scrape when DDGS library returns nothing."""
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    try:
        with httpx.Client(
            timeout=HTTP_FETCH_TIMEOUT_SEC,
            follow_redirects=True,
            headers=_HTTP_HEADERS,
        ) as client:
            r = client.get(url)
            r.raise_for_status()
            html = r.text
    except Exception:
        return []
    return _parse_ddg_html(html, max_results)


async def async_crawl4ai_search(query: str, max_results: int) -> list[dict[str, str]]:
    """Crawl DuckDuckGo HTML SERP with crawl4ai — search fallback after Tavily."""
    q = (query or "").strip()
    if not q:
        return []
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(q)}"
    try:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler(verbose=False) as crawler:
            try:
                from crawl4ai import CrawlerRunConfig

                run_cfg = CrawlerRunConfig(
                    page_timeout=int(CRAWL4AI_SEARCH_TIMEOUT_SEC * 1000),
                    wait_until="domcontentloaded",
                )
                result = await asyncio.wait_for(
                    crawler.arun(url=url, config=run_cfg),
                    timeout=CRAWL4AI_SEARCH_TIMEOUT_SEC + 5,
                )
            except (ImportError, TypeError):
                result = await asyncio.wait_for(
                    crawler.arun(url=url),
                    timeout=CRAWL4AI_SEARCH_TIMEOUT_SEC + 5,
                )
        html = str(getattr(result, "cleaned_html", None) or getattr(result, "html", None) or "")
        hits = _parse_ddg_html(html, max_results)
        if hits:
            return hits
        md = getattr(result, "markdown", None)
        md_text = str(getattr(md, "raw_markdown", None) or getattr(md, "fit_markdown", None) or md or "")
        if md_text.strip():
            return [_norm_hit(f"crawl4ai: {q[:80]}", url, md_text[:2000])]
        return []
    except Exception:
        return []


def merge_search_hits(
    *sources: list[dict[str, str]],
    max_results: int,
) -> list[dict[str, str]]:
    seen: set[str] = set()
    merged: list[dict[str, str]] = []
    for src in sources:
        for hit in src:
            url = hit.get("url", "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            merged.append(hit)
            if len(merged) >= max_results:
                return merged
    return merged


async def async_tavily(query: str, max_results: int) -> list[dict[str, str]]:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(tavily_search, query, max_results),
            timeout=SEARCH_TIMEOUT_SEC,
        )
    except Exception:
        return []


async def async_ddg(query: str, max_results: int) -> list[dict[str, str]]:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(ddg_search, query, max_results),
            timeout=SEARCH_TIMEOUT_SEC,
        )
    except Exception:
        return []


async def async_ddg_html(query: str, max_results: int) -> list[dict[str, str]]:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(ddg_html_fallback, query, max_results),
            timeout=SEARCH_TIMEOUT_SEC,
        )
    except Exception:
        return []


def _gemini_text_to_hits(text: str, query: str, max_results: int) -> list[dict[str, str]]:
    """Turn Gemini grounded prose into web_search-style hit dicts."""
    body = (text or "").strip()
    if not body or body.lower().startswith("gemini live search skipped"):
        return []
    if "unavailable" in body.lower()[:120] or body.lower().startswith("gemini live search failed"):
        return []

    hits: list[dict[str, str]] = []
    for line in body.splitlines():
        s = line.strip()
        if not s:
            continue
        if re.match(r"^[-*•]\s+", s) or re.match(r"^\d+[.)]\s+", s):
            s = re.sub(r"^[-*•]\s+", "", s)
            s = re.sub(r"^\d+[.)]\s+", "", s)
            hits.append(_norm_hit(s[:120], "", s))
        if len(hits) >= max_results:
            break

    if not hits:
        hits = [_norm_hit(f"Gemini live search: {query[:80]}", "", body[:4000])]
    return hits[:max_results]


def gemini_live_search_text(query: str) -> str:
    """Gemini Google Search grounding — used as web_search fallback before DDG."""
    q = (query or "").strip()
    if not q:
        return ""
    if not gemini_api_key():
        return ""
    client = shared_gemini_client()
    if client is None:
        return ""

    models = gemini_models_ordered()
    if not models:
        return ""

    try:
        from google.genai import types
    except ImportError:
        return ""

    last_err: Exception | None = None
    for model_id in models:
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=(
                    "Use Google Search grounding to answer this query with current web facts:\n"
                    f"{q}\n\n"
                    "Return concise bullet points. Mention source or site names when known. "
                    "Do not invent URLs or facts — if inconclusive, say so briefly."
                ),
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}],
                    temperature=0.1,
                ),
            )
            out = (response.text or "").strip()
            if out:
                return out
        except Exception as e:
            last_err = e
    return ""


async def async_gemini_live_search(query: str, max_results: int) -> list[dict[str, str]]:
    try:
        budget = mcp_tool_timeout_seconds("gemini_live_search")
        text = await asyncio.wait_for(
            asyncio.to_thread(gemini_live_search_text, query),
            timeout=budget,
        )
        return _gemini_text_to_hits(text, query, max_results)
    except Exception:
        return []


async def web_search_with_fallbacks(
    query: str,
    max_results: int,
    *,
    tavily_fn: SearchProviderFn | None = None,
    crawl_fn: SearchProviderFn | None = None,
    gemini_fn: SearchProviderFn | None = None,
    ddg_fn: SearchProviderFn | None = None,
    ddg_html_fn: SearchProviderFn | None = None,
) -> list[dict[str, str]]:
    """
    Search pipeline: Tavily → crawl4ai → Gemini live search → DuckDuckGo (library + httpx HTML).
    Optional provider overrides (e.g. MCP usage tracking for Tavily/DDG). Never raises.
    """
    tavily = tavily_fn or async_tavily
    crawl = crawl_fn or async_crawl4ai_search
    gemini = gemini_fn or async_gemini_live_search
    ddg = ddg_fn or async_ddg
    ddg_html = ddg_html_fn or async_ddg_html

    q = (query or "").strip()
    if not q:
        return [_norm_hit("web_search error", "", "Empty query.")]

    max_results = max(1, min(max_results, 5))
    try:
        tavily_hits = await tavily(q, max_results)
        if tavily_hits:
            return tavily_hits[:max_results]

        crawl_hits = await crawl(q, max_results)
        if crawl_hits:
            return crawl_hits[:max_results]

        gemini_hits = await gemini(q, max_results)
        if gemini_hits:
            return gemini_hits[:max_results]

        ddg_hits = await ddg(q, max_results)
        if ddg_hits:
            return ddg_hits[:max_results]

        html_hits = await ddg_html(q, max_results)
        if html_hits:
            return html_hits

        return [
            _norm_hit(
                "web_search error",
                "",
                f"No results from {SEARCH_PIPELINE_LABEL}.",
            )
        ]
    except Exception as e:
        return [_norm_hit("web_search error", "", f"{type(e).__name__}: {e}")]


def web_search_json(query: str, max_results: int) -> str:
    """Sync wrapper for MCP tools running in thread pool if needed."""
    return json.dumps(
        asyncio.run(web_search_with_fallbacks(query, max_results)),
        ensure_ascii=False,
    )


def httpx_plain_fetch(url: str, max_chars: int = 12_000) -> dict[str, Any]:
    """Lightweight HTTP fallback when crawl4ai is unavailable or times out."""
    target = (url or "").strip()
    if not target:
        return {
            "status": 0,
            "content_type": "text/plain",
            "length_bytes": 0,
            "text": "[httpx_fetch] Empty URL.",
            "error": "empty_url",
            "fallback": "httpx",
        }
    try:
        with httpx.Client(
            timeout=HTTP_FETCH_TIMEOUT_SEC,
            follow_redirects=True,
            headers=_HTTP_HEADERS,
        ) as client:
            r = client.get(target)
            r.raise_for_status()
            raw = r.text or ""
            text = unescape(re.sub(r"<script[^>]*>.*?</script>", " ", raw, flags=re.I | re.S))
            text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.I | re.S)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > max_chars:
                text = text[:max_chars]
            return {
                "status": r.status_code,
                "content_type": "text/plain",
                "length_bytes": len(text.encode("utf-8")),
                "text": text or "(empty body)",
                "fallback": "httpx",
            }
    except Exception as e:
        return {
            "status": 0,
            "content_type": "text/plain",
            "length_bytes": 0,
            "text": f"[httpx_fetch] Failed for {target!r}: {type(e).__name__}: {e}",
            "error": str(e),
            "fallback": "httpx",
        }


def is_search_error_payload(text: str) -> bool:
    """True when web_search JSON indicates failure (triggers action-layer retry)."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return True
    items = data if isinstance(data, list) else [data]
    if not items:
        return True
    first = items[0] if isinstance(items[0], dict) else {}
    title = str(first.get("title", "")).lower()
    if title == "web_search error":
        return True
    if first.get("url"):
        return False
    snippet = str(first.get("snippet", "")).strip()
    if snippet and "no results from" not in snippet.lower():
        return False
    return True


# --- Tool argument enrichment (was tool_enrichment.py) ------------------------


def derive_search_queries(user_query: str, goal_text: str = "", *, limit: int = 3) -> list[str]:
    """Build focused search strings from the user question and active goal."""
    uq = (user_query or "").strip()
    gt = (goal_text or "").strip()
    candidates: list[str] = []

    if uq:
        candidates.append(uq[:280])
    if gt and gt.lower() != uq.lower() and len(gt) > 12:
        candidates.append(gt[:280])

    lower = uq.lower()
    if "weather" in lower:
        city = _extract_place(uq) or "Tokyo"
        candidates.append(f"{city} weather forecast Saturday this weekend")
    if any(w in lower for w in ("family", "family-friendly", "kids", "children")):
        place = _extract_place(uq) or ""
        if place:
            candidates.append(f"family friendly things to do {place} weekend")

    seen: set[str] = set()
    out: list[str] = []
    for q in candidates:
        key = q.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(q)
        if len(out) >= limit:
            break
    return out or ([uq[:280]] if uq else [])


def _extract_place(text: str) -> str:
    m = re.search(r"\bin\s+([A-Z][a-zA-Z\s\-]+?)(?:\s+this|\s+weekend|[,.]|$)", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"\b(Tokyo|Delhi|Mumbai|Bangalore|London|Paris|New York|Sydney)\b", text, re.I)
    return m.group(1) if m else ""


def enrich_tool_call(tc: ToolCall, *, goal: Goal, user_query: str) -> ToolCall:
    """Ensure required tool fields are populated before MCP dispatch."""
    name = (tc.name or "").strip()
    args = dict(tc.arguments or {})
    queries = derive_search_queries(user_query, goal.text)

    if name in {"web_search", "gemini_live_search"}:
        q = str(args.get("query") or args.get("q") or "").strip()
        if not q:
            q = queries[0] if queries else (goal.text.strip()[:280] or user_query.strip()[:280])
            args["query"] = q
        if name == "web_search":
            try:
                mr = int(args.get("max_results", 5))
            except (TypeError, ValueError):
                mr = 5
            args["max_results"] = max(1, min(mr, 5))

    elif name == "fetch_url":
        url = str(args.get("url") or "").strip()
        if not url and args.get("query"):
            args["url"] = str(args["query"]).strip()
        if not url:
            urls = extract_http_urls(f"{user_query}\n{goal.text}")
            if urls:
                args["url"] = urls[0]

    elif name == "fetch_urls":
        urls = args.get("urls")
        if not isinstance(urls, list) or not any(str(u).strip() for u in urls):
            args["urls"] = []

    elif name == "index_document":
        p = str(args.get("path") or "").strip()
        if not p:
            found = extract_sandbox_paths(f"{user_query}\n{goal.text}")
            if found:
                p = found[0]
                args["path"] = p
        if p:
            from .documents import TEXT_SUFFIXES, suffix_for_path
            from .indexing import resolve_fast_text_sidecar

            if suffix_for_path(p) in TEXT_SUFFIXES:
                args["use_vlm"] = False
            else:
                sidecar = resolve_fast_text_sidecar(p)
                if sidecar:
                    args["use_vlm"] = False

    return ToolCall(name=name, arguments=args)


_SANDBOX_PREFIXES = ("papers", "research_papers", "rag_corpus", "uploads")
_SANDBOX_PATH_RE = re.compile(
    r"\b((?:papers|research_papers|rag_corpus|uploads)/[\w.\-]+(?:\.[\w.\-]+)?)\b",
    re.I,
)


_HTTP_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+", re.IGNORECASE)


def extract_http_urls(text: str) -> list[str]:
    """Return http(s) URLs from user text in document order (deduped)."""
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for m in _HTTP_URL_RE.finditer(text):
        url = m.group(0).rstrip(".,;)")
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(url)
    return out


def extract_sandbox_paths(text: str) -> list[str]:
    """Find sandbox-relative paths mentioned in user text."""
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for m in _SANDBOX_PATH_RE.finditer(text):
        p = m.group(1).replace("\\", "/").strip("/")
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _norm_path(path: str) -> str:
    return path.replace("\\", "/").strip().lower().lstrip("./")


def _path_in_hits(path: str, hits: list[Any] | None) -> bool:
    from .indexing import paths_share_index

    norm = _norm_path(path)
    for h in hits or []:
        desc = str(getattr(h, "descriptor", "") or "").lower()
        val = getattr(h, "value", None) or {}
        p = str(val.get("path", "") or "")
        if p and paths_share_index(path, p):
            return True
        pn = _norm_path(p)
        if pn == norm or norm in desc or f"[sandbox:{norm}" in desc or f"{norm} chunk" in desc:
            return True
    return False


def _index_in_history(path: str, history: list[dict[str, Any]] | None) -> bool:
    from .indexing import paths_share_index

    norm = _norm_path(path)
    prefix = norm.split("/", 1)[0] if "/" in norm else norm
    for entry in history or []:
        if entry.get("kind") != "action":
            continue
        tool = str(entry.get("tool") or "")
        args = entry.get("arguments") or {}
        if tool == "index_document":
            p = str(args.get("path") or "")
            if paths_share_index(path, p):
                return True
        if tool == "index_directory":
            d = _norm_path(str(args.get("path") or "")).strip("/")
            if norm.startswith(d + "/") or norm == d:
                return True
    return False


def heuristic_tool_call(
    *,
    goal: Goal,
    user_query: str,
    hits: list[Any] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> ToolCall | None:
    """Choose a local sandbox tool when Decision fails; None → caller may use web_search."""
    combined = f"{user_query}\n{goal.text}"
    paths = extract_sandbox_paths(combined)
    uq = user_query.lower()
    gt = goal.text.lower()
    wants_index = "index" in uq or ("index" in gt and paths)
    wants_recall = any(
        k in uq
        for k in (
            "indexed",
            "search_knowledge",
            "across these papers",
            "across the papers",
            "papers i have",
            "according to this paper",
            "from indexed",
        )
    )
    wants_bulk = wants_index and any(
        k in uq for k in ("every", "all ", "each ", "bulk", "directory", "under papers", "under research_papers")
    )

    if wants_bulk:
        dir_path = "papers"
        if "research_papers" in uq:
            dir_path = "research_papers"
        elif "rag_corpus" in uq:
            dir_path = "rag_corpus"
        elif paths:
            dir_path = paths[0].split("/", 1)[0]
        return ToolCall(name="index_directory", arguments={"path": dir_path})

    # Explicit "index … path" — index once, then recall on later decision failures.
    if wants_index and paths:
        path = paths[0]
        already = _index_in_history(path, history) or _path_in_hits(path, hits)
        if not already:
            args: dict[str, Any] = {"path": path}
            if path.lower().endswith((".md", ".txt")):
                args["use_vlm"] = False
            return ToolCall(name="index_document", arguments=args)
        query = user_query.strip()[:280] or goal.text.strip()[:280]
        return ToolCall(name="search_knowledge", arguments={"query": query, "k": 8})

    if paths:
        path = paths[0]
        indexed = _path_in_hits(path, hits) or _index_in_history(path, history)

        if indexed or wants_recall:
            query = user_query.strip()[:280] or goal.text.strip()[:280]
            return ToolCall(name="search_knowledge", arguments={"query": query, "k": 8})

        if any(k in gt for k in ("read", "extract", "open", "contents")):
            return ToolCall(name="read_file", arguments={"path": path})

    if wants_recall or any(k in uq for k in ("credit assignment", "chain-of-thought", "react paper")):
        query = user_query.strip()[:280] or goal.text.strip()[:280]
        return ToolCall(name="search_knowledge", arguments={"query": query, "k": 8})

    return None


def primary_search_query(user_query: str, goal_text: str = "") -> str:
    qs = derive_search_queries(user_query, goal_text, limit=1)
    return qs[0] if qs else (user_query or goal_text or "").strip()[:280]
