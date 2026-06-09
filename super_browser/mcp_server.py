"""
MCP server — web search, fetch, sandbox files, document indexing, knowledge search.

Tools (stdio transport):
    web_search, fetch_url, fetch_urls, analyze_image_url, query_database, get_time, currency_convert,
    validate_json_keys, count_syllables, safe_calculate,
    read_file, list_dir, create_file, update_file, edit_file,
    index_document, search_knowledge, index_directory

Run:  python -m super_browser.mcp_server
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
from typing import Any

import threading
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from .paths import ROOT, SANDBOX, STATE

os.environ["CRAWL4AI_BASE_DIRECTORY"] = str(ROOT / ".crawl4ai")

MAX_SEARCH_RESULTS = 5  # hard cap — Tavily prices per result
# Avoid huge JSON-RPC payloads that can drop the MCP stdio connection.
MAX_FETCH_MARKDOWN_CHARS = 50_000
# Parallel batch fetch: up to 3 URLs, 3 concurrent browser workers.
MAX_FETCH_URLS_BATCH = 3
MAX_FETCH_URL_CONCURRENCY = 3
MAX_FETCH_MARKDOWN_CHARS_BATCH_URL = 40_000
CRAWLER_POOL_SIZE = 2
PAGE_TIMEOUT_MS = 25_000

_crawler_pool: asyncio.Queue[Any] | None = None
_pool_init_lock = asyncio.Lock()
_crawl_io_lock = asyncio.Lock()

load_dotenv(ROOT / ".env")

from .llm_env import gemini_api_key, gemini_models_ordered, tavily_api_key
from .search_providers import httpx_plain_fetch, web_search_with_fallbacks

mcp = FastMCP("super-browser-mcp")

# NOTE: Do not import google.genai at module scope. A broken/partial google-genai install
# would crash this process on startup and kill MCP stdio — breaking fetch_url/web_search.
# Gemini is lazy-imported only inside analyze_image_url.

SANDBOX.mkdir(exist_ok=True)

USAGE_PATH = ROOT / "usage.json"
MONTHLY_CAP = 950  # leave 50/mo headroom on Tavily
_usage_lock = threading.Lock()


def _safe(path: str) -> Path:
    p = (SANDBOX / path).resolve()
    base = SANDBOX.resolve()
    if p != base and base not in p.parents:
        raise ValueError(f"Path '{path}' escapes the sandbox")
    return p


def _empty_usage(month: str) -> dict:
    return {
        "month": month,
        "tavily": {"count": 0, "errors": 0},
        "duckduckgo": {"count": 0, "errors": 0},
    }


def _load_usage() -> dict:
    month = datetime.now().strftime("%Y-%m")
    if not USAGE_PATH.exists():
        return _empty_usage(month)
    try:
        data = json.loads(USAGE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _empty_usage(month)
    if data.get("month") != month:
        return _empty_usage(month)
    for k in ("tavily", "duckduckgo"):
        data.setdefault(k, {"count": 0, "errors": 0})
    return data


def _save_usage(data: dict) -> None:
    USAGE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _bump(provider: str, field: str = "count") -> None:
    with _usage_lock:
        data = _load_usage()
        data[provider][field] = data[provider].get(field, 0) + 1
        _save_usage(data)


def _under_cap(provider: str) -> bool:
    return _load_usage()[provider]["count"] < MONTHLY_CAP


async def _async_tavily(query: str, max_results: int) -> list[dict]:
    from .search_providers import async_tavily as _at

    if not tavily_api_key() or not _under_cap("tavily"):
        return []
    results = await _at(query, max_results)
    if results:
        _bump("tavily")
    elif tavily_api_key():
        _bump("tavily", "errors")
    return results


async def _async_ddg(query: str, max_results: int) -> list[dict]:
    from .search_providers import async_ddg as _ad

    results = await _ad(query, max_results)
    if results:
        _bump("duckduckgo")
    return results


async def _init_crawler_pool() -> asyncio.Queue[Any] | None:
    global _crawler_pool
    async with _pool_init_lock:
        if _crawler_pool is not None:
            return _crawler_pool
        try:
            from crawl4ai import AsyncWebCrawler

            pool: asyncio.Queue[Any] = asyncio.Queue(maxsize=CRAWLER_POOL_SIZE)
            for _ in range(CRAWLER_POOL_SIZE):
                crawler = AsyncWebCrawler(verbose=False)
                await crawler.__aenter__()
                await pool.put(crawler)
            _crawler_pool = pool
            return pool
        except Exception:
            _crawler_pool = None
            return None


async def _borrow_crawler() -> Any | None:
    pool = await _init_crawler_pool()
    if pool is None:
        return None
    return await pool.get()


async def _return_crawler(crawler: Any | None) -> None:
    if crawler is not None and _crawler_pool is not None:
        await _crawler_pool.put(crawler)


def _httpx_fetch_fallback(url: str, cap: int, reason: str = "") -> dict:
    payload = httpx_plain_fetch(url, max_chars=cap)
    if reason:
        payload["crawl_fallback_reason"] = reason
    return payload


async def _crawl4ai_fetch(url: str, max_markdown_chars: int | None = None) -> dict:
    cap = max_markdown_chars if max_markdown_chars is not None else MAX_FETCH_MARKDOWN_CHARS
    crawler = await _borrow_crawler()
    if crawler is None:
        return _httpx_fetch_fallback(url, cap, reason="crawler_pool_unavailable")

    try:
        async with _crawl_io_lock:
            saved_fd = os.dup(1)
            os.dup2(2, 1)
            try:
                try:
                    from crawl4ai import CrawlerRunConfig

                    run_cfg = CrawlerRunConfig(page_timeout=PAGE_TIMEOUT_MS, wait_until="domcontentloaded")
                    r = await asyncio.wait_for(
                        crawler.arun(url=url, config=run_cfg),
                        timeout=PAGE_TIMEOUT_MS / 1000 + 10,
                    )
                except (ImportError, TypeError):
                    r = await asyncio.wait_for(
                        crawler.arun(url=url),
                        timeout=PAGE_TIMEOUT_MS / 1000 + 10,
                    )
            finally:
                os.dup2(saved_fd, 1)
                os.close(saved_fd)
        md = r.markdown
        raw = (
            getattr(md, "raw_markdown", None)
            or getattr(md, "fit_markdown", None)
            or md
            or r.cleaned_html
            or r.html
            or ""
        )
        text = str(raw).strip()
        if not text:
            return _httpx_fetch_fallback(url, cap, reason="empty_crawl_markdown")
        truncated = False
        if len(text) > cap:
            text = text[:cap]
            truncated = True
        payload: dict = {
            "status": int(getattr(r, "status_code", None) or 200),
            "content_type": "text/markdown",
            "length_bytes": len(text.encode("utf-8")),
            "text": text,
        }
        if truncated:
            payload["truncated"] = True
            payload["note"] = (
                f"Markdown truncated to {cap} chars for MCP transport; "
                "use a narrower fetch or search snippets if you need the tail."
            )
        return payload
    except asyncio.TimeoutError:
        return _httpx_fetch_fallback(url, cap, reason="crawl_timeout")
    except Exception as e:
        return _httpx_fetch_fallback(url, cap, reason=f"crawl_error:{type(e).__name__}")
    finally:
        await _return_crawler(crawler)


@mcp.tool()
async def web_search(query: str, max_results: int = 3) -> str:
    """Search: Tavily → crawl4ai → Gemini live search → DuckDuckGo. Returns JSON array string."""
    max_results = max(1, min(max_results, MAX_SEARCH_RESULTS))
    q = (query or "").strip()
    if not q:
        return json.dumps([{"title": "web_search error", "url": "", "snippet": "Empty query."}])
    hits = await web_search_with_fallbacks(
        q,
        max_results,
        tavily_fn=_async_tavily,
        ddg_fn=_async_ddg,
    )
    return json.dumps(hits, ensure_ascii=False)


@mcp.tool()
async def fetch_url(url: str, timeout: int = 20) -> str:
    """Fetch clean markdown from a URL via crawl4ai. Returns JSON object string."""
    return json.dumps(await _crawl4ai_fetch(url), ensure_ascii=False)


@mcp.tool()
async def fetch_urls(urls: list[str]) -> str:
    """Fetch up to 3 URLs in parallel via crawl4ai (warm browser pool). Returns JSON array string."""
    if not urls:
        return "[]"
    seen: set[str] = set()
    cleaned: list[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        s = u.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        cleaned.append(s)
        if len(cleaned) >= MAX_FETCH_URLS_BATCH:
            break

    sem = asyncio.Semaphore(MAX_FETCH_URL_CONCURRENCY)

    async def _one(target: str) -> dict:
        async with sem:
            try:
                payload = await _crawl4ai_fetch(
                    target, max_markdown_chars=MAX_FETCH_MARKDOWN_CHARS_BATCH_URL
                )
                return {"url": target, **payload}
            except Exception as e:
                fb = _httpx_fetch_fallback(
                    target, MAX_FETCH_MARKDOWN_CHARS_BATCH_URL, reason=f"fetch_exception:{type(e).__name__}"
                )
                return {"url": target, **fb}

    raw_pages = await asyncio.gather(*[_one(u) for u in cleaned], return_exceptions=True)
    pages: list[dict] = []
    for i, item in enumerate(raw_pages):
        target = cleaned[i]
        if isinstance(item, Exception):
            fb = _httpx_fetch_fallback(
                target, MAX_FETCH_MARKDOWN_CHARS_BATCH_URL, reason=f"gather_exception:{type(item).__name__}"
            )
            pages.append({"url": target, **fb})
        elif isinstance(item, dict):
            pages.append(item)
    return json.dumps(pages, ensure_ascii=False)


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    from .indexing import _chunk_text as _chunk_text_impl

    return _chunk_text_impl(text, chunk_size, overlap)


def _read_document_text(path: str) -> tuple[str, str]:
    from .indexing import _read_document_text as _read_document_text_impl

    return _read_document_text_impl(path)


def _chunk_source_tag(source_kind: str, path: str, chunk_index: int, total: int) -> str:
    from .indexing import _chunk_source_tag as _chunk_source_tag_impl

    return _chunk_source_tag_impl(source_kind, path, chunk_index, total)


def _index_document_path(path: str, chunk_size: int = 400, overlap: int = 80, use_vlm: bool | None = None) -> dict:
    from .indexing import index_document_path

    return index_document_path(path, chunk_size=chunk_size, overlap=overlap, use_vlm=use_vlm)


@mcp.tool()
def index_document(path: str, chunk_size: int = 400, overlap: int = 80, use_vlm: bool | None = None) -> dict:
    """Index any supported sandbox document into searchable Memory facts (FAISS).

    **Unified VLM pipeline** (default when no sidecar): file → PDF → page images → Gemini vision → ``page_N`` facts
    with ``citation`` metadata (e.g. ``papers/foo.pdf p.3/12``).

    **Fast path**: when a ``.md`` sidecar exists (e.g. ``papers/2605.23904v2.md`` or ``research_papers/2605.23904.md``
    for a PDF), indexing uses text chunks automatically — pass ``use_vlm=false`` explicitly to force this.

    Supported: ``.pdf``, images, ``.md``/``.txt``, ``.doc``/``.docx``, ``.ppt``/``.pptx``, and other
    Office formats (Office conversion requires LibreOffice on PATH). Also ``art:`` artifacts.

    Set ``use_vlm=false`` only to opt out to legacy text chunking (``.md``/``.txt``/``art:`` UTF-8).

    For one-shot inspection without indexing, use ``read_file``."""
    return _index_document_path(path, chunk_size=chunk_size, overlap=overlap, use_vlm=use_vlm)


@mcp.tool()
def index_directory(path: str = "rag_corpus", chunk_size: int = 400, overlap: int = 80) -> dict:
    """Bulk-index every supported document under a sandbox directory (recursive) via the VLM pipeline.

    Indexes ``.md``, ``.pdf``, images, Office files, etc. Default ``rag_corpus`` holds the five-item
    knowledge base (see ``corpus/MANIFEST.json``). Returns ``files_indexed``, ``chunks_indexed``,
    and per-file ``pages_indexed`` where applicable."""
    from .indexing import index_directory as index_directory_impl

    return index_directory_impl(path, chunk_size=chunk_size, overlap=overlap)


@mcp.tool()
def search_knowledge(query: str, k: int = 5) -> list[dict]:
    """Vector search over indexed document chunks and other Memory facts.

    Returns chunk previews with ``source_label``, ``descriptor``, ``metadata.path``, and for VLM-indexed
    PDFs/images also ``metadata.page_number``, ``metadata.citation``, and ``metadata.page_total``.
    Prefer this over re-fetching URLs when sources were already indexed via ``index_document``."""
    from .memory import _service

    hits = _service().read(query, [], kinds=["fact"], top_k=max(1, k))
    results: list[dict] = []
    for item in hits:
        preview = str(item.value.get("text") or item.descriptor)
        if len(preview) > 500:
            preview = preview[:497] + "..."
        source_label = ""
        if item.descriptor.startswith("["):
            end = item.descriptor.find("]")
            if end > 0:
                source_label = item.descriptor[: end + 1]
        results.append(
            {
                "memory_id": item.id,
                "source_label": source_label,
                "descriptor": item.descriptor,
                "preview": preview,
                "source": item.source,
                "metadata": {
                    "path": item.value.get("path"),
                    "source_kind": item.value.get("source_kind"),
                    "extraction": item.value.get("extraction"),
                    "page_number": item.value.get("page_number"),
                    "page_total": item.value.get("page_total"),
                    "citation": item.value.get("citation"),
                    "chunk_index": item.value.get("chunk_index"),
                    "chunk_total": item.value.get("chunk_total"),
                    "artifact_id": item.artifact_id,
                    "goal_id": item.goal_id,
                    "run_id": item.run_id,
                    "keywords": item.keywords,
                    "confidence": item.confidence,
                    "created_at": item.created_at.isoformat(),
                },
            }
        )
    return results


@mcp.tool()
def query_database(search_term: str = "") -> list[dict]:
    """Search cached products in state/commerce.db (same DB as the agent MemoryManager). Empty term returns recent rows (limit 50)."""
    db_path = STATE / "commerce.db"
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        q = (
            "SELECT url, platform, product_name, base_price, net_price, bank_offers_text, scraped_at "
            "FROM products"
        )
        params: list = []
        st = search_term.strip()
        if st:
            q += " WHERE product_name LIKE ? OR url LIKE ? OR platform LIKE ?"
            pat = f"%{st}%"
            params.extend([pat, pat, pat])
        q += " ORDER BY scraped_at DESC LIMIT 50"
        cur.execute(q, params)
        return [dict(row) for row in cur.fetchall()]


@mcp.tool()
def analyze_image_url(url: str, prompt: str = "Describe this image in detail, extracting any product information, brand, prices, and text.") -> str:
    """Download an image from a URL and analyze its content using Gemini. Example: analyze_image_url("https://example.com/image.jpg")"""
    import httpx

    try:
        from google.genai import Client, types as genai_types
    except ImportError as e:
        return (
            "Gemini SDK failed to import inside MCP server. "
            "Repair your environment with: `uv pip install --force-reinstall 'google-genai>=2.3.0'` "
            f"(import error: {e})."
        )

    models = gemini_models_ordered()
    if not models:
        return "Image analysis needs LLM models configured in `.env` (comma-separated list; see `.env.example`)."

    try:
        client = Client(api_key=gemini_api_key())
        with httpx.Client(timeout=30, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}) as hc:
            r = hc.get(url)
            r.raise_for_status()
            image_bytes = r.content
            mime_type = r.headers.get("content-type", "image/jpeg")

        last_err: Exception | None = None
        for model_id in models:
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=[
                        genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                        prompt,
                    ],
                )
                text = (response.text or "").strip()
                return text or "(Model returned an empty response.)"
            except Exception as e:
                last_err = e
        return f"Failed to analyze image from {url} after trying configured models: {last_err}"
    except Exception as e:
        return f"Failed to analyze image from {url}: {e}"



@mcp.tool()
def validate_json_keys(json_text: str, required_keys: str) -> dict:
    """Verify a JSON object contains required keys (comma-separated list).

    Example: validate_json_keys('{"a":1,"b":2}', "a,b,c") -> valid=false, missing=[c]
    Used by the Critic skill for grounded pass/fail verdicts."""
    keys = [k.strip() for k in required_keys.split(",") if k.strip()]
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        return {"valid": False, "error": str(e), "missing": keys, "present": []}
    if not isinstance(data, dict):
        return {"valid": False, "error": "root must be a JSON object", "missing": keys, "present": []}
    present = list(data.keys())
    missing = [k for k in keys if k not in data]
    return {"valid": len(missing) == 0, "missing": missing, "present": present}


def _syllables_in_word(word: str) -> int:
    w = re.sub(r"[^a-zA-Z']", "", word).lower()
    if not w:
        return 0
    count = 0
    prev_vowel = False
    for ch in w:
        is_vowel = ch in "aeiouy"
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if w.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


@mcp.tool()
def count_syllables(text: str) -> dict:
    """Count syllables per line (heuristic). Returns line counts for Critic verification.

    Example: count_syllables("hello world\\nfoo bar baz") -> {"lines": [3, 3], "total": 6}
    """
    lines = (text or "").splitlines() or [text or ""]
    line_counts: list[int] = []
    for line in lines:
        words = re.findall(r"[A-Za-z']+", line)
        line_counts.append(sum(_syllables_in_word(w) for w in words) if words else 0)
    return {"lines": line_counts, "total": sum(line_counts)}


@mcp.tool()
def safe_calculate(expression: str) -> dict:
    """Safely evaluate a numeric arithmetic expression (+ - * / // % ** parentheses).

    Example: safe_calculate("(17*23 + 41) / 7") -> {"value": 61.714..., "expression": "..."}
    Used by the calculator skill — no variables, no imports."""
    import ast
    import operator as op

    allowed = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.FloorDiv: op.floordiv,
        ast.Mod: op.mod,
        ast.Pow: op.pow,
        ast.USub: op.neg,
        ast.UAdd: op.pos,
    }

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed:
            return allowed[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in allowed:
            return allowed[type(node.op)](_eval(node.left), _eval(node.right))
        raise ValueError(f"disallowed expression: {ast.dump(node)}")

    expr = (expression or "").strip()
    if not expr:
        raise ValueError("empty expression")
    tree = ast.parse(expr, mode="eval")
    value = _eval(tree)
    return {"expression": expr, "value": value}


@mcp.tool()
def get_time(timezone: str = "UTC") -> dict:
    """Current time in a named IANA timezone. Example: get_time("Asia/Kolkata")."""
    tz = ZoneInfo(timezone)
    now = datetime.now(tz)
    offset = now.utcoffset()
    offset_hours = offset.total_seconds() / 3600 if offset else 0.0
    return {
        "iso": now.isoformat(),
        "human": now.strftime("%A, %d %B %Y %H:%M:%S %Z"),
        "timezone": timezone,
        "offset_hours": offset_hours,
    }


@mcp.tool()
def currency_convert(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert money between ISO-3 currencies via frankfurter.dev. Example: currency_convert(100, "USD", "INR")."""
    f = from_currency.upper()
    t = to_currency.upper()
    url = f"https://api.frankfurter.dev/v1/latest?amount={amount}&base={f}&symbols={t}"
    with httpx.Client(timeout=20, follow_redirects=True) as client:
        r = client.get(url)
        r.raise_for_status()
        data = r.json()
    converted = data["rates"][t]
    return {
        "amount": amount,
        "from": f,
        "to": t,
        "rate": converted / amount if amount else 0.0,
        "converted": converted,
        "date": data["date"],
        "source": "frankfurter.dev",
    }


@mcp.tool()
def read_file(path: str) -> dict:
    """Read a UTF-8 text or markdown (``.md``) file from the sandbox.

    Examples: ``read_file("papers/attention.md")``, ``read_file("rag_corpus/36_reciprocal_rank_fusion.md")``.
    Returns full ``content`` for one-shot inspection. When the file must remain searchable across
    later turns (RAG), use ``index_document`` or ``index_directory`` instead."""
    p = _safe(path)
    text = p.read_text(encoding="utf-8")
    return {
        "path": path,
        "size_bytes": p.stat().st_size,
        "content": text,
        "encoding": "utf-8",
    }


@mcp.tool()
def list_dir(path: str = ".") -> list[dict]:
    """List a directory inside the sandbox. Example: list_dir(".")."""
    p = _safe(path)
    out = []
    for child in sorted(p.iterdir()):
        is_dir = child.is_dir()
        out.append({
            "name": child.name,
            "type": "dir" if is_dir else "file",
            "size_bytes": 0 if is_dir else child.stat().st_size,
        })
    return out


@mcp.tool()
def create_file(path: str, content: str) -> dict:
    """Create a new file in the sandbox; errors if it exists. Example: create_file("hello.txt", "hi")."""
    p = _safe(path)
    if p.exists():
        raise ValueError(f"File '{path}' already exists")
    if not p.parent.exists():
        raise ValueError(f"Parent directory of '{path}' does not exist")
    p.write_text(content, encoding="utf-8")
    return {"ok": True, "path": path, "size_bytes": p.stat().st_size}


@mcp.tool()
def update_file(path: str, content: str) -> dict:
    """Overwrite an existing sandbox file. Example: update_file("hello.txt", "new body")."""
    p = _safe(path)
    if not p.exists():
        raise ValueError(f"File '{path}' does not exist")
    p.write_text(content, encoding="utf-8")
    return {"ok": True, "path": path, "size_bytes": p.stat().st_size}


@mcp.tool()
def edit_file(path: str, find: str, replace: str, replace_all: bool = False) -> dict:
    """Find-and-replace inside a sandbox file. Example: edit_file("hello.txt", "foo", "bar")."""
    p = _safe(path)
    text = p.read_text(encoding="utf-8")
    count = text.count(find)
    if count == 0:
        raise ValueError(f"'{find}' not found in '{path}'")
    if count > 1 and not replace_all:
        raise ValueError(
            f"'{find}' occurs {count} times in '{path}'; pass replace_all=True"
        )
    new_text = text.replace(find, replace) if replace_all else text.replace(find, replace, 1)
    p.write_text(new_text, encoding="utf-8")
    replacements = count if replace_all else 1
    return {
        "ok": True,
        "path": path,
        "replacements": replacements,
        "size_bytes": p.stat().st_size,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
