import asyncio
import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any

from loguru import logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .artifact_store import ARTIFACT_THRESHOLD_BYTES, ArtifactStore
from .llm_env import gemini_api_key, gemini_models_ordered, mcp_tool_timeout_seconds, shared_gemini_client, tavily_api_key
from .paths import ROOT
from .schemas import ToolCall
from .search_providers import (
    derive_search_queries,
    httpx_plain_fetch,
    is_search_error_payload,
    merge_search_hits,
    primary_search_query,
    web_search_with_fallbacks,
)

USAGE_PATH = ROOT / "usage.json"


def _project_venv_python(project_root: Path) -> str | None:
    """Prefer repo `.venv` so MCP matches `uv sync` deps even if another venv is activated."""
    if platform.system() == "Windows":
        exe = project_root / ".venv" / "Scripts" / "python.exe"
    else:
        exe = project_root / ".venv" / "bin" / "python"
    return str(exe) if exe.is_file() else None


def _mcp_subprocess_env() -> dict[str, str]:
    """Ensure MCP child inherits API keys and crawl cache paths."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    env = os.environ.copy()
    env["CRAWL4AI_BASE_DIRECTORY"] = str(ROOT / ".crawl4ai")
    for key, val in (
        ("TAVILY_API_KEY", tavily_api_key()),
        ("GEMINI_API_KEY", gemini_api_key()),
    ):
        if val:
            env[key] = val
    return env


def _flatten_mcp_error(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup):
        parts = [_flatten_mcp_error(e) for e in exc.exceptions]
        return " | ".join(p for p in parts if p)
    return f"{type(exc).__name__}: {exc}"


def _normalize_tool_text(tool_name: str, text: str, result: Any) -> str:
    """Normalize MCP tool output to a JSON string when the SDK returns structured data."""
    if text and text.strip() and text.strip() not in {
        "(MCP returned no result.)",
        "(MCP tool finished with no content blocks.)",
        "(MCP returned content blocks without text; check server/tool implementation.)",
    }:
        return text
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        try:
            if tool_name == "web_search" and isinstance(structured, dict):
                structured = [structured]
            return json.dumps(structured, ensure_ascii=False)
        except (TypeError, ValueError):
            pass
    if tool_name == "web_search" and text.strip().startswith("{"):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return json.dumps([obj], ensure_ascii=False)
        except json.JSONDecodeError:
            pass
    if tool_name == "web_search":
        return "[]"
    return text or "(MCP returned empty tool output.)"


def _extract_mcp_tool_text(result: Any) -> str:
    """Normalize MCP CallToolResult content; avoids IndexError on empty or non-text blocks."""
    if result is None:
        return "(MCP returned no result.)"
    blocks = getattr(result, "content", None)
    if not blocks:
        return "(MCP tool finished with no content blocks.)"
    texts: list[str] = []
    for block in blocks:
        chunk = getattr(block, "text", None)
        if chunk is not None and str(chunk).strip():
            texts.append(str(chunk))
    if texts:
        return "\n".join(texts)
    return "(MCP returned content blocks without text; check server/tool implementation.)"


def _args_contain_art_prefix(obj: Any) -> bool:
    if isinstance(obj, str):
        return obj.strip().startswith("art:")
    if isinstance(obj, dict):
        return any(_args_contain_art_prefix(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_args_contain_art_prefix(v) for v in obj)
    return False


def _log_usage_snapshot(prefix: str = "[Tavily/DDG usage]") -> None:
    """Surface MCP search billing counters from usage.json for the UI stream."""
    if not USAGE_PATH.exists():
        logger.info(f"{prefix} usage.json not present yet")
        return
    try:
        data = json.loads(USAGE_PATH.read_text(encoding="utf-8"))
        tv = data.get("tavily", {})
        dd = data.get("duckduckgo", {})
        logger.info(
            f"{prefix} month={data.get('month')} "
            f"tavily_calls={tv.get('count', 0)} tavily_errors={tv.get('errors', 0)} "
            f"ddg_calls={dd.get('count', 0)} ddg_errors={dd.get('errors', 0)}"
        )
    except Exception as e:
        logger.warning(f"{prefix} could not read usage.json: {e}")


def _format_size_label(nbytes: int) -> str:
    if nbytes >= 1024:
        return f"{max(1, round(nbytes / 1024))}KB"
    return f"{nbytes}B"


def format_artifact_size(nbytes: int) -> str:
    """Large Wikipedia-style blobs use raw bytes; smaller web artifacts use KB (Query D)."""
    if nbytes >= 100_000:
        return f"{nbytes} bytes"
    return _format_size_label(nbytes)


def _text_preview(text: str, *, max_len: int = 80) -> str:
    snippet = " ".join((text or "").replace("\n", " ").split())
    if len(snippet) > max_len:
        return snippet[: max_len - 3] + "..."
    return snippet


def summarize_tool_result(
    tool_name: str,
    arguments: dict[str, Any],
    text: str,
    artifact_id: str | None,
    *,
    artifact_bytes: int = 0,
) -> str:
    """One-line action summary for iteration logs."""
    tn = (tool_name or "").strip()
    body = (text or "").strip()
    nbytes = artifact_bytes or len(body.encode("utf-8"))

    if tn == "web_search":
        n = 0
        try:
            parsed = json.loads(body) if body.startswith("[") or body.startswith("{") else None
            if isinstance(parsed, dict):
                parsed = [parsed]
            if isinstance(parsed, list):
                n = sum(1 for x in parsed if isinstance(x, dict) and x.get("url"))
        except json.JSONDecodeError:
            pass
        if n == 0 and artifact_id:
            return f"[search stored as {artifact_id}, {_format_size_label(nbytes)}]"
        if n == 0:
            return "search failed (no results)"
        return f"[{n} URLs in descriptors]"

    if tn == "fetch_urls":
        if body.startswith("[artifact "):
            return body[:220] + ("..." if len(body) > 220 else "")
        if artifact_id:
            preview = _text_preview(body)
            return f"[artifact {artifact_id}, {format_artifact_size(nbytes)}] preview: {preview!r}"
        n = 0
        try:
            parsed = json.loads(body)
            if isinstance(parsed, list):
                n = len(parsed)
        except json.JSONDecodeError:
            pass
        if n:
            return f"[{n} pages fetched]"
        return body[:100] if body else "fetch completed"

    if tn == "fetch_url":
        if body.startswith("[artifact "):
            if artifact_id and nbytes:
                m = re.search(r"preview:\s*(.+?)(?:\.\.\.)?$", body)
                prev = m.group(1).strip() if m else _text_preview(body, max_len=55)
                return f"[artifact {artifact_id}, {format_artifact_size(nbytes)}] preview: {prev}..."
            return body[:200] + ("..." if len(body) > 200 else "")
        if artifact_id:
            preview = _text_preview(body, max_len=60)
            try:
                parsed = json.loads(body)
                if isinstance(parsed, dict):
                    preview = _text_preview(str(parsed.get("text") or body), max_len=60)
            except json.JSONDecodeError:
                pass
            return f"[artifact {artifact_id}, {format_artifact_size(nbytes)}] preview: {preview!r}..."
        snippet = body
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                snippet = str(parsed.get("text") or body)
        except json.JSONDecodeError:
            pass
        snippet = " ".join(snippet.split())
        if len(snippet) > 100:
            snippet = snippet[:97] + "..."
        return snippet or "page fetched"

    if tn == "create_file":
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict) and parsed.get("ok"):
                return "ok"
        except json.JSONDecodeError:
            pass
        return "ok" if '"ok": true' in body.lower() or body.lower().startswith("ok") else body[:80]

    if tn == "update_file" or tn == "edit_file":
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict) and parsed.get("ok"):
                return "ok"
        except json.JSONDecodeError:
            pass
        return "ok" if '"ok": true' in body.lower() else body[:80]

    if tn == "list_dir":
        try:
            parsed = json.loads(body) if body.lstrip().startswith("[") else None
            if isinstance(parsed, list) and parsed:
                names: list[str] = []
                for entry in parsed:
                    if not isinstance(entry, dict):
                        continue
                    name = str(entry.get("name") or entry.get("path") or "").strip()
                    if name:
                        names.append(name.rsplit("/", 1)[-1])
                if len(names) == 1:
                    return f"[file: {names[0]}]"
                if names:
                    return f"[{len(names)} files: {', '.join(names[:3])}]"
        except json.JSONDecodeError:
            pass
        preview = " ".join(body.split())
        return preview[:100] + ("..." if len(preview) > 100 else "") if preview else "directory listed"

    if tn == "read_file":
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                path = str(parsed.get("path") or arguments.get("path") or "")
                fname = path.rsplit("/", 1)[-1] if path else ""
                if fname:
                    return f"[file: {fname}]"
        except json.JSONDecodeError:
            pass

    if tn == "gemini_live_search":
        if artifact_id:
            return "live search summary stored as artifact"
        preview = " ".join(body.split())
        return preview[:100] + ("..." if len(preview) > 100 else "")

    if body.lower().startswith("tool ") or "failed" in body.lower()[:80]:
        return body[:120]
    if artifact_id:
        return f"{tn} completed, stored as artifact"
    preview = " ".join(body.split())
    return preview[:100] + ("..." if len(preview) > 100 else "") if preview else f"{tn} completed"


def _gemini_needs_web_search_fallback(text: str) -> bool:
    t = (text or "").lower()
    return any(
        phrase in t
        for phrase in (
            "timed out",
            "failed",
            "unavailable",
            "not configured",
            "skipped",
            "disabled",
        )
    )


async def _direct_web_search(query: str, max_results: int = 3) -> str:
    """Bypass MCP when transport fails or returns an error payload."""
    budget = mcp_tool_timeout_seconds("web_search")
    try:
        hits = await asyncio.wait_for(
            web_search_with_fallbacks(query, max_results),
            timeout=budget,
        )
        return json.dumps(hits, ensure_ascii=False)
    except Exception as e:
        return json.dumps([
            {"title": "web_search error", "url": "", "snippet": f"{type(e).__name__}: {e}"}
        ])


async def _direct_web_search_multi(queries: list[str], max_results: int = 5) -> str:
    """Try several query variants until one returns real URLs."""
    seen: set[str] = set()
    ordered: list[str] = []
    for q in queries:
        s = (q or "").strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            ordered.append(s)
    if not ordered:
        return json.dumps([{"title": "web_search error", "url": "", "snippet": "Empty query."}])

    budget = mcp_tool_timeout_seconds("web_search")
    per_query = max(8.0, budget / max(1, len(ordered[:3])))
    batches: list[list] = []
    for q in ordered[:3]:
        try:
            hits = await asyncio.wait_for(web_search_with_fallbacks(q, max_results), timeout=per_query)
            if isinstance(hits, list) and any(h.get("url") for h in hits):
                batches.append(hits)
        except Exception:
            continue
    merged = merge_search_hits(*batches, max_results=max_results) if batches else []
    if merged:
        return json.dumps(merged, ensure_ascii=False)
    return json.dumps([
        {
            "title": "web_search error",
            "url": "",
            "snippet": "No results after trying alternate search queries.",
        }
    ])


async def _direct_fetch_url(url: str, max_chars: int = 12_000) -> str:
    payload = await asyncio.to_thread(httpx_plain_fetch, url, max_chars)
    return json.dumps(payload, ensure_ascii=False)


async def _direct_fetch_urls(urls: list[str], max_chars: int = 12_000) -> str:
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
        if len(cleaned) >= 3:
            break

    async def _one(target: str) -> dict:
        raw = await asyncio.to_thread(httpx_plain_fetch, target, max_chars)
        return {"url": target, **raw}

    pages = await asyncio.gather(*[_one(u) for u in cleaned], return_exceptions=True)
    out: list[dict] = []
    for i, item in enumerate(pages):
        target = cleaned[i]
        if isinstance(item, Exception):
            fb = httpx_plain_fetch(target, max_chars)
            out.append({"url": target, **fb})
        elif isinstance(item, dict):
            out.append(item)
    return json.dumps(out, ensure_ascii=False)


def _resolve_search_query(args: dict[str, Any], fallback_query: str) -> str:
    q = str(args.get("query") or args.get("q") or "").strip()
    if q:
        return q
    return (fallback_query or "").strip()


class ActionActuator:
    def __init__(self):
        mcp_py = _project_venv_python(ROOT) or sys.executable
        if mcp_py != sys.executable:
            logger.debug(f"[MCP] Using project venv interpreter for subprocess: {mcp_py}")
        self.server_params = StdioServerParameters(
            command=mcp_py,
            args=["-m", "super_browser.mcp_server"],
            env=_mcp_subprocess_env(),
            cwd=str(ROOT),
        )

        self._mcp_lock = asyncio.Lock()
        self._stdio_cm: Any = None
        self._session_cm: Any = None
        self._mcp_session: ClientSession | None = None

    async def _reset_mcp_connection(self) -> None:
        """Close MCP stdio transport so the next tool call spawns a fresh server process."""
        if self._session_cm is not None:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._session_cm = None
            self._mcp_session = None
        if self._stdio_cm is not None:
            try:
                await self._stdio_cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._stdio_cm = None

    async def _ensure_mcp_session(self) -> ClientSession:
        if self._mcp_session is not None:
            return self._mcp_session
        self._stdio_cm = stdio_client(self.server_params)
        try:
            read, write = await self._stdio_cm.__aenter__()
            self._session_cm = ClientSession(read, write)
            self._mcp_session = await self._session_cm.__aenter__()
            await self._mcp_session.initialize()
            logger.info("[MCP] Session initialized (stdio subprocess).")
            return self._mcp_session
        except Exception:
            await self._reset_mcp_connection()
            raise

    async def aclose(self) -> None:
        """Release MCP stdio transport cleanly (must run before the event loop shuts down)."""
        async with self._mcp_lock:
            await self._reset_mcp_connection()

    def _pack_descriptor(self, text: str, artifact_id: str | None) -> str:
        if artifact_id:
            nbytes = len(text.encode("utf-8"))
            preview_src = text
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    preview_src = str(parsed.get("text") or text)
            except json.JSONDecodeError:
                pass
            prev = _text_preview(preview_src, max_len=55)
            return f"[artifact {artifact_id}, {format_artifact_size(nbytes)}] preview: {prev!r}..."
        return text

    async def execute(
        self,
        tool_call: ToolCall,
        *,
        store: ArtifactStore,
        fallback_query: str = "",
    ) -> tuple[str, str | None]:
        """Dispatch MCP tool call: returns ``(descriptor_text, optional_artifact_id)``."""
        tool_name = (tool_call.name or "").strip()
        tool_args: dict[str, Any] = dict(tool_call.arguments)
        fallback_query = (fallback_query or "").strip()

        if _args_contain_art_prefix(tool_args):
            msg = (
                "Refused tool dispatch: arguments contain an internal `art:` handle. "
                "Use ATTACHED ARTIFACT bytes from context instead of passing handles to MCP paths."
            )
            logger.warning(f"[Action] {msg}")
            return msg, None

        if tool_name == "gemini_live_search":
            q = _resolve_search_query(tool_args, fallback_query)
            if not q and fallback_query:
                q = fallback_query
            budget = mcp_tool_timeout_seconds("gemini_live_search")
            try:
                text = await asyncio.wait_for(
                    asyncio.to_thread(self.gemini_live_search, q),
                    timeout=budget,
                )
            except asyncio.TimeoutError:
                logger.error(f"[Gemini live search] timed out after {budget}s — falling back to web_search")
                text = await _direct_web_search(q)
            else:
                if _gemini_needs_web_search_fallback(text):
                    logger.warning("[Gemini live search] failed/unavailable — falling back to web_search")
                    text = await _direct_web_search(q)
            raw = text.encode("utf-8")
            if len(raw) > ARTIFACT_THRESHOLD_BYTES:
                aid = store.put(raw, content_type="text/plain; charset=utf-8", source="gemini_live_search")
                return self._pack_descriptor(text, aid or None), aid or None
            return text, None

        if tool_name == "web_search":
            q = _resolve_search_query(tool_args, fallback_query)
            if not q:
                q = primary_search_query(fallback_query)
            if q:
                tool_args["query"] = q
            try:
                tool_args["max_results"] = max(1, min(int(tool_args.get("max_results", 5)), 5))
            except (TypeError, ValueError):
                tool_args["max_results"] = 5

        logger.debug(f"[MCP] --> {tool_name} args={tool_args!r}")

        tool_fail_msg: str | None = None
        conn_fail_msg: str | None = None
        text = ""
        last_conn_err: BaseException | None = None

        async with self._mcp_lock:
            for attempt in range(2):
                try:
                    session = await self._ensure_mcp_session()
                    try:
                        budget = mcp_tool_timeout_seconds(tool_name)
                        result = await asyncio.wait_for(
                            session.call_tool(tool_name, tool_args),
                            timeout=budget,
                        )
                    except asyncio.TimeoutError:
                        logger.error(f"[MCP] {tool_name} timed out after {budget}s — resetting session")
                        await self._reset_mcp_connection()
                        tool_fail_msg = (
                            f"Tool '{tool_name}' timed out after {budget}s "
                            "(page crawl or search may be stuck — retry with fewer URLs or a simpler query)."
                        )
                        break
                    except Exception as e:
                        tool_fail_msg = f"Tool execution failed: {e}"
                        logger.error(f"[MCP] <-- {tool_name} FAILED: {e}")
                        break
                    text = _normalize_tool_text(tool_name, _extract_mcp_tool_text(result), result)
                    break
                except asyncio.CancelledError:
                    raise
                except BaseException as e:
                    last_conn_err = e
                    detail = _flatten_mcp_error(e)
                    logger.warning(
                        f"[MCP] transport/session error ({tool_name}), attempt {attempt + 1}/2: {detail}"
                    )
                    await self._reset_mcp_connection()
            else:
                detail = _flatten_mcp_error(last_conn_err) if last_conn_err else "unknown"
                logger.error(f"[MCP] subprocess/session failed ({tool_name}) after retries: {detail}")
                conn_fail_msg = (
                    f"MCP connection failed ({tool_name}): {detail}. "
                    "If imports failed in mcp_server.py, run `uv sync` and start the app with `uv run python app.py` "
                    "so the MCP child uses this project's `.venv`. Retry the step once the server stays up."
                )

        if conn_fail_msg or tool_fail_msg:
            if tool_name == "web_search":
                q = _resolve_search_query(tool_args, fallback_query)
                max_r = int(tool_args.get("max_results", 3))
                queries = derive_search_queries(fallback_query, q, limit=3) if fallback_query else ([q] if q else [])
                logger.warning(
                    f"[Action] MCP web_search failed ({conn_fail_msg or tool_fail_msg}) — direct fallback"
                )
                text = await _direct_web_search_multi(queries or [q], max_r)
            elif tool_name == "fetch_url":
                url = str(tool_args.get("url", "")).strip()
                logger.warning(
                    f"[Action] MCP fetch_url failed ({conn_fail_msg or tool_fail_msg}) — httpx fallback"
                )
                text = await _direct_fetch_url(url)
            elif tool_name == "fetch_urls":
                urls = tool_args.get("urls", [])
                if isinstance(urls, list):
                    logger.warning(
                        f"[Action] MCP fetch_urls failed ({conn_fail_msg or tool_fail_msg}) — httpx fallback"
                    )
                    text = await _direct_fetch_urls(urls)
                else:
                    return conn_fail_msg or tool_fail_msg or "Tool failed.", None
            else:
                return conn_fail_msg or tool_fail_msg or "Tool failed.", None
        elif tool_name == "web_search" and is_search_error_payload(text):
            q = _resolve_search_query(tool_args, fallback_query)
            max_r = int(tool_args.get("max_results", 3))
            queries = derive_search_queries(fallback_query, q, limit=3) if fallback_query else ([q] if q else [])
            if queries or q:
                logger.warning("[Action] MCP web_search error payload — multi-query direct fallback")
                text = await _direct_web_search_multi(queries or [q], max_r)
            else:
                logger.warning("[Action] web_search error payload with no query — cannot fallback")

        preview = text if len(text) <= 1200 else text[:1200] + "…"
        logger.debug(f"[MCP] <-- {tool_name} result_chars={len(text)} preview={preview!r}")

        if tool_name == "web_search":
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    if parsed.get("title") or parsed.get("url"):
                        parsed = [parsed]
                    elif isinstance(parsed.get("results"), list):
                        parsed = parsed["results"]
                if isinstance(parsed, list):
                    logger.debug(f"[MCP web_search] hits={len(parsed)}")
                    for i, hit in enumerate(parsed[:5], 1):
                        if isinstance(hit, dict):
                            logger.debug(
                                f"[MCP web_search] #{i} title={hit.get('title','')!r} url={hit.get('url','')!r}"
                            )
                    text = json.dumps(parsed, ensure_ascii=False)
            except json.JSONDecodeError:
                logger.debug("[MCP web_search] response was not JSON list")
            _log_usage_snapshot()

        if tool_name == "fetch_urls":
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    logger.debug(f"[MCP fetch_urls] pages={len(parsed)}")
                    for i, pg in enumerate(parsed[:6], 1):
                        if isinstance(pg, dict):
                            logger.debug(
                                f"[MCP fetch_urls] #{i} url={pg.get('url','')!r} "
                                f"status={pg.get('status')} chars={len(str(pg.get('text','')))}"
                            )
            except json.JSONDecodeError:
                logger.debug("[MCP fetch_urls] response was not JSON list")

        raw = text.encode("utf-8")
        if len(raw) > ARTIFACT_THRESHOLD_BYTES:
            aid = store.put(
                raw,
                content_type="application/json" if tool_name in {"web_search", "fetch_urls", "fetch_url"} else "text/plain; charset=utf-8",
                source=f"mcp:{tool_name}",
                descriptor=f"{tool_name} result",
            )
            return self._pack_descriptor(text, aid or None), aid or None
        return text, None

    async def execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        """Backward-compatible helper returning text only (used by tests / callers)."""
        desc, _aid = await self.execute(ToolCall(name=tool_name, arguments=tool_args), store=ArtifactStore())
        return desc

    def gemini_live_search(self, query: str) -> str:
        """Triggers Gemini's native Google Search tool for sanity-checking live prices."""
        from google.genai import types

        if not query or not str(query).strip():
            return "Gemini live search skipped: empty query."
        client = shared_gemini_client()
        if client is None:
            return (
                "Gemini live search unavailable (client failed to initialize). "
                "Configure `.env` for Gemini and retry, or use web_search / fetch_url."
            )
        if not gemini_api_key():
            return (
                "Gemini credentials are not configured in the environment; skipping grounded search. "
                "Use web_search or fetch_url instead."
            )

        logger.info(f"[Gemini google_search grounding] query={query!r}")
        models_to_try = gemini_models_ordered()
        if not models_to_try:
            return (
                "Gemini live search is disabled: set the comma-separated models list in `.env` "
                "(see `.env.example`)."
            )
        last_err: Exception | None = None
        for model_id in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=(
                        "You are assisting shoppers in India. Search for live listings related to:\n"
                        f"{query}\n\n"
                        "Prioritize Indian retailers and INR pricing (Amazon.in, Flipkart, official brand India pages). "
                        "Do not emphasize Amazon.com, Walmart, Best Buy, or Target unless the query explicitly asks for US stores. "
                        "Return concise bullets with price hints and platform names ONLY if search results support them. "
                        "If inconclusive, say so — do NOT invent SKUs, specs, or exact PDP facts."
                    ),
                    config=types.GenerateContentConfig(
                        tools=[{"google_search": {}}],
                        temperature=0.1,
                    ),
                )
                out = (response.text or "").strip()
                pv = out if len(out) <= 1500 else out[:1500] + "…"
                logger.info(
                    f"[Gemini google_search grounding] model={model_id} response_chars={len(out)} preview={pv!r}"
                )
                return out or "(Gemini returned an empty response.)"
            except Exception as e:
                last_err = e
                logger.warning(f"[Gemini google_search grounding] model={model_id} failed: {e}")
        logger.error(f"[Gemini google_search grounding] all models failed: {last_err!r}")
        return f"Gemini live search failed after fallbacks: {last_err or 'unknown error'}"
