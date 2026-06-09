"""
LLM credentials and model list: read from the process environment.

On import, loads repository-root `.env` via `python-dotenv`, then exposes accessors.
Variable names are constants below; never embed secrets or model IDs in code.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .paths import ROOT

# Resolve repo-root `.env` whenever this module is imported (before any LLM reads).
load_dotenv(ROOT / ".env")

# Environment variable names only — values come from `.env` / process env after load above.
_VAR_GEMINI_API_KEY = "GEMINI_API_KEY"
_VAR_GEMINI_MODELS = "GEMINI_MODELS"
_VAR_GEMINI_MODEL = "GEMINI_MODEL"
_VAR_TAVILY_API_KEY = "TAVILY_API_KEY"


def gemini_api_key() -> str:
    """Gemini API credential from the environment (empty if unset)."""
    return (os.environ.get(_VAR_GEMINI_API_KEY) or "").strip()


def tavily_api_key() -> str:
    """Tavily API credential from the environment (empty if unset)."""
    return (os.environ.get(_VAR_TAVILY_API_KEY) or "").strip()


def gemini_models_ordered() -> list[str]:
    """Models from env: `GEMINI_MODELS` (comma-separated) if set; else `GEMINI_MODEL` (single)."""
    raw = (os.environ.get(_VAR_GEMINI_MODELS) or "").strip()
    if raw:
        seen: set[str] = set()
        out: list[str] = []
        for part in raw.split(","):
            m = part.strip()
            if m and m not in seen:
                seen.add(m)
                out.append(m)
        return out
    single = (os.environ.get(_VAR_GEMINI_MODEL) or "").strip()
    return [single] if single else []


# Lazy shared Gemini HTTP client (used by perception, decision, action — direct google-genai SDK).
_gemini_lock = threading.Lock()
_gemini_client_singleton: Any = False  # False = not yet resolved


def shared_gemini_client() -> Any:
    """Return a single cached ``google.genai.Client`` or ``None``; builds on first use only."""
    global _gemini_client_singleton
    if _gemini_client_singleton is not False:
        return _gemini_client_singleton
    with _gemini_lock:
        if _gemini_client_singleton is not False:
            return _gemini_client_singleton
        key = gemini_api_key()
        if not key:
            _gemini_client_singleton = None
            return None
        try:
            from google.genai import Client

            _gemini_client_singleton = Client(api_key=key)
        except Exception as e:
            from loguru import logger

            logger.warning(f"Shared Gemini client init failed: {e}")
            _gemini_client_singleton = None
        return _gemini_client_singleton


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def agent_max_iterations() -> int:
    """Max perceive→decide→act loops per agent run (default 3). Override with AGENT_MAX_ITERATIONS."""
    return max(1, min(50, _int_env("AGENT_MAX_ITERATIONS", 3)))


def agent_iteration_ceiling() -> int:
    """Upper bound when auto-extending for multi-step queries (default 4). Override with AGENT_ITERATION_CEILING."""
    base = agent_max_iterations()
    return max(base, min(50, _int_env("AGENT_ITERATION_CEILING", 4)))


def estimate_iteration_need(user_query: str) -> int:
    """Heuristic step count from query shape — stay tight (2–3 turns) unless clearly multi-hop."""
    t = (user_query or "").lower().strip()
    need = agent_max_iterations()
    if not t:
        return need

    # Search + batch fetch + synthesize fits in 3 when fetch_urls batches top results.
    if any(k in t for k in ("top 3", "top three", "3 results", "three results")):
        need = max(need, 4)
    if "search for" in t and any(k in t for k in ("list", "advice", "summar", "agree", "read the", "read top")):
        need = max(need, 4)

    if "http://" in t or "https://" in t or "wikipedia" in t:
        need = max(need, 3)

    if "remember" in t and any(k in t for k in ("reminder", "calendar", "birthday")):
        need = max(need, 3)

    if any(k in t for k in ("weather", "forecast", "weekend", "activities", "family-friendly")):
        need = max(need, 3)

    if any(k in t for k in ("index", "indexed", "corpus", "papers/")):
        need = max(need, 3)

    return need


def resolve_iteration_budget(user_query: str, explicit: int | None = None) -> int:
    """Default 3; extend only slightly when the query clearly needs more tool steps."""
    if explicit is not None:
        return max(1, min(50, explicit))
    base = agent_max_iterations()
    ceiling = agent_iteration_ceiling()
    need = estimate_iteration_need(user_query)
    return min(ceiling, max(base, need))


def agent_run_max_seconds() -> float:
    """Hard cap for one agent job (wall clock). Prevents Run agent staying busy forever."""
    return max(120.0, _float_env("AGENT_RUN_MAX_SECONDS", 900.0))


def agent_llm_step_timeout_seconds() -> float:
    """Perception / decision LLM call budget (each)."""
    return max(15.0, _float_env("AGENT_LLM_STEP_TIMEOUT_SEC", 60.0))


def vlm_index_max_pages() -> int:
    """Max PDF pages to rasterize + VLM-extract per document (default 30)."""
    return max(1, min(100, _int_env("VLM_INDEX_MAX_PAGES", 30)))


def vlm_page_chunk_size() -> int:
    """Sub-chunk size (chars) when a VLM page exceeds this length (default 1200)."""
    return max(400, _int_env("VLM_PAGE_CHUNK_SIZE", 1200))


def vlm_index_dpi_scale() -> float:
    """PDF page rasterization scale for VLM (default 2.0 ≈ 144 DPI)."""
    return max(1.0, min(4.0, _float_env("VLM_INDEX_DPI_SCALE", 2.0)))


def mcp_tool_timeout_seconds(tool_name: str) -> float:
    """Per-tool MCP RPC budget; crawl-heavy tools get more time."""
    env_key = f"MCP_TIMEOUT_{tool_name.upper().replace('-', '_')}"
    if os.environ.get(env_key):
        return max(5.0, _float_env(env_key, 120.0))
    defaults: dict[str, float] = {
        "fetch_urls": 90.0,
        "fetch_url": 40.0,
        "web_search": 22.0,
        "query_database": 10.0,
        "analyze_image_url": 60.0,
        "gemini_live_search": 35.0,
        "get_time": 10.0,
        "currency_convert": 15.0,
        "read_file": 15.0,
        "list_dir": 10.0,
        "create_file": 15.0,
        "update_file": 15.0,
        "edit_file": 15.0,
        "index_document": 30.0,
        "index_directory": 120.0,
        "search_knowledge": 15.0,
    }
    return max(8.0, defaults.get(tool_name, 45.0))


def gateway_base_url() -> str | None:
    """Optional LLM gateway base URL — only used when ``GATEWAY_URL`` or ``GATEWAY_V3_URL`` is set."""
    raw = (os.environ.get("GATEWAY_URL") or os.environ.get("GATEWAY_V3_URL") or "").strip().rstrip("/")
    return raw or None


def gemini_embed_model() -> str:
    """Pinned embedding model — changing it invalidates all vectors in ``index.faiss``.

    **Remedy:** delete ``index.faiss`` + ``index_ids.json`` and rebuild from ``memory.json``
    (text preserved in each item's ``value``/``descriptor``), or ``scripts/clean.py`` + re-index.
    See ``docs/DEFERRALS.md``.
    """
    return (os.environ.get("GEMINI_EMBED_MODEL") or "gemini-embedding-2").strip()


def gemini_embed_output_dimensionality() -> int | None:
    """Optional MRL output size (768, 1536, or 3072). See Gemini Embeddings docs."""
    raw = (os.environ.get("GEMINI_EMBED_OUTPUT_DIM") or "").strip()
    if not raw:
        return None
    try:
        v = int(raw)
        return v if v > 0 else None
    except ValueError:
        return None


def _is_embed_v2_model(model: str) -> bool:
    return "embedding-2" in (model or "").lower()


def format_embed_input(
    text: str,
    *,
    task_type: str = "retrieval_document",
    title: str | None = None,
) -> str:
    """Format text for ``gemini-embedding-2`` task prefixes (Embedding 2 has no ``task_type`` param)."""
    snippet = (text or "").strip()
    if not snippet:
        return snippet
    is_query = "query" in task_type.lower() or task_type.upper() in (
        "RETRIEVAL_QUERY",
        "CODE_RETRIEVAL_QUERY",
    )
    if is_query:
        return f"task: search result | query: {snippet}"
    doc_title = (title or "none").strip() or "none"
    return f"title: {doc_title} | text: {snippet}"


def try_embed_text(
    text: str,
    *,
    task_type: str = "retrieval_document",
    title: str | None = None,
) -> list[float] | None:
    """Return an embedding vector via Gemini SDK, or gateway ``/v1/embed`` when configured."""
    from loguru import logger

    snippet = (text or "").strip()
    if not snippet:
        return None

    gateway_url = gateway_base_url()
    if gateway_url:
        try:
            import httpx

            payload = {"text": snippet, "task_type": task_type}
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(f"{gateway_url}/v1/embed", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                raw = data.get("embedding") if isinstance(data, dict) else None
                if isinstance(raw, list) and raw:
                    return [float(x) for x in raw]
            logger.debug(f"[embed] gateway returned HTTP {resp.status_code}; trying Gemini SDK.")
        except Exception as e:
            logger.debug(f"[embed] gateway unreachable ({e}); trying Gemini SDK.")

    client = shared_gemini_client()
    if client is None:
        return None

    from google.genai import types

    from .llm_retry import embed_content_with_retry

    model = gemini_embed_model()
    contents = format_embed_input(snippet, task_type=task_type, title=title) if _is_embed_v2_model(model) else snippet
    dim = gemini_embed_output_dimensionality()
    if _is_embed_v2_model(model):
        config = types.EmbedContentConfig(output_dimensionality=dim) if dim else None
    else:
        gemini_task = "RETRIEVAL_QUERY" if "query" in task_type.lower() else "RETRIEVAL_DOCUMENT"
        config = types.EmbedContentConfig(
            task_type=gemini_task,
            **({"output_dimensionality": dim} if dim else {}),
        )
    kwargs: dict[str, Any] = {"model": model, "contents": contents}
    if config is not None:
        kwargs["config"] = config

    try:
        response = embed_content_with_retry(
            model=model,
            contents=contents,
            config=config,
            label="embed",
        )
        if response.embeddings:
            values = response.embeddings[0].values
            if values:
                return [float(x) for x in values]
    except Exception as e:
        logger.warning(f"[embed] Gemini SDK failed after retries: {e}")
    return None
