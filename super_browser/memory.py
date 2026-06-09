"""
Memory service: typed ``MemoryItem`` rows in ``state/memory.json`` plus a FAISS
vector index (``state/index.faiss`` + ``state/index_ids.json``).

Reads are vector-first (embed query → FAISS dense search) with keyword overlap as fallback when
embeddings are unavailable, no items carry embeddings, or the index is empty. There is **no**
hybrid sparse retriever (BM25 / learned-sparse) or Reciprocal Rank Fusion yet —
see ``docs/DEFERRALS.md``. Writes embed
``fact`` / ``preference`` / ``tool_outcome`` descriptors at insertion; ``scratchpad`` rows
leave ``embedding`` null.

The FAISS index is reloaded from disk on every ``read()`` so MCP-subprocess writes from
``index_document`` are visible to the agent process without an in-process cache.

Cross-process contract: whatever is on disk after a write is what the next read sees.
``memory.json`` holds every item; ``index.faiss`` + ``index_ids.json`` hold only rows with
embeddings (scratchpad items appear in JSON but not in FAISS). Caching the index in-process
would break MCP visibility — the implementation pays a small disk-read cost per query instead.

Commerce SQLite catalog remains alongside episodic memory for PDP caching.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from loguru import logger
from pydantic import ValidationError

from .llm_env import gemini_models_ordered, shared_gemini_client, try_embed_text
from .paths import STATE as STATE_DIR
from .schemas import CachedProductRow, CommerceProduct, MemoryClassifyLLM, MemoryItem, MemoryKind, ToolCall
MEMORY_JSON_PATH = STATE_DIR / "memory.json"
INDEX_FAISS_PATH = STATE_DIR / "index.faiss"
INDEX_IDS_PATH = STATE_DIR / "index_ids.json"
DB_PATH = STATE_DIR / "commerce.db"

# Serialize cross-process / parallel index writes (memory.json + FAISS append).
_PERSIST_LOCK = threading.RLock()

_STOPWORDS = frozenset(
    """
    a an the and or but if to of in on for with as by at from into through during before after above below
    between under again further then once here there when where why how all both each few more most other some
    such no nor not only own same so than too very can will just don should now is are was were be been being
    it its this that these those my your our their me him her them us i you he she we they what which who whom
    """.split()
)


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"\W+", text.lower()) if t and t not in _STOPWORDS and len(t) > 1}


def memory_hit_content_excerpt(item: MemoryItem, *, max_len: int = 1200) -> str:
    """Salient text from a hit's ``value`` payload — not just the descriptor.

    Decision often needs ``value.text`` / ``value.chunk`` (indexed corpus), ``value.date``
    (birthday recall), or ``value.raw`` even when the descriptor omits them.
    """
    v = item.value or {}
    parts: list[str] = []

    entity = v.get("entity")
    date = v.get("date")
    citation = v.get("citation")
    if isinstance(citation, str) and citation.strip():
        parts.append(f"cite: {citation.strip()}")

    if entity and date:
        parts.append(f"{entity}'s birthday is on {date}")

    for key in ("text", "chunk", "raw", "preview"):
        val = v.get(key)
        if isinstance(val, str) and val.strip():
            s = val.strip()
            if s not in parts:
                parts.append(s)

    if isinstance(date, str) and date.strip() and date.strip() not in parts:
        parts.append(date.strip())
    if isinstance(entity, str) and entity.strip() and entity.strip() not in parts:
        parts.append(entity.strip())

    body = " | ".join(parts)
    if not body:
        return ""
    if len(body) > max_len:
        return body[: max_len - 3] + "..."
    return body


def _format_hits(
    hits: list[MemoryItem],
    *,
    max_hits: int = 24,
    max_chars: int = 16000,
    content_max_len: int = 1200,
) -> str:
    """Render memory hits for role prompts — expose descriptor **and** decisive value fields."""
    if not hits:
        return "  (no memory hits)"

    blocks: list[str] = []
    for i, h in enumerate(hits[:max_hits]):
        lines = [
            f"[hit {i}] id={h.id!r} kind={h.kind!r} confidence={h.confidence}",
            f"  descriptor: {h.descriptor}",
        ]
        excerpt = memory_hit_content_excerpt(h, max_len=content_max_len)
        if excerpt:
            lines.append(f"  content: {excerpt}")
        elif h.value:
            try:
                val_json = json.dumps(h.value, ensure_ascii=False, default=str)
                if len(val_json) > content_max_len:
                    val_json = val_json[: content_max_len - 3] + "..."
                lines.append(f"  value: {val_json}")
            except (TypeError, ValueError):
                pass
        if h.keywords:
            lines.append(f"  keywords: {h.keywords[:12]}")
        cite = (h.value or {}).get("citation")
        if isinstance(cite, str) and cite.strip():
            lines.append(f"  citation: {cite.strip()}")
        page_no = (h.value or {}).get("page_number")
        if page_no is not None:
            lines.append(f"  page: {page_no}")
        if h.artifact_id:
            lines.append(f"  artifact_id: {h.artifact_id!r}")
        blocks.append("\n".join(lines))

    text = "\n\n".join(blocks)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n… (memory hits truncated)"
    return text


def _memory_value_dict_from_json_blob(raw: str) -> dict[str, Any]:
    """Parse ``MemoryClassifyLLM.value_json`` into ``MemoryItem.value`` (Developer API cannot use map schemas)."""
    s = (raw or "").strip() or "{}"
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
        return {"value": obj}
    except json.JSONDecodeError:
        return {"text": s}


def _try_embed(descriptor: str, *, task_type: str = "retrieval_document") -> list[float] | None:
    return try_embed_text(descriptor, task_type=task_type)


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    out = vectors.astype(np.float32, copy=True)
    faiss.normalize_L2(out)
    return out


def _load_faiss_from_disk() -> tuple[faiss.Index | None, list[str]]:
    """Load the on-disk FAISS index and parallel id list (no in-process cache)."""
    if not INDEX_FAISS_PATH.exists() or not INDEX_IDS_PATH.exists():
        return None, []
    try:
        index = faiss.read_index(str(INDEX_FAISS_PATH))
        raw_ids = json.loads(INDEX_IDS_PATH.read_text(encoding="utf-8"))
        ids = [str(x) for x in raw_ids] if isinstance(raw_ids, list) else []
        if index.ntotal != len(ids):
            logger.warning(
                f"FAISS index size ({index.ntotal}) != id list ({len(ids)}); ignoring index."
            )
            return None, []
        return index, ids
    except Exception as e:
        logger.warning(f"FAISS load failed ({e}); treating index as empty.")
        return None, []


def _save_faiss_to_disk(index: faiss.Index | None, ids: list[str]) -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        if index is None or index.ntotal == 0:
            if INDEX_FAISS_PATH.exists():
                INDEX_FAISS_PATH.unlink()
            if INDEX_IDS_PATH.exists():
                INDEX_IDS_PATH.unlink()
            return
        faiss.write_index(index, str(INDEX_FAISS_PATH))
        INDEX_IDS_PATH.write_text(json.dumps(ids, indent=2), encoding="utf-8")
    except OSError as e:
        logger.error(f"FAISS write failed: {e}")


def _build_faiss_index(items: list[MemoryItem]) -> tuple[faiss.Index | None, list[str]]:
    ids: list[str] = []
    vectors: list[list[float]] = []
    for item in items:
        if item.embedding:
            ids.append(item.id)
            vectors.append(item.embedding)
    if not vectors:
        return None, []
    dim = len(vectors[0])
    index = faiss.IndexFlatIP(dim)
    arr = _normalize_vectors(np.array(vectors, dtype=np.float32))
    index.add(arr)
    return index, ids


def _append_to_faiss(index: faiss.Index | None, ids: list[str], item_id: str, embedding: list[float]) -> tuple[faiss.Index, list[str]]:
    vec = np.array([embedding], dtype=np.float32)
    _normalize_vectors(vec)
    if index is None:
        index = faiss.IndexFlatIP(len(embedding))
    index.add(vec)
    ids = list(ids) + [item_id]
    return index, ids


class MemoryService:
    def __init__(self) -> None:
        try:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"MemoryService: cannot create state dir: {e}")
            raise
        self._items: list[MemoryItem] = []
        self._load_disk()
        self._init_db()
        self._ensure_faiss_synced()

    # -- persistence ---------------------------------------------------------

    def _load_disk(self) -> None:
        with _PERSIST_LOCK:
            if not MEMORY_JSON_PATH.exists():
                self._items = []
                self._save_disk()
                return
            try:
                raw = json.loads(MEMORY_JSON_PATH.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"memory.json unreadable ({e}); starting empty.")
                self._items = []
                return

            items: list[MemoryItem] = []
            if isinstance(raw, dict) and isinstance(raw.get("items"), list):
                for row in raw["items"]:
                    try:
                        items.append(MemoryItem.model_validate(row))
                    except ValidationError:
                        continue
            elif isinstance(raw, dict) and isinstance(raw.get("facts"), list):
                for f in raw["facts"]:
                    if not isinstance(f, dict):
                        continue
                    text = str(f.get("text", "")).strip()
                    kws = [str(x).lower() for x in f.get("keywords", []) if x]
                    items.append(
                        MemoryItem(
                            id=new_id("mig"),
                            kind="fact",
                            keywords=kws,
                            descriptor=text[:240] or "(fact)",
                            value={"text": text},
                            artifact_id=None,
                            source="migrated_facts_v1",
                            run_id="",
                            goal_id=None,
                            confidence=1.0,
                            created_at=datetime.now(timezone.utc),
                        )
                    )
            self._items = items

    def _save_disk(self) -> None:
        try:
            MEMORY_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = {"items": [m.model_dump(mode="json") for m in self._items]}
            MEMORY_JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as e:
            logger.error(f"memory.json write failed: {e}")

    def _ensure_faiss_synced(self) -> None:
        """Rebuild the FAISS index from memory.json when disk index is missing but items have vectors."""
        index, ids = _load_faiss_from_disk()
        embedded = [m for m in self._items if m.embedding]
        if index is not None and index.ntotal > 0:
            return
        if not embedded:
            return
        rebuilt, rebuilt_ids = _build_faiss_index(self._items)
        _save_faiss_to_disk(rebuilt, rebuilt_ids)
        logger.info(f"[memory] rebuilt FAISS index from {len(rebuilt_ids)} embedded items.")

    def _persist_item(self, item: MemoryItem) -> MemoryItem:
        """Append to memory.json, then append to the on-disk FAISS index (synchronous)."""
        with _PERSIST_LOCK:
            self._load_disk()
            self._items.append(item)
            self._save_disk()
            if item.embedding:
                index, ids = _load_faiss_from_disk()
                index, ids = _append_to_faiss(index, ids, item.id, item.embedding)
                _save_faiss_to_disk(index, ids)
        return item

    def _init_db(self) -> None:
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS products (
                        url TEXT PRIMARY KEY,
                        platform TEXT,
                        product_name TEXT,
                        base_price REAL,
                        net_price REAL,
                        bank_offers_text TEXT,
                        scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"SQLite init failed ({DB_PATH}): {e}")

    # -- Vector-first read -------------------------------------------------------

    def read(
        self,
        query: str,
        history: list[dict[str, Any]],
        *,
        kinds: list[MemoryKind] | None = None,
        top_k: int = 8,
    ) -> list[MemoryItem]:
        """Vector-ranked recall with keyword fallback (no LLM).

        Reloads memory.json and the FAISS index from disk on every call so MCP-subprocess
        writes are visible without caching the index in this process.
        """
        self._load_disk()
        if kinds:
            pool = [m for m in self._items if m.kind in kinds]
        else:
            pool = list(self._items)

        k = max(1, top_k)
        by_id = {m.id: m for m in pool}
        has_any_embedding = any(m.embedding for m in self._items)

        query_embedding = _try_embed(query, task_type="retrieval_query")
        index, index_ids = _load_faiss_from_disk()

        if query_embedding and has_any_embedding and index is not None and index.ntotal > 0:
            try:
                qvec = np.array([query_embedding], dtype=np.float32)
                _normalize_vectors(qvec)
                distances, indices = index.search(qvec, min(k, index.ntotal))
                hits: list[MemoryItem] = []
                seen: set[str] = set()
                for idx, score in zip(indices[0], distances[0]):
                    if idx < 0:
                        continue
                    mem_id = index_ids[idx]
                    if mem_id in seen:
                        continue
                    item = by_id.get(mem_id)
                    if item is None:
                        continue
                    seen.add(mem_id)
                    hits.append(item)
                    if len(hits) >= k:
                        break
                if hits:
                    return hits
            except Exception as e:
                logger.warning(f"[memory.read] vector search failed ({e}); falling back to keywords.")

        return self._keyword_read(query, history, pool=pool, top_k=k)

    def _keyword_read(
        self,
        query: str,
        history: list[dict[str, Any]],
        *,
        pool: list[MemoryItem],
        top_k: int,
    ) -> list[MemoryItem]:
        ctx_bits: list[str] = []
        for h in history[-24:]:
            try:
                ctx_bits.append(json.dumps(h, default=str))
            except Exception:
                ctx_bits.append(str(h))
        ctx = " ".join(ctx_bits)
        qt = _tokens(query + " " + ctx)

        def score(m: MemoryItem) -> float:
            desc_t = _tokens(m.descriptor)
            key_t = set(m.keywords)
            return float(len(qt & desc_t) + len(qt & key_t) + 0.25 * len(qt & _tokens(json.dumps(m.value))))

        ranked = sorted(pool, key=score, reverse=True)
        return ranked[:top_k]

    def filter(
        self,
        *,
        kinds: list[MemoryKind] | None = None,
        goal_id: str | None = None,
        recent: int | None = None,
    ) -> list[MemoryItem]:
        self._load_disk()
        out = list(self._items)
        if kinds:
            out = [m for m in out if m.kind in kinds]
        if goal_id:
            out = [m for m in out if m.goal_id == goal_id]
        out.sort(key=lambda m: m.created_at, reverse=True)
        if recent is not None:
            out = out[:recent]
        return out

    def remember(
        self,
        raw_text: str,
        *,
        source: str,
        run_id: str,
        goal_id: str | None = None,
    ) -> None:
        """Classify free-form text via one structured LLM call, embed, then persist."""
        text = (raw_text or "").strip()
        if not text:
            return

        classified = self._classify_with_llm(text)
        descriptor = classified.descriptor or text[:240]
        embedding: list[float] | None = None
        if classified.kind != "scratchpad":
            embedding = _try_embed(descriptor, task_type="retrieval_document")

        item = MemoryItem(
            id=new_id("mem"),
            kind=classified.kind,
            keywords=[k.lower() for k in classified.keywords],
            descriptor=descriptor,
            value=_memory_value_dict_from_json_blob(classified.value_json),
            artifact_id=None,
            embedding=embedding,
            source=source,
            run_id=run_id,
            goal_id=goal_id,
            confidence=classified.confidence,
            created_at=datetime.now(timezone.utc),
        )
        self._persist_item(item)
        snippet = text if len(text) <= 80 else text[:77] + "..."
        kw_json = json.dumps(item.keywords[:8], ensure_ascii=False)
        logger.info(f'[memory.remember]  classified {snippet!r} as {item.kind}')
        logger.info(f'{" " * 19}keywords: {kw_json}')

    def _classify_with_llm(self, text: str) -> MemoryClassifyLLM:
        client = shared_gemini_client()
        models = gemini_models_ordered()
        prompt = f"""
Classify the following user content for a durable agent memory store.

Return JSON matching the schema with:
- kind: one of fact | preference | tool_outcome | scratchpad (use "fact" for birthdays and stated truths).
- keywords: short lowercase tokens useful for keyword recall.
- descriptor: ONE short human-readable line.
- value_json: ONE JSON **object** serialized as a string with canonical fields when obvious (e.g. {{"entity":"…","date":"…"}}).

Content:
{text}
"""
        if client is None or not models:
            return MemoryClassifyLLM(
                kind="fact",
                keywords=[w for w in _tokens(text)][:12],
                descriptor=text[:200],
                value_json=json.dumps({"text": text}, ensure_ascii=False),
                confidence=0.5,
            )

        try:
            from google.genai import types

            last_err: Exception | None = None
            for model_id in models:
                try:
                    response = client.models.generate_content(
                        model=model_id,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=MemoryClassifyLLM,
                            temperature=0.2,
                        ),
                    )
                    raw = (response.text or "").strip()
                    data = json.loads(raw)
                    return MemoryClassifyLLM.model_validate(data)
                except Exception as e:
                    last_err = e
                    logger.warning(f"[memory.classify] model={model_id} failed: {e}")
            logger.warning(f"[memory.classify] fallback heuristic after {last_err!r}")
        except Exception as e:
            logger.warning(f"[memory.classify] failed {e}")

        return MemoryClassifyLLM(
            kind="fact",
            keywords=[w for w in _tokens(text)][:12],
            descriptor=text[:200],
            value_json=json.dumps({"text": text}, ensure_ascii=False),
            confidence=0.5,
        )

    def record_outcome(
        self,
        *,
        tool_call: ToolCall,
        result_text: str,
        artifact_id: str | None,
        run_id: str,
        goal_id: str | None,
    ) -> MemoryItem:
        desc = f"{tool_call.name}({json.dumps(tool_call.arguments, default=str)[:180]}) → artifact={artifact_id}"
        embedding = _try_embed(desc[:500], task_type="retrieval_document")
        item = MemoryItem(
            id=new_id("out"),
            kind="tool_outcome",
            keywords=[tool_call.name.lower()]
            + [w for w in _tokens(json.dumps(tool_call.arguments, default=str))][:8],
            descriptor=desc[:500],
            value={
                "tool": tool_call.name,
                "arguments": tool_call.arguments,
                "preview": result_text[:4000],
            },
            artifact_id=artifact_id,
            embedding=embedding,
            source="mcp",
            run_id=run_id,
            goal_id=goal_id,
            confidence=1.0,
            created_at=datetime.now(timezone.utc),
        )
        return self._persist_item(item)

    # -- commerce catalog (concierge / PDP path) ------------------------------

    def upsert_product(self, product: CommerceProduct) -> None:
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO products (url, platform, product_name, base_price, net_price, bank_offers_text, scraped_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(url) DO UPDATE SET
                        platform=excluded.platform,
                        product_name=excluded.product_name,
                        base_price=excluded.base_price,
                        net_price=excluded.net_price,
                        bank_offers_text=excluded.bank_offers_text,
                        scraped_at=CURRENT_TIMESTAMP
                    """,
                    (
                        product.url,
                        product.platform,
                        product.product_name,
                        product.base_price,
                        product.net_price,
                        product.bank_offers_text,
                    ),
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"upsert_product failed: {e}")

    def query_products(self, search_term: str = "") -> list[CachedProductRow]:
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                query = "SELECT url, platform, product_name, base_price, net_price, bank_offers_text, scraped_at FROM products"
                params: list[Any] = []
                if search_term:
                    query += " WHERE product_name LIKE ? OR url LIKE ? OR platform LIKE ?"
                    pat = f"%{search_term}%"
                    params.extend([pat, pat, pat])
                query += " ORDER BY scraped_at DESC LIMIT 100"
                cursor.execute(query, params)
                rows = cursor.fetchall()
                out: list[CachedProductRow] = []
                for row in rows:
                    try:
                        out.append(CachedProductRow.model_validate(dict(row)))
                    except ValidationError:
                        continue
                return out
        except sqlite3.Error as e:
            logger.warning(f"query_products failed: {e}")
            return []


# Module-level helpers used by MCP ``index_document`` (cross-process writes).
_default_service: MemoryService | None = None


def _service() -> MemoryService:
    global _default_service
    if _default_service is None:
        _default_service = MemoryService()
    return _default_service


def add_fact(
    descriptor: str,
    *,
    value: dict,
    keywords: list[str],
    source: str,
    run_id: str,
    goal_id: str | None = None,
) -> MemoryItem:
    embedding = _try_embed(descriptor, task_type="retrieval_document")
    item = MemoryItem(
        id=new_id("mem"),
        kind="fact",
        keywords=[k.lower() for k in keywords],
        descriptor=descriptor,
        value=value,
        embedding=embedding,
        source=source,
        run_id=run_id,
        goal_id=goal_id,
    )
    return _service()._persist_item(item)


# Back-compat alias for older imports and docs
MemoryManager = MemoryService

# Legacy export: artifact text directory (binary store lives beside it)
ARTIFACTS_DIR = STATE_DIR / "artifacts"
