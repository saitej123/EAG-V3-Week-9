"""UI catalog: document inventory, design deferrals, eval query spec."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .documents import INDEXABLE_SUFFIXES
from .indexing import index_document_path
from .memory import MemoryService, _build_faiss_index, _load_faiss_from_disk, _save_faiss_to_disk
from .paths import SANDBOX

ROOT = Path(__file__).resolve().parent.parent
ASSIGNMENT_PATH = ROOT / "corpus" / "dag" / "ASSIGNMENT.json"
RESEARCH_CORPUS_DIR = "research_papers"

UPLOAD_DIR = "uploads"
SCAN_DIRS = ("research_papers", "papers", UPLOAD_DIR)


def _worked_query_ids() -> set[str]:
    spec = load_assignment_spec()
    ids = spec.get("worked_query_ids")
    if isinstance(ids, list) and ids:
        return {str(x) for x in ids}
    return {"COMP"}


def load_worked_queries() -> list[dict[str, Any]]:
    """Subset of assignment queries used for structural shape tests (browser comparison)."""
    wanted = _worked_query_ids()
    return [row for row in load_assignment_queries() if str(row.get("id", "")) in wanted]


def worked_queries_payload() -> dict[str, Any]:
    """DAG worked queries for eval runners and unit tests."""
    rows = load_worked_queries()
    spec = load_assignment_spec()
    return {
        "description": "Worked browser comparison query (COMP) — shape in demo query corpus.",
        "session_root": spec.get("session_root", "state/sessions"),
        "query_count": len(rows),
        "queries": rows,
    }


def get_dag_query(query_id: str) -> dict[str, Any] | None:
    qid = query_id.strip()
    for row in load_assignment_queries():
        if str(row.get("id", "")).lower() == qid.lower():
            return row
    return None


def load_assignment_spec() -> dict[str, Any]:
    if not ASSIGNMENT_PATH.is_file():
        return {"queries": []}
    return json.loads(ASSIGNMENT_PATH.read_text(encoding="utf-8"))


def load_assignment_queries() -> list[dict[str, Any]]:
    return list(load_assignment_spec().get("queries", []))


_REQUIRED_QUERY_FIELDS = ("id", "part", "title", "query", "wall_clock_sec")
_EXPECTED_QUERY_IDS = frozenset({"COMP", "DEAL", "TICKET", "STACK", "FORGE", "B1", "B2", "B3", "B4"})
_BROWSER_QUERY_IDS = _EXPECTED_QUERY_IDS


def _build_submission_outline(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Ordered submission sections for UI (browser-only assignment)."""
    outline = spec.get("submission_outline")
    if isinstance(outline, list) and outline:
        return [
            {
                "part": int(row.get("part") or 0),
                "title": str(row.get("title") or ""),
                "summary": str(row.get("summary") or ""),
                "query_ids": [str(x) for x in row.get("query_ids") or []],
                "design_id": row.get("design_id"),
            }
            for row in outline
        ]

    part_labels = {1: "Base", 2: "Parallel", 3: "Critic", 4: "Coder", 5: "New skill", 6: "Browser"}
    by_part: dict[int, list[str]] = {}
    for row in spec.get("queries") or []:
        part = int(row.get("part") or 0)
        by_part.setdefault(part, []).append(str(row["id"]))
    return [
        {
            "part": part,
            "title": part_labels.get(part, f"Group {part}"),
            "summary": "",
            "query_ids": by_part[part],
            "design_id": None,
        }
        for part in sorted(by_part)
    ]


def validate_assignment_corpus() -> list[str]:
    """Return human-readable errors; empty list means corpus is UI/API-ready."""
    errors: list[str] = []
    spec = load_assignment_spec()
    rows = spec.get("queries") or []
    ids = {str(r.get("id", "")) for r in rows}
    if ids != _EXPECTED_QUERY_IDS:
        missing = sorted(_EXPECTED_QUERY_IDS - ids)
        extra = sorted(ids - _EXPECTED_QUERY_IDS - {""})
        if missing:
            errors.append(f"Missing query ids: {', '.join(missing)}")
        if extra:
            errors.append(f"Unexpected query ids: {', '.join(extra)}")

    for row in rows:
        qid = str(row.get("id", ""))
        for field in _REQUIRED_QUERY_FIELDS:
            if row.get(field) in (None, ""):
                errors.append(f"{qid}: missing required field {field}")
        qtext = str(row.get("query") or "").strip()
        if not qtext:
            errors.append(f"{qid}: empty query text")
        try:
            if float(row.get("wall_clock_sec") or 0) <= 0:
                errors.append(f"{qid}: wall_clock_sec must be positive")
        except (TypeError, ValueError):
            errors.append(f"{qid}: invalid wall_clock_sec")
        min_actions = row.get("min_browser_actions")
        if min_actions is not None:
            try:
                if int(min_actions) < 3:
                    errors.append(f"{qid}: min_browser_actions must be >= 3")
            except (TypeError, ValueError):
                errors.append(f"{qid}: invalid min_browser_actions")
        featured = str(row.get("featured") or "")
        if featured in {"browser_design", "browser_creative"} and not min_actions:
            errors.append(f"{qid}: comparison task missing min_browser_actions")

    by_id = {str(r["id"]): r for r in rows if r.get("id")}
    for dq in spec.get("design_queries") or []:
        kind = str(dq.get("kind") or "")
        if kind == "parallel":
            ref = str(dq.get("query_id") or "")
            if ref not in by_id:
                errors.append(f"design_queries {dq.get('id')}: unknown query_id {ref}")
        elif kind == "critic":
            for ref in dq.get("query_ids") or []:
                if str(ref) not in by_id:
                    errors.append(f"design_queries {dq.get('id')}: unknown query_id {ref}")
        elif kind == "new_skill":
            ref = str(dq.get("query_id") or "")
            if ref not in by_id:
                errors.append(f"design_queries {dq.get('id')}: unknown query_id {ref}")
        elif kind == "browser":
            ref = str(dq.get("query_id") or "")
            if ref not in by_id:
                errors.append(f"design_queries {dq.get('id')}: unknown query_id {ref}")
        else:
            errors.append(f"design_queries {dq.get('id')}: unknown kind {kind}")

    outline = _build_submission_outline(spec)
    outlined_ids: list[str] = []
    for section in outline:
        outlined_ids.extend(section["query_ids"])
    if sorted(outlined_ids) != sorted(_EXPECTED_QUERY_IDS):
        errors.append("submission_outline must list every query id exactly once")
    design_by_id = {str(d["id"]): d for d in spec.get("design_queries") or [] if d.get("id")}
    for section in outline:
        did = section.get("design_id")
        if did and str(did) not in design_by_id:
            errors.append(f"outline part {section['part']}: unknown design_id {did}")

    return errors


def expected_flow_for_query(row: dict[str, Any]) -> str:
    """Human-readable skill chain for UI (inserts auto-spliced critic after distiller)."""
    explicit = str(row.get("expected_flow") or "").strip()
    if explicit:
        return explicit
    skills = list(row.get("expected_skills") or [])
    auto = list(row.get("expected_auto") or [])
    parts: list[str] = []
    for skill in skills:
        parts.append(str(skill))
        if skill == "distiller" and "critic" in auto and "critic" not in parts:
            parts.append("critic")
    return " → ".join(parts)


def min_browser_actions_for_text(text: str) -> int:
    """Resolve comparison-task action minimum from corpus query text."""
    blob = (text or "").strip()
    if not blob:
        return 0
    for row in load_assignment_queries():
        try:
            minimum = int(row.get("min_browser_actions") or 0)
        except (TypeError, ValueError):
            continue
        if minimum <= 0:
            continue
        corpus_q = str(row.get("query") or "").strip()
        if not corpus_q:
            continue
        if corpus_q == blob or corpus_q[:120] in blob or blob[:120] in corpus_q:
            return minimum
    return 0


def enrich_assignment_query(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["expected_flow"] = expected_flow_for_query(row)
    return out


def browser_queries_payload() -> dict[str, Any]:
    """Browser comparison corpus for Super Browser UI and /api/queries/dag."""
    spec = load_assignment_spec()
    queries = [enrich_assignment_query(q) for q in spec.get("queries", [])]
    outline = _build_submission_outline(spec)
    design = [
        dq for dq in spec.get("design_queries", []) if str(dq.get("kind") or "") == "browser"
    ]
    groups = [
        {
            "part": row["part"],
            "label": row["title"],
            "query_ids": row["query_ids"],
        }
        for row in outline
    ]
    return {
        "description": spec.get("description", ""),
        "session_root": spec.get("session_root", "state/sessions"),
        "log_dir": spec.get("log_dir", "logs/dag"),
        "query_count": len(queries),
        "queries": queries,
        "design_queries": design,
        "browser_findings": spec.get("browser_findings", {}),
        "browser_reference_runs": spec.get("browser_reference_runs", []),
        "groups": groups,
        "outline": outline,
        "browser_only": True,
    }


def assignment_payload() -> dict[str, Any]:
    """Alias for browser-only assignment corpus."""
    return browser_queries_payload()


# --- Design deferrals (see docs/DEFERRALS.md) ---


class DesignDeferral(BaseModel):
    id: str
    title: str
    summary: str
    where: str
    forward_phase: str
    forward_topic: str
    remedy: str | None = None


DESIGN_DEFERRALS: list[DesignDeferral] = [
    DesignDeferral(
        id="dense_only",
        title="Dense retrieval only",
        summary=(
            "Vector retrieval has no hybrid sparse partner. Production systems run BM25 or "
            "learned-sparse retrieval alongside dense FAISS and fuse ranked lists with "
            "Reciprocal Rank Fusion (RRF). This codebase uses dense retrieval alone."
        ),
        where="super_browser/memory.py — vector-first read()",
        forward_phase="Future release",
        forward_topic="Hybrid retrieval + RRF inside Memory.read()",
    ),
    DesignDeferral(
        id="heuristic_chunking",
        title="Heuristic sliding-window chunking",
        summary=(
            "Documents split at arbitrary word boundaries (default 400 words, 80 overlap). "
            "Simple and fast; may cut sentences mid-thought."
        ),
        where="super_browser/indexing.py — _chunk_text(), index_document_path()",
        forward_phase="Planned upgrade",
        forward_topic="Semantic chunking (sentence/paragraph/section aware)",
    ),
    DesignDeferral(
        id="faiss_reload",
        title="FAISS reload on every read",
        summary=(
            "The FAISS index reloads from disk on every memory.read() for MCP cross-process "
            "consistency. Cost is negligible at demo scale; mmap + mtime invalidation "
            "or inter-process locks matter at higher scale."
        ),
        where="super_browser/memory.py — _load_faiss_from_disk()",
        forward_phase="Future release",
        forward_topic="Memory-mapped index with mtime invalidation",
    ),
    DesignDeferral(
        id="fixed_embed_model",
        title="Fixed embedding model",
        summary=(
            "GEMINI_EMBED_MODEL (default gemini-embedding-2) pins the semantic space. "
            "Changing the model silently invalidates all stored vectors."
        ),
        where="super_browser/llm_env.py — gemini_embed_model(), try_embed_text()",
        forward_phase="Future release",
        forward_topic="Explicit re-embed / index-version migration",
        remedy=(
            "Delete state/index.faiss and state/index_ids.json, rebuild from memory.json "
            "(text preserved in value/descriptor), or run scripts/clean.py and re-index."
        ),
    ),
]


def deferrals_payload() -> dict[str, Any]:
    return {
        "scope": "current",
        "doc": "docs/DEFERRALS.md",
        "deferrals": [d.model_dump() for d in DESIGN_DEFERRALS],
        "forward_roadmap": [
            {"topic": "Semantic chunking", "phase": "Planned upgrade"},
            {"topic": "Hybrid retrieval + RRF", "phase": "Future release"},
            {"topic": "Parallel DAG fan-out", "phase": "Planned upgrade"},
            {"topic": "Skills abstraction", "phase": "Future release"},
            {"topic": "Cross-encoder reranking", "phase": "Future release"},
            {"topic": "FAISS mmap + mtime cache", "phase": "Future release"},
        ],
    }


# --- Document catalog ---


class IndexedDocumentRecord(BaseModel):
    path: str
    indexed: bool = True
    chunk_count: int = 0
    page_count: int | None = None
    extraction: str | None = None
    source_kind: str | None = None
    first_indexed_at: str | None = None
    last_indexed_at: str | None = None
    preview: str | None = None
    citation_sample: str | None = None
    memory_ids: list[str] = Field(default_factory=list)


class SandboxFileRecord(BaseModel):
    path: str
    name: str
    size_bytes: int
    suffix: str
    folder: str
    indexable: bool
    indexed: bool
    chunk_count: int = 0
    extraction: str | None = None
    modified_at: str | None = None


class DocumentCatalogStats(BaseModel):
    sandbox_files: int
    indexable_files: int
    indexed_documents: int
    total_chunks: int
    memory_items: int
    faiss_vectors: int
    upload_count: int


def _is_index_fact(item) -> bool:
    if item.kind != "fact":
        return False
    v = item.value or {}
    path = v.get("path")
    if not isinstance(path, str) or not path.strip():
        return False
    src = str(item.source or "")
    return src.startswith("mcp:") and "text" in v


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    try:
        return dt.isoformat()
    except (TypeError, ValueError):
        return None


def aggregate_indexed_documents(items: list) -> list[IndexedDocumentRecord]:
    groups: dict[str, list] = {}
    for item in items:
        if not _is_index_fact(item):
            continue
        path = str(item.value.get("path"))
        groups.setdefault(path, []).append(item)

    records: list[IndexedDocumentRecord] = []
    for path, facts in sorted(groups.items()):
        v0 = facts[0].value or {}
        pages = {f.value.get("page_number") for f in facts if f.value.get("page_number") is not None}
        page_count = len(pages) if pages else None
        previews = []
        for f in facts[:2]:
            t = (f.value or {}).get("text")
            if isinstance(t, str) and t.strip():
                previews.append(t.strip()[:160])
        cite = next(
            (str((f.value or {}).get("citation")) for f in facts if (f.value or {}).get("citation")),
            None,
        )
        created = min((f.created_at for f in facts if f.created_at), default=None)
        updated = max((f.created_at for f in facts if f.created_at), default=None)
        records.append(
            IndexedDocumentRecord(
                path=path,
                indexed=True,
                chunk_count=len(facts),
                page_count=page_count,
                extraction=str(v0.get("extraction") or "unknown"),
                source_kind=str(v0.get("source_kind") or ""),
                first_indexed_at=_iso(created),
                last_indexed_at=_iso(updated),
                preview=" … ".join(previews) if previews else None,
                citation_sample=cite,
                memory_ids=[f.id for f in facts],
            )
        )
    return records


def _scan_sandbox_files() -> list[SandboxFileRecord]:
    rows: list[SandboxFileRecord] = []
    for folder in SCAN_DIRS:
        base = SANDBOX / folder
        if not base.is_dir():
            continue
        for f in sorted(base.rglob("*")):
            if not f.is_file():
                continue
            rel = str(f.relative_to(SANDBOX)).replace("\\", "/")
            suffix = f.suffix.lower()
            if folder == UPLOAD_DIR or suffix in INDEXABLE_SUFFIXES:
                mtime = datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds")
                rows.append(
                    SandboxFileRecord(
                        path=rel,
                        name=f.name,
                        size_bytes=f.stat().st_size,
                        suffix=suffix or "(none)",
                        folder=folder,
                        indexable=suffix in INDEXABLE_SUFFIXES or suffix == "",
                        modified_at=mtime,
                        indexed=False,
                    )
                )
    return rows


def catalog_stats(service: MemoryService | None = None) -> DocumentCatalogStats:
    svc = service or MemoryService()
    svc._load_disk()
    indexed = aggregate_indexed_documents(svc._items)
    index_map = {r.path: r for r in indexed}
    sandbox_rows = _scan_sandbox_files()
    for row in sandbox_rows:
        idx = _index_record_for_path(row.path, index_map)
        if idx:
            row.indexed = True
            row.chunk_count = idx.chunk_count
            row.extraction = idx.extraction

    index, ids = _load_faiss_from_disk()
    faiss_n = index.ntotal if index is not None else 0
    uploads = [r for r in sandbox_rows if r.folder == UPLOAD_DIR]
    indexable = [r for r in sandbox_rows if r.indexable]

    return DocumentCatalogStats(
        sandbox_files=len(sandbox_rows),
        indexable_files=len(indexable),
        indexed_documents=len(indexed),
        total_chunks=sum(r.chunk_count for r in indexed),
        memory_items=len(svc._items),
        faiss_vectors=faiss_n if faiss_n == len(ids) else len(ids),
        upload_count=len(uploads),
    )


def _index_record_for_path(path: str, index_map: dict[str, IndexedDocumentRecord]) -> IndexedDocumentRecord | None:
    """Match indexed chunks to a catalog row (PDF ↔ sidecar aliases)."""
    if path in index_map:
        return index_map[path]
    from .indexing import related_index_paths

    for alt in related_index_paths(path):
        if alt in index_map:
            return index_map[alt]
    return None


def get_document_catalog(service: MemoryService | None = None) -> dict[str, Any]:
    svc = service or MemoryService()
    svc._load_disk()
    indexed = aggregate_indexed_documents(svc._items)
    index_map = {r.path: r for r in indexed}
    sandbox_files = _scan_sandbox_files()

    merged_paths = {f.path for f in sandbox_files}
    for rec in indexed:
        merged_paths.add(rec.path)

    documents: list[dict[str, Any]] = []
    for path in sorted(merged_paths):
        disk = next((f for f in sandbox_files if f.path == path), None)
        idx = _index_record_for_path(path, index_map)
        documents.append(
            {
                "path": path,
                "name": disk.name if disk else Path(path).name,
                "folder": disk.folder if disk else path.split("/", 1)[0],
                "size_bytes": disk.size_bytes if disk else None,
                "suffix": disk.suffix if disk else Path(path).suffix.lower(),
                "indexable": disk.indexable if disk else True,
                "on_disk": disk is not None,
                "indexed": idx is not None,
                "chunk_count": idx.chunk_count if idx else 0,
                "page_count": idx.page_count if idx else None,
                "extraction": idx.extraction if idx else None,
                "first_indexed_at": idx.first_indexed_at if idx else None,
                "last_indexed_at": idx.last_indexed_at if idx else None,
                "preview": idx.preview if idx else None,
                "citation_sample": idx.citation_sample if idx else None,
                "modified_at": disk.modified_at if disk else None,
            }
        )

    stats = catalog_stats(svc)
    return {
        "stats": stats.model_dump(),
        "documents": documents,
        "indexed": [r.model_dump() for r in indexed],
        "scan_dirs": list(SCAN_DIRS),
        **deferrals_payload(),
    }


def remove_document_index(path: str, service: MemoryService | None = None) -> dict[str, Any]:
    from .indexing import related_index_paths
    from .memory import _PERSIST_LOCK

    svc = service or MemoryService()
    target = path.strip().replace("\\", "/")
    paths_to_clear = related_index_paths(target)
    with _PERSIST_LOCK:
        svc._load_disk()
        before = len(svc._items)

        def keep(item) -> bool:
            if not _is_index_fact(item):
                return True
            p = str((item.value or {}).get("path") or "")
            return p not in paths_to_clear

        svc._items = [m for m in svc._items if keep(m)]
        removed = before - len(svc._items)
        svc._save_disk()
        rebuilt, ids = _build_faiss_index(svc._items)
        _save_faiss_to_disk(rebuilt, ids)
    return {"path": target, "facts_removed": removed, "status": "removed", "paths_cleared": sorted(paths_to_clear)}


def reindex_document(
    path: str,
    *,
    use_vlm: bool | None = None,
    service: MemoryService | None = None,
) -> dict[str, Any]:
    removed = remove_document_index(path, service=service)
    result = index_document_path(path, use_vlm=use_vlm)
    return {"removed": removed, "indexed": result}


def save_upload(filename: str, raw: bytes, *, subdir: str = UPLOAD_DIR) -> dict[str, Any]:
    safe = Path(filename).name
    if not safe or safe in {".", ".."}:
        raise ValueError("Invalid filename")
    dest_dir = SANDBOX / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / safe
    dest.write_bytes(raw)
    rel = f"{subdir}/{safe}".replace("\\", "/")
    return {
        "path": rel,
        "size_bytes": len(raw),
        "name": safe,
        "folder": subdir,
    }


def index_uploaded_file(path: str, *, use_vlm: bool | None = None) -> dict[str, Any]:
    return index_document_path(path, use_vlm=use_vlm)
