"""Serialize persisted DAG sessions for the Web UI (Cytoscape.js + dagre)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

from .dag_schemas import AgentResult, NodeState, NodeStatus
from . import persistence
from .persistence import SessionLoadError, SessionStore

_SKILL_COLORS: dict[str, str] = {
    "planner": "#18181b",
    "researcher": "#2563eb",
    "retriever": "#0891b2",
    "distiller": "#7c3aed",
    "summariser": "#6366f1",
    "critic": "#d97706",
    "coder": "#059669",
    "sandbox_executor": "#0d9488",
    "formatter": "#52525b",
    "calculator": "#db2777",
    "prosody_analyst": "#c026d3",
    "browser": "#0284c7",
}

_STATUS_COLORS: dict[str, str] = {
    "pending": "#e4e4e7",
    "running": "#fef3c7",
    "complete": "#dcfce7",
    "failed": "#fee2e2",
    "skipped": "#f4f4f5",
}

_STATUS_BORDERS: dict[str, str] = {
    "pending": "#a1a1aa",
    "running": "#f59e0b",
    "complete": "#22c55e",
    "failed": "#ef4444",
    "skipped": "#d4d4d8",
}

# UI legend: done / run / wait / fail (maps from NodeStatus values on disk)
_STATUS_LABEL: dict[str, str] = {
    "pending": "wait",
    "running": "run",
    "complete": "done",
    "failed": "fail",
    "skipped": "skip",
}


def _normalize_status(raw: str) -> str:
    s = (raw or "pending").strip().lower()
    if s in _STATUS_COLORS:
        return s
    aliases = {
        "done": "complete",
        "wait": "pending",
        "run": "running",
        "fail": "failed",
    }
    return aliases.get(s, "pending")


def _status_raw(status: Any) -> str:
    """NodeStatus enum and AgentResult.status (plain str) both normalize here."""
    if status is None:
        return "pending"
    if hasattr(status, "value"):
        return str(status.value)
    return str(status)


def status_ui_label(status: str) -> str:
    return _STATUS_LABEL.get(_normalize_status(status), status)


def _layered_positions(graph: nx.DiGraph) -> dict[str, dict[str, float]]:
    """Deterministic top-down coordinates — Cytoscape preset fallback if dagre stacks nodes."""
    if graph.number_of_nodes() == 0:
        return {}
    levels: dict[str, int] = {}
    try:
        order = list(nx.topological_sort(graph))
    except nx.NetworkXUnfeasible:
        order = list(graph.nodes())
    for nid in order:
        preds = list(graph.predecessors(nid))
        levels[nid] = max((levels[p] + 1 for p in preds), default=0)
    by_level: dict[int, list[str]] = {}
    for nid, lv in levels.items():
        by_level.setdefault(lv, []).append(nid)
    pos: dict[str, dict[str, float]] = {}
    for lv, nids in sorted(by_level.items()):
        for i, nid in enumerate(sorted(nids, key=str)):
            pos[nid] = {"x": float(i * 240), "y": float(lv * 150)}
    return pos


def _node_status(skill: str, graph: nx.DiGraph, nid: str, states: dict[str, Any]) -> str:
    """Prefer persisted node state files over embedded graph.result (stale mid-wave)."""
    if nid in states:
        st = states[nid]
        return _normalize_status(_status_raw(st.status))
    result = graph.nodes[nid].get("result")
    if isinstance(result, AgentResult):
        return _normalize_status(_status_raw(result.status))
    if isinstance(result, dict):
        return _normalize_status(str(result.get("status") or "pending"))
    return "pending"


def _memory_hits_for_ui(store: SessionStore) -> list[dict[str, str]]:
    """Compact memory hits for the graph sidebar."""
    rows: list[dict[str, str]] = []
    for raw in store.load_memory_hits()[:8]:
        if not isinstance(raw, dict):
            continue
        desc = str(raw.get("descriptor") or "memory")
        val = raw.get("value") if isinstance(raw.get("value"), dict) else {}
        chunk = val.get("chunk") or val.get("raw") or val.get("text") or ""
        preview = str(chunk).strip()[:200]
        if len(str(chunk).strip()) > 200:
            preview += "…"
        source = str(raw.get("source") or val.get("path") or "")
        rows.append({"descriptor": desc, "source": source, "preview": preview})
    return rows


def _session_stats(states: dict[str, Any]) -> dict[str, Any]:
    counts = {"complete": 0, "running": 0, "pending": 0, "failed": 0, "skipped": 0}
    wall = 0.0
    for st in states.values():
        raw = _normalize_status(_status_raw(st.status))
        if raw in counts:
            counts[raw] += 1
        if getattr(st, "elapsed_s", None):
            wall = max(wall, float(st.elapsed_s))
    return {"status_counts": counts, "max_node_elapsed_s": round(wall, 2)}


def _preview_text(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, dict):
        if not raw:
            return ""
        try:
            text = json.dumps(raw, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(raw)
    else:
        text = str(raw).strip()
    if text in ("{}", "[]", '""', "''", "None"):
        return ""
    return text[:280] + ("…" if len(text) > 280 else "")


def _result_preview(graph: nx.DiGraph, nid: str, states: dict[str, Any]) -> str:
    if nid in states:
        st = states[nid]
        if st.error:
            return f"Error: {st.error[:200]}"
        preview = _preview_text(st.output)
        if preview:
            return preview
        if _normalize_status(_status_raw(getattr(st, "status", None))) == "running":
            return "(running…)"
    result = graph.nodes[nid].get("result")
    if isinstance(result, AgentResult):
        if result.error:
            return f"Error: {result.error[:200]}"
        preview = _preview_text(result.output)
        if preview:
            return preview
        if _normalize_status(_status_raw(result.status)) == "running":
            return "(running…)"
    return "(no output yet)"


def _graph_node_count(graph_path: Path) -> int:
    try:
        data = json.loads(graph_path.read_text(encoding="utf-8"))
        return len(data.get("nodes", []))
    except (json.JSONDecodeError, OSError):
        return 0


def _status_counts_for_graph(graph: nx.DiGraph, states: dict[str, Any]) -> tuple[dict[str, int], dict[str, str]]:
    """Count statuses for every node in graph.json (not only state files on disk)."""
    counts = {"complete": 0, "running": 0, "pending": 0, "failed": 0, "skipped": 0}
    by_nid: dict[str, str] = {}
    for nid, data in graph.nodes(data=True):
        skill = str(data.get("skill") or "?")
        status = _node_status(skill, graph, nid, states)
        by_nid[nid] = status
        if status in counts:
            counts[status] += 1
    return counts, by_nid


def _formatter_terminal_complete(graph: nx.DiGraph, by_nid: dict[str, str]) -> bool:
    formatter_ids = [nid for nid, data in graph.nodes(data=True) if str(data.get("skill") or "") == "formatter"]
    if not formatter_ids:
        return False
    return all(by_nid.get(nid) == "complete" for nid in formatter_ids)


def _has_incomplete_work(by_nid: dict[str, str]) -> bool:
    return any(s in ("pending", "running", "failed") for s in by_nid.values())


def session_resume_meta(session_id: str) -> dict[str, Any]:
    """Resume eligibility from graph nodes + persisted node states (no query-id hardcoding)."""
    store = SessionStore(session_id)
    if not store.exists():
        return {
            "resumable": False,
            "run_complete": False,
            "status_counts": {},
            "running_count": 0,
            "pending_count": 0,
            "resume_enabled": False,
            "resume_action": "none",
        }
    try:
        states = store.load_all_node_states()
        graph = store.load_graph()
    except SessionLoadError:
        return {
            "resumable": False,
            "run_complete": False,
            "status_counts": {},
            "running_count": 0,
            "pending_count": 0,
            "load_error": True,
            "resume_enabled": False,
            "resume_action": "none",
        }

    counts, by_nid = _status_counts_for_graph(graph, states)
    formatter_complete = _formatter_terminal_complete(graph, by_nid)
    running_count = counts.get("running", 0)
    pending_count = counts.get("pending", 0)
    failed_count = counts.get("failed", 0)
    graph_nodes = graph.number_of_nodes()
    has_formatter = any(
        str(data.get("skill") or "") == "formatter" for _, data in graph.nodes(data=True)
    )

    # Incomplete nodes in the DAG → continue (SIGKILL mid-wave leaves running→pending on resume).
    resumable = graph_nodes > 0 and _has_incomplete_work(by_nid) and not formatter_complete

    base = {
        "resumable": resumable,
        "run_complete": formatter_complete,
        "status_counts": counts,
        "running_count": running_count,
        "pending_count": pending_count,
        "failed_count": failed_count,
        "graph_node_count": graph_nodes,
        "state_file_count": len(states),
        "has_formatter": has_formatter,
        "incomplete_node_count": sum(
            1 for s in by_nid.values() if s in ("pending", "running", "failed")
        ),
    }
    return {**base, **_resume_ui_fields(base)}


def _resume_hint_from_counts(meta: dict[str, Any]) -> str:
    from_nid = meta.get("resume_from_node_id")
    if from_nid:
        n = len(meta.get("resume_reset_node_ids") or [])
        return f"Continue from {from_nid}" + (f" ({n} node(s) reset)" if n else "")
    sc = meta.get("status_counts") or {}
    parts: list[str] = []
    if sc.get("running"):
        parts.append(f"{sc['running']} running → stopped on resume")
    if sc.get("pending"):
        parts.append(f"{sc['pending']} pending")
    if sc.get("failed"):
        parts.append(f"{sc['failed']} failed")
    if meta.get("run_complete") and meta.get("has_formatter"):
        parts.append("select formatter node to re-run")
    return "; ".join(parts) if parts else "Continues from disk (running → pending, then ready nodes)"


def _resume_ui_fields(meta: dict[str, Any]) -> dict[str, Any]:
    """UI fields derived only from node status counts on the graph."""
    if meta.get("load_error"):
        return {
            "resume_action": "none",
            "resume_label": "Resume session",
            "resume_enabled": False,
            "resume_disabled_reason": "Session data could not be loaded.",
        }

    has_formatter = meta.get("has_formatter")
    can_continue = meta.get("resumable")
    can_replay_terminal = meta.get("run_complete") and has_formatter
    resume_enabled = bool(can_continue or can_replay_terminal)

    if resume_enabled:
        return {
            "resume_action": "continue",
            "resume_label": "Resume",
            "resume_enabled": True,
            "resume_disabled_reason": None,
            "resume_hint": _resume_hint_from_counts(meta),
        }

    reason = "No incomplete nodes in graph"
    if meta.get("graph_node_count", 0) == 0:
        reason = "Graph has no nodes yet"
    elif meta.get("run_complete"):
        reason = "All nodes complete (including formatter)"
    return {
        "resume_action": "none",
        "resume_label": "Resume session",
        "resume_enabled": False,
        "resume_disabled_reason": reason + " — interrupt a run to persist partial state, then resume",
    }


def prepare_session_for_resume(
    session_id: str,
    *,
    from_node_id: str | None = None,
) -> dict[str, Any]:
    """Prepare disk for resume: stop running nodes; optionally rewind from a selected node."""
    store = SessionStore(session_id)
    if not store.exists():
        return session_resume_meta(session_id)
    if from_node_id:
        try:
            reset_ids = store.reset_from_node(from_node_id.strip())
        except SessionLoadError as e:
            meta = session_resume_meta(session_id)
            meta["resume_enabled"] = False
            meta["resume_disabled_reason"] = str(e)
            return meta
        meta = session_resume_meta(session_id)
        meta["resume_from_node_id"] = from_node_id.strip()
        meta["resume_reset_node_ids"] = reset_ids
        ui = _resume_ui_fields(meta)
        return {**meta, **ui}
    store.reset_running_to_pending()
    meta = session_resume_meta(session_id)
    if meta.get("resumable"):
        return meta
    if meta.get("run_complete") and meta.get("has_formatter"):
        if prepare_formatter_replay(session_id):
            return session_resume_meta(session_id)
    return meta


def prepare_formatter_replay(session_id: str) -> bool:
    """Reset formatter node(s) to pending so resume can re-execute the final step."""
    store = SessionStore(session_id)
    if not store.exists():
        return False
    try:
        states = store.load_all_node_states()
        graph = store.load_graph()
    except SessionLoadError:
        return False
    changed = False
    for nid, data in graph.nodes(data=True):
        if str(data.get("skill") or "") != "formatter":
            continue
        st = states.get(nid)
        if st is None:
            st = NodeState(node_id=nid, skill="formatter", metadata=dict(data.get("metadata") or {}))
            states[nid] = st
        if st.status == NodeStatus.complete:
            st.status = NodeStatus.pending
            st.output = None
            st.error = None
            st.elapsed_s = None
            st.started_at = None
            st.finished_at = None
            store.save_node_state(st)
            changed = True
    return changed


def list_dag_sessions(*, limit: int = 30) -> list[dict[str, Any]]:
    """Recent session folders that contain graph.json, newest first."""
    sessions_dir = persistence.SESSIONS_DIR
    if not sessions_dir.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sessions_dir.iterdir():
        if not path.is_dir():
            continue
        graph_path = path / "graph.json"
        if not graph_path.is_file():
            continue
        mtime = graph_path.stat().st_mtime
        query = ""
        qpath = path / "query.txt"
        if qpath.is_file():
            query = qpath.read_text(encoding="utf-8", errors="replace").strip()
            if len(query) > 120:
                query = query[:117] + "…"
        sid = path.name
        resume = session_resume_meta(sid)
        rows.append(
            {
                "session_id": sid,
                "modified_utc": datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
                "query_preview": query,
                "node_count": _graph_node_count(graph_path),
                **resume,
            }
        )
    rows.sort(key=lambda r: r["modified_utc"], reverse=True)
    return rows[:limit]


def latest_dag_session_id() -> str | None:
    sessions = list_dag_sessions(limit=1)
    return sessions[0]["session_id"] if sessions else None


def _merge_graph_with_states(graph: nx.DiGraph, states: dict[str, Any]) -> nx.DiGraph:
    """Include node state files not yet in graph.json (mid-wave planner extend race)."""
    merged = graph.copy()
    for nid, st in states.items():
        if nid in merged:
            continue
        meta = dict(st.metadata or {})
        merged.add_node(
            nid,
            skill=str(st.skill or "?"),
            label=str(meta.get("label") or nid),
            metadata=meta,
            inputs=list(st.inputs or []),
        )
    return merged


def graph_viz_payload(session_id: str) -> dict[str, Any]:
    """Build Cytoscape-ready nodes/edges from a persisted session."""
    store = SessionStore(session_id)
    if not store.exists():
        raise SessionLoadError(f"No graph for session {session_id}")
    graph = store.load_graph()
    try:
        states = store.load_all_node_states()
    except SessionLoadError:
        states = {}
    graph = _merge_graph_with_states(graph, states)

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    positions = _layered_positions(graph)

    for nid, data in graph.nodes(data=True):
        skill = str(data.get("skill") or "?")
        label = str(data.get("label") or nid)
        status = _node_status(skill, graph, nid, states)
        timing = ""
        if nid in states and getattr(states[nid], "elapsed_s", None) is not None:
            timing = f"<br/>Elapsed: {states[nid].elapsed_s:.2f}s"
        title = (
            f"<b>{escape_html(skill)}</b> · {escape_html(nid)}<br/>"
            f"Status: {escape_html(status)}{timing}<br/>"
            f"{escape_html(_result_preview(graph, nid, states))}"
        )
        node_elapsed = None
        if nid in states and states[nid].elapsed_s is not None:
            node_elapsed = round(float(states[nid].elapsed_s), 2)
        preview = _result_preview(graph, nid, states)
        nodes.append(
            {
                "id": nid,
                "label": f"{skill}\n{label}",
                "title": title,
                "result_preview": preview,
                "elapsed_s": node_elapsed,
                "color": {
                    "background": _STATUS_COLORS.get(status, "#f4f4f5"),
                    "border": _STATUS_BORDERS.get(status, "#a1a1aa"),
                    "highlight": {"background": "#e0e7ff", "border": "#4f46e5"},
                },
                "font": {"color": "#18181b", "size": 15, "face": "Inter, system-ui, sans-serif", "multi": True},
                "borderWidth": 2 if status == "running" else 1,
                "shape": "box",
                "margin": 12,
                "widthConstraint": {"minimum": 110, "maximum": 260},
                "skill": skill,
                "status": status,
                "status_label": status_ui_label(status),
                "position": positions.get(nid),
            }
        )

    for src, dst in graph.edges():
        edges.append(
            {
                "from": src,
                "to": dst,
                "arrows": "to",
                "color": {"color": "#a1a1aa", "highlight": "#6366f1"},
                "smooth": {"type": "cubicBezier", "roundness": 0.2},
            }
        )

    query = ""
    if store.query_path.is_file():
        query = store.query_path.read_text(encoding="utf-8", errors="replace").strip()

    resume = session_resume_meta(session_id)

    return {
        "session_id": session_id,
        "query": query,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
        "layout_hint": "dagre",
        "memory_hits": _memory_hits_for_ui(store),
        "stats": _session_stats(states),
        **resume,
    }


def escape_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
