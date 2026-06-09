"""Atomic session persistence for DAG runs."""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import networkx as nx
from pydantic import ValidationError

from .dag_schemas import AgentResult, NodeState, NodeStatus
from .paths import STATE

SESSIONS_DIR = STATE / "sessions"
_RESULT_TYPED = "_result_typed"


class SessionLoadError(Exception):
    """Raised when a persisted session file fails revival."""


def node_filename(node_id: str) -> str:
    """``n:1`` → ``n_001.json`` (DAG session on-disk layout)."""
    if node_id.startswith("n:"):
        try:
            num = int(node_id.split(":", 1)[1])
            return f"n_{num:03d}.json"
        except ValueError:
            pass
    safe = node_id.replace(":", "_")
    return f"{safe}.json"


def node_id_from_filename(name: str) -> str:
    stem = name.removesuffix(".json")
    if stem.startswith("n_") and stem[2:].isdigit():
        return f"n:{int(stem[2:])}"
    return stem.replace("_", ":")


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)


def _serialize_node_attr(value: Any) -> Any:
    if isinstance(value, AgentResult):
        payload = value.model_dump(mode="json")
        payload[_RESULT_TYPED] = True
        return payload
    return value


def _revive_node_attr(key: str, value: Any, *, path: Path, node_id: str) -> Any:
    if key == "result" and isinstance(value, dict) and value.get(_RESULT_TYPED):
        clean = {k: v for k, v in value.items() if k != _RESULT_TYPED}
        try:
            return AgentResult.model_validate(clean)
        except ValidationError as e:
            raise SessionLoadError(f"{path}: node {node_id} result revival failed: {e}") from e
    return value


def _export_graph(graph: nx.DiGraph) -> dict[str, Any]:
    export = graph.copy()
    for nid, attrs in list(export.nodes(data=True)):
        for key, val in list(attrs.items()):
            export.nodes[nid][key] = _serialize_node_attr(val)
    return nx.node_link_data(export)


def _import_graph(data: dict[str, Any], *, path: Path) -> nx.DiGraph:
    graph = nx.node_link_graph(data)
    for nid in list(graph.nodes):
        attrs = dict(graph.nodes[nid])
        for key, val in attrs.items():
            graph.nodes[nid][key] = _revive_node_attr(key, val, path=path, node_id=str(nid))
    return graph


class SessionStore:
    """``state/sessions/<sid>/`` — query.txt, graph.json, nodes/n_NNN.json."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.root = SESSIONS_DIR / session_id
        self.query_path = self.root / "query.txt"
        self.graph_path = self.root / "graph.json"
        self.legacy_pickle_path = self.root / "graph.pkl"
        self.nodes_dir = self.root / "nodes"

    def exists(self) -> bool:
        return self.graph_path.is_file() or self.legacy_pickle_path.is_file()

    def ensure_dirs(self) -> None:
        self.nodes_dir.mkdir(parents=True, exist_ok=True)

    def save_query(self, user_query: str) -> None:
        self.ensure_dirs()
        _atomic_write(self.query_path, user_query)

    def load_query(self) -> str:
        if not self.query_path.is_file():
            raise SessionLoadError(f"Missing query.txt for session {self.session_id}")
        try:
            return self.query_path.read_text(encoding="utf-8")
        except OSError as e:
            raise SessionLoadError(str(e)) from e

    def save_graph(self, graph: nx.DiGraph) -> None:
        self.ensure_dirs()
        data = _export_graph(graph)
        _atomic_write(self.graph_path, json.dumps(data, indent=2, ensure_ascii=False))

    def load_graph(self) -> nx.DiGraph:
        if self.graph_path.is_file():
            try:
                raw = json.loads(self.graph_path.read_text(encoding="utf-8"))
                return _import_graph(raw, path=self.graph_path)
            except (json.JSONDecodeError, OSError, KeyError) as e:
                raise SessionLoadError(f"{self.graph_path}: {e}") from e
        if self.legacy_pickle_path.is_file():
            print(
                f"[session] loading legacy pickle {self.legacy_pickle_path}",
                file=sys.stderr,
            )
            try:
                with self.legacy_pickle_path.open("rb") as fh:
                    obj = pickle.load(fh)
                if isinstance(obj, nx.DiGraph):
                    return obj
                raise SessionLoadError(f"{self.legacy_pickle_path}: not a DiGraph")
            except (OSError, pickle.UnpicklingError) as e:
                raise SessionLoadError(str(e)) from e
        raise SessionLoadError(f"Missing graph.json for session {self.session_id}")

    def save_node_state(self, state: NodeState) -> None:
        self.ensure_dirs()
        path = self.nodes_dir / node_filename(state.node_id)
        _atomic_write(path, state.model_dump_json(indent=2))

    def load_node_state(self, node_id: str) -> NodeState:
        path = self.nodes_dir / node_filename(node_id)
        if not path.is_file():
            raise SessionLoadError(f"Missing node state {node_id} at {path}")
        try:
            return NodeState.model_validate_json(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, ValidationError) as e:
            raise SessionLoadError(f"{path}: {e}") from e

    def load_all_node_states(self) -> dict[str, NodeState]:
        out: dict[str, NodeState] = {}
        if not self.nodes_dir.is_dir():
            return out
        for path in sorted(self.nodes_dir.glob("*.json")):
            nid = node_id_from_filename(path.name)
            try:
                out[nid] = NodeState.model_validate_json(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, ValidationError) as e:
                raise SessionLoadError(f"{path}: {e}") from e
        return out

    def reset_running_to_pending(self) -> None:
        for path in self.nodes_dir.glob("*.json"):
            state = NodeState.model_validate_json(path.read_text(encoding="utf-8"))
            if state.status == NodeStatus.running:
                state.status = NodeStatus.pending
                state.started_at = None
                self.save_node_state(state)

    def reset_from_node(self, node_id: str) -> list[str]:
        """Stop in-flight work and reset *node_id* plus all descendants to pending (upstream stays done)."""
        self.reset_running_to_pending()
        graph = self.load_graph()
        if node_id not in graph:
            raise SessionLoadError(f"Unknown node {node_id!r} in session {self.session_id}")
        targets = set(nx.descendants(graph, node_id)) | {node_id}
        states = self.load_all_node_states()
        reset_ids: list[str] = []
        for nid in nx.topological_sort(graph):
            if nid not in targets:
                continue
            data = graph.nodes[nid]
            st = states.get(nid)
            if st is None:
                st = NodeState(
                    node_id=nid,
                    skill=str(data.get("skill") or "?"),
                    inputs=list(data.get("inputs") or []),
                    metadata=dict(data.get("metadata") or {}),
                    status=NodeStatus.pending,
                )
            else:
                st.status = NodeStatus.pending
                st.output = None
                st.error = None
                st.elapsed_s = None
                st.started_at = None
                st.finished_at = None
                st.artifact_id = None
            self.save_node_state(st)
            reset_ids.append(nid)
        return reset_ids

    @property
    def memory_hits_path(self) -> Path:
        return self.root / "memory_hits.json"

    def save_memory_hits(self, hits: list[Any]) -> None:
        """Snapshot FAISS hits at session start for the graph viewer."""
        self.ensure_dirs()
        rows: list[dict[str, Any]] = []
        for h in hits:
            if hasattr(h, "model_dump"):
                rows.append(h.model_dump(mode="json"))
            elif isinstance(h, dict):
                rows.append(h)
        _atomic_write(self.memory_hits_path, json.dumps(rows, indent=2, ensure_ascii=False))

    def load_memory_hits(self) -> list[dict[str, Any]]:
        if not self.memory_hits_path.is_file():
            return []
        try:
            raw = json.loads(self.memory_hits_path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, list) else []
        except (json.JSONDecodeError, OSError):
            return []
