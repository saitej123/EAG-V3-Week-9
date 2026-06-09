"""DAG graph visualization payload tests."""

from __future__ import annotations

import json
import os
import time

import networkx as nx

from super_browser.dag_schemas import AgentResult, NodeSpec, NodeState, NodeStatus, PlannerOutput
from super_browser.flow import Graph
from super_browser.graph_viz import graph_viz_payload, list_dag_sessions
from super_browser.persistence import SessionStore
from super_browser.skills import SkillRegistry


def test_graph_viz_payload_hello_shape(tmp_path, monkeypatch):
    from super_browser import persistence as pers_mod

    monkeypatch.setattr(pers_mod, "SESSIONS_DIR", tmp_path / "sessions")
    sid = "dag_test_viz"
    store = SessionStore(sid)
    g = Graph(SkillRegistry())
    plan = PlannerOutput(
        rationale="hi",
        nodes=[NodeSpec(skill="formatter", inputs=["USER_QUERY"], metadata={"label": "out"})],
    )
    p = g.add_node_from_spec(NodeSpec(skill="planner", inputs=["USER_QUERY"], metadata={"label": "planner"}), node_id="n:1")
    g.extend_from(p, plan)
    store.save_query("Say hello.")
    store.save_graph(g.dg)
    store.save_node_state(
        NodeState(node_id="n:1", skill="planner", status=NodeStatus.complete, output="{}")
    )

    store.save_memory_hits(
        [
            {
                "descriptor": "faiss:chunk",
                "source": "papers/foo.md",
                "value": {"chunk": "Attention is all you need preview text."},
            }
        ]
    )

    payload = graph_viz_payload(sid)
    assert payload["node_count"] == 2
    assert payload["edge_count"] == 1
    assert len(payload["nodes"]) == 2
    assert any(n["skill"] == "formatter" for n in payload["nodes"])
    assert payload["stats"]["status_counts"]["complete"] == 1
    assert len(payload["memory_hits"]) == 1
    assert "Attention" in payload["memory_hits"][0]["preview"]
    assert "resumable" in payload
    assert payload["nodes"][0].get("result_preview") is not None
    assert all(n.get("position") for n in payload["nodes"])
    coords = {(n["position"]["x"], n["position"]["y"]) for n in payload["nodes"]}
    assert len(coords) == len(payload["nodes"])
    assert payload["nodes"][0]["status_label"] == "done"


def test_graph_viz_shows_running_from_disk(tmp_path, monkeypatch):
    from super_browser import persistence as pers_mod

    monkeypatch.setattr(pers_mod, "SESSIONS_DIR", tmp_path / "sessions")
    sid = "dag_running_viz"
    store = SessionStore(sid)
    g = Graph(SkillRegistry())
    p = g.add_node_from_spec(NodeSpec(skill="planner", inputs=["USER_QUERY"], metadata={"label": "planner"}), node_id="n:1")
    f = g.add_node_from_spec(NodeSpec(skill="formatter", inputs=["n:1"], metadata={"label": "out"}), node_id="n:2")
    g.dg.add_edge(p, f)
    store.save_query("test")
    store.save_graph(g.dg)
    store.save_node_state(NodeState(node_id="n:1", skill="planner", status=NodeStatus.complete, output="{}"))
    store.save_node_state(NodeState(node_id="n:2", skill="formatter", status=NodeStatus.running, output=None))

    payload = graph_viz_payload(sid)
    fmt = next(n for n in payload["nodes"] if n["id"] == "n:2")
    assert fmt["status"] == "running"
    assert fmt["status_label"] == "run"
    assert payload["stats"]["status_counts"]["running"] == 1


def test_graph_viz_agent_result_str_status_on_graph_only(tmp_path, monkeypatch):
    """Mid-wave poll: graph.json has AgentResult before node state file exists."""
    from super_browser import persistence as pers_mod

    monkeypatch.setattr(pers_mod, "SESSIONS_DIR", tmp_path / "sessions")
    sid = "dag_graph_result_only"
    store = SessionStore(sid)
    g = nx.DiGraph()
    g.add_node(
        "n:4",
        skill="formatter",
        label="out",
        metadata={"label": "out"},
        result=AgentResult(success=True, agent_name="formatter", status="running"),
    )
    store.save_query("test")
    store.save_graph(g)

    payload = graph_viz_payload(sid)
    fmt = next(n for n in payload["nodes"] if n["id"] == "n:4")
    assert fmt["status"] == "running"
    assert fmt["status_label"] == "run"
    assert fmt["result_preview"] == "(running…)"


def test_api_dag_graph_formatter_running_mid_wave(tmp_path, monkeypatch):
    """GET /api/dag/graph must not 500 when AgentResult.status is a str (UI live poll)."""
    from super_browser import persistence as pers_mod

    monkeypatch.setattr(pers_mod, "SESSIONS_DIR", tmp_path / "sessions")
    sid = "dag_M_api_live"
    store = SessionStore(sid)
    g = nx.DiGraph()
    g.add_node("n:1", skill="planner", label="planner", metadata={"label": "planner"})
    g.add_node(
        "n:4",
        skill="formatter",
        label="out",
        metadata={"label": "out"},
        result=AgentResult(success=True, agent_name="formatter", status="running"),
    )
    g.add_edge("n:1", "n:4")
    store.save_query("test query M")
    store.save_graph(g)

    from app import app
    from fastapi.testclient import TestClient

    client = TestClient(app)
    res = client.get(f"/api/dag/graph?session_id={sid}")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    assert body["session_id"] == sid
    fmt = next(n for n in body["nodes"] if n["id"] == "n:4")
    assert fmt["status"] == "running"
    assert fmt["status_label"] == "run"
    assert fmt["color"]["background"] == "#fef3c7"


def test_list_dag_sessions_orders_newest(tmp_path, monkeypatch):
    from super_browser import persistence as pers_mod

    monkeypatch.setattr(pers_mod, "SESSIONS_DIR", tmp_path / "sessions")
    base = time.time()
    for i, sid in enumerate(("dag_old", "dag_new")):
        store = SessionStore(sid)
        store.ensure_dirs()
        store.save_query(sid)
        store.graph_path.write_text(
            json.dumps(nx.node_link_data(nx.DiGraph()), indent=2),
            encoding="utf-8",
        )
        os.utime(store.graph_path, (base + i, base + i))
    rows = list_dag_sessions()
    assert rows[0]["session_id"] == "dag_new"
