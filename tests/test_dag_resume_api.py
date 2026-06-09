"""DAG resume API and session metadata."""

from __future__ import annotations

import json

import networkx as nx
from fastapi.testclient import TestClient

from super_browser.dag_schemas import NodeSpec, NodeState, NodeStatus, PlannerOutput
from super_browser.flow import Graph
from super_browser.graph_viz import list_dag_sessions, session_resume_meta
from super_browser.persistence import SessionStore
from super_browser.skills import SkillRegistry


def _partial_k_session(tmp_path, monkeypatch) -> str:
    from super_browser import persistence as pers_mod

    monkeypatch.setattr(pers_mod, "SESSIONS_DIR", tmp_path / "sessions")
    sid = "dag_K_partial_ui"
    store = SessionStore(sid)
    g = Graph(SkillRegistry())
    plan = PlannerOutput(
        rationale="three cities",
        nodes=[
            NodeSpec(skill="researcher", inputs=["USER_QUERY"], metadata={"label": "lagos"}),
            NodeSpec(skill="researcher", inputs=["USER_QUERY"], metadata={"label": "cairo"}),
            NodeSpec(skill="researcher", inputs=["USER_QUERY"], metadata={"label": "kinshasa"}),
            NodeSpec(skill="coder", inputs=["n:lagos", "n:cairo", "n:kinshasa"], metadata={"label": "fast"}),
            NodeSpec(skill="formatter", inputs=["n:fast"], metadata={"label": "out"}),
        ],
    )
    p = g.add_node_from_spec(
        NodeSpec(skill="planner", inputs=["USER_QUERY"], metadata={"label": "planner"}),
        node_id="n:1",
    )
    g.extend_from(p, plan)
    store.save_query(
        "For Lagos, Cairo, and Kinshasa, find current populations and growth rates "
        "and tell me which is growing fastest."
    )
    store.save_graph(g.dg)
    store.save_node_state(NodeState(node_id="n:1", skill="planner", status=NodeStatus.complete, output="{}"))
    for nid, label in [("n:2", "lagos"), ("n:3", "cairo")]:
        store.save_node_state(
            NodeState(node_id=nid, skill="researcher", status=NodeStatus.complete, output="ok")
        )
    store.save_node_state(NodeState(node_id="n:4", skill="researcher", status=NodeStatus.running))
    return sid


def test_session_resume_meta_partial_k(tmp_path, monkeypatch):
    sid = _partial_k_session(tmp_path, monkeypatch)
    meta = session_resume_meta(sid)
    assert meta["resumable"] is True
    assert meta["run_complete"] is False
    assert meta["running_count"] == 1
    assert meta["pending_count"] >= 0
    assert meta["resume_action"] == "continue"
    assert meta["resume_enabled"] is True


def test_list_dag_sessions_includes_resume_flags(tmp_path, monkeypatch):
    sid = _partial_k_session(tmp_path, monkeypatch)
    rows = list_dag_sessions()
    row = next(r for r in rows if r["session_id"] == sid)
    assert row["resumable"] is True


def test_resume_endpoint_rejects_fully_complete_without_formatter(tmp_path, monkeypatch):
    from super_browser import persistence as pers_mod

    monkeypatch.setattr(pers_mod, "SESSIONS_DIR", tmp_path / "sessions")
    sid = "dag_done_no_fmt"
    store = SessionStore(sid)
    store.save_query("done")
    g = nx.DiGraph()
    g.add_node("n:1", skill="planner")
    store.save_graph(g)
    store.save_node_state(NodeState(node_id="n:1", skill="planner", status=NodeStatus.complete, output="{}"))

    from app import app

    client = TestClient(app)
    res = client.post("/run-agent/resume", json={"session_id": sid})
    assert res.status_code == 400
    res2 = client.post("/run-agent/resume", json={"session_id": sid, "from_node_id": "n:missing"})
    assert res2.status_code == 400

    meta = session_resume_meta(sid)
    assert meta["resume_enabled"] is False


def test_resume_meta_enabled_when_formatter_complete(tmp_path, monkeypatch):
    from super_browser import persistence as pers_mod

    monkeypatch.setattr(pers_mod, "SESSIONS_DIR", tmp_path / "sessions")
    sid = "dag_done"
    store = SessionStore(sid)
    store.save_query("done")
    g = nx.DiGraph()
    g.add_node("n:1", skill="planner")
    g.add_node("n:2", skill="formatter")
    g.add_edge("n:1", "n:2")
    store.save_graph(g)
    store.save_node_state(NodeState(node_id="n:2", skill="formatter", status=NodeStatus.complete, output='{"text":"ok"}'))

    meta = session_resume_meta(sid)
    assert meta["resume_action"] == "continue"
    assert meta["resume_enabled"] is True
    assert meta["has_formatter"] is True


def test_resume_replay_formatter_on_complete_session(tmp_path, monkeypatch):
    import app as app_mod

    from super_browser import persistence as pers_mod

    monkeypatch.setattr(pers_mod, "SESSIONS_DIR", tmp_path / "sessions")
    sid = "dag_done_replay"
    store = SessionStore(sid)
    store.save_query("done")
    g = nx.DiGraph()
    g.add_node("n:1", skill="planner")
    g.add_node("n:2", skill="formatter")
    g.add_edge("n:1", "n:2")
    store.save_graph(g)
    store.save_node_state(NodeState(node_id="n:2", skill="formatter", status=NodeStatus.complete, output='{"text":"ok"}'))

    with app_mod._ops_lock:
        app_mod._run_busy = False

    def _noop_task(coro):
        if hasattr(coro, "close"):
            coro.close()
        return None

    monkeypatch.setattr(app_mod.asyncio, "create_task", _noop_task)

    client = TestClient(app_mod.app)
    res = client.post("/run-agent/resume", json={"session_id": sid, "from_node_id": "n:2"})
    assert res.status_code == 200
    st = store.load_node_state("n:2")
    assert st.status == NodeStatus.pending


def test_prepare_session_for_resume_resets_running_on_disk(tmp_path, monkeypatch):
    from super_browser import persistence as pers_mod
    from super_browser.graph_viz import prepare_session_for_resume

    monkeypatch.setattr(pers_mod, "SESSIONS_DIR", tmp_path / "sessions")
    sid = _partial_k_session(tmp_path, monkeypatch)
    store = SessionStore(sid)
    st = store.load_node_state("n:4")
    assert st.status == NodeStatus.running
    meta = prepare_session_for_resume(sid)
    st2 = store.load_node_state("n:4")
    assert st2.status == NodeStatus.pending
    assert meta["running_count"] == 0
    assert meta["pending_count"] >= 1

def test_reset_from_node_keeps_upstream_complete(tmp_path, monkeypatch):
    from super_browser import persistence as pers_mod

    monkeypatch.setattr(pers_mod, "SESSIONS_DIR", tmp_path / "sessions")
    sid = _partial_k_session(tmp_path, monkeypatch)
    store = SessionStore(sid)
    reset_ids = store.reset_from_node("n:4")
    assert "n:4" in reset_ids
    assert store.load_node_state("n:2").status == NodeStatus.complete
    assert store.load_node_state("n:3").status == NodeStatus.complete
    assert store.load_node_state("n:4").status == NodeStatus.pending


def test_dag_unlock_clears_busy_flag(tmp_path, monkeypatch):
    import app as app_mod

    with app_mod._ops_lock:
        app_mod._run_busy = True
        app_mod._run_task = None
    client = TestClient(app_mod.app)
    res = client.post("/api/dag/unlock")
    assert res.status_code == 200
    assert res.json().get("agent_busy") is False


def test_api_agent_stop_clears_busy(tmp_path, monkeypatch):
    import app as app_mod

    with app_mod._ops_lock:
        app_mod._run_busy = True
        app_mod._run_task = None
    client = TestClient(app_mod.app)
    res = client.post("/api/agent/stop")
    assert res.status_code == 200
    body = res.json()
    assert body.get("status") == "success"
    assert body.get("agent_busy") is False


def test_resume_endpoint_accepts_resumable_session(tmp_path, monkeypatch):
    import app as app_mod

    sid = _partial_k_session(tmp_path, monkeypatch)
    with app_mod._ops_lock:
        app_mod._run_busy = False

    def _noop_task(coro):
        if hasattr(coro, "close"):
            coro.close()
        return None

    monkeypatch.setattr(app_mod.asyncio, "create_task", _noop_task)

    client = TestClient(app_mod.app)
    res = client.post("/run-agent/resume", json={"session_id": sid})
    assert res.status_code == 200
    assert res.json().get("status") == "Agent resumed"
    assert res.json().get("session_id") == sid
    st = SessionStore(sid).load_node_state("n:4")
    assert st.status == NodeStatus.pending
