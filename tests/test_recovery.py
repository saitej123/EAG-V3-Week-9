"""Persistence, recovery classification, and critic splice tests."""

from __future__ import annotations

import asyncio
import json
import pickle

import networkx as nx
import pytest

from super_browser.dag_schemas import AgentResult, NodeSpec, NodeState, NodeStatus, PlannerOutput
from super_browser.flow import CRITIC_FAIL_CAP, Executor, Graph
from super_browser.persistence import SessionLoadError, SessionStore, node_filename, node_id_from_filename
from super_browser.recovery import classify_failure
from super_browser.skills import SkillRegistry

# --- classify_failure (pinned gateway error strings) ---------------------------------

@pytest.mark.parametrize(
    "error,expected",
    [
        ("HTTP 503 Service Unavailable", "transient"),
        ("upstream returned 502 Bad Gateway", "transient"),
        ("504 Gateway Timeout from worker", "transient"),
        ("request timeout after 120s", "transient"),
        ("connection reset by peer", "transient"),
        ("ConnectionError: failed to connect", "transient"),
        ("HTTPStatusError: 503", "transient"),
        ("service unavailable — try again", "transient"),
        ("malformed JSON in planner output", "validation_error"),
        ("1 validation error for NodeSpec", "validation_error"),
        ("ValidationError: field required", "validation_error"),
        ("JSON validation error at nodes[0]", "validation_error"),
        ("distiller produced wrong field types", "upstream_failure"),
        ("KeyError: 'population'", "upstream_failure"),
        ("sandbox exit code 1", "upstream_failure"),
        ("unexpected tool response", "upstream_failure"),
        ("empty researcher output", "upstream_failure"),
        ("RuntimeError: formatter missing", "upstream_failure"),
    ],
)
def test_classify_failure(error: str, expected: str):
    assert classify_failure(error) == expected


def test_classify_failure_case_insensitive_timeout():
    assert classify_failure("TIMEOUT waiting for gateway") == "transient"


# --- critic splice (4 tests) -------------------------------------------------------

def _distiller_planner_output() -> PlannerOutput:
    return PlannerOutput(
        rationale="extract",
        nodes=[
            NodeSpec(skill="distiller", inputs=["USER_QUERY"], metadata={"label": "d"}),
            NodeSpec(skill="formatter", inputs=["n:d"], metadata={"label": "out"}),
        ],
    )


def test_auto_inserted_critic_path_target_and_child_ids():
    reg = SkillRegistry()
    g = Graph(reg)
    p = g.add_node_from_spec(NodeSpec(skill="planner", metadata={"label": "p"}))
    g.extend_from(p, _distiller_planner_output())
    critics = [n for n, d in g.dg.nodes(data=True) if d.get("skill") == "critic"]
    assert len(critics) == 1
    meta = g.dg.nodes[critics[0]]["metadata"]
    assert meta.get("target") == meta.get("child")


def test_planner_emitted_critic_not_doubled():
    reg = SkillRegistry()
    g = Graph(reg)
    p = g.add_node_from_spec(NodeSpec(skill="planner", metadata={"label": "p"}))
    output = PlannerOutput(
        rationale="constrained write",
        nodes=[
            NodeSpec(skill="distiller", inputs=["USER_QUERY"], metadata={"label": "d"}),
            NodeSpec(
                skill="critic",
                inputs=["n:d"],
                metadata={"label": "crit", "target": "out", "child": "out", "question": "4-6-4 syllables"},
            ),
            NodeSpec(skill="formatter", inputs=["n:crit"], metadata={"label": "out"}),
        ],
    )
    g.extend_from(p, output)
    assert sum(1 for _, d in g.dg.nodes(data=True) if d.get("skill") == "critic") == 1


def test_critic_fail_recovery_cap_per_target():
    async def _run() -> None:
        ex = Executor(registry=SkillRegistry())
        g = ex.graph
        d = g.add_node_from_spec(NodeSpec(skill="distiller", metadata={"label": "d"}))
        c = g.add_node_from_spec(
            NodeSpec(skill="critic", inputs=["n:d"], metadata={"label": "crit", "target": "out", "child": "out"})
        )
        f = g.add_node_from_spec(NodeSpec(skill="formatter", inputs=["n:crit"], metadata={"label": "out"}))
        g.dg.add_edge(d, c)
        g.dg.add_edge(c, f)
        ex.states = {
            d: NodeState(node_id=d, skill="distiller", status=NodeStatus.complete, output="{}"),
            c: NodeState(node_id=c, skill="critic", status=NodeStatus.complete),
            f: NodeState(node_id=f, skill="formatter", status=NodeStatus.pending),
        }
        fail_json = '{"verdict": "fail", "rationale": "syllable mismatch"}'
        await ex._handle_critic(c, fail_json)
        assert sum(1 for _, nd in g.dg.nodes(data=True) if nd.get("skill") == "planner") == 1
        ex.graph.critic_fail_counts["out"] = CRITIC_FAIL_CAP
        await ex._handle_critic(c, fail_json)
        assert sum(1 for _, nd in g.dg.nodes(data=True) if nd.get("skill") == "planner") == 1

    asyncio.run(_run())


def test_critic_pass_leaves_graph_unchanged():
    async def _run() -> None:
        ex = Executor(registry=SkillRegistry())
        g = ex.graph
        c = g.add_node_from_spec(NodeSpec(skill="critic", metadata={"label": "crit", "target": "out", "child": "out"}))
        f = g.add_node_from_spec(NodeSpec(skill="formatter", inputs=["n:crit"], metadata={"label": "out"}))
        g.dg.add_edge(c, f)
        ex.states = {
            c: NodeState(node_id=c, skill="critic", status=NodeStatus.complete),
            f: NodeState(node_id=f, skill="formatter", status=NodeStatus.pending),
        }
        before = g.dg.number_of_nodes()
        await ex._handle_critic(c, '{"verdict": "pass", "rationale": "ok"}')
        assert g.dg.number_of_nodes() == before
        assert ex.states[f].status == NodeStatus.pending

    asyncio.run(_run())


# --- recovery classifier integration -------------------------------------------------

def _executor_with_store(tmp_path, monkeypatch, *, sid: str = "recovery_test"):
    from super_browser import persistence as pers_mod
    from super_browser.persistence import SessionStore

    monkeypatch.setattr(pers_mod, "SESSIONS_DIR", tmp_path / "sessions")
    store = SessionStore(sid)
    store.save_query("test")
    ex = Executor(registry=SkillRegistry())
    ex.store = store
    return ex, store


def test_transient_failure_does_not_queue_recovery_planner(tmp_path, monkeypatch):
    async def _run() -> None:
        ex, store = _executor_with_store(tmp_path, monkeypatch)
        nid = ex.graph.add_node_from_spec(NodeSpec(skill="researcher", metadata={"label": "r1"}))
        store.save_graph(ex.graph.dg)
        st = NodeState(node_id=nid, skill="researcher", status=NodeStatus.running)
        ex.states[nid] = st
        before = ex.graph.dg.number_of_nodes()
        await ex._handle_node_failure(nid, "researcher", "HTTP 503 Service Unavailable", st)
        assert ex.graph.dg.number_of_nodes() == before
        assert ex._fatal_error is not None

    asyncio.run(_run())


def test_upstream_failure_queues_one_recovery_planner(tmp_path, monkeypatch):
    async def _run() -> None:
        ex, store = _executor_with_store(tmp_path, monkeypatch)
        nid = ex.graph.add_node_from_spec(NodeSpec(skill="distiller", metadata={"label": "d"}))
        store.save_graph(ex.graph.dg)
        st = NodeState(node_id=nid, skill="distiller", status=NodeStatus.running)
        ex.states[nid] = st
        await ex._handle_node_failure(nid, "distiller", "wrong schema in output", st)
        assert ex._fatal_error is None
        planners = [n for n, d in ex.graph.dg.nodes(data=True) if d.get("skill") == "planner"]
        assert len(planners) == 1
        meta = ex.graph.dg.nodes[planners[0]]["metadata"]
        assert meta.get("failure_report", {}).get("skill") == "distiller"

    asyncio.run(_run())


def test_planner_failure_surfaces_without_recovery(tmp_path, monkeypatch):
    async def _run() -> None:
        ex, store = _executor_with_store(tmp_path, monkeypatch)
        nid = ex.graph.add_node_from_spec(NodeSpec(skill="planner", metadata={"label": "p"}))
        store.save_graph(ex.graph.dg)
        st = NodeState(node_id=nid, skill="planner", status=NodeStatus.running)
        ex.states[nid] = st
        before = ex.graph.dg.number_of_nodes()
        await ex._handle_node_failure(nid, "planner", "distiller produced wrong fields", st)
        assert ex.graph.dg.number_of_nodes() == before
        assert ex._fatal_error is not None

    asyncio.run(_run())


# --- persistence layout --------------------------------------------------------------

def test_node_filename_mapping():
    assert node_filename("n:1") == "n_001.json"
    assert node_filename("n:7") == "n_007.json"
    assert node_id_from_filename("n_003.json") == "n:3"


def test_session_query_txt_and_agent_result_revival(tmp_path, monkeypatch):
    from super_browser import persistence as pers_mod

    monkeypatch.setattr(pers_mod, "SESSIONS_DIR", tmp_path / "sessions")
    store = SessionStore("populations_dag")
    store.save_query("Find populations of London, Paris, Berlin")
    g = nx.DiGraph()
    g.add_node("n:1", skill="planner", label="planner", result=AgentResult(status="complete", output='{"nodes":[]}'))
    store.save_graph(g)
    loaded = store.load_graph()
    result = loaded.nodes["n:1"]["result"]
    assert isinstance(result, AgentResult)
    assert result.status == "complete"
    assert store.load_query().startswith("Find populations")


def test_legacy_pickle_fallback(tmp_path, monkeypatch, capsys):
    from super_browser import persistence as pers_mod

    monkeypatch.setattr(pers_mod, "SESSIONS_DIR", tmp_path / "sessions")
    store = SessionStore("legacy")
    store.ensure_dirs()
    g = nx.DiGraph()
    g.add_node("n:1", skill="planner")
    with (store.root / "graph.pkl").open("wb") as fh:
        pickle.dump(g, fh)
    loaded = store.load_graph()
    assert "n:1" in loaded
    assert "legacy pickle" in capsys.readouterr().err.lower()


def test_resume_resets_running_to_pending(tmp_path, monkeypatch):
    from super_browser import persistence as pers_mod

    monkeypatch.setattr(pers_mod, "SESSIONS_DIR", tmp_path / "sessions")
    store = SessionStore("dag-test")
    store.save_query("q")
    g = nx.DiGraph()
    g.add_node("n:2", skill="researcher", label="london")
    store.save_graph(g)
    running = NodeState(node_id="n:2", skill="researcher", status=NodeStatus.running)
    store.save_node_state(running)
    store.reset_running_to_pending()
    st = store.load_node_state("n:2")
    assert st.status == NodeStatus.pending


def test_agent_result_revival_failure_raises(tmp_path, monkeypatch):
    from super_browser import persistence as pers_mod

    monkeypatch.setattr(pers_mod, "SESSIONS_DIR", tmp_path / "sessions")
    store = SessionStore("bad")
    store.ensure_dirs()
    bad = {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": [{"id": "n:1", "result": {"_result_typed": True, "status": 123}}],
        "links": [],
    }
    store.graph_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(SessionLoadError):
        store.load_graph()
