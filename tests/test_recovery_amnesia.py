"""Recovery amnesia — recovery Planner must receive completed sibling inputs."""

from __future__ import annotations

import asyncio

from super_browser.dag_schemas import NodeSpec, NodeState, NodeStatus
from super_browser.flow import Executor
from super_browser.recovery import recovery_planner_inputs
from super_browser.skills import SkillRegistry


def test_recovery_planner_carries_completed_siblings():
    ex = Executor(registry=SkillRegistry())
    g = ex.graph
    r1 = g.add_node_from_spec(NodeSpec(skill="researcher", metadata={"label": "2"}))
    r2 = g.add_node_from_spec(NodeSpec(skill="researcher", metadata={"label": "3"}))
    fail = g.add_node_from_spec(NodeSpec(skill="distiller", metadata={"label": "4"}))
    ex.states = {
        r1: NodeState(node_id=r1, skill="researcher", status=NodeStatus.complete, output="ok1"),
        r2: NodeState(node_id=r2, skill="researcher", status=NodeStatus.complete, output="ok2"),
        fail: NodeState(node_id=fail, skill="distiller", status=NodeStatus.failed, error="bad"),
    }
    inputs, reused = recovery_planner_inputs(g, ex.states, failed_node_id=fail)
    assert inputs[0] == "USER_QUERY"
    assert "n:2" in inputs
    assert "n:3" in inputs
    assert "n:4" in inputs
    assert set(reused) == {"n:2", "n:3"}


def test_recovery_planner_excludes_planner_and_critic():
    ex = Executor(registry=SkillRegistry())
    g = ex.graph
    p = g.add_node_from_spec(NodeSpec(skill="planner", metadata={"label": "p"}))
    c = g.add_node_from_spec(NodeSpec(skill="critic", metadata={"label": "c"}))
    d = g.add_node_from_spec(NodeSpec(skill="distiller", metadata={"label": "d"}))
    ex.states = {
        p: NodeState(node_id=p, skill="planner", status=NodeStatus.complete),
        c: NodeState(node_id=c, skill="critic", status=NodeStatus.complete),
        d: NodeState(node_id=d, skill="distiller", status=NodeStatus.failed),
    }
    inputs, reused = recovery_planner_inputs(g, ex.states, failed_node_id=d)
    assert "n:p" not in inputs
    assert "n:c" not in inputs
    assert reused == []


def test_recovery_legacy_fallback_when_no_priors(tmp_path, monkeypatch):
    from super_browser import persistence as pers_mod
    from super_browser.persistence import SessionStore

    monkeypatch.setattr(pers_mod, "SESSIONS_DIR", tmp_path / "sessions")

    async def _run() -> None:
        store = SessionStore("recovery_legacy")
        store.save_query("test")
        ex = Executor(registry=SkillRegistry())
        ex.store = store
        nid = ex.graph.add_node_from_spec(NodeSpec(skill="distiller", metadata={"label": "d"}))
        store.save_graph(ex.graph.dg)
        st = NodeState(node_id=nid, skill="distiller", status=NodeStatus.running)
        ex.states[nid] = st
        await ex._handle_node_failure(nid, "distiller", "wrong schema", st)
        planners = [n for n, d in ex.graph.dg.nodes(data=True) if d.get("skill") == "planner"]
        assert len(planners) == 1
        inputs = ex.graph.dg.nodes[planners[0]].get("inputs") or []
        assert inputs[0] == "USER_QUERY"
        assert "n:d" in inputs

    asyncio.run(_run())
