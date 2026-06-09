"""Unit tests for DAG orchestration."""

from __future__ import annotations

import json

import networkx as nx
import pytest

from super_browser.dag_schemas import NodeSpec, NodeStatus, PlannerOutput
from super_browser.flow import Graph
from super_browser.persistence import SessionStore, node_filename
from super_browser.skills import SkillRegistry


def test_node_spec_validation():
    spec = NodeSpec.model_validate(
        {"skill": "researcher", "inputs": ["USER_QUERY"], "metadata": {"label": "r1", "question": "London population"}}
    )
    assert spec.skill == "researcher"
    assert spec.metadata["label"] == "r1"


def test_planner_output_parses_populations_dag():
    raw = {
        "rationale": "Parallel lookups then compare.",
        "nodes": [
            {"skill": "researcher", "inputs": ["USER_QUERY"], "metadata": {"label": "london", "question": "London population"}},
            {"skill": "researcher", "inputs": ["USER_QUERY"], "metadata": {"label": "paris", "question": "Paris population"}},
            {"skill": "researcher", "inputs": ["USER_QUERY"], "metadata": {"label": "berlin", "question": "Berlin population"}},
            {"skill": "coder", "inputs": ["n:london", "n:paris", "n:berlin"], "metadata": {"label": "compare"}},
            {"skill": "formatter", "inputs": ["n:compare"], "metadata": {"label": "out"}},
        ],
    }
    out = PlannerOutput.model_validate(raw)
    assert len(out.nodes) == 5
    assert out.nodes[-1].skill == "formatter"


def test_coder_internal_successor_sandbox_auto_inserted():
    reg = SkillRegistry()
    g = Graph(reg)
    p = g.add_node_from_spec(NodeSpec(skill="planner", metadata={"label": "p"}))
    output = PlannerOutput(
        rationale="compare",
        nodes=[
            NodeSpec(skill="coder", inputs=["n:a"], metadata={"label": "compare"}),
            NodeSpec(skill="formatter", inputs=["n:compare"], metadata={"label": "out"}),
        ],
    )
    g.extend_from(p, output)
    sandboxes = [n for n, d in g.dg.nodes(data=True) if d.get("skill") == "sandbox_executor"]
    assert len(sandboxes) == 1
    coders = [n for n, d in g.dg.nodes(data=True) if d.get("skill") == "coder"]
    assert g.dg.has_edge(coders[0], sandboxes[0])


def test_critic_splice_yields_ready_critic_node():
    """Auto-spliced critic must appear in extend_from added list (Executor registers state)."""
    reg = SkillRegistry()
    g = Graph(reg)
    p = g.add_node_from_spec(NodeSpec(skill="planner", metadata={"label": "p"}))
    output = PlannerOutput(
        rationale="validate json",
        nodes=[
            NodeSpec(
                skill="distiller",
                inputs=["USER_QUERY"],
                metadata={"label": "d1", "required_keys": "author,title,year"},
            ),
            NodeSpec(skill="formatter", inputs=["n:d1"], metadata={"label": "out"}),
        ],
    )
    added = g.extend_from(p, output)
    critics = [n for n in added if g.dg.nodes[n].get("skill") == "critic"]
    assert len(critics) == 1
    from super_browser.dag_schemas import NodeState

    states = {p: NodeState(node_id=p, skill="planner", status=NodeStatus.complete)}
    for nid in added:
        states[nid] = NodeState(node_id=nid, skill=str(g.dg.nodes[nid]["skill"]), status=NodeStatus.pending)
    # distiller complete → critic ready (not formatter yet)
    dist = next(n for n in added if g.dg.nodes[n].get("skill") == "distiller")
    states[dist] = NodeState(node_id=dist, skill="distiller", status=NodeStatus.complete)
    ready = g.ready_nodes(states)
    assert critics[0] in ready
    assert not any(g.dg.nodes[n].get("skill") == "formatter" for n in ready)


def test_render_prompt_omits_empty_memory_hits():
    from super_browser.skills import SkillRegistry

    prompt = SkillRegistry().get("planner").render_prompt(
        user_query="hello",
        inputs_block="USER_QUERY",
        memory_hits="",
    )
    assert "\nMEMORY HITS:\n" not in prompt
    assert "INPUTS:" in prompt
    assert "USER QUERY:" in prompt


def test_format_memory_hits_preview():
    from super_browser.schemas import MemoryItem
    from super_browser.skills import format_memory_hits

    item = MemoryItem(
        id="m1",
        kind="fact",
        descriptor="chunk doc",
        source="sandbox/papers/x.md",
        value={"chunk": "x" * 500},
    )
    block = format_memory_hits([item])
    assert "source=sandbox/papers/x.md" in block
    assert "…" in block
    assert format_memory_hits([]) == ""


def test_critic_splice_on_distiller_outgoing_edge():
    reg = SkillRegistry()
    g = Graph(reg)
    planner_id = g.add_node_from_spec(NodeSpec(skill="planner", inputs=["USER_QUERY"], metadata={"label": "p"}))
    output = PlannerOutput(
        rationale="extract",
        nodes=[
            NodeSpec(skill="distiller", inputs=["USER_QUERY"], metadata={"label": "d"}),
            NodeSpec(skill="formatter", inputs=["n:d"], metadata={"label": "out"}),
        ],
    )
    g.extend_from(planner_id, output)
    distillers = [n for n, d in g.dg.nodes(data=True) if d.get("skill") == "distiller"]
    critics = [n for n, d in g.dg.nodes(data=True) if d.get("skill") == "critic"]
    formatters = [n for n, d in g.dg.nodes(data=True) if d.get("skill") == "formatter"]
    assert len(distillers) == 1
    assert len(critics) == 1
    assert len(formatters) == 1
    d, c, f = distillers[0], critics[0], formatters[0]
    assert g.dg.has_edge(d, c)
    assert g.dg.has_edge(c, f)
    assert not g.dg.has_edge(d, f)


def test_ready_nodes_parallel_researchers():
    reg = SkillRegistry()
    g = Graph(reg)
    p = g.add_node_from_spec(NodeSpec(skill="planner", metadata={"label": "p"}))
    output = PlannerOutput(
        rationale="fan out",
        nodes=[
            NodeSpec(skill="researcher", inputs=["USER_QUERY"], metadata={"label": "a"}),
            NodeSpec(skill="researcher", inputs=["USER_QUERY"], metadata={"label": "b"}),
        ],
    )
    g.extend_from(p, output)
    from super_browser.dag_schemas import NodeState

    states = {
        p: NodeState(node_id=p, skill="planner", status=NodeStatus.complete),
    }
    for nid in g.dg.nodes:
        if nid == p:
            continue
        states[nid] = NodeState(node_id=nid, skill=g.dg.nodes[nid]["skill"], status=NodeStatus.pending)
    ready = g.ready_nodes(states)
    assert len(ready) == 2


def test_session_store_atomic_roundtrip(tmp_path, monkeypatch):
    from super_browser import persistence as pers_mod
    from super_browser.dag_schemas import AgentResult

    monkeypatch.setattr(pers_mod, "SESSIONS_DIR", tmp_path / "sessions")
    store = SessionStore("test-sess")
    store.save_query("hello")
    g = nx.DiGraph()
    g.add_node("n:1", skill="planner", label="planner", result=AgentResult(status="pending"))
    store.save_graph(g)
    loaded = store.load_graph()
    assert "n:1" in loaded
    assert isinstance(loaded.nodes["n:1"]["result"], AgentResult)
    assert store.load_query() == "hello"
    assert (store.nodes_dir / "n_001.json").name == node_filename("n:1")


def test_prime_memory_read_uses_empty_history_and_top_k(monkeypatch):
    """Regression: memory.read(query, 12) passed int as history (Session8StartingCode-class bug)."""
    import asyncio
    from unittest.mock import MagicMock

    from super_browser.flow import Executor

    ex = Executor()
    ex.graph.user_query = "Say hello."
    mock_read = MagicMock(return_value=[])
    ex.memory.read = mock_read
    ex.memory.remember = MagicMock()
    monkeypatch.setattr("super_browser.flow.format_memory_hits", lambda hits: "")

    async def _sync_to_thread(fn, *a, **kw):
        return fn(*a, **kw)

    monkeypatch.setattr("super_browser.flow.asyncio.to_thread", _sync_to_thread)
    asyncio.run(ex._prime_memory())
    mock_read.assert_called_once_with("Say hello.", [], top_k=12)
