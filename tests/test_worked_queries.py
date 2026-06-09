"""Structural tests for DAG worked queries (no live LLM)."""

from __future__ import annotations

import pytest

from super_browser.catalog import get_dag_query, load_worked_queries, worked_queries_payload
from super_browser.dag_schemas import NodeSpec, NodeState, NodeStatus, PlannerOutput
from super_browser.flow import Graph
from super_browser.persistence import SessionStore
from super_browser.skills import SkillRegistry

# Canonical planner outputs matching worked-query shapes --------------------------------

HELLO_PLAN = PlannerOutput(
    rationale="Trivial greeting.",
    nodes=[NodeSpec(skill="formatter", inputs=["USER_QUERY"], metadata={"label": "out"})],
)

SHANNON_PLAN = PlannerOutput(
    rationale="Fetch and extract Shannon bio.",
    nodes=[
        NodeSpec(skill="researcher", inputs=["USER_QUERY"], metadata={"label": "fetch", "question": "Shannon Wikipedia page"}),
        NodeSpec(skill="distiller", inputs=["n:fetch"], metadata={"label": "fields", "question": "birth, death, three contributions"}),
        NodeSpec(skill="formatter", inputs=["n:fields"], metadata={"label": "out"}),
    ],
)

POPULATIONS_PLAN = PlannerOutput(
    rationale="Three populations, then pairwise comparison.",
    nodes=[
        NodeSpec(skill="researcher", inputs=["USER_QUERY"], metadata={"label": "london", "question": "London population"}),
        NodeSpec(skill="researcher", inputs=["USER_QUERY"], metadata={"label": "paris", "question": "Paris population"}),
        NodeSpec(skill="researcher", inputs=["USER_QUERY"], metadata={"label": "berlin", "question": "Berlin population"}),
        NodeSpec(skill="coder", inputs=["n:london", "n:paris", "n:berlin"], metadata={"label": "compare"}),
        NodeSpec(skill="formatter", inputs=["n:compare"], metadata={"label": "out"}),
    ],
)

FAILFAST_PLAN = PlannerOutput(
    rationale="Path does not exist.",
    nodes=[
        NodeSpec(
            skill="formatter",
            inputs=["USER_QUERY"],
            metadata={"label": "out", "note": "Cannot read /nonexistent/path.txt — file not accessible"},
        ),
    ],
)

K_PLAN = PlannerOutput(
    rationale="Three African cities in parallel.",
    nodes=[
        NodeSpec(skill="researcher", inputs=["USER_QUERY"], metadata={"label": "lagos", "question": "Lagos population and growth rate"}),
        NodeSpec(skill="researcher", inputs=["USER_QUERY"], metadata={"label": "cairo", "question": "Cairo population and growth rate"}),
        NodeSpec(skill="researcher", inputs=["USER_QUERY"], metadata={"label": "kinshasa", "question": "Kinshasa population and growth rate"}),
        NodeSpec(skill="coder", inputs=["n:lagos", "n:cairo", "n:kinshasa"], metadata={"label": "fastest"}),
        NodeSpec(skill="formatter", inputs=["n:fastest"], metadata={"label": "out"}),
    ],
)

PLANS = {
    "hello": HELLO_PLAN,
    "A": SHANNON_PLAN,
    "I": POPULATIONS_PLAN,
    "J": FAILFAST_PLAN,
    "K": K_PLAN,
}


def _skills_in_graph(g: Graph) -> set[str]:
    return {str(d.get("skill")) for _, d in g.dg.nodes(data=True)}


def _extend(g: Graph, plan: PlannerOutput) -> None:
    p = g.add_node_from_spec(NodeSpec(skill="planner", inputs=["USER_QUERY"], metadata={"label": "planner"}), node_id="n:1")
    g.extend_from(p, plan)


@pytest.mark.parametrize("query_id", ["hello", "A", "I", "J", "K"])
def test_worked_queries_spec_lists_five(query_id: str):
    payload = worked_queries_payload()
    assert payload["query_count"] == 5
    assert get_dag_query(query_id) is not None


@pytest.mark.parametrize("query_id", ["hello", "A", "I", "J", "K"])
def test_worked_query_dag_shape(query_id: str):
    row = get_dag_query(query_id)
    assert row is not None
    g = Graph(SkillRegistry())
    _extend(g, PLANS[query_id])

    skills = _skills_in_graph(g)
    for skill in row["expected_skills"]:
        assert skill in skills

    assert g.dg.number_of_nodes() >= int(row["expected_nodes_min"])

    if row.get("parallel_researchers"):
        researchers = [n for n, d in g.dg.nodes(data=True) if d.get("skill") == "researcher"]
        assert len(researchers) == row["parallel_researchers"]

    if "critic" in row.get("expected_auto", []):
        assert any(d.get("skill") == "critic" for _, d in g.dg.nodes(data=True))

    if "sandbox_executor" in row["expected_skills"]:
        assert any(d.get("skill") == "sandbox_executor" for _, d in g.dg.nodes(data=True))


def test_hello_minimal_two_nodes():
    g = Graph(SkillRegistry())
    _extend(g, HELLO_PLAN)
    assert g.dg.number_of_nodes() == 2
    assert _skills_in_graph(g) == {"planner", "formatter"}


def test_query_j_degenerate_planner_to_formatter():
    g = Graph(SkillRegistry())
    _extend(g, FAILFAST_PLAN)
    planner = "n:1"
    formatters = [n for n, d in g.dg.nodes(data=True) if d.get("skill") == "formatter"]
    assert len(formatters) == 1
    assert not any(d.get("skill") == "researcher" for _, d in g.dg.nodes(data=True))
    assert g.dg.has_edge(planner, formatters[0])


def test_query_k_resume_from_partial_researcher_layer(tmp_path, monkeypatch):
    """Simulate SIGKILL mid-gather: one researcher running → resume completes pending."""
    from super_browser import persistence as pers_mod

    monkeypatch.setattr(pers_mod, "SESSIONS_DIR", tmp_path / "sessions")
    sid = "dag_K_resumed_v2"
    row = get_dag_query("K")
    assert row is not None

    store = SessionStore(sid)
    store.save_query(str(row["query"]))
    g = Graph(SkillRegistry())
    _extend(g, K_PLAN)
    store.save_graph(g.dg)

    researchers = sorted(n for n, d in g.dg.nodes(data=True) if d.get("skill") == "researcher")
    assert len(researchers) == 3
    for i, rid in enumerate(researchers):
        status = NodeStatus.complete if i < 2 else NodeStatus.running
        store.save_node_state(
            NodeState(
                node_id=rid,
                skill="researcher",
                status=status,
                output="pop=1M" if status == NodeStatus.complete else None,
            )
        )
    store.save_node_state(NodeState(node_id="n:1", skill="planner", status=NodeStatus.complete, output=K_PLAN.model_dump_json()))

    store.reset_running_to_pending()
    states = store.load_all_node_states()
    pending_researchers = [n for n, s in states.items() if s.skill == "researcher" and s.status == NodeStatus.pending]
    assert len(pending_researchers) == 1
    complete_researchers = [n for n, s in states.items() if s.skill == "researcher" and s.status == NodeStatus.complete]
    assert len(complete_researchers) == 2

    from super_browser.graph_viz import session_resume_meta

    meta = session_resume_meta(sid)
    assert meta["resumable"] is True
    assert meta["resume_action"] == "continue"
    assert meta["incomplete_node_count"] >= 1


def test_session_resume_meta_counts_graph_nodes_without_state_files(tmp_path, monkeypatch):
    from super_browser import persistence as pers_mod
    from super_browser.graph_viz import session_resume_meta

    monkeypatch.setattr(pers_mod, "SESSIONS_DIR", tmp_path / "sessions")
    sid = "dag_partial_graph"
    store = SessionStore(sid)
    g = Graph(SkillRegistry())
    _extend(g, K_PLAN)
    store.save_graph(g.dg)
    store.save_node_state(NodeState(node_id="n:1", skill="planner", status=NodeStatus.complete, output="{}"))
    meta = session_resume_meta(sid)
    assert meta["graph_node_count"] >= 5
    assert meta["incomplete_node_count"] >= 4
    assert meta["resumable"] is True


def test_load_session_hydrates_downstream_nodes(tmp_path, monkeypatch):
    """Coder/formatter on graph without node files stay pending until researchers finish."""
    from super_browser import persistence as pers_mod
    from super_browser.flow import Executor

    monkeypatch.setattr(pers_mod, "SESSIONS_DIR", tmp_path / "sessions")
    sid = "dag_K_hydrate"
    store = SessionStore(sid)
    store.save_query("K hydrate test")
    g = Graph(SkillRegistry())
    _extend(g, K_PLAN)
    store.save_graph(g.dg)
    store.save_node_state(NodeState(node_id="n:1", skill="planner", status=NodeStatus.complete, output="{}"))
    researchers = sorted(n for n, d in g.dg.nodes(data=True) if d.get("skill") == "researcher")
    for i, rid in enumerate(researchers):
        st = NodeStatus.complete if i < 2 else NodeStatus.running
        store.save_node_state(NodeState(node_id=rid, skill="researcher", status=st, output="x" if st == NodeStatus.complete else None))

    ex = Executor()
    ex.store = store
    ex._load_session()
    coder = next(n for n, d in ex.graph.dg.nodes(data=True) if d.get("skill") == "coder")
    fmt = next(n for n, d in ex.graph.dg.nodes(data=True) if d.get("skill") == "formatter")
    assert ex.states[coder].status == NodeStatus.pending
    assert ex.states[fmt].status == NodeStatus.pending
    assert ex.states[researchers[2]].status == NodeStatus.pending
    ready = ex.graph.ready_nodes(ex.states)
    assert researchers[2] in ready
    assert coder not in ready
    assert fmt not in ready


def test_query_a_graph_shape_planner_researcher_distiller_critic_formatter():
    """Query A: planner → researcher → distiller → (auto critic) → formatter (5 nodes)."""
    g = Graph(SkillRegistry())
    _extend(g, SHANNON_PLAN)
    skills = _skills_in_graph(g)
    assert skills == {"planner", "researcher", "distiller", "critic", "formatter"}
    assert g.dg.number_of_nodes() == 5
    distiller = next(n for n, d in g.dg.nodes(data=True) if d.get("skill") == "distiller")
    formatter = next(n for n, d in g.dg.nodes(data=True) if d.get("skill") == "formatter")
    critic = next(n for n, d in g.dg.nodes(data=True) if d.get("skill") == "critic")
    assert g.dg.has_edge(distiller, critic)
    assert g.dg.has_edge(critic, formatter)


def test_query_a_has_wikipedia_target():
    dag_a = get_dag_query("A")
    assert dag_a is not None
    assert "wikipedia.org/wiki/Claude_Shannon" in dag_a["query"]
