"""Critic auto-insertion on pre-planned graphs + USER_QUERY in critic inputs."""

from __future__ import annotations

from super_browser.dag_schemas import NodeSpec, PlannerOutput
from super_browser.flow import Graph
from super_browser.skills import SkillRegistry


def _distiller_formatter_plan() -> PlannerOutput:
    return PlannerOutput(
        rationale="extract",
        nodes=[
            NodeSpec(skill="distiller", inputs=["USER_QUERY"], metadata={"label": "d"}),
            NodeSpec(skill="formatter", inputs=["n:d"], metadata={"label": "out"}),
        ],
    )


def test_preplanned_distiller_formatter_gets_critic():
    g = Graph(SkillRegistry())
    p = g.add_node_from_spec(NodeSpec(skill="planner", metadata={"label": "p"}))
    g.extend_from(p, _distiller_formatter_plan())
    critics = [n for n, d in g.dg.nodes(data=True) if d.get("skill") == "critic"]
    assert len(critics) == 1
    d = next(n for n, nd in g.dg.nodes(data=True) if nd.get("skill") == "distiller")
    f = next(n for n, nd in g.dg.nodes(data=True) if nd.get("skill") == "formatter")
    c = critics[0]
    assert g.dg.has_edge(d, c)
    assert g.dg.has_edge(c, f)


def test_explicit_planner_critic_not_duplicated():
    g = Graph(SkillRegistry())
    p = g.add_node_from_spec(NodeSpec(skill="planner", metadata={"label": "p"}))
    output = PlannerOutput(
        rationale="constrained",
        nodes=[
            NodeSpec(skill="distiller", inputs=["USER_QUERY"], metadata={"label": "d"}),
            NodeSpec(
                skill="critic",
                inputs=["n:d"],
                metadata={"label": "crit", "target": "out", "child": "out"},
            ),
            NodeSpec(skill="formatter", inputs=["n:crit"], metadata={"label": "out"}),
        ],
    )
    g.extend_from(p, output)
    assert sum(1 for _, d in g.dg.nodes(data=True) if d.get("skill") == "critic") == 1


def test_multiple_outgoing_edges_each_get_critic():
    g = Graph(SkillRegistry())
    p = g.add_node_from_spec(NodeSpec(skill="planner", metadata={"label": "p"}))
    output = PlannerOutput(
        rationale="fan out distill",
        nodes=[
            NodeSpec(skill="distiller", inputs=["USER_QUERY"], metadata={"label": "d"}),
            NodeSpec(skill="formatter", inputs=["n:d"], metadata={"label": "out_a"}),
            NodeSpec(skill="summariser", inputs=["n:d"], metadata={"label": "out_b"}),
        ],
    )
    g.extend_from(p, output)
    critics = [n for n, d in g.dg.nodes(data=True) if d.get("skill") == "critic"]
    assert len(critics) == 2


def test_non_critic_skill_does_not_trigger_insertion():
    g = Graph(SkillRegistry())
    p = g.add_node_from_spec(NodeSpec(skill="planner", metadata={"label": "p"}))
    output = PlannerOutput(
        rationale="research",
        nodes=[
            NodeSpec(skill="researcher", inputs=["USER_QUERY"], metadata={"label": "r"}),
            NodeSpec(skill="formatter", inputs=["n:r"], metadata={"label": "out"}),
        ],
    )
    g.extend_from(p, output)
    assert not any(d.get("skill") == "critic" for _, d in g.dg.nodes(data=True))


def test_auto_inserted_critic_receives_user_query_input():
    g = Graph(SkillRegistry())
    p = g.add_node_from_spec(NodeSpec(skill="planner", metadata={"label": "p"}))
    g.extend_from(p, _distiller_formatter_plan())
    critic = next(n for n, d in g.dg.nodes(data=True) if d.get("skill") == "critic")
    distiller = next(n for n, d in g.dg.nodes(data=True) if d.get("skill") == "distiller")
    inputs = g.dg.nodes[critic].get("inputs") or []
    assert "USER_QUERY" in inputs
    assert distiller in inputs
