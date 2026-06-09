"""DAG assignment corpus and critic metadata propagation."""

from __future__ import annotations

from super_browser.catalog import (
    assignment_payload,
    load_assignment_queries,
    validate_assignment_corpus,
)
from super_browser.dag_schemas import NodeSpec, PlannerOutput
from super_browser.flow import Graph
from super_browser.skills import SkillRegistry


def test_assignment_corpus_has_all_parts():
    rows = load_assignment_queries()
    ids = {str(r["id"]) for r in rows}
    assert {"hello", "A", "I", "J", "K", "P", "C_pass", "C_fail", "M", "PROS", "COMP", "B1", "B2", "B3", "B4"}.issubset(ids)
    parts = {int(r["part"]) for r in rows if r.get("part")}
    assert parts == {1, 2, 3, 4, 5, 6}


def test_critic_splice_propagates_required_keys():
    reg = SkillRegistry()
    g = Graph(reg)
    p = g.add_node_from_spec(NodeSpec(skill="planner", metadata={"label": "p"}))
    output = PlannerOutput(
        rationale="validate json",
        nodes=[
            NodeSpec(
                skill="distiller",
                inputs=["USER_QUERY"],
                metadata={
                    "label": "d1",
                    "required_keys": "author,title,year",
                    "verbatim_json": True,
                },
            ),
            NodeSpec(skill="formatter", inputs=["n:d1"], metadata={"label": "out"}),
        ],
    )
    g.extend_from(p, output)
    critics = [d for _, d in g.dg.nodes(data=True) if d.get("skill") == "critic"]
    assert len(critics) == 1
    assert critics[0]["metadata"].get("required_keys") == "author,title,year"


def test_calculator_skill_registered():
    reg = SkillRegistry()
    assert reg.get("calculator").tools_allowed == ["safe_calculate"]


def test_prosody_analyst_skill_registered():
    reg = SkillRegistry()
    assert reg.get("prosody_analyst").tools_allowed == ["count_syllables"]


def test_assignment_payload():
    payload = assignment_payload()
    assert payload["query_count"] >= 15
    assert payload["log_dir"] == "logs/dag"
    design = payload.get("design_queries") or []
    assert len(design) >= 4
    kinds = {d["kind"] for d in design}
    assert "parallel" in kinds and "critic" in kinds and "new_skill" in kinds and "browser" in kinds
    assert len(payload.get("groups") or []) == 6
    assert payload.get("browser_reference_runs")


def test_assignment_corpus_validates():
    assert validate_assignment_corpus() == []
