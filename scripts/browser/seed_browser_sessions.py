#!/usr/bin/env python3
"""Seed browser reference sessions from corpus/dag/ASSIGNMENT.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from super_browser.catalog import load_assignment_spec
from super_browser.dag_schemas import AgentResult, NodeSpec, NodeState, NodeStatus, PlannerOutput
from super_browser.flow import Graph
from super_browser.persistence import SessionStore
from super_browser.skills import SkillRegistry

FIXTURE_B4 = ROOT / "sandbox" / "browser" / "canvas-only.html"
HF_URL = "https://huggingface.co/models"

URLS = {
    "B1": "https://news.ycombinator.com",
    "B2": "https://www.amazon.com/s?k=laptop",
    "B3": HF_URL,
    "B4": "file://" + str(FIXTURE_B4.resolve()),
    "COMP": HF_URL,
}

SAMPLE_ACTIONS = {
    "B3": [
        {"turn": 1, "action": "click", "note": "clicked:Text Generation▾"},
        {"turn": 2, "action": "click", "note": "clicked:Transformers▾"},
        {"turn": 3, "action": "click", "note": "clicked:Sort: Most likes▾"},
        {"turn": 4, "action": "click", "note": "clicked:Most likes"},
        {"turn": 5, "action": "extract", "note": "read top 3 model card titles from filtered list"},
    ],
    "B4": [
        {"turn": 1, "action": "a11y", "note": "a11y:empty tree — done(success=false)"},
        {"turn": 7, "action": "click_coord", "note": "click_coord:320,240"},
    ],
    "COMP": [
        {"turn": 1, "action": "click", "note": "clicked:Text Generation▾"},
        {"turn": 2, "action": "click", "note": "clicked:Transformers▾"},
        {"turn": 3, "action": "click", "note": "clicked:Sort: Most likes▾"},
        {"turn": 4, "action": "click", "note": "clicked:Most likes"},
        {"turn": 5, "action": "click", "note": "opened: meta-llama/Llama-3.1-8B-Instruct card"},
        {"turn": 6, "action": "click", "note": "opened: Qwen/Qwen2.5-72B-Instruct card"},
    ],
}

COMP_EXTRACTED = """Top models (text-generation · transformers · sort=likes):
1. meta-llama/Llama-3.1-8B-Instruct — 12.4k likes — Meta 8B instruction-tuned LLM
2. Qwen/Qwen2.5-72B-Instruct — 9.8k likes — Alibaba 72B multilingual model
3. mistralai/Mistral-7B-Instruct-v0.3 — 8.1k likes — Mistral 7B instruction model"""

COMP_TABLE = """| Model | Likes | Description |
|-------|-------|-------------|
| meta-llama/Llama-3.1-8B-Instruct | 12.4k | Meta 8B instruction-tuned LLM |
| Qwen/Qwen2.5-72B-Instruct | 9.8k | Alibaba 72B multilingual model |
| mistralai/Mistral-7B-Instruct-v0.3 | 8.1k | Mistral 7B instruction model |"""


def _browser_output(*, qid: str, query: dict, run: dict | None = None) -> dict:
    run = run or {}
    actions = SAMPLE_ACTIONS.get(qid, [])
    path = run.get("path", "a11y" if qid in {"B3", "COMP"} else "extract")
    turns = run.get("turns", len(actions) if path in ("a11y", "vision") else 0)
    content = COMP_EXTRACTED if qid == "COMP" else f"[reference run {qid}]"
    return {
        "url": URLS[qid],
        "goal": query["query"],
        "path": path,
        "turns": turns,
        "content": content,
        "actions": actions,
        "page_state_logs": actions,
        "final_url": f"{HF_URL}?pipeline_tag=text-generation&library=transformers&sort=likes"
        if qid == "COMP"
        else URLS[qid],
        "elapsed_s": run.get("wall_clock_sec", 5.6 if qid == "COMP" else 2.0),
        "llm_calls": turns if path in ("a11y", "vision") else 0,
        "input_tokens": run.get("input_tokens", 9620 if qid in {"B3", "COMP"} else 0),
        "output_tokens": run.get("output_tokens", 408 if qid in {"B3", "COMP"} else 0),
        "cost_usd": run.get("cost_usd", 0.0),
    }


def seed_layer_session(*, qid: str, run: dict, query: dict) -> str:
    sid = f"dag_{qid}_ref"
    store = SessionStore(sid)
    store.save_query(query["query"])

    reg = SkillRegistry()
    g = Graph(reg)
    p = g.add_node_from_spec(NodeSpec(skill="planner", metadata={"label": "planner"}))
    plan = PlannerOutput(
        rationale=f"Reference browser run {qid}",
        nodes=[
            NodeSpec(
                skill="browser",
                inputs=["USER_QUERY"],
                metadata={"label": qid, "url": URLS[qid], "goal": query["query"]},
            )
        ],
    )
    g.extend_from(p, plan)
    browser_id = next(n for n, d in g.dg.nodes(data=True) if d.get("skill") == "browser")
    out = _browser_output(qid=qid, query=query, run=run)
    elapsed = float(out["elapsed_s"])

    store.save_graph(g.dg)
    store.save_node_state(
        NodeState(
            node_id="n:1",
            skill="planner",
            status=NodeStatus.complete,
            output=plan.model_dump_json(),
            elapsed_s=1.0,
        )
    )
    store.save_node_state(
        NodeState(
            node_id=browser_id,
            skill="browser",
            status=NodeStatus.complete,
            output=json.dumps(out),
            elapsed_s=elapsed,
        )
    )
    g.dg.nodes[browser_id]["result"] = AgentResult(
        status="complete",
        output=json.dumps(out),
        elapsed_s=elapsed,
    )
    store.save_graph(g.dg)
    store.root.joinpath("memory_hits.json").write_text("[]\n", encoding="utf-8")
    return sid


def seed_comp_session(query: dict) -> str:
    sid = "dag_COMP_ref"
    store = SessionStore(sid)
    store.save_query(query["query"])

    reg = SkillRegistry()
    g = Graph(reg)
    p = g.add_node_from_spec(NodeSpec(skill="planner", metadata={"label": "planner"}))
    plan = PlannerOutput(
        rationale="HF model comparison needs interactive filters — browser then distiller → formatter.",
        nodes=[
            NodeSpec(
                skill="browser",
                inputs=["USER_QUERY"],
                metadata={
                    "label": "browser",
                    "url": HF_URL,
                    "goal": query["query"],
                },
            ),
            NodeSpec(
                skill="distiller",
                inputs=["n:browser"],
                metadata={
                    "label": "extract",
                    "question": "Extract top 3 model names, like counts, and one-line descriptions.",
                },
            ),
            NodeSpec(
                skill="formatter",
                inputs=["n:extract"],
                metadata={"label": "out", "question": "Render a markdown comparison table."},
            ),
        ],
    )
    g.extend_from(p, plan)
    by_label = {
        (d.get("metadata") or {}).get("label"): nid
        for nid, d in g.dg.nodes(data=True)
        if d.get("skill") != "planner"
    }
    browser_id = by_label.get("browser") or next(
        n for n, d in g.dg.nodes(data=True) if d.get("skill") == "browser"
    )
    dist_id = by_label.get("extract")
    fmt_id = by_label.get("out")
    if dist_id:
        g.splice_critics_on_outgoing_edges([dist_id])

    browser_out = _browser_output(qid="COMP", query=query)
    dist_out = json.dumps({"text": COMP_EXTRACTED, "models": 3})
    fmt_out = json.dumps({"text": COMP_TABLE})

    store.save_graph(g.dg)
    store.save_node_state(
        NodeState(
            node_id="n:1",
            skill="planner",
            status=NodeStatus.complete,
            output=plan.model_dump_json(),
            elapsed_s=2.1,
        )
    )
    store.save_node_state(
        NodeState(
            node_id=browser_id,
            skill="browser",
            status=NodeStatus.complete,
            output=json.dumps(browser_out),
            elapsed_s=5.6,
        )
    )
    if dist_id:
        store.save_node_state(
            NodeState(
                node_id=dist_id,
                skill="distiller",
                status=NodeStatus.complete,
                output=dist_out,
                elapsed_s=1.8,
            )
        )
    if fmt_id:
        store.save_node_state(
            NodeState(
                node_id=fmt_id,
                skill="formatter",
                status=NodeStatus.complete,
                output=fmt_out,
                elapsed_s=1.2,
            )
        )
    for nid, out, elapsed in (
        (browser_id, json.dumps(browser_out), 5.6),
        (dist_id, dist_out, 1.8),
        (fmt_id, fmt_out, 1.2),
    ):
        if nid:
            g.dg.nodes[nid]["result"] = AgentResult(
                status="complete",
                output=out,
                elapsed_s=elapsed,
            )
    store.save_graph(g.dg)
    store.root.joinpath("memory_hits.json").write_text("[]\n", encoding="utf-8")
    replay_md = ROOT / "state" / "sessions" / sid / "browser_replay.md"
    replay_md.parent.mkdir(parents=True, exist_ok=True)
    from super_browser.browser.replay import build_browser_replay_report, replay_report_markdown

    replay_md.write_text(replay_report_markdown(build_browser_replay_report(sid)), encoding="utf-8")
    return sid


def seed_browser_reference_sessions() -> list[str]:
    """Create dag_COMP_ref and dag_B1_ref…B4_ref for UI replay demos."""
    spec = load_assignment_spec()
    queries = {q["id"]: q for q in spec.get("queries", [])}
    created: list[str] = []

    if "COMP" in queries:
        created.append(seed_comp_session(queries["COMP"]))

    layer_queries = {qid: q for qid, q in queries.items() if str(qid).startswith("B")}
    for run in spec.get("browser_reference_runs") or []:
        qid = run["query_id"]
        if qid not in layer_queries:
            continue
        created.append(seed_layer_session(qid=qid, run=run, query=layer_queries[qid]))

    return created


def main() -> int:
    created = seed_browser_reference_sessions()
    print("Seeded browser reference sessions:")
    for sid in created:
        print(f"  state/sessions/{sid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
