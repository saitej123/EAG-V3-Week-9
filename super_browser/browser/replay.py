"""Browser replay viewer — evidence report from persisted DAG sessions."""

from __future__ import annotations

import json
from typing import Any

from ..persistence import SessionStore


def _parse_json(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def browser_replay_payload(output: dict[str, Any] | None) -> dict[str, Any]:
    """Compact evidence block for UI / formatter (path, turns, actions, final URL)."""
    if not isinstance(output, dict):
        return {"available": False}
    actions = output.get("actions") or []
    page_states = output.get("page_state_logs") or actions
    return {
        "available": True,
        "path": output.get("path"),
        "turns": output.get("turns", 0),
        "final_url": output.get("final_url") or output.get("url"),
        "elapsed_s": output.get("elapsed_s"),
        "llm_calls": output.get("llm_calls", 0),
        "input_tokens": output.get("input_tokens", 0),
        "output_tokens": output.get("output_tokens", 0),
        "cost_usd": output.get("cost_usd", 0.0),
        "actions": actions,
        "page_state_logs": page_states,
        "content_preview": (output.get("content") or "")[:500],
    }


def _planner_dag_summary(store: SessionStore) -> dict[str, Any]:
    graph = store.load_graph()
    nodes: list[dict[str, Any]] = []
    for nid, data in graph.nodes(data=True):
        nodes.append(
            {
                "id": nid,
                "skill": data.get("skill"),
                "label": (data.get("metadata") or {}).get("label") or data.get("label") or nid,
                "inputs": list(data.get("inputs") or []),
            }
        )
    edges = [{"source": u, "target": v} for u, v in graph.edges()]
    ordered_skills = [n["skill"] for n in sorted(nodes, key=lambda x: str(x["id"])) if n.get("skill")]
    flow = " → ".join(ordered_skills)
    return {"nodes": nodes, "edges": edges, "flow": flow}


def _comparison_table_from_session(states: dict[str, Any]) -> str | None:
    for st in states.values():
        if getattr(st, "skill", None) != "formatter" or not st.output:
            continue
        parsed = _parse_json(st.output)
        text = str(parsed.get("text") or parsed.get("answer") or st.output).strip()
        if text:
            return text
    return None


def _extracted_data_from_session(states: dict[str, Any], browser_node_id: str | None) -> str | None:
    if browser_node_id:
        st = states.get(browser_node_id)
        if st and st.output:
            content = _parse_json(st.output).get("content")
            if content:
                return str(content)
    for st in states.values():
        if getattr(st, "skill", None) == "distiller" and st.output:
            parsed = _parse_json(st.output)
            text = parsed.get("text") or parsed.get("summary") or st.output
            if text:
                return str(text)
    return None


def _format_action_line(action: Any) -> str:
    if isinstance(action, str):
        return action
    if isinstance(action, dict):
        turn = action.get("turn")
        note = action.get("note") or action.get("action") or json.dumps(action, ensure_ascii=False)
        return f"turn {turn}: {note}" if turn is not None else str(note)
    return str(action)


def format_replay_sections(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Eight numbered replay sections for the browser replay viewer."""
    run = (report.get("browser_runs") or [{}])[0] if report.get("browser_runs") else {}
    dag = report.get("planner_dag") or {}
    cost = report.get("cost_summary") or {}
    actions = run.get("page_state_logs") or run.get("actions") or []
    action_lines = [_format_action_line(a) for a in actions]
    extracted = run.get("extracted_data") or report.get("extracted_data")
    comparison = report.get("comparison_table") or ""

    cost_line = (
        f"turns={cost.get('total_turns', run.get('turns', 0))} · "
        f"llm_calls={run.get('llm_calls', 0)} · "
        f"tokens={cost.get('total_input_tokens', run.get('input_tokens', 0))}/"
        f"{cost.get('total_output_tokens', run.get('output_tokens', 0))} · "
        f"cost=${cost.get('total_cost_usd', run.get('cost_usd', 0)):.4f}"
    )

    dag_lines = [f"{n.get('id')} ({n.get('skill')})" for n in dag.get("nodes") or []]
    dag_body = (dag.get("flow") or " → ".join(dag_lines)) + (
        "\n" + "\n".join(f"  {e.get('source')} → {e.get('target')}" for e in dag.get("edges") or [])
        if dag.get("edges")
        else ""
    )

    return [
        {"n": 1, "title": "Original user goal", "body": report.get("user_goal") or "(none)"},
        {"n": 2, "title": "Planner DAG", "body": dag_body.strip() or "(none)"},
        {
            "n": 3,
            "title": "Browser path chosen",
            "body": str(run.get("path") or "unknown"),
            "badge": run.get("path"),
        },
        {
            "n": 4,
            "title": "Browser actions taken",
            "body": "\n".join(action_lines) if action_lines else "(no actions logged)",
            "count": len(action_lines),
        },
        {
            "n": 5,
            "title": "Page-state logs",
            "body": "\n".join(action_lines) if action_lines else "(screenshots not persisted — see action log)",
        },
        {"n": 6, "title": "Extracted data", "body": str(extracted or run.get("content_preview") or "(none)")},
        {"n": 7, "title": "Final comparison table", "body": comparison or "(formatter output pending)"},
        {"n": 8, "title": "Turn count and cost summary", "body": cost_line},
    ]


def replay_report_markdown(report: dict[str, Any]) -> str:
    """Markdown export for GitHub submission / replay trace."""
    lines = [
        "# Browser comparison replay report",
        "",
        f"**Session:** `{report.get('session_id', '')}`",
        "",
    ]
    for sec in format_replay_sections(report):
        lines.append(f"## {sec['n']}. {sec['title']}")
        lines.append("")
        lines.append("```text")
        lines.append(str(sec.get("body") or "").strip() or "(empty)")
        lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_browser_replay_report(session_id: str, *, node_id: str | None = None) -> dict[str, Any]:
    """Full browser replay report for a persisted session."""
    store = SessionStore(session_id)
    if not store.exists():
        raise FileNotFoundError(f"No session at {store.root}")

    query = store.load_query() or ""
    states = store.load_all_node_states()
    graph = store.load_graph()

    browser_runs: list[dict[str, Any]] = []
    total_cost = 0.0
    total_turns = 0
    total_input = 0
    total_output = 0
    primary_browser_id: str | None = None

    for nid, data in graph.nodes(data=True):
        if data.get("skill") != "browser":
            continue
        if node_id and nid != node_id:
            continue
        primary_browser_id = nid
        st = states.get(nid)
        out = _parse_json(st.output if st else None)
        payload = browser_replay_payload(out)
        if not payload.get("available"):
            continue
        browser_runs.append(
            {
                "node_id": nid,
                "label": (data.get("metadata") or {}).get("label") or nid,
                **payload,
                "extracted_data": out.get("content"),
                "status": st.status.value if st and hasattr(st.status, "value") else None,
            }
        )
        total_cost += float(payload.get("cost_usd") or 0)
        total_turns += int(payload.get("turns") or 0)
        total_input += int(payload.get("input_tokens") or 0)
        total_output += int(payload.get("output_tokens") or 0)

    comparison = _comparison_table_from_session(states)
    extracted = _extracted_data_from_session(states, primary_browser_id)

    report: dict[str, Any] = {
        "available": bool(browser_runs) or bool(query),
        "session_id": session_id,
        "user_goal": query,
        "planner_dag": _planner_dag_summary(store),
        "browser_runs": browser_runs,
        "extracted_data": extracted,
        "comparison_table": comparison,
        "cost_summary": {
            "total_cost_usd": round(total_cost, 4),
            "total_turns": total_turns,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
        },
    }
    report["sections"] = format_replay_sections(report)
    return report
