#!/usr/bin/env python3
"""Extract Browser skill metrics from a persisted DAG session."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from super_browser.persistence import SessionStore


def _parse_output(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def analyze(session_id: str) -> dict:
    store = SessionStore(session_id)
    if not store.exists():
        raise FileNotFoundError(f"No session at {store.root}")

    graph = store.load_graph()
    states = store.load_all_node_states()
    rows: list[dict] = []

    for nid, data in graph.nodes(data=True):
        if data.get("skill") != "browser":
            continue
        st = states.get(nid)
        out = _parse_output(st.output if st else None)
        rows.append(
            {
                "node_id": nid,
                "label": data.get("label", nid),
                "path": out.get("path"),
                "turns": out.get("turns", 0),
                "llm_calls": out.get("llm_calls", 0),
                "input_tokens": out.get("input_tokens", 0),
                "output_tokens": out.get("output_tokens", 0),
                "cost_usd": out.get("cost_usd", 0.0),
                "elapsed_s": out.get("elapsed_s") or (st.elapsed_s if st else None),
                "final_url": out.get("final_url"),
                "actions": len(out.get("actions") or []),
            }
        )

    total_cost = sum(float(r.get("cost_usd") or 0) for r in rows)
    return {
        "session_id": session_id,
        "browser_nodes": len(rows),
        "runs": rows,
        "total_cost_usd": round(total_cost, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Browser cascade output in a DAG session")
    parser.add_argument("session_id", help="Session id under state/sessions/")
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args()
    payload = analyze(args.session_id)
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    print(f"Session: {payload['session_id']}")
    print(f"Browser nodes: {payload['browser_nodes']}")
    print(f"Total cost: ${payload['total_cost_usd']:.4f}")
    for row in payload["runs"]:
        print(
            f"  {row['node_id']} path={row['path']} turns={row['turns']} "
            f"cost=${float(row.get('cost_usd') or 0):.4f} wall={row.get('elapsed_s')}s "
            f"tokens={row.get('input_tokens')}/{row.get('output_tokens')} actions={row.get('actions')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
