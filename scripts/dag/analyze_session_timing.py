#!/usr/bin/env python3
"""Analyze parallel fan-out timing from a persisted DAG session."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from super_browser.persistence import SessionStore


def analyze(session_id: str) -> dict:
    store = SessionStore(session_id)
    if not store.exists():
        raise FileNotFoundError(f"No session at {store.root}")

    states = store.load_all_node_states()
    graph = store.load_graph()

    researchers: list[dict] = []
    for nid, data in graph.nodes(data=True):
        if data.get("skill") != "researcher":
            continue
        st = states.get(nid)
        if not st or st.elapsed_s is None:
            continue
        researchers.append(
            {
                "node_id": nid,
                "label": data.get("label", nid),
                "elapsed_s": round(st.elapsed_s, 3),
                "started_at": st.started_at,
                "finished_at": st.finished_at,
            }
        )

    # Group researchers that started in the same executor wave (within 0.05s)
    waves: list[list[dict]] = []
    for r in sorted(researchers, key=lambda x: x["started_at"] or 0):
        placed = False
        for wave in waves:
            if wave and abs((r["started_at"] or 0) - (wave[0]["started_at"] or 0)) < 0.05:
                wave.append(r)
                placed = True
                break
        if not placed:
            waves.append([r])

    parallel_waves: list[dict] = []
    for wave in waves:
        if len(wave) < 2:
            continue
        elapsed = [w["elapsed_s"] for w in wave]
        wall = max(elapsed)
        summed = sum(elapsed)
        parallel_waves.append(
            {
                "count": len(wave),
                "labels": [w["label"] for w in wave],
                "branch_elapsed_s": elapsed,
                "wall_clock_max_s": wall,
                "sum_if_serial_s": round(summed, 3),
                "speedup_vs_serial": round(summed / wall, 2) if wall > 0 else 0,
                "parallel_confirmed": wall < summed * 0.85,
            }
        )

    total_wall = 0.0
    for st in states.values():
        if st.elapsed_s:
            total_wall += st.elapsed_s

    return {
        "session_id": session_id,
        "researcher_count": len(researchers),
        "parallel_waves": parallel_waves,
        "researchers": researchers,
        "note": (
            "Parallel layer wall-clock ≈ max(branch elapsed), not sum. "
            "parallel_confirmed=true when max < 85% of sum."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze DAG session parallel timing")
    parser.add_argument("session_id", help="Session id under state/sessions/")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        report = analyze(args.session_id)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Session: {report['session_id']}")
        print(f"Researchers: {report['researcher_count']}")
        for i, wave in enumerate(report["parallel_waves"], 1):
            print(f"\nParallel wave {i} ({wave['count']} branches):")
            print(f"  labels: {wave['labels']}")
            print(f"  branch elapsed (s): {wave['branch_elapsed_s']}")
            print(f"  wall ≈ max = {wave['wall_clock_max_s']}s  (serial sum = {wave['sum_if_serial_s']}s)")
            print(f"  speedup: {wave['speedup_vs_serial']}x  parallel_confirmed={wave['parallel_confirmed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
