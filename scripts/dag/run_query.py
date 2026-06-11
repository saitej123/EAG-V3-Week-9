#!/usr/bin/env python3
"""DAG CLI — run demo query by id, free-text query, or resume a session."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from super_browser.catalog import get_dag_query, load_assignment_queries
from super_browser.flow import Executor
from super_browser.persistence import SessionLoadError


def _query_ids() -> list[str]:
    return [str(q["id"]) for q in load_assignment_queries()]


async def _run(args: argparse.Namespace) -> str:
    executor = Executor()
    try:
        if args.resume:
            return await executor.resume(args.resume)
        query = args.query
        if args.query_id:
            row = get_dag_query(args.query_id)
            if row is None:
                raise SystemExit(f"Unknown query id: {args.query_id}")
            query = str(row["query"])
            if args.session is None and not args.resume:
                args.session = f"dag_{args.query_id}"
        if not query or not query.strip():
            raise SystemExit("query required unless --resume is set")
        return await executor.run(query.strip(), session_id=args.session)
    finally:
        await executor.aclose()


def main() -> int:
    ids = _query_ids()
    parser = argparse.ArgumentParser(description="DAG orchestrator CLI")
    parser.add_argument("query_id", nargs="?", choices=ids, help="Browser query id (COMP, DEAL, TICKET, STACK, FORGE, B1–B4)")
    parser.add_argument("--query", "-q", default="", help="Free-text user query")
    parser.add_argument("--session", default=None, help="Session id for a new run")
    parser.add_argument("--resume", metavar="SID", default=None, help="Resume session id (reads query.txt)")
    args = parser.parse_args()

    if args.resume and args.query_id:
        parser.error("use either query_id or --resume, not both")
    if not args.resume and not args.query_id and not args.query.strip():
        parser.error("query_id, --query, or --resume required")

    try:
        answer = asyncio.run(_run(args))
    except SessionLoadError as e:
        print(f"SessionLoadError: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"Run stopped: {e}", file=sys.stderr)
        return 1

    print(answer)
    if args.session or args.query_id:
        sid = args.resume or args.session or (f"dag_{args.query_id}" if args.query_id else "")
        if sid:
            print(f"\n[session] state/sessions/{sid}/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
