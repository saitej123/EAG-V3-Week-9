#!/usr/bin/env python3
"""Export browser replay report as Markdown."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from super_browser.browser.replay import build_browser_replay_report, replay_report_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description="Export browser replay report (Markdown)")
    parser.add_argument("session_id", help="Session id under state/sessions/")
    parser.add_argument(
        "-o",
        "--output",
        help="Write to file (default: stdout)",
    )
    args = parser.parse_args()
    report = build_browser_replay_report(args.session_id)
    md = replay_report_markdown(report)
    if args.output:
        Path(args.output).write_text(md, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(md, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
