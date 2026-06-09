#!/usr/bin/env python3
"""Remove runtime artifacts for a fresh agent run.

Preserves:
  - sandbox/papers/, sandbox/research_papers/, sandbox/uploads/, sandbox/browser/
  - source code, templates, corpus/

Removes:
  - state/ (memory.json, FAISS index, artifacts, SQLite)
  - logs/, sandbox_home/, .crawl4ai/, usage.json
  - generated files under sandbox/ (reminders, agent-created files)
  - __pycache__, .pytest_cache, *.pyc
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SANDBOX = ROOT / "sandbox"
KEEP_SANDBOX_DIRS = {"papers", "research_papers", "uploads", "browser"}


def _rm_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.is_file():
        try:
            path.unlink()
        except OSError:
            pass


def clean_sandbox_generated() -> list[str]:
    """Delete sandbox files/dirs except static corpora."""
    removed: list[str] = []
    if not SANDBOX.is_dir():
        return removed
    for child in SANDBOX.iterdir():
        if child.name in KEEP_SANDBOX_DIRS:
            continue
        removed.append(str(child.relative_to(ROOT)))
        _rm_path(child)
    return removed


def clean_caches() -> None:
    """Clear Python/pytest caches without scanning .venv."""
    _rm_path(ROOT / "__pycache__")
    _rm_path(ROOT / ".pytest_cache")
    for sub in ("super_browser", "runs", "tests", "scripts", "tools"):
        base = ROOT / sub
        if not base.is_dir():
            continue
        _rm_path(base / "__pycache__")
        _rm_path(base / ".pytest_cache")
        for cache in base.rglob("__pycache__"):
            if ".venv" in cache.parts:
                continue
            shutil.rmtree(cache, ignore_errors=True)
    for pyc in ROOT.glob("*.pyc"):
        try:
            pyc.unlink()
        except OSError:
            pass


def clean_workspace(*, keep_logs: bool = False) -> dict[str, list[str]]:
    """Return summary of removed paths."""
    removed: dict[str, list[str]] = {"runtime": [], "sandbox": [], "cache": []}

    for name in ("state", "sandbox_home", ".crawl4ai", "usage.json"):
        path = ROOT / name
        if path.exists():
            removed["runtime"].append(name + ("/" if path.is_dir() else ""))
            _rm_path(path)

    if not keep_logs:
        logs = ROOT / "logs"
        if logs.exists():
            removed["runtime"].append("logs/")
            _rm_path(logs)

    removed["sandbox"] = clean_sandbox_generated()
    clean_caches()
    removed["cache"].append("__pycache__ / .pytest_cache / *.pyc (under repo, not .venv)")

    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean runtime artifacts for a fresh run")
    parser.add_argument("--keep-logs", action="store_true", help="Keep logs/ directory")
    args = parser.parse_args()

    summary = clean_workspace(keep_logs=args.keep_logs)
    print("Workspace cleaned for fresh run.\n")
    if summary["runtime"]:
        print("Removed runtime:")
        for item in summary["runtime"]:
            print(f"  - {item}")
    if summary["sandbox"]:
        print("Removed sandbox (generated):")
        for item in summary["sandbox"]:
            print(f"  - {item}")
    else:
        print("Sandbox: no extra generated files (papers/ + browser/ kept).")
    print("Cache: cleared Python/pytest caches outside .venv.")
    print("\nKept: sandbox/papers/, sandbox/research_papers/, sandbox/uploads/, sandbox/browser/, source, templates, corpus/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
