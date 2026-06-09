"""Repository root paths — all durable data lives under ``ROOT``."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SANDBOX = ROOT / "sandbox"
STATE = ROOT / "state"


def resolve_sandbox_subdir(directory: str) -> Path:
    """Resolve a subdirectory strictly under ``SANDBOX`` (no path traversal)."""
    rel = (directory or "").strip().replace("\\", "/").strip("/")
    if not rel or rel.startswith("/") or ".." in rel.split("/"):
        raise ValueError("Invalid sandbox directory path")
    target = (SANDBOX / rel).resolve()
    sandbox = SANDBOX.resolve()
    if target != sandbox and sandbox not in target.parents:
        raise ValueError("Directory must stay under sandbox/")
    if not target.is_dir():
        raise ValueError(f"Directory '{rel}' does not exist in the sandbox")
    return target
