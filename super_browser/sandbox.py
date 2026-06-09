"""
Isolated subprocess runner for the sandbox_executor skill.

Usability boundary, not a security sandbox: scrubs env to a small allowlist,
runs in a fresh temp directory (deleted on exit), caps stdout/stderr at 1 MiB
each, and enforces a 30-second timeout. Does not isolate the network or restrict
filesystem access outside the temp dir.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

_SANDBOX_ENV_KEYS = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE")
_MAX_IO_BYTES = 1_048_576
_TIMEOUT_SEC = 30


def _scrubbed_env() -> dict[str, str]:
    out: dict[str, str] = {}
    for key in _SANDBOX_ENV_KEYS:
        val = os.environ.get(key)
        if val:
            out[key] = val
    if "PATH" not in out:
        out["PATH"] = os.defpath
    return out


def run_python(code: str) -> dict[str, Any]:
    """Execute *code* in a subprocess; return stdout, stderr, exit_code."""
    with tempfile.TemporaryDirectory(prefix="dag_sandbox_") as tmp:
        script = Path(tmp) / "main.py"
        script.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [os.environ.get("SANDBOX_PYTHON", "python3"), str(script)],
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_SEC,
                cwd=tmp,
                env=_scrubbed_env(),
            )
            stdout = (proc.stdout or "")[:_MAX_IO_BYTES]
            stderr = (proc.stderr or "")[:_MAX_IO_BYTES]
            return {"stdout": stdout, "stderr": stderr, "exit_code": proc.returncode}
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": f"timeout after {_TIMEOUT_SEC}s", "exit_code": -1}
        except OSError as e:
            return {"stdout": "", "stderr": str(e), "exit_code": -1}
