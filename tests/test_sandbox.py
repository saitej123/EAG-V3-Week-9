"""Sandbox subprocess runner tests."""

from __future__ import annotations

from super_browser.sandbox import run_python


def test_run_python_stdout():
    result = run_python('print(1640000)')
    assert result["exit_code"] == 0
    assert "1640000" in result["stdout"]
