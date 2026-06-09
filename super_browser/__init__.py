"""Super Browser Agent — vector memory, MCP tools, iteration loop or DAG orchestrator."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["SuperBrowserAgent", "DagAgent"]

if TYPE_CHECKING:
    from super_browser.agent import SuperBrowserAgent
    from super_browser.flow import DagAgent


def __getattr__(name: str):
    if name == "SuperBrowserAgent":
        from super_browser.agent import SuperBrowserAgent

        return SuperBrowserAgent
    if name == "DagAgent":
        from super_browser.flow import DagAgent

        return DagAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
