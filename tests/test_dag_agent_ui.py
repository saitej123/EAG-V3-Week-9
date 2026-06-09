"""DagAgent UI integration — fresh executor per run."""

from __future__ import annotations

import asyncio

from super_browser import flow as flow_mod
from super_browser.flow import DagAgent


async def _run_fresh_executor_test(monkeypatch):
    created: list[object] = []
    closed: list[object] = []

    class FakeExecutor:
        async def run(self, user_query: str, session_id: str | None = None) -> str:
            return f"answer:{user_query[:8]}"

        async def aclose(self) -> None:
            closed.append(self)

    def factory() -> FakeExecutor:
        ex = FakeExecutor()
        created.append(ex)
        return ex

    monkeypatch.setattr(flow_mod, "Executor", factory)
    monkeypatch.setattr(flow_mod, "log_final_answer", lambda _a: None)

    agent = DagAgent()
    await agent.run("Fetch https://example.com")
    await agent.run("Say hello.")
    assert len(created) == 2
    assert len(closed) == 2


def test_dag_agent_creates_fresh_executor_each_run(monkeypatch):
    asyncio.run(_run_fresh_executor_test(monkeypatch))
