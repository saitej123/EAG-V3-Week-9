"""Tests for browser screenshot persistence and replay section 5."""

from __future__ import annotations

from pathlib import Path

from super_browser.browser.page_capture import PageCapture, resolve_screenshot_path
from super_browser.browser.replay import format_replay_sections


def test_resolve_screenshot_path_rejects_traversal(tmp_path, monkeypatch):
    from super_browser import persistence

    monkeypatch.setattr(persistence, "SESSIONS_DIR", tmp_path)
    sid = "dag_test_ref"
    rel = "browser_screenshots/n_002/001.png"
    shot_dir = tmp_path / sid / "browser_screenshots" / "n_002"
    shot_dir.mkdir(parents=True)
    (shot_dir / "001.png").write_bytes(b"\x89PNG\r\n")

    ok = resolve_screenshot_path(sid, rel)
    assert ok is not None
    assert ok.name == "001.png"
    assert resolve_screenshot_path(sid, "../secrets.png") is None
    assert resolve_screenshot_path(sid, "nodes/n_001.json") is None


def test_replay_section_five_includes_screenshot_urls():
    report = {
        "session_id": "dag_COMP_ref",
        "user_goal": "Compare models",
        "planner_dag": {"nodes": [], "edges": []},
        "browser_runs": [
            {
                "path": "a11y",
                "turns": 2,
                "page_state_logs": [
                    {
                        "turn": 1,
                        "note": "clicked:Sort",
                        "screenshot": "browser_screenshots/n_002/001.png",
                    }
                ],
            }
        ],
        "cost_summary": {"total_turns": 2, "total_cost_usd": 0.0},
    }
    sections = format_replay_sections(report)
    sec5 = next(s for s in sections if s["n"] == 5)
    assert sec5["screenshots"]
    assert "/api/dag/browser-screenshot" in sec5["screenshots"][0]["url"]
    assert "clicked:Sort" in sec5["body"]


def test_page_capture_writes_relative_path(tmp_path, monkeypatch):
    from super_browser import persistence

    monkeypatch.setattr(persistence, "SESSIONS_DIR", tmp_path)

    class FakePage:
        async def screenshot(self, **kwargs):
            Path(kwargs["path"]).write_bytes(b"\x89PNG\r\n")

    cap = PageCapture("dag_demo", "n:2")
    import asyncio

    entry = asyncio.run(cap.log_from_page(FakePage(), turn=1, note="clicked:Filter"))
    assert entry["screenshot"] == "browser_screenshots/n_2/001.png"
    assert (tmp_path / "dag_demo" / entry["screenshot"]).is_file()
