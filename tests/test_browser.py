"""Unit tests for Browser skill cascade helpers."""

from __future__ import annotations

import base64

import pytest

from super_browser.browser.driver import fence_actions, is_dropdown_trigger, normalize_actions
from super_browser.browser.extract import content_is_useful, goal_keywords
from super_browser.browser.highlight import dedupe_clickables, draw_marks
from super_browser.vision_api import decode_image_payload


def test_goal_keywords_strips_stopwords():
    keys = goal_keywords("Find the population of London and Paris")
    assert "london" in keys
    assert "paris" in keys
    assert "find" not in keys


def test_content_is_useful_requires_length_and_keyword():
    goal = "Hacker News top stories"
    short = "x" * 100
    assert content_is_useful(short, goal) is False
    long = ("story " * 50) + "hacker news listings"
    assert content_is_useful(long, goal) is True


def test_dedupe_prefers_outer_box():
    outer = {"tag": "button", "label": "Rectangle", "box": {"x": 10, "y": 10, "width": 40, "height": 40}}
    inner = {"tag": "rect", "label": "", "box": {"x": 15, "y": 15, "width": 8, "height": 8}}
    kept = dedupe_clickables([inner, outer])
    assert len(kept) == 1
    assert kept[0]["label"] == "Rectangle"
    assert kept[0]["mark"] == 1


def test_draw_marks_returns_png():
    from io import BytesIO

    from PIL import Image

    img = Image.new("RGB", (20, 20), color=(255, 255, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    png = buf.getvalue()
    items = [{"mark": 1, "box": {"x": 1, "y": 1, "width": 4, "height": 4}, "label": "A"}]
    out = draw_marks(png, items, device_pixel_ratio=1.0)
    assert out[:8] == b"\x89PNG\r\n\x1a\n"


def test_draw_marks_handles_inverted_boxes():
    from io import BytesIO

    from PIL import Image

    from super_browser.browser.highlight import normalize_box

    assert normalize_box({"x": 10, "y": 10, "width": -5, "height": -3}) == {
        "x": 5.0,
        "y": 7.0,
        "width": 5.0,
        "height": 3.0,
    }
    img = Image.new("RGB", (40, 40), color=(255, 255, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    png = buf.getvalue()
    items = [{"mark": 1, "box": {"x": 2, "y": 0, "width": 8, "height": -2}, "label": "B"}]
    out = draw_marks(png, items, device_pixel_ratio=1.0)
    assert out[:8] == b"\x89PNG\r\n\x1a\n"


def test_decode_image_payload_accepts_data_uri():
    raw = base64.b64encode(b"hello").decode()
    data_uri = f"data:image/png;base64,{raw}"
    assert decode_image_payload(image=data_uri) == b"hello"


def test_decode_image_payload_requires_payload():
    with pytest.raises(ValueError):
        decode_image_payload()


def test_is_dropdown_trigger():
    assert is_dropdown_trigger("Sort: Most likes▾") is True
    assert is_dropdown_trigger("Libraries:") is True
    assert is_dropdown_trigger("text-generation") is False


def test_fence_actions_dropdown_is_solo():
    actions = normalize_actions(
        {
            "actions": [
                {"action": "click", "target": "Sort: Most likes▾"},
                {"action": "click", "target": "Most likes"},
            ]
        }
    )
    fenced = fence_actions(actions)
    assert len(fenced) == 1
    assert fenced[0]["target"] == "Sort: Most likes▾"


def test_fence_actions_caps_at_two():
    actions = [
        {"action": "click", "target": "text-generation"},
        {"action": "click", "target": "transformers"},
        {"action": "click", "target": "extra"},
    ]
    assert len(fence_actions(actions)) == 2


def test_detect_gateway_block():
    from super_browser.browser.dom import detect_gateway_block

    assert detect_gateway_block("Let's confirm you are human before continuing") is True
    assert detect_gateway_block("Hacker News top stories") is False


def test_layer_extract_escalates_on_captcha_html(monkeypatch):
    import asyncio

    from super_browser.browser import extract as extract_mod

    async def fake_get(*args, **kwargs):
        class Resp:
            text = "<html><div class=\"g-recaptcha\">confirm you are human</div></html>"

            def raise_for_status(self):
                return None

        return Resp()

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        get = fake_get

    monkeypatch.setattr(extract_mod.httpx, "AsyncClient", lambda **kwargs: FakeClient())
    assert asyncio.run(extract_mod.layer_extract("https://huggingface.co/models", "models")) is None


def test_browser_output_shape():
    from super_browser.browser.skill import to_browser_output

    out = to_browser_output(
        url="https://huggingface.co/models",
        goal="filter and read top 3",
        raw={
            "path": "a11y",
            "url": "https://huggingface.co/models?sort=likes",
            "content": "1. model-a",
            "turns": 5,
            "transcript": ["clicked:Sort:"],
            "llm_calls": 5,
        },
    )
    assert out.path == "a11y"
    assert out.turns == 5
    assert out.final_url.startswith("https://")
    assert len(out.actions) == 1


def test_cost_fields_default_zero():
    from super_browser.browser.ledger import apply_cost_fields, estimate_cost_usd

    assert estimate_cost_usd(input_tokens=9620, output_tokens=408) == 0.0
    raw = apply_cost_fields({"path": "a11y", "input_tokens": 100, "output_tokens": 20})
    assert raw["cost_usd"] == 0.0


def test_gateway_blocked_path():
    from super_browser.browser.skill import to_browser_output

    out = to_browser_output(
        url="https://www.redfin.com/x",
        goal="extract beds",
        raw={"path": "gateway_blocked", "content": None, "llm_calls": 0},
    )
    assert out.path == "gateway_blocked"
    assert out.cost_usd == 0.0


def test_browser_replay_payload():
    from super_browser.browser.replay import browser_replay_payload

    payload = browser_replay_payload({"path": "a11y", "turns": 5, "content": "x" * 600})
    assert payload["available"] is True
    assert payload["path"] == "a11y"
    assert len(payload["content_preview"]) == 500


def test_force_path_runs_single_layer(monkeypatch):
    import asyncio

    calls: list[str] = []

    async def fake_extract(url: str, goal: str):
        calls.append("extract")
        return {"path": "extract", "url": url, "content": "ok" * 100, "llm_calls": 0}

    async def fake_deterministic(url: str, goal: str):
        calls.append("deterministic")
        return None

    monkeypatch.setattr("super_browser.browser.skill.layer_extract", fake_extract)
    monkeypatch.setattr("super_browser.browser.skill.layer_deterministic", fake_deterministic)

    async def _run() -> None:
        from super_browser.browser.skill import run_browser_cascade

        out, err = await run_browser_cascade(
            "https://example.com",
            "example content",
            llm=None,
            force_path="deterministic",
        )
        assert calls == ["deterministic"]
        assert out.path == "failed"
        assert err is not None

    asyncio.run(_run())


def test_reference_sessions_match_corpus():
    import json
    from pathlib import Path

    from super_browser.catalog import load_assignment_spec
    from scripts.browser.analyze_browser_session import analyze

    spec = load_assignment_spec()
    by_id = {r["query_id"]: r for r in spec.get("browser_reference_runs") or []}
    for qid in ("B1", "B2", "B3", "B4"):
        payload = analyze(f"dag_{qid}_ref")
        assert payload["browser_nodes"] == 1
        row = payload["runs"][0]
        ref = by_id[qid]
        assert row["path"] == ref["path"]
        assert row["turns"] == ref["turns"]
        assert row["cost_usd"] == ref["cost_usd"]
        assert abs(float(row["elapsed_s"]) - ref["wall_clock_sec"]) < 0.05


def test_browser_replay_report_on_reference_session():
    from super_browser.browser.replay import build_browser_replay_report

    report = build_browser_replay_report("dag_B3_ref")
    assert report["available"] is True
    assert report["browser_runs"][0]["path"] == "a11y"
    assert report["cost_summary"]["total_turns"] == 5
    assert len(report["sections"]) == 8
    assert report["sections"][0]["title"] == "Original user goal"


def test_comp_replay_has_comparison_table():
    from super_browser.browser.replay import build_browser_replay_report

    report = build_browser_replay_report("dag_COMP_ref")
    assert report["available"] is True
    assert report["browser_runs"][0]["path"] == "a11y"
    assert len(report["browser_runs"][0]["actions"]) >= 3
    assert report["comparison_table"]
    assert "| Model |" in report["comparison_table"]
    assert report["sections"][3]["count"] >= 3
    assert report["sections"][4]["title"] == "Screenshots or page-state logs"


def test_comparison_task_skips_static_extract_success(monkeypatch):
    import asyncio

    calls: list[str] = []

    async def fake_extract(url: str, goal: str):
        calls.append("extract")
        return {
            "path": "extract",
            "url": url,
            "content": "models " * 100,
            "llm_calls": 0,
            "transcript": [],
        }

    async def fake_render(url: str, goal: str, *, page=None):
        calls.append("render")
        return {
            "path": "extract",
            "url": url,
            "content": "models " * 100,
            "llm_calls": 0,
            "transcript": ["render:playwright"],
        }

    async def fake_a11y(url: str, goal: str, llm, *, page=None):
        calls.append("a11y")
        return {
            "path": "a11y",
            "url": url,
            "content": "top 3 models",
            "turns": 3,
            "transcript": ["clicked:a", "clicked:b", "clicked:c"],
            "llm_calls": 3,
        }

    async def fake_nav(page, url):
        return url

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def fake_session():
        class FakePage:
            url = "https://huggingface.co/models"

        yield FakePage()

    monkeypatch.setattr("super_browser.browser.skill.layer_extract", fake_extract)
    monkeypatch.setattr("super_browser.browser.skill.layer_render", fake_render)
    monkeypatch.setattr("super_browser.browser.skill.layer_a11y", fake_a11y)
    monkeypatch.setattr("super_browser.browser.skill.layer_deterministic", lambda *a, **k: None)
    monkeypatch.setattr("super_browser.browser.skill.layer_vision", lambda *a, **k: None)
    monkeypatch.setattr("super_browser.browser.skill.navigate_robust", fake_nav)
    monkeypatch.setattr("super_browser.browser.skill.live_page_blocked", lambda page: False)
    monkeypatch.setattr("super_browser.browser.skill.browser_session", fake_session)

    async def _run() -> None:
        from super_browser.browser.skill import run_browser_cascade

        out, err = await run_browser_cascade(
            "https://huggingface.co/models",
            "Compare top 3 Hugging Face text-generation models",
            llm=object(),
            min_browser_actions=3,
        )
        assert "extract" not in calls
        assert "a11y" in calls
        assert out.path == "a11y"
        assert len(out.actions) >= 3
        assert err is None

    asyncio.run(_run())


def test_action_count_includes_vision_turns():
    from super_browser.browser.skill import _action_count

    assert _action_count({"transcript": ["vision_turn:1", "vision_turn:2", "vision_turn:3"]}) == 3
    assert _action_count({"transcript": ["render:playwright", "vision_turn:1"]}) == 1


def test_format_browser_path_labels():
    from super_browser.browser.replay import format_browser_path

    assert "blocked" in format_browser_path("gateway_blocked")
    assert format_browser_path("a11y").startswith("a11y")
