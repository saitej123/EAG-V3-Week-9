"""Unit tests for Browser skill cascade helpers."""

from __future__ import annotations

import base64

import pytest

from super_browser.browser.turn_rules import fence_actions, is_dropdown_trigger, normalize_actions
from super_browser.browser.extract import content_is_useful, goal_keywords
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


def test_annotate_marks_returns_png():
    from io import BytesIO

    from PIL import Image

    from super_browser.browser.drivers.elements import Element
    from super_browser.browser.drivers.marks import annotate

    img = Image.new("RGB", (80, 60), color=(255, 255, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    el = Element(id=1, tag="button", role="button", name="Go", x=10, y=10, w=20, h=15)
    out = annotate(buf.getvalue(), [el], dpr=1.0)
    assert out[:8] == b"\x89PNG\r\n\x1a\n"


def test_element_center_coords():
    from super_browser.browser.drivers.elements import Element

    el = Element(id=2, tag="a", role="link", name="Pricing", x=0, y=0, w=100, h=40)
    assert el.cx == 50.0
    assert el.cy == 20.0


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
    from super_browser.browser.gateway import detect_gateway_block

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

    seen: list[str] = []

    async def fake_run(self, url, goal, *, force_path=None, min_browser_actions=0, all_urls=None):
        seen.append(str(force_path))
        return {"path": "failed", "url": url, "content": None, "error": "deterministic miss"}, "interaction_failed"

    monkeypatch.setattr("super_browser.browser.drivers.cascade.BrowserSkill.run", fake_run)

    async def _run() -> None:
        from super_browser.browser.skill import run_browser_cascade

        out, err = await run_browser_cascade(
            "https://example.com",
            "example content",
            llm=None,
            force_path="deterministic",
        )
        assert seen == ["deterministic"]
        assert out.path == "failed"
        assert err is not None

    asyncio.run(_run())


def test_multi_page_partial_continues_to_vlm(monkeypatch):
    import asyncio

    async def fake_run(self, url, goal, *, force_path=None, min_browser_actions=0, all_urls=None):
        if all_urls and len(all_urls) > 1:
            return {
                "path": "vision",
                "url": url,
                "content": "| tool | price |\n| --- | --- |\n| X | $1 |",
                "transcript": ["vision_turn:1", "click_mark:1", "click_mark:2"],
                "turns": 1,
                "llm_calls": 1,
            }, None
        return {"path": "failed", "content": None, "error": "n/a"}, "interaction_failed"

    monkeypatch.setattr("super_browser.browser.drivers.cascade.BrowserSkill.run", fake_run)

    async def _run() -> None:
        from super_browser.browser.skill import run_browser_cascade

        out, err = await run_browser_cascade(
            "https://cursor.com/pricing",
            "Compare 5 AI coding tools pricing",
            llm=object(),
            min_browser_actions=3,
            all_urls=[
                "https://cursor.com/pricing",
                "https://github.com/features/copilot/plans",
            ],
        )
        assert out.path == "vision"
        assert err is None

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
    sec5 = report["sections"][4]
    assert sec5.get("screenshots") or "screenshot" in str(sec5.get("body") or "").lower()


def test_comparison_task_skips_static_extract_success(monkeypatch):
    import asyncio

    async def fake_run(self, url, goal, *, force_path=None, min_browser_actions=0, all_urls=None):
        return {
            "path": "a11y",
            "url": url,
            "content": "top 3 models table",
            "turns": 3,
            "transcript": ["a11y_turn:1", "click_mark:1", "click_mark:2", "click_mark:3"],
            "llm_calls": 3,
        }, None

    monkeypatch.setattr("super_browser.browser.drivers.cascade.BrowserSkill.run", fake_run)

    async def _run() -> None:
        from super_browser.browser.skill import run_browser_cascade

        out, err = await run_browser_cascade(
            "https://huggingface.co/models",
            "Compare top 3 Hugging Face text-generation models",
            llm=object(),
            min_browser_actions=3,
        )
        assert out.path == "a11y"
        assert len(out.actions) >= 3
        assert err is None

    asyncio.run(_run())


def test_action_count_includes_vision_turns():
    from super_browser.browser.validation import action_count

    assert action_count({"transcript": ["vision_turn:1", "vision_turn:2", "vision_turn:3"]}) == 3
    assert action_count({"transcript": ["render:playwright", "vision_turn:1"]}) == 1
    assert action_count({"transcript": ["vlm_live:extract", "vision_turn:1"]}) == 2


def test_action_count_ignores_multi_page_synthetic_scroll():
    from super_browser.browser.validation import action_count

    assert action_count({"transcript": ["opened:flipkart.com", "scroll:multi_page"]}) == 1


def test_action_count_includes_navigation_actions():
    from super_browser.browser.validation import action_count

    assert action_count({"transcript": ["go_to_url:https://github.com/trending", "opened:https://github.com/a/b"]}) == 2


def test_github_trending_content_formats_required_rows():
    from super_browser.browser.drivers.cascade import _github_trending_content
    from super_browser.comparison_format import format_comparison_answer

    content = _github_trending_content(
        [
            {
                "repository_name": "freeCodeCamp/freeCodeCamp",
                "star_count": "432k",
                "primary_language": "TypeScript",
            },
            {
                "repository_name": "swc-project/swc",
                "star_count": "33k",
                "primary_language": "Rust",
            },
            {
                "repository_name": "puppeteer/puppeteer",
                "star_count": "92k",
                "primary_language": "TypeScript",
            },
        ]
    )
    table = format_comparison_answer(
        "Compare 3 trending open-source repositories on GitHub with columns: repository name, star count, primary language",
        [{"kind": "upstream", "skill": "browser", "output": {"content": content}}],
    )

    assert table is not None
    assert "freeCodeCamp/freeCodeCamp" in table
    assert "432k" in table
    assert "TypeScript" in table


def test_hf_models_content_formats_required_rows():
    from super_browser.browser.drivers.cascade import _hf_models_content
    from super_browser.comparison_format import format_comparison_answer

    content = _hf_models_content(
        [
            {
                "model": "deepseek-ai/DeepSeek-R1",
                "likes": "13.4k",
                "one_line_description": "Text Generation • 685B",
            },
            {
                "model": "meta-llama/Meta-Llama-3-8B",
                "likes": "6.58k",
                "one_line_description": "Text Generation • 8B",
            },
            {
                "model": "bigscience/bloom",
                "likes": "5.01k",
                "one_line_description": "Text Generation • 176B",
            },
        ]
    )
    table = format_comparison_answer(
        "Compare the top 3 Hugging Face text-generation models sorted by likes: return a table (model, likes, one-line description)",
        [{"kind": "upstream", "skill": "browser", "output": {"content": content}}],
    )

    assert table is not None
    assert "deepseek-ai/DeepSeek-R1" in table
    assert "13.4k" in table
    assert "Text Generation" in table


def test_urbanpro_training_content_formats_provider_rows():
    from super_browser.browser.drivers.cascade import _urbanpro_training_content
    from super_browser.comparison_format import format_comparison_answer

    content = _urbanpro_training_content(
        [
            {
                "institute": "Satya Anand Kumar",
                "course_duration": "not listed",
                "approximate_fee": "not listed",
            },
            {
                "institute": "Sathya Narayanan",
                "course_duration": "not listed",
                "approximate_fee": "not listed",
            },
            {
                "institute": "Hemanth kumar N",
                "course_duration": "not listed",
                "approximate_fee": "not listed",
            },
            {
                "institute": "Joshua F",
                "course_duration": "not listed",
                "approximate_fee": "not listed",
            },
            {
                "institute": "Sunilkumar A s",
                "course_duration": "not listed",
                "approximate_fee": "not listed",
            },
        ]
    )
    table = format_comparison_answer(
        "Compare 5 CNC/VMC training institutes in Bangalore: return a table (institute, course duration, approximate fee)",
        [{"kind": "upstream", "skill": "browser", "output": {"content": content}}],
    )

    assert table is not None
    assert "Satya Anand Kumar" in table
    assert "not listed" in table


def test_amazon_product_content_formats_required_fields():
    import json

    from super_browser.browser.drivers.cascade import _amazon_product_content

    content = _amazon_product_content(
        {
            "url": "https://www.amazon.com/dp/B0DHC5CXKR",
            "title": "Gaming Laptop, 15.6 Inch FHD",
            "price": "not listed",
            "brand": "DUNHOO",
            "description": "16GB RAM and 512GB SSD laptop computer.",
        }
    )
    start = content.index("{")
    end = content.index("\n}\n") + 2
    data = json.loads(content[start:end])

    assert data["rows"][0]["title"] == "Gaming Laptop, 15.6 Inch FHD"
    assert data["rows"][0]["brand"] == "DUNHOO"
    assert "| Gaming Laptop, 15.6 Inch FHD | not listed | DUNHOO |" in content


def test_resolve_cnc_bangalore_uses_live_urbanpro_category():
    from super_browser.browser.urls import resolve_browser_urls

    urls = resolve_browser_urls(
        "",
        "Compare 5 CNC/VMC training institutes in Bangalore on UrbanPro",
    )

    assert "https://www.urbanpro.com/bangalore/cad-cam-training" in urls
    assert all("cnc-programming-training" not in url for url in urls)


def test_comparison_content_ready_rejects_homepage_blob():
    from super_browser.browser.validation import comparison_content_ready

    goal = "Compare 3 laptops under ₹80,000 on Flipkart"
    homepage = "Online Shopping India | Buy Mobiles, Electronics, Appliances & More" * 20
    assert comparison_content_ready(homepage, goal, min_browser_actions=3) is False
    table = "| Name | Price |\n| --- | --- |\n| A | ₹50,000 |\n| B | ₹60,000 |\n| C | ₹70,000 |"
    assert comparison_content_ready(table, goal, min_browser_actions=3) is True


def test_comparison_content_ready_rejects_homepage_prices_on_listing():
    from super_browser.browser.validation import comparison_content_ready

    goal = "Compare 3 laptops under ₹80,000 on Flipkart: search laptop, filter price"
    promos = "Deal ₹59,990 | Offer ₹54,999 | Sale ₹49,999 " * 5
    assert comparison_content_ready(promos, goal, min_browser_actions=3) is False


def test_comparison_content_ready_accepts_listing_with_specs():
    from super_browser.browser.validation import comparison_content_ready

    goal = "Compare 3 laptops under ₹80,000 on Flipkart"
    blob = (
        "HP Laptop ₹59,990 Intel Core i5 16GB RAM 4.3 stars\n"
        "Lenovo ₹54,999 Ryzen 5 8GB RAM 4.2 stars\n"
        "Acer ₹49,999 Core i3 8GB SSD 4.1 stars"
    )
    assert comparison_content_ready(blob, goal, min_browser_actions=3) is True


def test_layer_succeeded_rejects_homepage_despite_actions():
    from super_browser.browser.validation import layer_succeeded

    goal = "Compare 3 laptops under ₹80,000 on Flipkart: search laptop, filter price"
    homepage = "Flipkart homepage " * 120
    result = {
        "path": "agent",
        "content": homepage,
        "transcript": ["click_index:1", "click_index:2", "scroll:down"],
    }
    assert layer_succeeded(result, min_browser_actions=3, goal=goal) is False


def test_layer_succeeded_accepts_vlm_markdown_table():
    from super_browser.browser.validation import layer_succeeded

    goal = "Compare 3 laptops under ₹80,000 on Flipkart"
    table = (
        "| Name | Price | CPU/RAM | Rating |\n"
        "| --- | --- | --- | --- |\n"
        "| HP 15 | ₹59,990 | i5/16GB | 4.3 |\n"
        "| Lenovo | ₹54,999 | Ryzen5/8GB | 4.2 |\n"
        "| Acer | ₹49,999 | i3/8GB | 4.1 |"
    )
    result = {
        "path": "vision",
        "content": table,
        "transcript": ["vision_turn:1", "vision_turn:2", "click_mark:3"],
    }
    assert layer_succeeded(result, min_browser_actions=3, goal=goal) is True


def test_format_browser_path_labels():
    from super_browser.browser.replay import format_browser_path

    assert "blocked" in format_browser_path("gateway_blocked")
    assert format_browser_path("a11y").startswith("a11y")
    assert "agent" in format_browser_path("agent")


def test_to_browser_output_accepts_agent_path():
    from super_browser.browser.skill import to_browser_output

    out = to_browser_output(
        url="https://example.com",
        goal="compare pricing",
        raw={
            "path": "agent",
            "content": "Pro plan $20",
            "transcript": ["click_index:1", "click_index:2", "scroll:down"],
            "llm_calls": 2,
        },
    )
    assert out.path == "agent"
    assert out.content == "Pro plan $20"
    assert len(out.actions) == 3

