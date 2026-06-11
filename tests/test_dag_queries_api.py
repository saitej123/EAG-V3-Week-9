"""Browser assignment query corpus and /api/queries/* contract tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from super_browser.catalog import (
    assignment_payload,
    browser_queries_payload,
    expected_flow_for_query,
    get_dag_query,
    load_assignment_queries,
    validate_assignment_corpus,
)

EXPECTED_IDS = ["COMP", "DEAL", "TICKET", "STACK", "FORGE", "B1", "B2", "B3", "B4"]
CREATIVE_IDS = ["DEAL", "TICKET", "STACK", "FORGE"]


def test_validate_assignment_corpus_clean():
    assert validate_assignment_corpus() == []


def test_every_demo_query_has_query_text_and_bounds():
    for row in load_assignment_queries():
        assert str(row["query"]).strip()
        assert float(row["wall_clock_sec"]) > 0
        assert row.get("title")
        assert int(row["part"]) == 1


def test_design_queries_reference_real_ids():
    payload = assignment_payload()
    ids = {q["id"] for q in payload["queries"]}
    for dq in payload["design_queries"]:
        assert dq["kind"] == "browser"
        assert dq["query_id"] in ids
        assert set(dq.get("creative_comparisons") or []).issubset(ids)


def test_groups_cover_all_queries():
    payload = assignment_payload()
    grouped = [qid for g in payload["groups"] for qid in g["query_ids"]]
    assert sorted(grouped) == sorted(EXPECTED_IDS)


def test_submission_outline_order_matches_checklist():
    payload = assignment_payload()
    outline = payload["outline"]
    assert len(outline) == 4
    assert outline[0]["title"] == "Anchor mission"
    assert outline[0]["query_ids"] == ["COMP"]
    assert outline[0]["design_id"] == "browser_design"
    assert outline[1]["query_ids"] == ["DEAL", "TICKET"]
    assert outline[2]["query_ids"] == ["STACK", "FORGE"]
    assert outline[3]["query_ids"] == ["B1", "B2", "B3", "B4"]


@pytest.mark.parametrize("qid", EXPECTED_IDS)
def test_get_dag_query_lookup(qid: str):
    row = get_dag_query(qid)
    assert row is not None
    assert row["id"] == qid


def test_api_dag_queries_success():
    from app import app

    client = TestClient(app)
    res = client.get("/api/queries/dag")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    assert body["query_count"] == 9
    assert len(body["queries"]) == 9
    assert len(body["design_queries"]) == 1
    assert len(body["groups"]) == 4
    assert len(body["outline"]) == 4
    assert body.get("browser_only") is True
    assert body["outline"][0]["query_ids"][0] == "COMP"

    ids = [q["id"] for q in body["queries"]]
    assert sorted(ids) == sorted(EXPECTED_IDS)

    for q in body["queries"]:
        assert q["query"].strip()
        assert "wall_clock_sec" in q
        if q.get("expected_skills"):
            assert q.get("expected_flow")


def test_comp_expected_flow_and_min_actions():
    row = get_dag_query("COMP")
    assert row is not None
    assert "huggingface.co/models" in row["query"]
    assert row.get("min_browser_actions", 0) >= 3
    assert expected_flow_for_query(row) == "planner → browser → distiller → critic → formatter"
    payload = assignment_payload()
    api_comp = next(q for q in payload["queries"] if q["id"] == "COMP")
    assert api_comp["expected_flow"] == expected_flow_for_query(row)


def test_creative_queries_require_browser_actions():
    for qid in CREATIVE_IDS:
        row = get_dag_query(qid)
        assert row is not None
        assert row.get("min_browser_actions", 0) >= 3
        assert "browser" in row.get("expected_skills", [])
        assert row.get("featured") == "browser_creative"


def test_api_dag_queries_render_fields_for_ui():
    from app import app

    client = TestClient(app)
    body = client.get("/api/queries/dag").json()
    by_id = {q["id"]: q for q in body["queries"]}

    assert by_id["COMP"]["expected_flow"] == "planner → browser → distiller → critic → formatter"
    assert by_id["DEAL"]["featured"] == "browser_creative"
    assert by_id["TICKET"]["title"] == "IMAX showdown — BookMyShow"
    assert by_id["B1"]["expected_path"] == "extract"
    assert by_id["B4"]["expected_path"] == "vision"


def test_browser_queries_payload_matches_assignment():
    dag = assignment_payload()
    browser = browser_queries_payload()
    assert dag["query_count"] == browser["query_count"]
    assert {q["id"] for q in dag["queries"]} == {q["id"] for q in browser["queries"]}


def test_api_browser_queries_success():
    from app import app

    client = TestClient(app)
    res = client.get("/api/queries/browser")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    assert body["query_count"] == 9
    ids = {q["id"] for q in body["queries"]}
    assert ids == set(EXPECTED_IDS)
    assert len(body["design_queries"]) == 1
    assert body["design_queries"][0]["kind"] == "browser"
    assert body.get("browser_only") is True
    assert len(body["outline"]) == 4
    assert body["outline"][0]["title"] == "Anchor mission"
    assert body["outline"][1]["query_ids"] == ["DEAL", "TICKET"]
    assert body["outline"][2]["query_ids"] == ["STACK", "FORGE"]
    assert body["outline"][3]["query_ids"] == ["B1", "B2", "B3", "B4"]


def test_api_browser_reseed_sessions():
    from app import app

    client = TestClient(app)
    res = client.post("/api/browser/reseed-sessions")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    assert "dag_COMP_ref" in body["session_ids"]
    assert len(body["session_ids"]) == 5


def test_api_browser_queries_html_page_includes_loader():
    from app import app

    client = TestClient(app)
    html = client.get("/").text
    assert "loadBrowserQueries" in html
    assert "dagQueriesScroll" in html
    assert "/api/queries/browser" in html
    assert "chatWelcomeChips" in html
    assert "renderWelcomeDemoChips" in html
    assert "sidebarTabTasksBtn" in html
    assert "panelTasks" in html
    assert "mainTopTablist" not in html
    assert "dagGraphDownloadBtn" in html
    assert "dagGraphResumeBtn" in html
    assert "/run-agent/resume" in html
    assert "dagGraphResumeHint" in html
    assert "DAG Queries" not in html
    assert "RAG Queries" not in html
