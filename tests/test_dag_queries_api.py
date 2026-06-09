"""DAG demo query corpus and /api/queries/dag contract tests."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from super_browser.catalog import (
    assignment_payload,
    expected_flow_for_query,
    get_dag_query,
    load_assignment_queries,
    validate_assignment_corpus,
)

EXPECTED_IDS = [
    "hello",
    "A",
    "I",
    "J",
    "K",
    "P",
    "C_pass",
    "C_fail",
    "M",
    "PROS",
    "COMP",
    "B1",
    "B2",
    "B3",
    "B4",
]


def test_validate_assignment_corpus_clean():
    assert validate_assignment_corpus() == []


def test_every_demo_query_has_query_text_and_bounds():
    for row in load_assignment_queries():
        assert str(row["query"]).strip()
        assert float(row["wall_clock_sec"]) > 0
        assert row.get("title")
        assert int(row["part"]) in {1, 2, 3, 4, 5, 6}


def test_design_queries_reference_real_ids():
    payload = assignment_payload()
    ids = {q["id"] for q in payload["queries"]}
    for dq in payload["design_queries"]:
        if dq["kind"] == "parallel":
            assert dq["query_id"] in ids
        if dq["kind"] == "critic":
            assert set(dq["query_ids"]).issubset(ids)
        if dq["kind"] == "new_skill":
            assert dq["query_id"] in ids
        if dq["kind"] == "browser":
            assert dq["query_id"] in ids


def test_groups_cover_all_queries():
    payload = assignment_payload()
    grouped = [qid for g in payload["groups"] for qid in g["query_ids"]]
    assert sorted(grouped) == sorted(EXPECTED_IDS)


def test_submission_outline_order_matches_checklist():
    payload = assignment_payload()
    outline = payload["outline"]
    assert len(outline) == 6
    assert outline[0]["part"] == 1
    assert outline[0]["query_ids"] == ["hello", "A", "I", "J", "K"]
    assert outline[1]["part"] == 2
    assert outline[1]["query_ids"] == ["P"]
    assert outline[1]["design_id"] == "parallel_design"
    assert outline[2]["query_ids"] == ["C_pass", "C_fail"]
    assert outline[2]["design_id"] == "critic_design"
    assert outline[3]["query_ids"] == ["M"]
    assert outline[4]["query_ids"] == ["PROS"]
    assert outline[4]["design_id"] == "prosody_design"
    assert outline[5]["part"] == 6
    assert outline[5]["query_ids"] == ["COMP", "B1", "B2", "B3", "B4"]
    assert outline[5]["design_id"] == "browser_design"


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
    assert body["query_count"] == 15
    assert len(body["queries"]) == 15
    assert len(body["design_queries"]) == 4
    assert len(body["groups"]) == 6
    assert len(body["outline"]) == 6
    assert body["outline"][0]["query_ids"][0] == "hello"

    ids = [q["id"] for q in body["queries"]]
    assert sorted(ids) == sorted(EXPECTED_IDS)

    for q in body["queries"]:
        assert q["query"].strip()
        assert "wall_clock_sec" in q
        if q.get("expected_skills"):
            assert q.get("expected_flow")


def test_query_a_expected_flow_and_wikipedia_url():
    row = get_dag_query("A")
    assert row is not None
    assert "wikipedia.org/wiki/Claude_Shannon" in row["query"]
    assert expected_flow_for_query(row) == "planner → researcher → distiller → critic → formatter"
    payload = assignment_payload()
    api_a = next(q for q in payload["queries"] if q["id"] == "A")
    assert api_a["expected_flow"] == expected_flow_for_query(row)
    assert "fetch_url" in api_a.get("verify_hint", "").lower() or "fetch" in api_a.get("ui_hint", "").lower()


def test_api_dag_queries_render_fields_for_ui():
    from app import app

    client = TestClient(app)
    body = client.get("/api/queries/dag").json()
    by_id = {q["id"]: q for q in body["queries"]}

    assert by_id["P"]["parallel_researchers"] == 3
    assert by_id["C_pass"]["critic_expect"] == "pass"
    assert by_id["C_fail"]["critic_expect"] == "fail_then_recovery"
    assert "validate_json_keys" in by_id["C_pass"]["query"]
    assert by_id["M"]["ui_hint"]
    assert by_id["PROS"]["ui_hint"]
    assert by_id["PROS"]["expected_flow"] == "planner → prosody_analyst → formatter"
    assert "prosody_analyst" in by_id["PROS"]["expected_skills"]
    assert by_id["COMP"]["expected_flow"] == "planner → browser → distiller → formatter"
    assert by_id["B1"]["expected_path"] == "extract"
    assert re.search(r"150769", by_id["M"]["ui_hint"])


def test_api_browser_queries_success():
    from app import app

    client = TestClient(app)
    res = client.get("/api/queries/browser")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    assert body["query_count"] == 5
    ids = {q["id"] for q in body["queries"]}
    assert ids == {"COMP", "B1", "B2", "B3", "B4"}
    assert len(body["design_queries"]) == 1
    assert body["design_queries"][0]["kind"] == "browser"


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
