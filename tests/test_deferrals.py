"""Design deferral documentation and structured payload."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_deferrals_doc_exists() -> None:
    path = ROOT / "docs" / "DEFERRALS.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    for phrase in (
        "Dense retrieval only",
        "Heuristic chunking",
        "FAISS reloaded from disk",
        "Fixed embedding model",
        "Reciprocal Rank Fusion",
        "future release",
    ):
        assert phrase in text, f"missing {phrase!r} in DEFERRALS.md"


def test_deferrals_payload() -> None:
    from super_browser.catalog import DESIGN_DEFERRALS, deferrals_payload

    assert len(DESIGN_DEFERRALS) == 4
    payload = deferrals_payload()
    assert payload["scope"] == "current"
    ids = {d["id"] for d in payload["deferrals"]}
    assert ids == {"dense_only", "heuristic_chunking", "faiss_reload", "fixed_embed_model"}

