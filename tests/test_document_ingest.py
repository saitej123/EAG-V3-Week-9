"""Tests for unified document conversion and VLM ingestion."""

from __future__ import annotations

from unittest.mock import patch

from super_browser.documents import (
    INDEXABLE_SUFFIXES,
    DocumentExtract,
    PageExtract,
    citation_label,
    extract_document_vlm,
    is_indexable_document,
    normalize_to_pdf,
    suffix_from_content_type,
    vlm_page_extraction_prompt,
)
from super_browser.indexing import _page_source_tag, index_document_path


def test_indexable_suffixes_cover_office_and_md():
    assert ".md" in INDEXABLE_SUFFIXES
    assert ".docx" in INDEXABLE_SUFFIXES
    assert ".pptx" in INDEXABLE_SUFFIXES
    assert ".pdf" in INDEXABLE_SUFFIXES


def test_is_indexable_document():
    assert is_indexable_document("papers/foo.pdf")
    assert is_indexable_document("notes.md")
    assert is_indexable_document("deck.pptx")
    assert not is_indexable_document("binary.exe")


def test_suffix_from_content_type():
    assert suffix_from_content_type("application/pdf") == ".pdf"
    assert suffix_from_content_type("text/markdown; charset=utf-8") == ".md"


def test_citation_label():
    assert citation_label("papers/a.pdf", 2, 10) == "papers/a.pdf p.2/10"


def test_page_source_tag():
    assert _page_source_tag("sandbox", "p.pdf", 1, 5, 1, 1) == "[sandbox:p.pdf p.1/5]"
    assert "part 2/3" in _page_source_tag("sandbox", "p.pdf", 3, 5, 2, 3)


def test_vlm_prompt_covers_checkboxes_and_plots():
    prompt = vlm_page_extraction_prompt(page_number=1, page_total=3, path="form.pdf")
    lower = prompt.lower()
    assert "checkbox" in lower
    assert "tick" in lower
    assert "plot" in lower or "chart" in lower
    assert "figure" in lower


def test_text_md_normalizes_to_pdf():
    raw = b"# Title\n\nParagraph with enough content for a page.\n"
    pdf = normalize_to_pdf("note.md", raw)
    assert pdf[:4] == b"%PDF"


def test_document_extract_page_map():
    doc = DocumentExtract(
        path="x.pdf",
        extraction="vlm",
        pages=[PageExtract(page_number=1, page_total=1, text="hello")],
    )
    assert doc.page_map == {"page_1": "hello"}


@patch("super_browser.documents._vlm_extract_page", return_value="Page one text")
@patch("super_browser.documents._pdf_page_images", return_value=[(b"png", "image/png")])
@patch("super_browser.documents.normalize_to_pdf", return_value=b"%PDF-stub")
def test_extract_document_vlm_unified(mock_norm, mock_pdf, mock_vlm):
    out = extract_document_vlm("test.md", b"# Hi")
    assert out.extraction == "vlm"
    assert out.page_map == {"page_1": "Page one text"}
    mock_norm.assert_called_once()
    mock_vlm.assert_called_once()


def test_index_pdf_with_sidecar_stores_pdf_path(tmp_path, monkeypatch):
    from super_browser import indexing

    rp = tmp_path / "research_papers"
    rp.mkdir(parents=True)
    (rp / "2605.23904.pdf").write_bytes(b"%PDF-1.4")
    (rp / "2605.23904.md").write_text("# Paper\n\nSidecar body for indexing.", encoding="utf-8")
    monkeypatch.setattr(indexing, "SANDBOX", tmp_path)

    stored_paths: list[str] = []

    def capture(desc, **kw):
        stored_paths.append(str((kw.get("value") or {}).get("path")))

    with patch("super_browser.memory.add_fact", side_effect=capture):
        result = index_document_path("research_papers/2605.23904.pdf")

    assert result["path"] == "research_papers/2605.23904.pdf"
    assert result["read_path"] == "research_papers/2605.23904.md"
    assert result["chunks_indexed"] >= 1
    assert stored_paths
    assert all(p == "research_papers/2605.23904.pdf" for p in stored_paths)


def test_index_markdown_uses_text_by_default(tmp_path, monkeypatch):
    from super_browser import indexing

    monkeypatch.setattr(indexing, "SANDBOX", tmp_path)
    (tmp_path / "note.md").write_text("# Title\n\nBody paragraph.", encoding="utf-8")

    with patch("super_browser.memory.add_fact") as mock_add:
        result = index_document_path("note.md")

    assert result["extraction"] == "text"
    assert result["chunks_indexed"] >= 1
    assert mock_add.called


def test_index_markdown_text_opt_out(tmp_path, monkeypatch):
    from super_browser import indexing

    monkeypatch.setattr(indexing, "SANDBOX", tmp_path)
    (tmp_path / "note.md").write_text("# Title\n\nBody paragraph.", encoding="utf-8")

    with patch("super_browser.memory.add_fact") as mock_add:
        result = index_document_path("note.md", use_vlm=False)

    assert result["extraction"] == "text"
    assert result["chunks_indexed"] >= 1
    assert mock_add.called


def test_index_pdf_vlm_path(tmp_path, monkeypatch):
    from super_browser import indexing

    monkeypatch.setattr(indexing, "SANDBOX", tmp_path)
    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4")

    fake = DocumentExtract(
        path="doc.pdf",
        extraction="vlm",
        pages=[PageExtract(page_number=1, page_total=1, text="Attention is all you need")],
    )

    with patch("super_browser.indexing.extract_document", return_value=fake):
        with patch("super_browser.memory.add_fact") as mock_add:
            result = index_document_path("doc.pdf")

    assert result["extraction"] == "vlm"
    assert result["pages_indexed"] == 1
    assert "page_1" in result["page_map"]
    mock_add.assert_called_once()
    call_kw = mock_add.call_args.kwargs
    assert call_kw["value"]["page_number"] == 1
    assert call_kw["value"]["citation"] == "doc.pdf p.1/1"
