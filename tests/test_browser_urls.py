"""Tests for multi-URL resolution and live browser fetch helpers."""

from __future__ import annotations

from super_browser.browser.urls import resolve_browser_urls


def test_resolve_stack_pricing_urls():
    query = (
        "Compare 5 AI coding assistants by free plan vs paid plan: use the browser to visit "
        "official pricing pages for Cursor, GitHub Copilot, Codeium, Tabnine, and Continue.dev."
    )
    urls = resolve_browser_urls("", query, query)
    hosts = [u.split("/")[2] for u in urls]
    assert "cursor.com" in hosts
    assert "github.com" in hosts
    assert "codeium.com" in hosts
    assert any("tabnine.com" in h for h in hosts)
    assert "continue.dev" in hosts
    assert len(urls) >= 5


def test_resolve_merges_explicit_urls_first():
    query = "Compare tools at https://cursor.com/pricing and https://tabnine.com/pricing"
    urls = resolve_browser_urls("https://cursor.com/pricing", query, query)
    assert urls[0].startswith("https://cursor.com")
    assert any("tabnine.com" in u for u in urls)


def test_single_url_unchanged():
    urls = resolve_browser_urls("https://example.com/page", "fetch example", "")
    assert urls == ["https://example.com/page"]
