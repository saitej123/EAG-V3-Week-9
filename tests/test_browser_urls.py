"""Tests for multi-URL resolution and live browser fetch helpers."""

from __future__ import annotations

import os

from super_browser.browser.urls import (
    browser_max_urls,
    is_blocked_browser_portal,
    resolve_browser_urls,
    resolve_listing_portal_url,
)


def test_resolve_stack_pricing_urls():
    query = (
        "Compare 3 AI coding assistants by free plan vs paid plan: use the browser to visit "
        "official pricing pages for Cursor, GitHub Copilot, and Codeium."
    )
    urls = resolve_browser_urls("", query, query)
    hosts = [u.split("/")[2] for u in urls]
    assert "cursor.com" in hosts
    assert "github.com" in hosts
    assert "windsurf.com" in hosts
    assert len(urls) == browser_max_urls()
    assert len(urls) <= 3


def test_resolve_stack_caps_extra_named_targets(monkeypatch):
    monkeypatch.delenv("BROWSER_MAX_URLS", raising=False)
    query = (
        "Compare 5 AI coding assistants: visit official pricing pages for "
        "Cursor, GitHub Copilot, Codeium, Tabnine, and Continue.dev."
    )
    urls = resolve_browser_urls("", query, query)
    assert len(urls) == 5
    hosts = [u.split("/")[2] for u in urls]
    assert "cursor.com" in hosts
    assert "github.com" in hosts
    assert "tabnine.com" in hosts or any("tabnine" in h for h in hosts)


def test_resolve_merges_explicit_urls_first():
    query = "Compare tools at https://cursor.com/pricing and https://tabnine.com/pricing"
    urls = resolve_browser_urls("https://cursor.com/pricing", query, query)
    assert urls[0].startswith("https://cursor.com")
    assert any("tabnine.com" in u for u in urls)


def test_resolve_forge_urbanpro_url():
    query = "Compare 5 CNC/VMC training institutes in Bangalore on UrbanPro."
    urls = resolve_browser_urls("", query, query)
    assert urls
    assert any("urbanpro.com" in u for u in urls)
    assert not any("google.com" in u for u in urls)


def test_resolve_stack_five_tool_urls():
    query = (
        "Compare 5 AI coding tools by free plan and paid plan: visit official pricing for "
        "Cursor, GitHub Copilot, Codeium, Tabnine, and Continue.dev."
    )
    urls = resolve_browser_urls("", query, query)
    hosts = [u.split("/")[2] for u in urls]
    assert "cursor.com" in hosts
    assert len(urls) == 5


def test_resolve_forge_blocks_google():
    query = "Compare 3 CNC/VMC training institutes in Bangalore"
    urls = resolve_browser_urls("https://www.google.com", query, query)
    assert not any("google.com" in u for u in urls)


def test_single_url_unchanged():
    urls = resolve_browser_urls("https://example.com/page", "fetch example", "")
    assert urls == ["https://example.com/page"]


def test_resolve_dedupes_trailing_slash_variants():
    query = (
        "Compare 3 laptops under ₹80,000 on Flipkart: go to https://www.flipkart.com, "
        "search laptop, filter price up to ₹80,000."
    )
    urls = resolve_browser_urls("https://www.flipkart.com", query, query)
    assert urls == ["https://www.flipkart.com"]
