"""DAG LLM client — Gemini by default; gateway not required."""

from __future__ import annotations

import pytest

from super_browser.gateway_client import SkillLLMClient, gateway_url, gateway_v8_url, gateway_v9_url


def test_dag_ignores_gateway_url_env(monkeypatch):
    monkeypatch.setenv("GATEWAY_URL", "http://127.0.0.1:8101")
    monkeypatch.delenv("GATEWAY_V8_URL", raising=False)
    assert gateway_v8_url() is None
    client = SkillLLMClient("test")
    assert client.base_url is None


def test_dag_gateway_only_with_v8_env(monkeypatch):
    monkeypatch.setenv("GATEWAY_V8_URL", "http://127.0.0.1:8108")
    monkeypatch.delenv("GATEWAY_V9_URL", raising=False)
    assert gateway_v8_url() == "http://127.0.0.1:8108"
    assert gateway_url() == "http://127.0.0.1:8108"
    client = SkillLLMClient("test")
    assert client.base_url == "http://127.0.0.1:8108"


def test_dag_gateway_prefers_v9_over_v8(monkeypatch):
    monkeypatch.setenv("GATEWAY_V9_URL", "http://127.0.0.1:8109")
    monkeypatch.setenv("GATEWAY_V8_URL", "http://127.0.0.1:8108")
    assert gateway_v9_url() == "http://127.0.0.1:8109"
    assert gateway_url() == "http://127.0.0.1:8109"
    client = SkillLLMClient("test")
    assert client.base_url == "http://127.0.0.1:8109"
