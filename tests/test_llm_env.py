"""LLM env helpers — gateway is optional; Gemini SDK is the default embed path."""

from __future__ import annotations

import super_browser.llm_env as llm_env


def test_gateway_base_url_unset_by_default(monkeypatch) -> None:
    monkeypatch.delenv("GATEWAY_URL", raising=False)
    monkeypatch.delenv("GATEWAY_V3_URL", raising=False)
    assert llm_env.gateway_base_url() is None


def test_gateway_base_url_from_env(monkeypatch) -> None:
    monkeypatch.setenv("GATEWAY_URL", "http://127.0.0.1:8101/")
    assert llm_env.gateway_base_url() == "http://127.0.0.1:8101"


def test_try_embed_skips_gateway_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("GATEWAY_URL", raising=False)
    monkeypatch.delenv("GATEWAY_V3_URL", raising=False)
    gateway_calls: list[str] = []

    class FakeHttpxClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, url: str, **kwargs) -> None:
            gateway_calls.append(url)

    import httpx

    monkeypatch.setattr(httpx, "Client", FakeHttpxClient)
    monkeypatch.setattr(llm_env, "shared_gemini_client", lambda: None)

    assert llm_env.try_embed_text("hello world") is None
    assert gateway_calls == []


def test_default_embed_model_is_v2(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_EMBED_MODEL", raising=False)
    assert llm_env.gemini_embed_model() == "gemini-embedding-2"


def test_format_embed_input_query() -> None:
    out = llm_env.format_embed_input("What is DPO?", task_type="retrieval_query")
    assert out == "task: search result | query: What is DPO?"


def test_format_embed_input_document() -> None:
    out = llm_env.format_embed_input("chunk body", task_type="retrieval_document", title="DPO paper")
    assert out == "title: DPO paper | text: chunk body"
