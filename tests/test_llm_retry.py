"""Tests for LLM retry helpers."""

from __future__ import annotations

import json

import pytest

from super_browser.llm_retry import call_with_retry, loads_json_lenient, vlm_page_batch_size


def test_call_with_retry_recovers_after_transient_failure():
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] < 2:
            raise RuntimeError("transient")
        return "ok"

    assert call_with_retry(flaky, label="test", max_attempts=3, sleep_sec=0.01, backoff=1.0) == "ok"
    assert state["n"] == 2


def test_call_with_retry_raises_after_max_attempts():
    def always_fail():
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        call_with_retry(always_fail, label="test", max_attempts=2, sleep_sec=0.01, backoff=1.0)


def test_vlm_page_batch_size_default_ten():
    assert vlm_page_batch_size() == 10


def test_loads_json_lenient_salvages_truncated_object():
    truncated = (
        '{"reasoning": "done", "branch": "answer", "answer_text": "Indoor museums are best when it rains'
    )
    data = loads_json_lenient(truncated)
    assert data["branch"] == "answer"
    assert "Indoor museums" in data["answer_text"]


def test_loads_json_lenient_strips_fence():
    raw = '```json\n{"a": 1}\n```'
    assert loads_json_lenient(raw) == {"a": 1}


def test_loads_json_lenient_raises_on_empty():
    with pytest.raises(json.JSONDecodeError):
        loads_json_lenient("")
