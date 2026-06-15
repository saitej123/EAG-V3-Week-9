"""Tests for LLM retry helpers."""

from __future__ import annotations

import pytest

from super_browser.llm_retry import call_with_retry, is_transient_network_error


def test_is_transient_network_error():
    assert is_transient_network_error(OSError(-3, "Temporary failure in name resolution"))
    assert not is_transient_network_error(ValueError("bad json"))


def test_call_with_retry_fast_fails_on_dns():
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        raise OSError(-3, "Temporary failure in name resolution")

    with pytest.raises(OSError):
        call_with_retry(flaky, label="test-dns")
    assert calls["n"] == 1
