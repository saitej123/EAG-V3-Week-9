"""Tests for vision_api — thinking-token budget and model fallback."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from super_browser.vision_api import (
    _extract_response_text,
    _thinking_config_for_model,
    gemini_vision_model,
    gemini_vision_models_ordered,
    vision_analyze,
)


def test_gemini_vision_model_defaults_to_stable_flash(monkeypatch):
    monkeypatch.delenv("GEMINI_VISION_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash")
    assert gemini_vision_model() == "gemini-2.5-flash"


def test_gemini_vision_models_respects_override(monkeypatch):
    monkeypatch.setenv("GEMINI_VISION_MODEL", "gemini-2.5-flash-lite")
    models = gemini_vision_models_ordered()
    assert models[0] == "gemini-2.5-flash-lite"


def test_thinking_config_for_35_flash():
    cfg = _thinking_config_for_model("gemini-3.5-flash")
    assert cfg is not None
    assert getattr(cfg, "thinking_level", None) == "MINIMAL"


def test_thinking_config_for_25_flash():
    cfg = _thinking_config_for_model("gemini-2.5-flash")
    assert cfg is not None
    assert getattr(cfg, "thinking_budget", None) == 0


def test_extract_response_text_from_candidate_parts():
    part = MagicMock()
    part.text = "Red"
    content = MagicMock()
    content.parts = [part]
    candidate = MagicMock()
    candidate.content = content
    response = MagicMock()
    response.text = ""
    response.candidates = [candidate]
    assert _extract_response_text(response) == "Red"


def test_vision_analyze_retries_on_empty_text(monkeypatch):
    monkeypatch.setenv("GEMINI_VISION_MODEL", "gemini-3.5-flash")

    empty = MagicMock()
    empty.text = ""
    empty.candidates = []
    empty.usage_metadata = MagicMock(prompt_token_count=10, candidates_token_count=0, thoughts_token_count=40)

    good = MagicMock()
    good.text = "table data"
    good.candidates = []
    good.usage_metadata = MagicMock(prompt_token_count=12, candidates_token_count=5, thoughts_token_count=0)

    client = MagicMock()
    client.models.generate_content.side_effect = [empty, good]

    with patch("super_browser.vision_api.shared_gemini_client", return_value=client):
        with patch("super_browser.vision_api.gemini_vision_models_ordered", return_value=["bad-model", "good-model"]):
            result = vision_analyze(image_bytes=b"png", prompt="read this", max_tokens=512)

    assert result["text"] == "table data"
    assert result["model"] == "good-model"
    assert client.models.generate_content.call_count == 2
