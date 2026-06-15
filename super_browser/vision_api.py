"""Vision API — shared helper for POST /v1/vision and Browser Layer 3."""

from __future__ import annotations

import base64
import os
from typing import Any

from loguru import logger

from .llm_env import gemini_models_ordered, shared_gemini_client
from .llm_retry import call_with_retry, is_transient_network_error

_VISION_DEFAULT_MODEL = "gemini-2.5-flash"
_VISION_MIN_OUTPUT_TOKENS = 1024


def gemini_vision_model() -> str:
    """Primary vision-capable model id (env override or reliable default)."""
    override = (os.environ.get("GEMINI_VISION_MODEL") or "").strip()
    if override:
        return override
    models = gemini_vision_models_ordered()
    return models[0] if models else _VISION_DEFAULT_MODEL


def gemini_vision_models_ordered() -> list[str]:
    """Vision model fallback chain — prefers env override, then known vision models."""
    override = (os.environ.get("GEMINI_VISION_MODEL") or "").strip()
    preferred = [
        override,
        _VISION_DEFAULT_MODEL,
        "gemini-2.5-flash-lite",
    ]
    preferred.extend(gemini_models_ordered())
    seen: set[str] = set()
    out: list[str] = []
    for model in preferred:
        m = (model or "").strip()
        if not m or m in seen:
            continue
        seen.add(m)
        out.append(m)
    return out or [_VISION_DEFAULT_MODEL]


def _is_thinking_capable_model(model_id: str) -> bool:
    mid = (model_id or "").lower()
    return any(tag in mid for tag in ("2.5", "2.0", "3.", "3-"))


def _vision_generate_config(*, temperature: float, max_tokens: int) -> Any:
    """Build Gemini config that leaves room for visible output on thinking models."""
    from google.genai import types

    budget = max(max_tokens, _VISION_MIN_OUTPUT_TOKENS)
    kwargs: dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": budget,
    }
    return types.GenerateContentConfig(**kwargs)


def _thinking_config_for_model(model_id: str) -> Any | None:
    """Disable or minimize internal reasoning so vision answers are not empty."""
    from google.genai import types

    if not _is_thinking_capable_model(model_id):
        return None
    mid = model_id.lower()
    if "3.5" in mid or "3-" in mid or mid.startswith("gemini-3"):
        return types.ThinkingConfig(thinking_level="MINIMAL")
    return types.ThinkingConfig(thinking_budget=0)


def _apply_thinking_config(config: Any, model_id: str) -> Any:
    thinking = _thinking_config_for_model(model_id)
    if thinking is None:
        return config
    try:
        return config.model_copy(update={"thinking_config": thinking})
    except Exception:
        try:
            config.thinking_config = thinking
        except Exception:
            pass
    return config


def _extract_response_text(response: Any) -> str:
    text = (getattr(response, "text", None) or "").strip()
    if text:
        return text
    candidates = getattr(response, "candidates", None) or []
    parts: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if content is None:
            continue
        for part in getattr(content, "parts", None) or []:
            chunk = getattr(part, "text", None)
            if chunk:
                parts.append(str(chunk))
    return "\n".join(parts).strip()


def vision_extract_prompt(*, goal: str, url: str) -> str:
    """Single-shot screenshot read prompt shared by Playwright VLM layers."""
    from .comparison_format import parse_comparison_spec

    spec = parse_comparison_spec(f"{goal}\n{url}")
    if spec.is_comparison and spec.columns:
        cols = " | ".join(spec.columns)
        return f"""Screenshot of a live web page.

GOAL: {goal}
URL: {url}

Extract a markdown comparison table with **{spec.row_count} rows** and columns: {cols}
Include any location/context named in the goal. Use only text visible in the screenshot.
Plain markdown is fine — JSON not required.
"""
    return f"""Screenshot of a live web page.

GOAL: {goal}
URL: {url}

Read the screenshot and extract the fields needed to answer the goal.
Return markdown (use a table when comparing items). Plain text is fine — JSON not required.
"""


def vision_analyze(
    *,
    image_bytes: bytes,
    prompt: str,
    mime_type: str = "image/png",
    temperature: float = 0.2,
    max_tokens: int = 512,
    label: str = "vision",
) -> dict[str, Any]:
    """Route an image + prompt to a vision-capable Gemini model."""
    client = shared_gemini_client()
    if client is None:
        raise RuntimeError("Gemini not configured — set GEMINI_API_KEY for vision calls")

    from google.genai import types

    models = gemini_vision_models_ordered()
    last_error: Exception | None = None
    input_tokens = 0
    output_tokens = 0
    used_model = models[0]

    for model_id in models:
        used_model = model_id
        config = _apply_thinking_config(
            _vision_generate_config(temperature=temperature, max_tokens=max_tokens),
            model_id,
        )
        try:
            response = call_with_retry(
                lambda mid=model_id, cfg=config: client.models.generate_content(
                    model=mid,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                        prompt,
                    ],
                    config=cfg,
                ),
                label=f"{label}:{model_id}",
                fast_fail_on=is_transient_network_error,
            )
        except Exception as e:
            last_error = e
            logger.warning(f"[vision] model {model_id} failed: {e}")
            continue

        text = _extract_response_text(response)
        usage = getattr(response, "usage_metadata", None)
        input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        thoughts = int(getattr(usage, "thoughts_token_count", 0) or 0)
        finish = ""
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            finish = str(getattr(candidates[0], "finish_reason", "") or "")
        logger.info(
            f"[vision] model={model_id} in={input_tokens} out={output_tokens} "
            f"thoughts={thoughts} finish={finish} label={label}"
        )
        if text:
            return {
                "text": text,
                "model": model_id,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        if output_tokens == 0 and thoughts > 0:
            logger.warning(
                f"[vision] model={model_id} returned empty text "
                f"(thoughts={thoughts} consumed output budget)"
            )

    if last_error is not None:
        raise RuntimeError(f"vision_analyze failed: {last_error}")
    raise RuntimeError(f"vision_analyze returned empty text for all models ({models})")


def decode_image_payload(*, image_base64: str | None = None, image: str | None = None) -> bytes:
    """Accept base64 image from API clients (either field name)."""
    raw = (image_base64 or image or "").strip()
    if not raw:
        raise ValueError("image_base64 or image is required")
    if raw.startswith("data:"):
        _, _, payload = raw.partition(",")
        raw = payload or raw
    try:
        return base64.b64decode(raw, validate=False)
    except Exception as e:
        raise ValueError(f"invalid base64 image: {e}") from e
