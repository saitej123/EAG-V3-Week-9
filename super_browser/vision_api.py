"""Vision API — shared helper for POST /v1/vision and Browser Layer 3."""

from __future__ import annotations

import base64
import os
from typing import Any

from loguru import logger

from .llm_env import gemini_models_ordered, shared_gemini_client
from .llm_retry import generate_content_with_retry


def gemini_vision_model() -> str:
    """Vision-capable model id (env override or first GEMINI_MODELS entry)."""
    override = (os.environ.get("GEMINI_VISION_MODEL") or "").strip()
    if override:
        return override
    models = gemini_models_ordered()
    if models:
        return models[0]
    return "gemini-2.0-flash"


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

    model = gemini_vision_model()
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    response = generate_content_with_retry(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt,
        ],
        config=config,
        label=label,
    )
    text = (response.text or "").strip()
    usage = getattr(response, "usage_metadata", None)
    input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    logger.info(f"[vision] model={model} in={input_tokens} out={output_tokens} label={label}")
    return {
        "text": text,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


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
