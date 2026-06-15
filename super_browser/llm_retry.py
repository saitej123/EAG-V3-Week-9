"""Shared retry / backoff helpers for Gemini indexing and inference calls."""

from __future__ import annotations

import errno
import json
import re
import socket
import time
from collections.abc import Callable
from typing import Any, TypeVar

from loguru import logger
from pydantic import BaseModel, ValidationError

from .llm_env import (
    _float_env,
    _int_env,
    gemini_models_ordered,
    gemini_models_with_fallbacks,
    shared_gemini_client,
)

T = TypeVar("T")


def is_transient_network_error(exc: BaseException) -> bool:
    """True for DNS / connectivity failures where immediate retry rarely helps."""
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    if isinstance(exc, OSError):
        code = getattr(exc, "errno", None)
        if code in {
            errno.ENETUNREACH,
            errno.EHOSTUNREACH,
            errno.ECONNREFUSED,
            errno.ETIMEDOUT,
            errno.ECONNRESET,
        }:
            return True
        if code in (-2, -3):  # ENOENT / EAI_NONAME — common on WSL DNS blips
            return True
    msg = str(exc).lower()
    needles = (
        "name resolution",
        "temporary failure",
        "network is unreachable",
        "connection refused",
        "connection reset",
        "failed to resolve",
        "nodename nor servname",
        "getaddrinfo failed",
        "timed out",
        "timeout",
    )
    return any(n in msg for n in needles)


def llm_retry_max_attempts() -> int:
    return max(1, min(10, _int_env("LLM_RETRY_MAX", 3)))


def llm_retry_sleep_seconds() -> float:
    return max(0.25, _float_env("LLM_RETRY_SLEEP_SEC", 2.0))


def llm_retry_backoff_factor() -> float:
    return max(1.0, min(4.0, _float_env("LLM_RETRY_BACKOFF", 1.5)))


def vlm_page_batch_size() -> int:
    return max(1, min(20, _int_env("VLM_PAGE_BATCH_SIZE", 10)))


def vlm_batch_sleep_seconds() -> float:
    return max(0.0, _float_env("VLM_BATCH_SLEEP_SEC", 1.0))


def index_file_sleep_seconds() -> float:
    return max(0.0, _float_env("INDEX_FILE_SLEEP_SEC", 0.35))


def call_with_retry(
    fn: Callable[[], T],
    *,
    label: str = "llm",
    max_attempts: int | None = None,
    sleep_sec: float | None = None,
    backoff: float | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    fast_fail_on: Callable[[Exception], bool] | None = None,
) -> T:
    """Call ``fn`` up to ``max_attempts`` times with sleep between failures."""
    attempts = max_attempts if max_attempts is not None else llm_retry_max_attempts()
    wait = sleep_sec if sleep_sec is not None else llm_retry_sleep_seconds()
    factor = backoff if backoff is not None else llm_retry_backoff_factor()
    last: Exception | None = None
    fail_fast = fast_fail_on or is_transient_network_error

    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            last = e
            if fail_fast(e):
                logger.warning(f"[{label}] transient network error — skipping retries: {e}")
                break
            if attempt >= attempts:
                break
            if on_retry:
                on_retry(attempt, e, wait)
            else:
                logger.warning(f"[{label}] attempt {attempt}/{attempts} failed: {e} — retry in {wait:.1f}s")
            time.sleep(wait)
            wait *= factor

    assert last is not None
    raise last


def generate_content_with_retry(
    *,
    model: str,
    contents: Any,
    config: Any,
    label: str = "generate",
    max_attempts: int | None = None,
) -> Any:
    """Gemini ``generate_content`` with retry across models on persistent failure."""
    client = shared_gemini_client()
    if client is None:
        raise RuntimeError("Gemini client unavailable")

    models = gemini_models_with_fallbacks(model)
    last: Exception | None = None

    for model_id in models:
        try:
            return call_with_retry(
                lambda mid=model_id: client.models.generate_content(
                    model=mid,
                    contents=contents,
                    config=config,
                ),
                label=f"{label}:{model_id}",
                max_attempts=max_attempts,
                fast_fail_on=is_transient_network_error,
            )
        except Exception as e:
            last = e
            if is_transient_network_error(e):
                logger.warning(f"[{label}] model {model_id} network error — trying next model: {e}")
            else:
                logger.warning(f"[{label}] model {model_id} exhausted retries: {e}")

    raise RuntimeError(f"generate_content failed after retries: {last}")


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", t, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return t


def _close_truncated_json_object(text: str) -> str | None:
    """Best-effort repair when the model truncates mid-string (common on long answers)."""
    if not text.startswith("{"):
        return None
    depth = 0
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text
    candidate = text
    if in_string:
        candidate += '"'
    candidate += "}" * max(depth, 0)
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        return None


def loads_json_lenient(raw: str) -> Any:
    """Parse model JSON text; salvage truncated objects when possible."""
    text = _strip_json_fence((raw or "").strip())
    if not text:
        raise json.JSONDecodeError("empty response", raw or "", 0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        fixed = _close_truncated_json_object(text)
        if fixed is not None:
            return json.loads(fixed)
        raise


def extract_pydantic_model(response: Any, model: type[BaseModel]) -> BaseModel:
    """Build a Pydantic model from a Gemini response (parsed field or JSON text)."""
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        if isinstance(parsed, model):
            return parsed
        return model.model_validate(parsed)
    raw = (getattr(response, "text", None) or "").strip()
    data = loads_json_lenient(raw)
    return model.model_validate(data)


def generate_structured_with_retry(
    *,
    model: str,
    contents: Any,
    config: Any,
    schema_model: type[BaseModel],
    label: str = "structured",
    max_attempts: int | None = None,
) -> BaseModel:
    """Generate schema-constrained JSON and parse it; retries on API and parse failures."""
    client = shared_gemini_client()
    if client is None:
        raise RuntimeError("Gemini client unavailable")

    attempts = max_attempts if max_attempts is not None else llm_retry_max_attempts()
    sleep = llm_retry_sleep_seconds()
    factor = llm_retry_backoff_factor()
    models = gemini_models_with_fallbacks(model)
    last: Exception | None = None

    for model_id in models:
        wait = sleep
        for attempt in range(1, attempts + 1):
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=contents,
                    config=config,
                )
                return extract_pydantic_model(response, schema_model)
            except (json.JSONDecodeError, ValidationError) as e:
                last = e
                if attempt >= attempts:
                    logger.warning(
                        f"[{label}] model={model_id} parse/validation failed after {attempts} attempts: {e}"
                    )
                    break
                logger.warning(
                    f"[{label}] model={model_id} parse attempt {attempt}/{attempts} failed: {e} — retry in {wait:.1f}s"
                )
                time.sleep(wait)
                wait *= factor
            except Exception as e:
                last = e
                if attempt >= attempts:
                    logger.warning(f"[{label}] model={model_id} API failed after {attempts} attempts: {e}")
                    break
                logger.warning(
                    f"[{label}] model={model_id} API attempt {attempt}/{attempts} failed: {e} — retry in {wait:.1f}s"
                )
                time.sleep(wait)
                wait *= factor

    raise RuntimeError(f"{label} structured generation failed: {last}")


def embed_content_with_retry(
    *,
    model: str,
    contents: Any,
    config: Any | None = None,
    label: str = "embed",
) -> Any:
    client = shared_gemini_client()
    if client is None:
        raise RuntimeError("Gemini client unavailable")

    kwargs: dict[str, Any] = {"model": model, "contents": contents}
    if config is not None:
        kwargs["config"] = config

    return call_with_retry(
        lambda: client.models.embed_content(**kwargs),
        label=label,
    )
