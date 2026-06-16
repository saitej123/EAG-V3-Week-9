"""Direct Gemini client for browser drivers — no LLM gateway required."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger

from ...llm_retry import loads_json_lenient
from ...vision_api import vision_analyze


@dataclass
class GeminiResult:
    """Normalised reply from chat or vision calls."""

    parsed: dict[str, Any] | None
    text: str
    provider: str = "gemini"
    model: str = ""
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class GeminiClient:
    """Async Gemini adapter for A11yDriver and SetOfMarksDriver."""

    def __init__(self, llm: Any | None = None, *, session: str | None = None) -> None:
        self._llm = llm
        self.session = session

    def _chat_sync(
        self,
        prompt: str,
        *,
        system: str | None,
        max_tokens: int,
    ) -> str:
        if self._llm is not None:
            return self._llm.chat(
                agent="browser",
                prompt=prompt,
                system=system,
                temperature=0.0,
                max_tokens=max_tokens,
            )
        from ...gateway_client import SkillLLMClient

        client = SkillLLMClient(self.session or "browser")
        return client.chat(
            agent="browser",
            prompt=prompt,
            system=system,
            temperature=0.0,
            max_tokens=max_tokens,
        )

    @staticmethod
    def _parse_actions(text: str, schema: dict | None) -> dict | None:
        if not text.strip():
            return None
        data = loads_json_lenient(text)
        if isinstance(data, dict) and isinstance(data.get("actions"), list):
            return data
        if schema:
            logger.debug("[browser-drivers] model reply was not valid action JSON")
        return None

    async def chat(
        self,
        prompt: str,
        *,
        schema: dict | None = None,
        schema_name: str = "AgentOutput",
        system: str | None = None,
        max_tokens: int = 1024,
        session: str | None = None,
        model: str | None = None,
        provider: str | None = None,
    ) -> GeminiResult:
        del schema_name, session, model, provider
        full = prompt
        if schema:
            full = (
                f"{prompt}\n\nReturn ONLY valid JSON with keys `thinking` (string) "
                f"and `actions` (array). Schema:\n{json.dumps(schema)}"
            )
        started = time.time()
        text = await asyncio.to_thread(
            self._chat_sync,
            full,
            system=system,
            max_tokens=max_tokens,
        )
        parsed = self._parse_actions(text, schema)
        return GeminiResult(
            parsed=parsed,
            text=text,
            latency_ms=int((time.time() - started) * 1000),
            input_tokens=max(len(full) // 4, 1),
            output_tokens=max(len(text) // 4, 1),
        )

    async def vision(
        self,
        image_data_url: str,
        prompt: str,
        *,
        schema: dict | None = None,
        schema_name: str = "AgentOutput",
        system: str | None = None,
        max_tokens: int = 1024,
        session: str | None = None,
        model: str | None = None,
        provider: str | None = None,
    ) -> GeminiResult:
        del schema_name, session, model, provider
        if "," in image_data_url:
            b64 = image_data_url.split(",", 1)[1]
        else:
            b64 = image_data_url
        image_bytes = base64.b64decode(b64)
        parts = [p for p in (system, prompt) if p]
        if schema:
            parts.append(
                "Return ONLY valid JSON with keys `thinking` and `actions`. "
                f"Schema:\n{json.dumps(schema)}"
            )
        combined = "\n\n".join(parts)
        started = time.time()
        result = await asyncio.to_thread(
            vision_analyze,
            image_bytes=image_bytes,
            prompt=combined,
            label="browser-drivers-vision",
            max_tokens=max_tokens,
        )
        text = str(result.get("text") or "")
        parsed = self._parse_actions(text, schema)
        return GeminiResult(
            parsed=parsed,
            text=text,
            model=str(result.get("model") or ""),
            latency_ms=int((time.time() - started) * 1000),
            input_tokens=int(result.get("input_tokens") or 0),
            output_tokens=int(result.get("output_tokens") or 0),
        )
