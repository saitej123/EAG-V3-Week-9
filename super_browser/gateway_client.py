"""LLM dispatch for DAG skills — Gemini SDK by default; gateway is opt-in only."""

from __future__ import annotations

import os
from typing import Any

import httpx
import yaml
from loguru import logger

from .llm_env import gemini_models_ordered, shared_gemini_client
from .llm_retry import generate_content_with_retry, loads_json_lenient
from .paths import ROOT

ROUTING_PATH = ROOT / "agent_routing.yaml"


def gateway_v8_url() -> str | None:
    """Gateway is optional. Only ``GATEWAY_V8_URL`` enables it (not ``GATEWAY_URL``)."""
    raw = (os.environ.get("GATEWAY_V8_URL") or "").strip().rstrip("/")
    return raw or None


def gateway_v9_url() -> str | None:
    """V9 gateway (cost ledger, Browser cascade) — prefer over V8 when set."""
    raw = (os.environ.get("GATEWAY_V9_URL") or "").strip().rstrip("/")
    return raw or None


def gateway_url() -> str | None:
    """Single gateway entry point — V9 wins over V8 so all skills share one ledger."""
    return gateway_v9_url() or gateway_v8_url()


def _load_agent_routing() -> dict[str, dict[str, str]]:
    if not ROUTING_PATH.is_file():
        return {}
    raw = yaml.safe_load(ROUTING_PATH.read_text(encoding="utf-8")) or {}
    agents = raw.get("agents") or {}
    return {k: v if isinstance(v, dict) else {} for k, v in agents.items()}


class SkillLLMClient:
    """Call Gemini directly; use gateway when ``GATEWAY_V9_URL`` or ``GATEWAY_V8_URL`` is set."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.base_url = gateway_url()
        self.routing = _load_agent_routing() if self.base_url else {}
        self._retry_5xx = max(1, int(os.environ.get("GATEWAY_RETRY_5XX", "2")))

    def _provider_for(self, agent: str) -> str | None:
        row = self.routing.get(agent) or {}
        return (row.get("provider") or "").strip() or None

    def chat(
        self,
        *,
        agent: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_schema: type | None = None,
        tools: list[dict] | None = None,
    ) -> str:
        if self.base_url:
            return self._chat_gateway(
                agent=agent,
                prompt=prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                response_schema=response_schema,
            )
        return self._chat_gemini(
            agent=agent,
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            response_schema=response_schema,
        )

    def _chat_gateway(
        self,
        *,
        agent: str,
        prompt: str,
        system: str | None,
        temperature: float,
        max_tokens: int,
        tools: list[dict] | None,
        response_schema: type | None,
    ) -> str:
        provider = self._provider_for(agent)
        body: dict[str, Any] = {
            "prompt": prompt,
            "system": system,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "agent": agent,
            "session": self.session_id,
            "stream": False,
        }
        if provider:
            body["provider"] = provider
        if tools:
            body["tools"] = tools
        last_err: Exception | None = None
        for attempt in range(self._retry_5xx + 1):
            try:
                with httpx.Client(timeout=120.0) as client:
                    resp = client.post(f"{self.base_url}/v1/chat", json=body)
                if resp.status_code >= 500 and attempt < self._retry_5xx:
                    logger.warning(f"[gateway] {resp.status_code} on {agent}, retry {attempt + 1}")
                    continue
                resp.raise_for_status()
                data = resp.json()
                return str(data.get("text") or "")
            except Exception as e:
                last_err = e
                if attempt >= self._retry_5xx:
                    break
        logger.warning(f"[gateway] falling back to Gemini SDK: {last_err}")
        return self._chat_gemini(
            agent=agent,
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            response_schema=response_schema,
        )

    def _chat_gemini(
        self,
        *,
        agent: str,
        prompt: str,
        system: str | None,
        temperature: float,
        max_tokens: int,
        response_schema: type | None,
    ) -> str:
        client = shared_gemini_client()
        models = gemini_models_ordered()
        if client is None or not models:
            raise RuntimeError(
                "Gemini not configured — set GEMINI_API_KEY in .env (gateway not required for DAG mode)"
            )
        from google.genai import types

        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system:
            config_kwargs["system_instruction"] = system
        if response_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema
        config = types.GenerateContentConfig(**config_kwargs)
        response = generate_content_with_retry(
            model=models[0],
            contents=prompt,
            config=config,
            label=f"dag:{agent}",
        )
        return (response.text or "").strip()

    @staticmethod
    def parse_json(text: str) -> dict[str, Any]:
        return loads_json_lenient(text)


# Back-compat alias
GatewayClient = SkillLLMClient
