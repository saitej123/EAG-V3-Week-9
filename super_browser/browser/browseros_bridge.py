"""BrowserOS MCP bridge — alternative to bundled Chromium when BrowserOS is running.

Enable with ``BROWSER_BACKEND=browseros`` or ``BROWSEROS_ENABLED=1``.
MCP URL defaults to ``http://127.0.0.1:9239/mcp`` (copy from chrome://browseros/mcp).

For full Playwright cascade layers, also set ``BROWSEROS_CDP_URL`` to BrowserOS CDP endpoint.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from .browser_backend import browseros_enabled, browseros_mcp_url
from .browseros_mcp import BrowserOsMcpClient, BrowserOsMcpError, parse_snapshot_index_map
from .driver import agent_turn_prompt, fence_actions, normalize_actions
from .vlm_parse import parse_action_json

_MAX_TURNS = 10
_MIN_CONTENT = 80


async def try_browseros_task(*, task: str, url: str, llm: Any | None = None) -> dict[str, Any] | None:
    """Run BrowserOS MCP automation; return cascade-shaped payload or None."""
    if not browseros_enabled():
        return None

    client = BrowserOsMcpClient(browseros_mcp_url())
    if not await client.ping():
        logger.info(
            "[browser] BrowserOS MCP unavailable — start BrowserOS and open chrome://browseros/mcp"
        )
        return None

    notes: list[str] = ["browseros:mcp"]
    try:
        await client.call_tool("navigate_page", {"url": url})
        notes.append(f"browseros:navigate:{url}")

        if llm is not None:
            agent = await _run_browseros_agent(client, task=task, url=url, llm=llm)
            if agent:
                return agent

        content = await client.call_tool("get_page_content", {})
        text = str(content or "").strip()
        if len(text) >= _MIN_CONTENT:
            notes.append("browseros:get_page_content")
            return {
                "path": "agent",
                "url": url,
                "content": text[:48000],
                "content_type": "text/markdown",
                "transcript": notes,
                "llm_calls": 0,
            }

        snap = await client.call_tool("take_enhanced_snapshot", {})
        snap_text = str(snap or "").strip()
        if len(snap_text) >= _MIN_CONTENT:
            notes.append("browseros:snapshot")
            return {
                "path": "agent",
                "url": url,
                "content": snap_text[:48000],
                "content_type": "text/plain",
                "transcript": notes,
                "llm_calls": 0,
            }
    except BrowserOsMcpError as e:
        logger.warning(f"[browser] BrowserOS MCP error: {e}")
    except Exception as e:
        logger.warning(f"[browser] BrowserOS bridge failed: {e}")
    return None


async def _run_browseros_agent(
    client: BrowserOsMcpClient,
    *,
    task: str,
    url: str,
    llm: Any,
) -> dict[str, Any] | None:
    notes: list[str] = ["browseros:agent"]
    llm_calls = 0
    input_tokens = 0
    output_tokens = 0

    for turn in range(1, _MAX_TURNS + 1):
        snapshot = await client.call_tool("take_enhanced_snapshot", {})
        state_text = str(snapshot or "").strip()
        id_map = parse_snapshot_index_map(state_text)
        if not state_text:
            break

        prompt = agent_turn_prompt(
            goal=task,
            state=state_text[:12000],
            url=url,
            turn=turn,
            element_count=len(id_map) or max(1, state_text.count("\n")),
        )
        try:
            raw_text = llm.chat(agent="browser", prompt=prompt, temperature=0.15, max_tokens=640)
        except Exception as e:
            notes.append(f"browseros:llm_error:t{turn}:{e}")
            continue

        llm_calls += 1
        input_tokens += max(len(prompt) // 4, 1)
        output_tokens += max(len(str(raw_text)) // 4, 48)
        raw = parse_action_json(str(raw_text or ""), allow_prose_done=True)

        kind = (raw.get("action") or "").lower()
        if kind in {"done", "extract"}:
            answer = str(raw.get("answer") or raw_text or "").strip()
            if len(answer) >= 40:
                notes.append("browseros:done")
                return _agent_payload(
                    url=url,
                    content=answer,
                    notes=notes,
                    llm_calls=llm_calls,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    turns=turn,
                )

        actions = fence_actions(normalize_actions(raw))
        if not actions:
            if len(str(raw_text or "")) >= 40:
                return _agent_payload(
                    url=url,
                    content=str(raw_text).strip(),
                    notes=notes,
                    llm_calls=llm_calls,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    turns=turn,
                )
            continue

        for action in actions:
            kind = (action.get("action") or "").lower()
            if kind in {"done", "extract"}:
                answer = str(action.get("answer") or "").strip()
                if answer:
                    notes.append("browseros:done")
                    return _agent_payload(
                        url=url,
                        content=answer,
                        notes=notes,
                        llm_calls=llm_calls,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        turns=turn,
                    )
            note = await _execute_mcp_action(client, action, id_map)
            notes.append(note)

    content = await client.call_tool("get_page_content", {})
    text = str(content or "").strip()
    if len(text) >= _MIN_CONTENT:
        notes.append("browseros:final_content")
        return _agent_payload(
            url=url,
            content=text,
            notes=notes,
            llm_calls=llm_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            turns=_MAX_TURNS,
        )
    return None


async def _execute_mcp_action(
    client: BrowserOsMcpClient,
    action: dict[str, Any],
    id_map: dict[int, str],
) -> str:
    kind = (action.get("action") or "").lower()
    try:
        if kind == "click_index":
            idx = int(action.get("index") or 0)
            element_id = id_map.get(idx) or str(idx)
            await client.call_tool("click", {"elementId": element_id})
            return f"browseros:click:{idx}"
        if kind == "scroll":
            direction = str(action.get("direction") or "down").lower()
            await client.call_tool("scroll", {"direction": direction})
            return f"browseros:scroll:{direction}"
        if kind in {"go_to_url", "navigate"}:
            target = str(action.get("url") or "").strip()
            if target:
                await client.call_tool("navigate_page", {"url": target})
                return f"browseros:go_to_url:{target[:80]}"
        if kind == "wait":
            import asyncio

            secs = float(action.get("seconds") or 1.0)
            await asyncio.sleep(min(max(secs, 0.2), 8.0))
            return f"browseros:wait:{secs}"
        if kind in {"type", "fill"}:
            idx = int(action.get("index") or action.get("element") or 0)
            text = str(action.get("text") or action.get("value") or "")
            element_id = id_map.get(idx) or str(idx)
            await client.call_tool("fill", {"elementId": element_id, "text": text, "clear": True})
            return f"browseros:fill:{idx}"
    except Exception as e:
        return f"browseros:action_failed:{kind}:{e}"
    return f"browseros:unsupported:{kind}"


def _agent_payload(
    *,
    url: str,
    content: str,
    notes: list[str],
    llm_calls: int,
    input_tokens: int,
    output_tokens: int,
    turns: int,
) -> dict[str, Any]:
    return {
        "path": "agent",
        "url": url,
        "content": content[:48000],
        "content_type": "text/plain",
        "transcript": notes,
        "turns": turns,
        "llm_calls": llm_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
