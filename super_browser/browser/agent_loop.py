"""Turn loop with indexed interactive elements (browser-use CLI pattern)."""

from __future__ import annotations

from loguru import logger

from .driver import TurnResult, agent_turn_prompt, execute_action, fence_actions, normalize_actions
from .indexed_dom import build_indexed_interactive_state
from .navigation import dismiss_cookie_banners, page_live_extract, wait_for_page_ready
from .vlm_parse import parse_action_json

_MAX_TURNS = 10
_MIN_DONE_CHARS = 40


async def run_indexed_agent_loop(page, goal: str, llm: Any) -> TurnResult | None:
    """Fresh indexed state each turn → LLM → execute actions → repeat."""
    notes: list[str] = []
    llm_calls = 0
    input_tokens = 0
    output_tokens = 0

    for turn in range(1, _MAX_TURNS + 1):
        await wait_for_page_ready(page)
        await dismiss_cookie_banners(page)
        state_text, selector_map = await build_indexed_interactive_state(page)
        if not selector_map:
            logger.info(f"[browser] agent t{turn}: no interactive elements")
            break

        prompt = agent_turn_prompt(
            goal=goal,
            state=state_text,
            url=page.url,
            turn=turn,
            element_count=len(selector_map),
        )
        try:
            raw_text = llm.chat(agent="browser", prompt=prompt, temperature=0.15, max_tokens=640)
        except Exception as e:
            notes.append(f"llm_error:t{turn}:{e}")
            continue

        llm_calls += 1
        input_tokens += max(len(prompt) // 4, 1)
        output_tokens += max(len(str(raw_text)) // 4, 48)
        raw = parse_action_json(str(raw_text or ""), allow_prose_done=True)

        kind = (raw.get("action") or "").lower()
        if kind in {"done", "extract"}:
            answer = str(raw.get("answer") or raw_text or "").strip()
            if len(answer) >= _MIN_DONE_CHARS:
                notes.append("agent:done")
                return TurnResult(
                    notes=notes,
                    done=True,
                    answer=answer[:12000],
                    turns=turn,
                    llm_calls=llm_calls,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

        actions = fence_actions(normalize_actions(raw))
        if not actions:
            if len(str(raw_text or "")) >= _MIN_DONE_CHARS:
                return TurnResult(
                    notes=notes,
                    done=True,
                    answer=str(raw_text).strip()[:12000],
                    turns=turn,
                    llm_calls=llm_calls,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            continue

        for action in actions:
            action["_selector_map"] = selector_map
            kind = (action.get("action") or "").lower()
            if kind in {"done", "extract"}:
                answer = str(action.get("answer") or "").strip()
                if answer:
                    return TurnResult(
                        notes=notes,
                        done=True,
                        answer=answer[:12000],
                        turns=turn,
                        llm_calls=llm_calls,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
            note = await execute_action(page, action)
            notes.append(note)

        if turn >= _MAX_TURNS - 1:
            live = await page_live_extract(page, goal)
            if live and len(live) >= _MIN_DONE_CHARS:
                notes.append("agent:final_live_extract")
                return TurnResult(
                    notes=notes,
                    done=True,
                    answer=live[:12000],
                    turns=turn,
                    llm_calls=llm_calls,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

    return None
