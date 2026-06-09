"""
Super Browser Agent loop — wires Memory → Perception → Decision → Action (MCP).

Vector-first memory (FAISS + embeddings), ``index_document`` / ``search_knowledge`` MCP tools.
History entries are plain dicts mirroring typed boundaries (answer vs action events).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from .paths import ROOT

os.environ["CRAWL4AI_BASE_DIRECTORY"] = str(ROOT / ".crawl4ai")

from .action import ActionActuator, format_artifact_size, summarize_tool_result
from .artifact_store import ArtifactStore
from .decision import DecisionModule, fallback_iteration_budget_markdown
from .llm_env import (
    agent_iteration_ceiling,
    agent_llm_step_timeout_seconds,
    agent_max_iterations,
    resolve_iteration_budget,
)
from .memory import MemoryManager, memory_hit_content_excerpt
from .perception import PerceptionModule
from .schemas import DecisionOutput, Goal, MemoryItem, Observation, ToolCall
from .search_providers import (
    derive_search_queries,
    enrich_tool_call,
    heuristic_tool_call,
    merge_search_hits,
    primary_search_query,
    web_search_with_fallbacks,
)

# Cap artifact bytes sent to Decision LLM (large Wikipedia pages slow synthesis ~80s+).
MAX_DECISION_ATTACH_CHARS = 12_000
_CONTINUATION = " " * 16
_SYNTH_KW = (
    "synthes",
    "extract",
    "compare",
    "decide",
    "recommend",
    "analyze",
    "present",
    "summar",
    "formulate",
)
_MULTI_SOURCE_KW = (
    "search for",
    "web search",
    "fetch top",
    "read top",
    "top 3 result",
    "top three result",
    "top 3 source",
)


def _is_synthesis_goal_text(text: str) -> bool:
    t = text.lower()
    if any(k in t for k in _SYNTH_KW):
        return True
    if "list" in t and any(w in t for w in ("advice", "common", "numbered", "agree", "summar")):
        return True
    return False


def _is_multi_source_pipeline_goal(text: str) -> bool:
    """Search / serial top-N fetch goals in a multi-source synthesis run (Query D)."""
    t = text.lower()
    if _is_synthesis_goal_text(text):
        return False
    if any(k in t for k in _MULTI_SOURCE_KW):
        return True
    if t.startswith("search "):
        return True
    if ("fetch" in t or "read" in t) and "top" in t and "result" in t:
        return True
    if "from search" in t or "search result" in t:
        return True
    if ("fetch" in t or "read" in t) and any(w in t for w in ("result", "descriptor", "hit")):
        return True
    return False


def _is_multi_source_user_query(user_query: str) -> bool:
    """True for Query D-style search + read top-N + synthesize tasks."""
    t = (user_query or "").lower()
    if "search for" not in t and "web search" not in t:
        return False
    return any(
        k in t
        for k in (
            "top 3",
            "top three",
            "3 results",
            "three results",
            "read the top",
            "read top",
        )
    )


def _should_log_perception(goal: Goal | None, obs: Observation, user_query: str = "") -> bool:
    if not obs.goals:
        return False
    if obs.all_done():
        return True
    if goal is None:
        return True
    if _is_synthesis_goal_text(goal.text):
        return True
    if _is_multi_source_user_query(user_query) and _is_multi_source_pipeline_goal(goal.text):
        return False
    return True


def _fact_display_text(item: MemoryItem) -> str:
    excerpt = memory_hit_content_excerpt(item, max_len=200)
    if excerpt:
        return excerpt
    return (item.descriptor or "fact").strip()[:200]


def _log_memory_read(hits: list[MemoryItem]) -> None:
    logger.info(f"[memory.read]   {len(hits)} hits")
    if len(hits) != 1:
        return
    h = hits[0]
    if h.kind == "fact":
        logger.info(f'{_CONTINUATION}fact: "{_fact_display_text(h)}"')
    elif h.kind == "preference":
        logger.info(f'{_CONTINUATION}preference: "{_fact_display_text(h)}"')


def _answer_closes_run(ans: str, goal: Goal, obs: Observation) -> bool:
    """True when a substantive answer already satisfies remaining open goals."""
    text = (ans or "").strip()
    if len(text) < 50:
        return False
    open_goals = [g for g in obs.goals if not g.done and g.id != goal.id]
    if not open_goals:
        return True
    if all(_is_synthesis_goal_text(g.text) for g in open_goals):
        return len(text) >= 60
    return False


def _should_finish_after_answer(
    ans: str,
    goal: Goal,
    obs: Observation,
    *,
    iter_index: int,
    cap: int,
) -> bool:
    """Stop as soon as a substantive answer satisfies the plan — no extra iterations."""
    text = (ans or "").strip()
    if not text:
        return False
    if iter_index >= cap - 1:
        return True

    open_goals = [g for g in obs.goals if not g.done]

    if len(obs.goals) <= 1:
        return len(text) >= 15

    if len(open_goals) == 1 and open_goals[0].id == goal.id:
        gt = goal.text.lower()
        if any(k in gt for k in ("answer", "when", "what", "tell me", "recall", "remember")):
            return len(text) >= 10
        if any(k in gt for k in _SYNTH_KW):
            return len(text) >= 60
        return len(text) >= 30

    # Substantive answer with only synthesis-style goals still open — stop early.
    if len(open_goals) <= 2 and len(text) >= 80:
        if any(k in goal.text.lower() for k in _SYNTH_KW + ("explain", "compare", "summar")):
            return True

    return False


def _log_attach(artifact_id: str, nbytes: int) -> None:
    logger.info(f"[attach]        {artifact_id} ({format_artifact_size(nbytes)})")


def _should_log_attach(goal: Goal, attached: list[tuple[str, bytes]]) -> bool:
    """Log attach whenever Decision receives artifact bytes (extract / synthesis goals)."""
    if not attached or not goal.attach_artifact_id:
        return False
    if any(k in goal.text.lower() for k in _SYNTH_KW):
        return True
    if "attach" in goal.text.lower() or "artifact" in goal.text.lower():
        return True
    return len(attached) > 0


def _log_iter_header(iter_num: int) -> None:
    logger.info(f"─── iter {iter_num} ───")


def _log_perception(goals: list[Goal]) -> None:
    for idx, g in enumerate(goals):
        status = "done" if g.done else "open"
        text = g.text.strip()
        if len(text) > 100:
            text = text[:97] + "..."
        if idx == 0:
            logger.info(f"[perception]    [{status}] {text}")
        else:
            logger.info(f"{_CONTINUATION}[{status}] {text}")
        if g.attach_artifact_id:
            logger.info(f"{_CONTINUATION}  attach={g.attach_artifact_id}")


def _log_decision_tool(tc: ToolCall) -> None:
    args_json = json.dumps(tc.arguments, ensure_ascii=False)
    logger.info(f"[decision]      TOOL_CALL: {tc.name}({args_json})")


def _log_decision_answer(answer: str) -> None:
    lines = (answer or "").strip().splitlines()
    if not lines:
        logger.info("[decision]      ANSWER: (empty)")
        return
    logger.info(f"[decision]      ANSWER: {lines[0]}")
    for line in lines[1:]:
        logger.info(f"{_CONTINUATION}{line}")


def _log_action(summary: str) -> None:
    s = (summary or "ok").strip()
    if len(s) > 220:
        s = s[:217] + "..."
    logger.info(f"[action]        → {s}")


def _log_all_goals_done(count: int) -> None:
    logger.info(f"[done] all {count} goals satisfied")


def _truncate_attachment_blob(blob: bytes) -> bytes:
    try:
        text = blob.decode("utf-8", errors="replace")
    except Exception:
        return blob[:MAX_DECISION_ATTACH_CHARS]
    if len(text) <= MAX_DECISION_ATTACH_CHARS:
        return blob
    return text[:MAX_DECISION_ATTACH_CHARS].encode("utf-8")


def _log_final_answer(answer: str) -> None:
    from .flow import log_final_answer

    log_final_answer(answer)


def _final_text_from_history(history: list[dict]) -> str | None:
    answers = [h.get("text") for h in history if h.get("kind") == "answer"]
    answers = [a for a in answers if isinstance(a, str) and a.strip()]
    return answers[-1].strip() if answers else None


def _search_hits_from_history(history: list[dict]) -> list[dict]:
    hits: list[dict] = []
    for entry in history:
        if entry.get("kind") != "action" or entry.get("tool") != "web_search":
            continue
        desc = entry.get("result_descriptor") or ""
        if not desc.strip().startswith("["):
            try:
                parsed = json.loads(desc)
                if isinstance(parsed, list):
                    hits.extend(x for x in parsed if isinstance(x, dict) and x.get("url"))
            except json.JSONDecodeError:
                pass
    return hits


class SuperBrowserAgent:
    def __init__(self) -> None:
        self.memory = MemoryManager()
        self.perception = PerceptionModule()
        self.decision = DecisionModule()
        self.action = ActionActuator()
        self.artifacts = ArtifactStore()

    async def _run_heuristic_fallback(
        self,
        *,
        goal: Goal,
        user_query: str,
        hits: list[MemoryItem],
        history: list[dict],
        run_id: str,
        iter_index: int,
        note: str,
    ) -> None:
        """When Decision fails, prefer local index/read/search over web_search."""
        tc = heuristic_tool_call(goal=goal, user_query=user_query, hits=hits, history=history)
        if tc is None:
            logger.warning(f"[decision] Heuristic web_search ({note})")
            tc = ToolCall(name="web_search", arguments={})
        else:
            logger.warning(f"[decision] Heuristic {tc.name} ({note})")
        tc = enrich_tool_call(tc, goal=goal, user_query=user_query)
        _log_decision_tool(tc)
        desc, art_id = await self.action.execute(
            tc,
            store=self.artifacts,
            fallback_query=primary_search_query(user_query, goal.text),
        )
        art_bytes = len(self.artifacts.get_bytes(art_id) or b"") if art_id else 0
        _log_action(
            summarize_tool_result(tc.name, tc.arguments, desc, art_id, artifact_bytes=art_bytes)
        )
        await asyncio.to_thread(
            lambda: self.memory.record_outcome(
                tool_call=tc,
                result_text=desc,
                artifact_id=art_id,
                run_id=run_id,
                goal_id=goal.id,
            ),
        )
        history.append(
            {
                "iter": iter_index + 1,
                "kind": "action",
                "goal_id": goal.id,
                "tool": tc.name,
                "arguments": tc.arguments,
                "result_descriptor": desc[:800],
                "artifact_id": art_id,
                "note": note,
            }
        )

    async def _emergency_rescue_answer(self, user_query: str, goals: list[Goal], history: list[dict]) -> str | None:
        """Run focused searches and synthesize when the loop exits without a user-facing answer."""
        goal_text = next((g.text for g in goals if not g.done), goals[0].text if goals else "")
        queries = derive_search_queries(user_query, goal_text, limit=2)
        if not queries:
            queries = [primary_search_query(user_query, goal_text)]
        logger.warning(f"[emergency] Running rescue searches: {queries!r}")

        batches = await asyncio.gather(
            *[web_search_with_fallbacks(q, 5) for q in queries],
            return_exceptions=True,
        )
        lists = [b for b in batches if isinstance(b, list)]
        merged = merge_search_hits(*lists, max_results=10) if lists else []
        merged = [h for h in merged if h.get("url") and "web_search error" not in str(h.get("title", "")).lower()]
        if not merged:
            prior = _search_hits_from_history(history)
            merged = prior

        if not merged:
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(self.decision.synthesize_best_effort_answer, user_query, goals),
                    timeout=agent_llm_step_timeout_seconds(),
                )
            except asyncio.TimeoutError:
                return self.decision.synthesize_best_effort_answer(user_query, goals)

        try:
            answer = await asyncio.wait_for(
                asyncio.to_thread(
                    self.decision.synthesize_from_search_hits,
                    user_query,
                    goals,
                    merged,
                ),
                timeout=agent_llm_step_timeout_seconds(),
            )
            return answer
        except asyncio.TimeoutError:
            logger.error("[emergency] synthesis timed out — returning snippet list")
            lines = [f"## Recommendations (search-based)\n", f"**Your question:** {user_query}\n"]
            for h in merged[:6]:
                lines.append(f"- **{h.get('title', 'Result')}** — {h.get('snippet', '')[:220]}")
            lines.append(
                "\n*Weather and activity fit could not be fully synthesized in time; "
                "review snippets above or retry.*"
            )
            return "\n".join(lines)

    async def run(self, user_query: str, max_iterations: int | None = None) -> None:
        from .artifact_store import ensure_state_dirs

        ensure_state_dirs()
        cap = resolve_iteration_budget(user_query, max_iterations)
        run_id = uuid.uuid4().hex[:8]
        history: list[dict] = []
        prior_goals: list[Goal] = []

        try:
            await self._run_loop(user_query, cap, run_id, history, prior_goals)
        finally:
            try:
                await self.action.aclose()
            except Exception as e:
                logger.warning(f"[MCP] cleanup after run failed: {e}")

    async def _run_loop(
        self,
        user_query: str,
        cap: int,
        run_id: str,
        history: list[dict],
        prior_goals: list[Goal],
    ) -> None:
        logger.info(f"Query: {user_query}")
        base = agent_max_iterations()
        if cap > base:
            logger.info(
                f"run_id={run_id} | max iterations: {cap} "
                f"(auto-extended from {base}, ceiling {agent_iteration_ceiling()})"
            )
        else:
            logger.info(f"run_id={run_id} | max iterations: {cap}")

        # Durable-memory classification on raw user text (PoP contract).
        await asyncio.to_thread(
            lambda: self.memory.remember(user_query, source="user_query", run_id=run_id),
        )

        consecutive_errors = 0
        max_consecutive = 4

        for i in range(cap):
            try:
                _log_iter_header(i + 1)

                hits = await asyncio.to_thread(
                    lambda: self.memory.read(user_query, history, top_k=10),
                )
                _log_memory_read(hits)

                try:
                    obs = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.perception.observe,
                            user_query,
                            hits,
                            history,
                            prior_goals,
                            run_id,
                        ),
                        timeout=agent_llm_step_timeout_seconds(),
                    )
                except asyncio.TimeoutError:
                    logger.error("[perception] LLM step timed out — reusing prior goal plan")
                    history.append({"iter": i + 1, "kind": "error", "detail": "perception_llm_timeout"})
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive:
                        _log_final_answer(
                            "## Run stopped (errors)\n\nPerception timed out repeatedly. "
                            "See **Live console** and retry."
                        )
                        logger.info("[agent] RUN_COMPLETE reason=error_abort")
                        break
                    if prior_goals:
                        obs = Observation(goals=list(prior_goals))
                    elif self.perception.state.goals:
                        obs = Observation(goals=list(self.perception.state.goals))
                    else:
                        obs = Observation(
                            goals=[
                                Goal(
                                    id=f"g-{uuid.uuid4().hex[:8]}",
                                    text=(user_query or "Fulfill the user request.")[:800],
                                    done=False,
                                )
                            ]
                        )
                    prior_goals = list(obs.goals)
                    goal = obs.next_unfinished()
                    if _should_log_perception(goal, obs, user_query):
                        _log_perception(obs.goals)
                    continue
                prior_goals = list(obs.goals)

                if obs.all_done():
                    _log_perception(obs.goals)
                    _log_all_goals_done(len(obs.goals))
                    ft = _final_text_from_history(history)
                    if ft:
                        _log_final_answer(ft)
                    else:
                        _log_final_answer(
                            "All goals were marked complete, but no decision **answer** was recorded in "
                            "this run’s history. Open **Live console** for tool outputs and Perception "
                            "goal lines, or retry with a clearer intent."
                        )
                    logger.info("[agent] RUN_COMPLETE reason=all_goals_done")
                    break

                goal = obs.next_unfinished()
                if goal is None:
                    logger.warning("No unfinished goal in this iteration — stopping.")
                    _log_final_answer(
                        "The agent stopped: no unfinished goal was available (empty or inconsistent plan). "
                        "Check **Live console** and retry."
                    )
                    logger.info("[agent] RUN_COMPLETE reason=no_active_goal")
                    break

                if any(k in goal.text.lower() for k in _SYNTH_KW):
                    art_hits = [h for h in hits if h.artifact_id]
                    if art_hits and not goal.attach_artifact_id:
                        goal.attach_artifact_id = art_hits[-1].artifact_id
                        for g in obs.goals:
                            if g.id == goal.id:
                                g.attach_artifact_id = goal.attach_artifact_id
                                break

                if _should_log_perception(goal, obs, user_query):
                    _log_perception(obs.goals)

                attached: list[tuple[str, bytes]] = []
                attach_full_bytes = 0
                if goal.attach_artifact_id and self.artifacts.exists(goal.attach_artifact_id):
                    blob = self.artifacts.get_bytes(goal.attach_artifact_id)
                    if blob:
                        attach_full_bytes = len(blob)
                        trimmed = _truncate_attachment_blob(blob)
                        attached.append((goal.attach_artifact_id, trimmed))
                        logger.debug(
                            f"[attach] bytes for {goal.attach_artifact_id} "
                            f"({len(blob)} → {len(trimmed)} bytes for decision)"
                        )
                if _should_log_attach(goal, attached):
                    _log_attach(goal.attach_artifact_id or attached[0][0], attach_full_bytes)

                db_rows = await asyncio.to_thread(self.memory.query_products, "")

                try:
                    decision_out = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.decision.next_step,
                            goal,
                            hits,
                            attached,
                            history,
                            user_query,
                            db_rows,
                            iteration_cap=cap,
                        ),
                        timeout=agent_llm_step_timeout_seconds(),
                    )
                except asyncio.TimeoutError:
                    logger.error("[decision] LLM step timed out — running heuristic fallback")
                    history.append({"iter": i + 1, "kind": "error", "detail": "decision_llm_timeout"})
                    await self._run_heuristic_fallback(
                        goal=goal,
                        user_query=user_query,
                        hits=hits,
                        history=history,
                        run_id=run_id,
                        iter_index=i,
                        note="heuristic_after_decision_timeout",
                    )
                    consecutive_errors = 0
                    continue

                ans, tc = decision_out.resolved()
                if ans is not None:
                    history.append(
                        {
                            "iter": i + 1,
                            "kind": "answer",
                            "goal_id": goal.id,
                            "text": ans,
                        }
                    )
                    _log_decision_answer(ans)
                    consecutive_errors = 0

                    synth_done = any(k in goal.text.lower() for k in _SYNTH_KW) and len(ans.strip()) > 80
                    if (
                        _should_finish_after_answer(ans, goal, obs, iter_index=i, cap=cap)
                        or synth_done
                        or _answer_closes_run(ans, goal, obs)
                    ):
                        _log_all_goals_done(len(obs.goals))
                        _log_final_answer(ans)
                        logger.info("[agent] RUN_COMPLETE reason=all_goals_done")
                        break
                    continue

                if tc is None:
                    logger.warning("[decision] No answer and no tool_call — skipping.")
                    history.append({"iter": i + 1, "kind": "error", "detail": "empty_decision"})
                    if consecutive_errors == 0:
                        await self._run_heuristic_fallback(
                            goal=goal,
                            user_query=user_query,
                            hits=hits,
                            history=history,
                            run_id=run_id,
                            iter_index=i,
                            note="heuristic_after_empty_decision",
                        )
                        consecutive_errors = 0
                        continue
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive:
                        _log_final_answer(
                            "## Run stopped (errors)\n\nToo many consecutive failures or empty "
                            "decisions. See **Live console** for details."
                        )
                        logger.info("[agent] RUN_COMPLETE reason=error_abort")
                        break
                    continue

                tc = enrich_tool_call(tc, goal=goal, user_query=user_query)
                _log_decision_tool(tc)
                desc = ""
                art_id: str | None = None
                art_bytes = 0
                try:
                    desc, art_id = await self.action.execute(
                        tc,
                        store=self.artifacts,
                        fallback_query=primary_search_query(user_query, goal.text),
                    )
                    art_bytes = len(self.artifacts.get_bytes(art_id) or b"") if art_id else 0
                    _log_action(
                        summarize_tool_result(
                            tc.name, tc.arguments, desc, art_id, artifact_bytes=art_bytes
                        )
                    )
                except Exception as e:
                    _log_action(f"{tc.name} failed: {type(e).__name__}: {e}")
                    raise

                await asyncio.to_thread(
                    lambda: self.memory.record_outcome(
                        tool_call=tc,
                        result_text=desc,
                        artifact_id=art_id,
                        run_id=run_id,
                        goal_id=goal.id,
                    ),
                )

                history.append(
                    {
                        "iter": i + 1,
                        "kind": "action",
                        "goal_id": goal.id,
                        "tool": tc.name,
                        "arguments": tc.arguments,
                        "result_descriptor": desc[:800],
                        "artifact_id": art_id,
                    }
                )
                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                logger.exception(f"[agent] iteration {i + 1} failed: {e}")
                history.append({"iter": i + 1, "kind": "iteration_error", "error": str(e)})
                if consecutive_errors >= max_consecutive:
                    _log_final_answer(
                        "## Run stopped (errors)\n\nRepeated iteration exceptions. See **Live console**."
                    )
                    logger.info("[agent] RUN_COMPLETE reason=error_abort")
                    break

        else:
            logger.warning("Reached max iterations without completion.")
            ft = _final_text_from_history(history)
            if ft:
                _log_final_answer(ft)
                logger.info("[agent] RUN_COMPLETE reason=answer_in_history")
                return

            obs_goals = list(self.perception.state.goals)
            rescue = await self._emergency_rescue_answer(user_query, obs_goals, history)
            if rescue:
                _log_final_answer(rescue)
                logger.info("[agent] RUN_COMPLETE reason=emergency_rescue")
                return

            hits = await asyncio.to_thread(
                lambda: self.memory.read(user_query, history, top_k=12),
            )
            tail = json.dumps(history[-16:], indent=2, default=str)
            attached_txt = ""
            for g in obs_goals:
                if g.attach_artifact_id and self.artifacts.exists(g.attach_artifact_id):
                    b = self.artifacts.get_bytes(g.attach_artifact_id)
                    if b:
                        attached_txt += b.decode("utf-8", errors="replace")[:20000]
            db_rows = await asyncio.to_thread(self.memory.query_products, "")
            try:
                summary_md = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.decision.summarize_partial_progress,
                        user_query,
                        obs_goals,
                        hits,
                        db_rows,
                        tail,
                        attached_txt,
                        iteration_cap=cap,
                    ),
                    timeout=agent_llm_step_timeout_seconds(),
                )
            except asyncio.TimeoutError:
                summary_md = fallback_iteration_budget_markdown(user_query, obs_goals, tail, cap)
            _log_final_answer(summary_md)
            logger.info("[agent] RUN_COMPLETE reason=max_iterations")


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    q = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his birth date."
    )
    asyncio.run(SuperBrowserAgent().run(q))
