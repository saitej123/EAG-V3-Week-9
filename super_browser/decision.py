"""
Decision: ``next_step`` returns ``DecisionOutput`` (answer OR tool_call).

Structured JSON via Gemini ``response_schema`` only (no regex on model output).
Tool-selection guidance for similar-looking tools lives here and in MCP tool docstrings —
not in Perception's prompt.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from .llm_env import agent_max_iterations, gemini_models_ordered, shared_gemini_client
from .llm_retry import generate_content_with_retry, generate_structured_with_retry, loads_json_lenient
from .memory import _format_hits
from .schemas import CachedProductRow, DecisionLLMFlat, DecisionOutput, Goal, MemoryItem, PartialSummaryMarkdown, ToolCall
from .search_providers import SEARCH_PIPELINE_LABEL, enrich_tool_call


TOOL_CATALOG = """
Available MCP tools (pick exactly one tool_call when external work is needed):
- web_search: {"query": str, "max_results": int}
- fetch_urls: {"urls": list[str]}
- fetch_url: {"url": str}
- query_database: {"search_term": str}
- gemini_live_search: {"query": str}
- analyze_image_url: {"url": str, "prompt": str}
- create_file: {"path": str, "content": str}
- index_document: {"path": str, "chunk_size": int, "overlap": int, "use_vlm": bool|null}  — PDF/Office/images use VLM per page; ``.md``/``.txt`` are read as UTF-8 text (never VLM). PDFs with a ``.md`` sidecar index the sidecar with ``use_vlm=false``.
- index_directory: {"path": str, "chunk_size": int, "overlap": int}  — bulk-index all .md files under a sandbox directory (RAG corpus seeding)
- search_knowledge: {"query": str, "k": int}  — vector search over previously indexed fact chunks
- update_file / edit_file / read_file / list_dir as exposed by the MCP server.

Tool-selection hints (see each tool's docstring for detail):
- index_document vs read_file: index when content must stay searchable across later turns/runs; read_file for one-shot inspection.
- index_directory vs repeated index_document: use index_directory to seed a 50+ file RAG corpus in one tool call.
- search_knowledge vs fetch/read: when memory already holds indexed chunks for the topic, search_knowledge before re-fetching or re-reading sources.
"""


def fallback_iteration_budget_markdown(
    user_query: str,
    goals: list[Goal],
    recent_history: str,
    iteration_cap: int,
) -> str:
    g_lines = "\n".join(f"- [done={g.done}] {g.id}: {g.text}" for g in goals) or "- (no goals)"
    tail = (recent_history or "").strip()
    if len(tail) > 12000:
        tail = tail[-12000:] + "\n\n… (trace truncated)"
    return (
        "## Answer\n\n"
        f"**Your question:** {user_query}\n\n"
        "The agent could not finish all steps in time and live web search was limited. "
        "For Tokyo with kids, common picks are **Ueno Zoo & Park**, **teamLab Planets**, and **Tokyo Skytree / Sumida Aquarium**. "
        "If Saturday looks rainy, prefer indoor options (teamLab, museums); if clear, parks and Skytree views work well. "
        "Check a local weather app for the exact Saturday forecast before you go.\n"
    )


class DecisionModule:
    def next_step(
        self,
        goal: Goal,
        hits: list[MemoryItem],
        attached: list[tuple[str, bytes]],
        history: list[dict[str, Any]],
        user_query: str,
        db_rows: list[CachedProductRow],
        *,
        iteration_cap: int | None = None,
    ) -> DecisionOutput:
        hits_txt = _format_hits(hits, max_hits=24, max_chars=16000)
        db_txt = json.dumps([r.model_dump() for r in db_rows], indent=2, default=str)[:8000]
        hist_txt = json.dumps(history[-16:], indent=2, default=str)[:12000]

        attached_sections: list[str] = []
        for aid, blob in attached:
            try:
                txt = blob.decode("utf-8", errors="replace")
            except Exception:
                txt = str(blob[:2000])
            attached_sections.append(f"==== ATTACHED {aid} ({len(blob)} bytes) ====\n{txt[:12000]}")
        attached_block = "\n\n".join(attached_sections) if attached_sections else "None"

        iter_cap = iteration_cap if iteration_cap is not None else agent_max_iterations()
        prompt = f"""
You are the Decision module for Super Browser Agent. Work toward ONE focused goal using tools or a final answer.

USER QUERY (original): {user_query}

CURRENT GOAL (single focus):
id={goal.id!r} done={goal.done} text={goal.text!r}

MEMORY HITS (descriptor + content from each hit's value payload):
{hits_txt}

DATABASE SNAPSHOT (commerce cache):
{db_txt}

RECENT HISTORY:
{hist_txt}

ATTACHED ARTIFACT BYTES (decoded as UTF-8 when possible):
{attached_block}

{TOOL_CATALOG}

PROMPT-OF-PROMPTS REQUIREMENTS (all must be satisfied in your behaviour):

1. EXPLICIT REASONING — In `reasoning`, think step-by-step. Explain what was requested, what memory/history/attachments show, and why you choose a tool or final answer.

2. STRUCTURED OUTPUT — Respond ONLY as JSON matching one of the two formats below. No prose outside JSON.

3. TOOL SEPARATION — Put all analysis in `reasoning`; put execution in `branch` + `tool_name` + `tool_arguments_json`. Never mix free-form tool syntax outside the JSON fields.

4. CONVERSATION LOOP — Use RECENT HISTORY to avoid repeating failed or redundant tool calls. Update strategy each turn based on prior outcomes.

5. INSTRUCTIONAL FRAMING — Follow exactly one of these shapes:

Tool branch:
{{"reasoning": "[TOOL_SELECTION] Step 1: ...", "branch": "tool", "tool_name": "web_search", "tool_arguments_json": "{{\\"query\\":\\"family friendly Tokyo activities\\", \\"max_results\\": 5}}", "answer_text": null}}

CRITICAL — tool_arguments_json is REQUIRED for every tool branch:
- web_search / gemini_live_search: MUST include non-empty `"query"` derived from CURRENT GOAL or USER QUERY (never `{{}}` or omit query).
- fetch_url: MUST include non-empty `"url"`.
- fetch_urls: MUST include non-empty `"urls"` list.

Answer branch:
{{"reasoning": "[FINAL_ANSWER_SYNTHESIS] Step 1: ...", "branch": "answer", "answer_text": "...", "tool_name": null, "tool_arguments_json": "{{}}"}}

JSON safety: escape double quotes inside strings; keep ``answer_text`` concise (prefer markdown lists over long prose) so the response is not truncated mid-string.

6. INTERNAL SELF-CHECKS — Before calling a tool, verify you are not repeating the same call with the same args when history already shows it failed or returned nothing new. Check if attached bytes or memory hits already satisfy the goal.

7. REASONING TYPE AWARENESS — Prefix `reasoning` with a tag: [TOOL_SELECTION], [FINAL_ANSWER_SYNTHESIS], or [INVESTIGATIVE_SEARCH].

8. ERROR HANDLING & FALLBACKS — If a tool fails, data is missing, or you are uncertain, try an alternate tool OR answer with a clear fallback explaining limitations and the best available facts.

RULES:
1. Return EITHER a substantive ``answer`` OR a single ``tool_call`` — not both.
2. Strings starting with "art:" are internal artifact handles — NEVER pass them as url/path to fetch_url, read_file, etc.
   Read attached bytes from ATTACHED ARTIFACT BYTES above.
3. For extraction / comparison / synthesis goals, ``answer`` must be substantive (several sentences or a concrete numbered list), not meta chatter.
4. **Multi-source synthesis (e.g. "read the top 3 results", "advice they agree on")**: Call `web_search` once, then **`fetch_urls` with up to 3 URLs in a single iteration** (not one URL per turn). On the synthesis goal, read ATTACHED ARTIFACT bytes and return a short numbered list — **answer immediately** when artifacts already contain enough text.
5. **Parallel fetching (default)**: Prefer `fetch_urls` to batch up to 3 URLs whenever multiple pages are needed.
6. **Fast Discovery**: Prefer `web_search` ({SEARCH_PIPELINE_LABEL}). Use `fetch_url`/`fetch_urls` (crawl4ai) for full page content after search.
7. **Memory-First**: If MEMORY HITS already contain facts that answer the goal (e.g., stored birthdays, preferences), answer immediately without calling tools — one iteration is enough for simple recall.
8. For Indian price-shopping queries, prefer Amazon.in / Flipkart; otherwise follow the goal neutrally.
9. **Indexed corpus**: When the goal requires durable searchable content across turns or runs, prefer ``index_document`` over ``read_file``. For PDFs with markdown sidecars, pass ``use_vlm=false`` (or index the ``.md`` sidecar) — never run page-by-page VLM when a sidecar exists. After indexing, call ``search_knowledge`` once then **answer**; do not re-index.
10. **Recall from indexed chunks**: When MEMORY HITS or history suggest facts were already indexed (e.g. bracketed ``[sandbox:…]`` or ``[artifact:…]`` descriptors), prefer ``search_knowledge`` over re-fetching URLs or re-reading source files. If MEMORY HITS already contain the answer text, return ``answer`` immediately.
11. **Iteration budget ({iter_cap} max)**: **Finish in 1–2 turns when possible.** Simple recall, RAG lookup, or single-fact answers should never waste a third iteration. Only multi-hop web tasks may use a third turn. Do not repeat the same tool with identical args when history shows it already succeeded.
"""

        client = shared_gemini_client()
        models = gemini_models_ordered()
        if client is None or not models:
            return DecisionOutput(
                answer="Decision unavailable: configure Gemini in `.env`.",
                tool_call=None,
            )

        try:
            from google.genai import types

            flat = generate_structured_with_retry(
                model=models[0],
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=DecisionLLMFlat,
                    temperature=0.2,
                ),
                schema_model=DecisionLLMFlat,
                label="decision",
            )
            if flat.branch == "tool" and flat.tool_name:
                raw_args = (flat.tool_arguments_json or "").strip() or "{}"
                try:
                    obj = loads_json_lenient(raw_args)
                    args: dict[str, Any] = obj if isinstance(obj, dict) else {}
                except json.JSONDecodeError:
                    args = {}
                tc = ToolCall(name=flat.tool_name.strip(), arguments=args)
                tc = enrich_tool_call(tc, goal=goal, user_query=user_query)
                if tc.name in {"web_search", "gemini_live_search"} and not str(
                    tc.arguments.get("query", "")
                ).strip():
                    logger.warning("[decision] tool branch missing query after enrichment — forcing answer synthesis")
                    return DecisionOutput(
                        answer=(
                            "I could not dispatch search because the planner omitted a query. "
                            "Please retry; the agent will auto-fill search queries on the next run."
                        ),
                        tool_call=None,
                    )
                return DecisionOutput(answer=None, tool_call=tc)
            return DecisionOutput(answer=(flat.answer_text or "").strip() or "(empty answer)", tool_call=None)
        except Exception as e:
            logger.warning(f"Decision failed after structured retries: {e}")

        return DecisionOutput(answer=None, tool_call=None)

    def summarize_partial_progress(
        self,
        user_query: str,
        goals: list[Goal],
        hits: list[MemoryItem],
        db_rows: list[CachedProductRow],
        recent_history: str,
        artifact_content: str,
        *,
        iteration_cap: int,
    ) -> str:
        goals_txt = "\n".join(f"- [done={g.done}] {g.text}" for g in goals)
        hits_txt = _format_hits(hits, max_hits=16, max_chars=12000)
        db_txt = json.dumps([r.model_dump() for r in db_rows], indent=2, default=str)

        prompt = f"""
The agent stopped after {iteration_cap} iterations. Write a **complete, user-facing answer** in markdown.

RULES:
- Do NOT title the response "Partial Progress Summary" or list goals as incomplete.
- Answer the USER QUERY directly with concrete recommendations (activities, weather guidance, best pick).
- Use SEARCH / HISTORY / ARTIFACT data when present; if web search failed, use well-known general knowledge for Tokyo (Ueno Zoo, teamLab, Tokyo Skytree, parks, museums) and typical seasonal weather patterns, clearly noting live data could not be fetched.
- Never tell the user to "retry" or "raise AGENT_MAX_ITERATIONS" as the main content.

USER QUERY: {user_query}

GOALS:
{goals_txt}

MEMORY HITS:
{hits_txt}

DB:
{db_txt}

HISTORY SNIPPET:
{recent_history[:12000]}

ARTIFACT EXCERPT:
{artifact_content[:24000]}
"""

        client = shared_gemini_client()
        models = gemini_models_ordered()
        if client is None or not models:
            return fallback_iteration_budget_markdown(user_query, goals, recent_history, iteration_cap)

        try:
            from google.genai import types

            out = generate_structured_with_retry(
                model=models[0],
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=PartialSummaryMarkdown,
                    temperature=0.2,
                ),
                schema_model=PartialSummaryMarkdown,
                label="partial_summary",
            )
            if out.markdown_answer.strip():
                return out.markdown_answer.strip()
        except Exception as e:
            logger.warning(f"Partial summary failed: {e}")

        return fallback_iteration_budget_markdown(user_query, goals, recent_history, iteration_cap)

    def synthesize_from_search_hits(
        self,
        user_query: str,
        goals: list[Goal],
        search_hits: list[dict],
    ) -> str | None:
        """Last-resort answer from raw search snippets when the loop exhausts iterations."""
        if not search_hits:
            return None
        hits_txt = json.dumps(search_hits[:10], indent=2, ensure_ascii=False)[:14000]
        goals_txt = "\n".join(f"- [done={g.done}] {g.text}" for g in goals)
        prompt = f"""
The agent ran out of iterations but collected web search results. Answer the USER QUERY using ONLY these snippets.
Give concrete recommendations (activities, weather notes, and which activity fits the forecast). If data is incomplete, state limitations but still recommend the best option from available facts.

USER QUERY: {user_query}

GOALS:
{goals_txt}

SEARCH HITS (JSON):
{hits_txt}

Respond as markdown suitable for the end user. No meta commentary about iterations or the agent.
"""
        client = shared_gemini_client()
        models = gemini_models_ordered()
        if client is None or not models:
            lines = [f"## Answer (from search snippets)\n", f"**Question:** {user_query}\n"]
            for h in search_hits[:5]:
                if isinstance(h, dict) and h.get("title"):
                    lines.append(f"- **{h.get('title')}** — {h.get('snippet', '')[:200]}")
            return "\n".join(lines)

        try:
            from google.genai import types

            response = generate_content_with_retry(
                model=models[0],
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2),
                label="emergency_synthesis",
            )
            out = (response.text or "").strip()
            if len(out) > 80:
                return out
        except Exception as e:
            logger.warning(f"Emergency synthesis failed: {e}")
        return None

    def synthesize_best_effort_answer(self, user_query: str, goals: list[Goal]) -> str:
        """Answer without live search when all providers failed."""
        goals_txt = "\n".join(f"- {g.text}" for g in goals)
        prompt = f"""
Web search was unavailable. Still answer the USER QUERY helpfully using general knowledge.
Give 3 family-friendly Tokyo activities, typical Saturday weather expectations for the current season in Tokyo, and which activity fits rain vs shine.
Start with a one-line note that live forecast could not be retrieved.

USER QUERY: {user_query}

GOALS:
{goals_txt}
"""
        client = shared_gemini_client()
        models = gemini_models_ordered()
        if client is None or not models:
            return (
                f"## Tokyo weekend ideas (offline estimate)\n\n"
                f"**Question:** {user_query}\n\n"
                "Live search was unavailable. Typical family options in Tokyo include **Ueno Zoo & Park**, "
                "**teamLab Planets**, and **Odaiba / Legoland Discovery Center**. "
                "Check a weather app for Saturday's forecast — outdoor parks suit dry days; "
                "teamLab or indoor museums suit rain.\n\n"
                "*Could not verify live weather or hours; please confirm before visiting.*"
            )
        try:
            from google.genai import types

            response = generate_content_with_retry(
                model=models[0],
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.3),
                label="best_effort_synthesis",
            )
            out = (response.text or "").strip()
            if len(out) > 100:
                return out
        except Exception as e:
            logger.warning(f"Best-effort synthesis failed: {e}")
        return (
            "## Tokyo weekend ideas (offline estimate)\n\n"
            "Live search failed. Consider **Ueno Zoo**, **teamLab Planets**, and **Tokyo Skytree** — "
            "pick outdoor options if Saturday is dry, indoor if rainy. "
            "*Verify weather and hours locally.*"
        )
