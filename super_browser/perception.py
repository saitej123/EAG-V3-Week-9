"""
Perception: ``observe`` returns an ``Observation`` (ordered goals).

LLM emits drafts without stable ids; ``artifact_index`` refers to enumerated MEMORY HITS
that carry ``artifact_id``. The outer loop assigns stable ``Goal.id`` by position.

Architectural constraint (tool-blindness): Perception's SYSTEM prompt must never name MCP
tools or include a tool catalogue. Goals are intent-level only; tool selection is Decision's
responsibility (via its SYSTEM prompt and MCP tool docstrings).
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from loguru import logger

from .llm_env import gemini_models_ordered, shared_gemini_client
from .llm_retry import generate_structured_with_retry
from .search_providers import extract_sandbox_paths
from .schemas import Goal, MemoryItem, Observation, PerceptionGoalDraft, PerceptionLLMResponse


class PerceptionModule:
    def __init__(self) -> None:
        self.state = Observation(goals=[])

    def observe(
        self,
        query: str,
        hits: list[MemoryItem],
        history: list[dict[str, Any]],
        prior_goals: list[Goal],
        run_id: str,
    ) -> Observation:
        hits_with_art: list[tuple[int, MemoryItem]] = [(i, h) for i, h in enumerate(hits) if h.artifact_id]

        hits_lines = []
        for j, (_i, h) in enumerate(hits_with_art):
            hits_lines.append(
                f"  artifact_index={j} memory_index={_i} artifact_id={h.artifact_id!r} descriptor={h.descriptor[:160]!r}"
            )
        hits_block = "\n".join(hits_lines) if hits_lines else "  (no memory hits currently carry artifact handles)"

        prior_lines = "\n".join(
            f"  pos={p}: id={g.id!r} done={g.done} text={g.text!r} attach={g.attach_artifact_id!r}"
            for p, g in enumerate(prior_goals)
        ) or "  (none — first decomposition)"

        hist_txt = json.dumps(history[-16:], indent=2, default=str)[:12000]

        prompt = f"""
You are the Perception module for Super Browser Agent. Maintain an ordered goal list across a multi-turn loop.

USER QUERY:
{query}

RUN ID: {run_id}

PRIOR GOALS (preserve order; same positions unless goals complete):
{prior_lines}

MEMORY HITS WITH ARTIFACTS (use artifact_index ONLY from this list; integers 0..{max(0, len(hits_with_art)-1)}):
{hits_block}

RECENT HISTORY (JSON):
{hist_txt}

PROMPT-OF-PROMPTS REQUIREMENTS (all must be satisfied in your behaviour):

1. EXPLICIT REASONING — In `reasoning`, think step-by-step before updating goals. Explain what history shows, what changed, and why each goal is or is not done.

2. STRUCTURED OUTPUT — Respond ONLY as JSON matching the schema below. No prose outside JSON. Output must be easy to parse and validate.

3. TOOL SEPARATION — Perception PLANS only; you never call tools and you never name tools in goal text. Decision EXECUTES tools. Use `artifact_index` (integer or null) to tell Decision which memory artifact bytes to attach.

4. CONVERSATION LOOP — Each turn receives PRIOR GOALS + RECENT HISTORY. Reconcile `done` flags from new evidence. When prior_goals is non-empty, output the same number of goals in the same order.

5. INSTRUCTIONAL FRAMING — Follow this exact response shape:
{{
  "reasoning": "[PLANNING] Step 1: review history. Step 2: update goals.",
  "goals": [
    {{"text": "Search and extract source content", "done": false, "artifact_index": null}},
    {{"text": "Synthesize final answer for the user", "done": false, "artifact_index": null}}
  ]
}}

6. INTERNAL SELF-CHECKS — Before marking `done=true`, verify history contains successful outcomes for that step. If a tool failed, keep `done=false`. Sanity-check `artifact_index` is in range or null.

7. REASONING TYPE AWARENESS — Prefix `reasoning` with a tag: [PLANNING], [RECONCILIATION], or [ATTACHMENT_RESOLUTION].

8. ERROR HANDLING & FALLBACKS — If history shows repeated failures, ambiguity, or missing data, adjust goals to include fallbacks (e.g., "Search alternate source" or "Provide partial summary from available facts") instead of stalling.

RULES:
1. If prior_goals is empty: decompose the query into a highly concise ordered list of imperative goals (ideally **no more than 2 goals**, and at most 3, to respect the tight 3-iteration budget). Group related tasks together (e.g., search and extraction can be a single goal, and final summary the second goal) to ensure the agent converges extremely quickly.
2. If prior_goals is non-empty: output EXACTLY len(prior_goals) goals in the SAME ORDER.
   Update ``done`` when history shows the step satisfied. Done goals stay done.
3. For the first unfinished goal, set ``artifact_index`` ONLY when Decision needs fetched bytes now.
   Use the integer from MEMORY HITS WITH ARTIFACTS. Otherwise null.
4. Never invent artifact handles as strings — only integer artifact_index or null.
5. Preserve semantics of each goal; refine ``text`` lightly if needed but do not drop goals.
6. Goal text is intent-level only — describe WHAT to accomplish (e.g. "Index the paper for later search", "Answer from indexed knowledge"), never HOW via specific tool or API names.

Respond as JSON: {{"reasoning": "<tagged step-by-step reasoning>", "goals": [{{"text": "...", "done": false, "artifact_index": null}}]}}
"""

        client = shared_gemini_client()
        models = gemini_models_ordered()
        llm_goals: list[PerceptionGoalDraft] = []

        if client is None or not models:
            logger.warning("Perception: Gemini unavailable; heuristic fallback.")
            llm_goals = self._fallback_drafts(query, prior_goals)
        else:
            try:
                from google.genai import types

                parsed = generate_structured_with_retry(
                    model=models[0],
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=PerceptionLLMResponse,
                        temperature=0.2,
                    ),
                    schema_model=PerceptionLLMResponse,
                    label="perception",
                )
                llm_goals = parsed.goals
            except Exception as e:
                logger.warning(f"Perception failed: {e}")
                llm_goals = self._fallback_drafts(query, prior_goals)

        merged = self._merge_goals(prior_goals, llm_goals, hits_with_art)
        if not merged:
            merged = [
                Goal(
                    id=f"g-{uuid.uuid4().hex[:8]}",
                    text=(query or "").strip()[:800] or "Fulfill the user request.",
                    done=False,
                    attach_artifact_id=None,
                )
            ]
            logger.warning("Perception returned no goals — using single catch-all goal.")
        self.state = Observation(goals=merged)
        return self.state

    def _fallback_drafts(self, query: str, prior: list[Goal]) -> list[PerceptionGoalDraft]:
        if prior:
            return [PerceptionGoalDraft(text=g.text, done=g.done, artifact_index=None) for g in prior]
        q = (query or "").strip()[:800]
        ql = q.lower()
        paths = extract_sandbox_paths(q)
        if paths and "index" in ql:
            path = paths[0]
            if any(k in ql for k in ("every", "all ", "each ", "bulk")) and "/" in path:
                folder = path.split("/", 1)[0] + "/"
                return [
                    PerceptionGoalDraft(text=f"Index every file under {folder}", done=False),
                    PerceptionGoalDraft(text="Answer the user's question from indexed content", done=False),
                ]
            return [
                PerceptionGoalDraft(text=f"Index the file {path}", done=False),
                PerceptionGoalDraft(text="Answer the user's question from indexed content", done=False),
            ]
        return [PerceptionGoalDraft(text=q or "Fulfill the user request.", done=False)]

    def _merge_goals(
        self,
        prior: list[Goal],
        drafts: list[PerceptionGoalDraft],
        hits_with_art: list[tuple[int, MemoryItem]],
    ) -> list[Goal]:
        if not prior:
            out: list[Goal] = []
            for d in drafts:
                gid = f"g-{uuid.uuid4().hex[:8]}"
                attach = self._resolve_attach(d.artifact_index, hits_with_art)
                out.append(Goal(id=gid, text=d.text, done=d.done, attach_artifact_id=attach))
            return out

        n = len(prior)
        padded = list(drafts[:n])
        while len(padded) < n:
            idx = len(padded)
            padded.append(
                PerceptionGoalDraft(text=prior[idx].text, done=prior[idx].done, artifact_index=None)
            )
        padded = padded[:n]

        merged: list[Goal] = []
        for idx in range(n):
            pr = prior[idx]
            d = padded[idx]
            done = bool(pr.done or d.done)
            attach = self._resolve_attach(d.artifact_index, hits_with_art)
            merged.append(
                Goal(
                    id=pr.id,
                    text=d.text or pr.text,
                    done=done,
                    attach_artifact_id=attach if attach else pr.attach_artifact_id,
                )
            )
        return merged

    def _resolve_attach(
        self,
        artifact_index: int | None,
        hits_with_art: list[tuple[int, MemoryItem]],
    ) -> str | None:
        if artifact_index is None:
            return None
        if artifact_index < 0 or artifact_index >= len(hits_with_art):
            return None
        return hits_with_art[artifact_index][1].artifact_id
