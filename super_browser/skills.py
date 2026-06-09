"""DAG skill registry + per-skill execution.

The orchestrator (flow.py) treats every node as a `Skill` object loaded
from agent_config.yaml. There is no Python class per skill — that
abstraction would have to be added at the point where a skill needs
behaviour the orchestrator can't infer from the yaml. Today every skill
either calls the LLM gateway or (for sandbox_executor) calls sandbox.py.

What lives here:
  - Skill / SkillRegistry
  - input resolution (`n:...`, `art:...`, `USER_QUERY`, literals)
  - prompt rendering (template + inputs + optional failure report)
  - JSON parsing of the model's reply (single top-level object)
  - the MCP tool loop for tool-using skills
  - `run_skill(...)` — the dispatcher
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml
from loguru import logger
from pydantic import ValidationError

from .dag_schemas import AgentResult, CoderOutput, CriticVerdict, NodeSpec, PlannerOutput, SkillConfig
from .paths import ROOT
from .sandbox import run_python
from .schemas import Goal, MemoryItem, ToolCall
from .search_providers import enrich_tool_call, extract_http_urls

CONFIG_PATH = ROOT / "agent_config.yaml"
PROMPTS_DIR = ROOT / "prompts"
MEMORY_PREVIEW_CHARS = 400
MAX_TOOL_ROUNDS = 6
FETCH_PAGE_CHARS_FOR_DOWNSTREAM = 20_000


def _parse_fetch_json_text(raw: str) -> str:
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return str(data.get("text") or raw)
    except json.JSONDecodeError:
        pass
    return raw


def _fetch_page_body(
    desc: str,
    art_id: str | None,
    get_bytes: Callable[[str], bytes | None] | None,
) -> str:
    """Full page text for distiller — load artifact when MCP stored a large fetch."""
    if art_id and get_bytes:
        blob = get_bytes(art_id)
        if blob:
            return _parse_fetch_json_text(blob.decode("utf-8", errors="replace"))[
                :FETCH_PAGE_CHARS_FOR_DOWNSTREAM
            ]
    if desc and not desc.startswith("[artifact "):
        return _parse_fetch_json_text(desc)[:FETCH_PAGE_CHARS_FOR_DOWNSTREAM]
    return (desc or "").strip()


def fetch_url_succeeded(desc: str, art_id: str | None, *, body: str | None = None) -> bool:
    """True when fetch_url returned enough content for downstream distiller."""
    text = (body if body is not None else desc or "").strip()
    if art_id:
        return len(text) >= 200 or not text.startswith("[")
    low = text.lower()
    if not text or low.startswith("[search stored") or "connection failed" in low[:120]:
        return False
    return len(text) >= 400


def explicit_url_fetch_mode(skill_name: str, user_query: str, sub_query: str) -> bool:
    return skill_name == "researcher" and bool(extract_http_urls(f"{sub_query}\n{user_query}"))


# ── catalogue ────────────────────────────────────────────────────────────────


class Skill:
    """One yaml catalogue entry — behaviour inferred from config, not subclasses."""

    def __init__(self, name: str, cfg: SkillConfig) -> None:
        self.name = name
        self.config = cfg
        self.prompt_path = ROOT / cfg.prompt if not Path(cfg.prompt).is_absolute() else Path(cfg.prompt)
        self.description = str(cfg.description or "").strip()
        self.tools_allowed: list[str] = list(cfg.tools)
        self.internal_successors: list[str] = list(cfg.internal_successors)
        self.critic: bool = bool(cfg.critic)
        self.extends_graph: bool = bool(cfg.extends_graph)
        self.provider_pin: str | None = cfg.provider
        self.temperature: float = float(cfg.temperature)
        self.max_tokens: int = int(cfg.max_tokens)

    @property
    def tools(self) -> list[str]:
        return self.tools_allowed

    def prompt_template(self) -> str:
        path = self.prompt_path
        if not path.is_file():
            alt = PROMPTS_DIR / path.name
            if alt.is_file():
                path = alt
        if not path.is_file():
            return f"You are the {self.name} skill. (Prompt file missing.)"
        return path.read_text(encoding="utf-8").strip()

    def render_prompt(
        self,
        *,
        user_query: str,
        inputs_block: str,
        memory_hits: str = "",
        failure_context: str = "",
    ) -> str:
        """Backward-compatible wrapper used by unit tests."""
        parts = [self.prompt_template(), "", "USER QUERY:", user_query]
        if memory_hits.strip():
            parts.extend(["", "MEMORY HITS:", memory_hits.strip()])
        parts.extend(["", "INPUTS:", inputs_block.strip() or "(none)"])
        if failure_context.strip():
            parts.extend(["", "FAILURE:", failure_context.strip()])
        return "\n".join(parts)


class SkillRegistry:
    def __init__(self, config_path: Path | None = None) -> None:
        path = config_path or CONFIG_PATH
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        skills_raw: dict[str, Any] = raw.get("skills") or {}
        self._skills: dict[str, Skill] = {
            name: Skill(name, SkillConfig.model_validate(row)) for name, row in skills_raw.items()
        }

    def get(self, name: str) -> Skill:
        if name not in self._skills:
            raise KeyError(f"unknown skill: {name}")
        return self._skills[name]

    def names(self) -> list[str]:
        return list(self._skills.keys())

    def has_critic_splice(self, name: str) -> bool:
        return bool(self.get(name).critic)


# ── input resolution + prompt rendering ──────────────────────────────────────


def format_memory_hits(hits: list[MemoryItem]) -> str:
    """Compact FAISS hit block for prompts — omitted when empty."""
    if not hits:
        return ""
    lines: list[str] = []
    for h in hits[:8]:
        v = h.value or {}
        chunk = v.get("chunk")
        raw = v.get("raw")
        if isinstance(chunk, str) and chunk.strip():
            preview_src = chunk.strip()
        elif isinstance(raw, str) and raw.strip():
            preview_src = raw.strip()
        else:
            preview_src = str(v.get("text") or h.descriptor or "")
        preview = preview_src[:MEMORY_PREVIEW_CHARS]
        if len(preview_src) > MEMORY_PREVIEW_CHARS:
            preview += "…"
        source = (h.source or v.get("path") or "").strip()
        lines.append(f"- {h.descriptor} | source={source} | {preview}")
    return "\n".join(lines)


def _resolve_node_id(ref: str, graph_nodes: dict, label_map: dict[str, str]) -> str | None:
    if ref in graph_nodes:
        return ref
    if ref.startswith("n:"):
        suffix = ref[2:]
        if suffix in label_map:
            return label_map[suffix]
        if ref in graph_nodes:
            return ref
    if ref in label_map:
        return label_map[ref]
    return None


def resolve_inputs(
    node_inputs: list[str],
    graph_nodes: dict[str, Any],
    query: str,
    *,
    label_map: dict[str, str] | None = None,
    artifacts_get_bytes: Callable[[str], bytes | None] | None = None,
) -> list[dict[str, Any]]:
    """Materialise each input id into a dict the prompt can serialise."""
    labels = label_map or {}
    out: list[dict[str, Any]] = []
    for inp in node_inputs:
        if inp == "USER_QUERY":
            out.append({"id": "USER_QUERY", "kind": "query", "value": query})
        elif inp.startswith("n:") or inp in labels:
            nid = _resolve_node_id(inp, graph_nodes, labels)
            if nid and nid in graph_nodes:
                upstream = graph_nodes[nid].get("result")
                if isinstance(upstream, AgentResult):
                    out.append(
                        {
                            "id": inp,
                            "kind": "upstream",
                            "skill": upstream.agent_name,
                            "output": upstream.output,
                        }
                    )
                else:
                    out.append({"id": inp, "kind": "upstream-missing", "output": None})
            else:
                out.append({"id": inp, "kind": "upstream-missing", "output": None})
        elif inp.startswith("art:"):
            aid = inp[4:]
            try:
                blob = artifacts_get_bytes(aid) if artifacts_get_bytes else None
                text = (blob or b"").decode("utf-8", errors="replace")
                out.append({"id": inp, "kind": "artifact", "text": text[:20_000]})
            except Exception as e:
                out.append({"id": inp, "kind": "artifact-missing", "error": str(e)})
        else:
            out.append({"id": inp, "kind": "literal", "value": inp})
    return out


def render_prompt(
    skill: Skill,
    query: str,
    resolved: list[dict[str, Any]],
    failure_report: str | None = None,
    memory_hits: list[MemoryItem] | None = None,
) -> str:
    parts = [skill.prompt_template().rstrip(), "", f"USER QUERY: {query}"]
    if failure_report:
        parts += ["", f"FAILURE:\n{failure_report}"]
    hits_block = format_memory_hits(memory_hits or [])
    if hits_block:
        parts += ["", f"MEMORY HITS ({len(memory_hits or [])} from FAISS):", hits_block]
    parts += ["", "INPUTS:", json.dumps(resolved, indent=2, default=str)[:20_000]]
    return "\n".join(parts)


def parse_skill_json(text: str) -> dict[str, Any]:
    """Skills return a single top-level JSON object. Strip markdown fences if present."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.endswith("```"):
            t = t[:-3]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(t[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {}


def _parse_tool_call(text: str) -> ToolCall | None:
    from .llm_retry import loads_json_lenient

    try:
        data = loads_json_lenient(text)
        if isinstance(data, dict) and data.get("tool_name"):
            args = data.get("tool_arguments") or data.get("tool_arguments_json") or {}
            if isinstance(args, str):
                args = loads_json_lenient(args)
            return ToolCall(name=str(data["tool_name"]), arguments=args if isinstance(args, dict) else {})
    except Exception:
        pass
    return None


def _lift_successors(skill: Skill, parsed: dict[str, Any]) -> tuple[list[NodeSpec], list[str]]:
    """Validate planner/skill-emitted NodeSpec rows; collect rejection messages."""
    raw_successors = list(parsed.pop("successors", []) or [])
    successors: list[NodeSpec] = []
    rejected: list[str] = []
    for s in raw_successors:
        try:
            successors.append(NodeSpec.model_validate(s))
        except ValidationError as ve:
            rejected.append(f"successor={s!r}  error={ve}")
    if skill.name == "planner":
        for s in parsed.get("nodes", []) or []:
            try:
                successors.append(NodeSpec.model_validate(s))
            except ValidationError as ve:
                rejected.append(f"node={s!r}  error={ve}")
    return successors, rejected


@dataclass
class SkillRunContext:
    """Runtime dependencies injected by the Executor."""

    user_query: str
    session_id: str
    memory_hits: list[MemoryItem]
    llm: Any
    action: Any
    artifacts: Any
    label_map: dict[str, str]
    failure_report: str | None = None


def _dag_enrich_tool_call(tc: ToolCall, *, user_query: str, sub_query: str) -> ToolCall:
    goal = Goal(id="dag", text=(sub_query or user_query).strip()[:500] or user_query)
    return enrich_tool_call(tc, goal=goal, user_query=user_query)


def _auto_tool_for_web_skill(
    skill: Skill,
    *,
    user_query: str,
    sub_query: str,
    skip_fetch: bool = False,
) -> ToolCall | None:
    """When the model skips tools, still fetch/search if the sub-question needs the web."""
    if skill.name not in {"researcher", "browser"}:
        return None
    combined = f"{sub_query}\n{user_query}"
    urls = extract_http_urls(combined)
    if skip_fetch and urls:
        return None
    if urls and "fetch_url" in skill.tools_allowed:
        return ToolCall(name="fetch_url", arguments={"url": urls[0]})
    if "web_search" in skill.tools_allowed and any(
        k in combined.lower()
        for k in ("population", "fetch", "http", "wikipedia", "current", "find ", "search")
    ):
        q = (sub_query or user_query).strip()[:280]
        return ToolCall(name="web_search", arguments={"query": q, "max_results": 5})
    return None


async def _tool_loop(
    skill: Skill,
    prompt: str,
    ctx: SkillRunContext,
    *,
    node_id: str,
    sub_query: str,
) -> tuple[str, str | None]:
    context = prompt
    last_art_id: str | None = None
    single_url = explicit_url_fetch_mode(skill.name, ctx.user_query, sub_query)
    fetched_once = False
    last_fetch_body = ""
    get_bytes = getattr(ctx.artifacts, "get_bytes", None)

    for round_i in range(MAX_TOOL_ROUNDS):
        text = await asyncio.to_thread(
            ctx.llm.chat,
            agent=skill.name,
            prompt=context,
            temperature=skill.temperature,
            max_tokens=skill.max_tokens,
        )
        tc = _parse_tool_call(text)
        if tc is None or tc.name not in skill.tools_allowed:
            if fetched_once and single_url:
                return (last_fetch_body or text, last_art_id)
            tc = _auto_tool_for_web_skill(
                skill,
                user_query=ctx.user_query,
                sub_query=sub_query,
                skip_fetch=fetched_once and single_url,
            )
            if tc is None:
                return (text, last_art_id)
            logger.info(f"[dag] {node_id} ({skill.name}) auto tool: {tc.name}")
        elif fetched_once and single_url and tc.name == "fetch_url":
            logger.info(f"[dag] {node_id} ({skill.name}) ignoring repeat fetch_url — page already loaded")
            return (last_fetch_body, last_art_id)

        tc = _dag_enrich_tool_call(tc, user_query=ctx.user_query, sub_query=sub_query)
        desc, art_id = await ctx.action.execute(
            tc,
            store=ctx.artifacts,
            fallback_query=sub_query or ctx.user_query,
        )
        if art_id:
            last_art_id = art_id
        logger.info(f"[dag] {node_id} ({skill.name}) tool round {round_i + 1}: {tc.name}")
        if tc.name == "fetch_url":
            body = _fetch_page_body(desc, art_id, get_bytes)
            if fetch_url_succeeded(desc, art_id, body=body):
                fetched_once = True
                last_fetch_body = body or desc
                if single_url:
                    logger.info(
                        f"[dag] {node_id} ({skill.name}) fetch_url complete — single fetch for explicit URL"
                    )
                    return (last_fetch_body, last_art_id)
        context = f"{context}\n\nTOOL {tc.name} result:\n{desc[:8000]}\n\nContinue or give final answer."
    return (text, last_art_id)


async def run_skill(
    skill: Skill,
    node_id: str,
    graph_nodes: dict[str, Any],
    session_id: str,
    query: str,
    failure_report: str | None,
    ctx: SkillRunContext,
) -> tuple[AgentResult, str]:
    """Dispatch one node. Returns (result, rendered_prompt)."""
    resolved = resolve_inputs(
        graph_nodes[node_id]["inputs"],
        graph_nodes,
        query,
        label_map=ctx.label_map,
        artifacts_get_bytes=ctx.artifacts.get_bytes,
    )
    rendered = render_prompt(skill, query, resolved, failure_report, memory_hits=ctx.memory_hits)
    started = time.time()

    if skill.name == "browser":
        from .browser import run_browser_cascade
        from .search_providers import extract_http_urls

        meta = graph_nodes[node_id].get("metadata") or {}
        sub_q = str(meta.get("question") or meta.get("goal") or query)
        meta_url = str(meta.get("url") or "").strip()
        combined = f"{sub_q}\n{query}\n{meta_url}"
        urls = [meta_url] if meta_url.startswith("http") else extract_http_urls(combined)
        if not urls:
            return (
                AgentResult(
                    success=False,
                    agent_name=skill.name,
                    status="failed",
                    error="browser skill requires url in metadata or an http(s) URL in the query",
                    error_code="extraction_failed",
                    elapsed_s=time.time() - started,
                ),
                rendered,
            )
        force_path = str(meta.get("force_path") or "").strip().lower() or None
        browser_out, error_code = await run_browser_cascade(
            urls[0],
            sub_q or query,
            llm=ctx.llm,
            force_path=force_path,
        )
        parsed = browser_out.model_dump()
        ok = browser_out.path not in {"failed", "gateway_blocked"} and bool(browser_out.content)
        return (
            AgentResult(
                success=ok,
                agent_name=skill.name,
                status="complete" if ok else "failed",
                output=parsed,
                elapsed_s=time.time() - started,
                error=None if ok else str(parsed.get("error") or "browser cascade failed"),
                error_code=None if ok else error_code,
            ),
            rendered,
        )

    if skill.name == "sandbox_executor":
        code = ""
        for r in resolved:
            if r.get("kind") == "upstream":
                upstream_out = r.get("output")
                if isinstance(upstream_out, dict):
                    code = upstream_out.get("code") or code
                elif isinstance(upstream_out, str):
                    parsed = parse_skill_json(upstream_out)
                    code = parsed.get("code") or code
                    if not code:
                        try:
                            co = CoderOutput.model_validate(json.loads(upstream_out))
                            code = co.code
                        except Exception:
                            pass
        if not code:
            return (
                AgentResult(
                    success=False,
                    agent_name=skill.name,
                    status="failed",
                    error="no code in upstream coder output",
                    elapsed_s=time.time() - started,
                ),
                rendered,
            )
        out = await asyncio.to_thread(run_python, code)
        return (
            AgentResult(
                success=(out["exit_code"] == 0 and not out.get("timed_out")),
                agent_name=skill.name,
                status="complete" if out["exit_code"] == 0 else "failed",
                output=out,
                elapsed_s=time.time() - started,
            ),
            rendered,
        )

    schema: type | None = None
    if skill.name == "planner":
        schema = PlannerOutput
    elif skill.name == "critic":
        schema = CriticVerdict
    elif skill.name == "coder":
        schema = CoderOutput

    tool_art_id: str | None = None
    try:
        if skill.tools_allowed:
            text, tool_art_id = await _tool_loop(
                skill,
                rendered,
                ctx,
                node_id=node_id,
                sub_query=str(graph_nodes[node_id].get("metadata", {}).get("question") or query),
            )
            if skill.name == "critic":
                text = await asyncio.to_thread(
                    ctx.llm.chat,
                    agent=skill.name,
                    prompt=(
                        "Using the tool results above, emit the final verdict as JSON only.\n\n"
                        f"{text[-6000:]}"
                    ),
                    temperature=0.0,
                    max_tokens=skill.max_tokens,
                    response_schema=CriticVerdict,
                )
            parsed = parse_skill_json(text)
            if skill.name == "critic" and not parsed:
                try:
                    parsed = CriticVerdict.model_validate(json.loads(text)).model_dump()
                except Exception:
                    parsed = {"raw": text}
        else:
            text = await asyncio.to_thread(
                ctx.llm.chat,
                agent=skill.name,
                prompt=rendered,
                temperature=skill.temperature,
                max_tokens=skill.max_tokens,
                response_schema=schema,
            )
            if schema is not None:
                from .gateway_client import SkillLLMClient

                data = SkillLLMClient.parse_json(text)
                if skill.name == "planner":
                    parsed = PlannerOutput.model_validate(data).model_dump()
                elif skill.name == "coder":
                    parsed = CoderOutput.model_validate(data).model_dump()
                elif skill.name == "critic":
                    parsed = CriticVerdict.model_validate(data).model_dump()
                else:
                    parsed = data if isinstance(data, dict) else parse_skill_json(text)
            else:
                parsed = parse_skill_json(text)
                if not parsed and text.strip():
                    parsed = {"text": text.strip()}

        successors, rejected = _lift_successors(skill, dict(parsed))
        if skill.extends_graph and not successors:
            raise ValueError(f"{skill.name} emitted zero nodes")

        if rejected:
            err = (
                f"{skill.name}: {len(rejected)} malformed NodeSpec(s) emitted.\n"
                + "\n".join(f"  - {line}" for line in rejected)
            )
            logger.error(f"[skills] {err}")
            return (
                AgentResult(
                    success=False,
                    agent_name=skill.name,
                    status="failed",
                    output=parsed,
                    successors=successors,
                    elapsed_s=time.time() - started,
                    error=err,
                ),
                rendered,
            )

        return (
            AgentResult(
                success=True,
                agent_name=skill.name,
                status="complete",
                output=parsed,
                artifact_id=tool_art_id if skill.tools_allowed else None,
                successors=successors,
                elapsed_s=time.time() - started,
            ),
            rendered,
        )
    except Exception as e:
        return (
            AgentResult(
                success=False,
                agent_name=skill.name,
                status="failed",
                error=f"exception: {type(e).__name__}: {e}",
                elapsed_s=time.time() - started,
            ),
            rendered,
        )
