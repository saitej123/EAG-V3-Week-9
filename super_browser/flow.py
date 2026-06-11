"""Growing-graph DAG orchestrator.

The agent's loop becomes a NetworkX DiGraph. Each node is a skill; edges
carry typed AgentResult payloads. The graph GROWS at runtime via five
actors: the Planner's seed plan, dynamic successors from any skill,
static `internal_successors` from the yaml, Critic auto-insertion on
edges out of `critic:true` skills, and Planner re-invocation on node
failure (gated by `recovery.plan_recovery`). Perception's tool-blindness
contract from S7 is preserved — Planner names skills, never tools.

Persistence lives in persistence.py; skill execution in skills.py;
failure-policy in recovery.py; sandbox in sandbox.py.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

import networkx as nx
from loguru import logger

from .action import ActionActuator
from .artifact_store import ArtifactStore
from .dag_schemas import (
    AgentResult,
    NodeSpec,
    NodeState,
    NodeStatus,
    PlannerOutput,
)
from .gateway_client import SkillLLMClient
from .memory import MemoryManager
from .persistence import SessionLoadError, SessionStore
from .recovery import handle_critic_verdict, plan_recovery, recovery_planner_inputs
from .schemas import MemoryItem
from .search_providers import extract_http_urls
from .skills import SkillRegistry, SkillRunContext, format_memory_hits, run_skill

CRITIC_FAIL_CAP = 1
MAX_NODES = 60
_BROWSER_GRACEFUL_SKIP = frozenset({"browser_exhausted", "missing_dependency"})


def coerce_planner_successors(user_query: str, successors: list[NodeSpec]) -> list[NodeSpec]:
    """Inject fetch/comparison metadata when the planner under-specifies the plan."""
    from .comparison_format import enrich_planner_nodes

    successors = enrich_planner_nodes(user_query, successors)
    urls = extract_http_urls(user_query)
    if not urls:
        return successors
    skills = {s.skill for s in successors}
    if "researcher" in skills or "browser" in skills:
        return successors
    if skills != {"formatter"}:
        return successors
    logger.warning(
        "[dag] planner emitted formatter-only for URL query — injecting researcher→distiller→formatter"
    )
    return [
        NodeSpec(
            skill="researcher",
            inputs=["USER_QUERY"],
            metadata={"label": "fetch", "question": f"Fetch and summarize {urls[0]}"},
        ),
        NodeSpec(
            skill="distiller",
            inputs=["n:fetch"],
            metadata={
                "label": "extract",
                "question": "Extract birth date, death date, and three key contributions",
            },
        ),
        NodeSpec(skill="formatter", inputs=["n:extract"], metadata={"label": "out"}),
    ]


def _log_node(node_id: str, skill: str, msg: str) -> None:
    logger.info(f"[dag] {node_id} ({skill}) {msg}")


def _agent_text_output(result: AgentResult | None) -> str:
    if result is None:
        return ""
    out = result.output
    if isinstance(out, dict):
        for key in ("text", "final_answer", "summary"):
            if key in out and out[key]:
                return str(out[key])
        return json.dumps(out, ensure_ascii=False)
    return str(out or "")


def _state_output_from_result(result: AgentResult) -> str:
    out = result.output
    if isinstance(out, dict):
        return json.dumps(out, ensure_ascii=False)
    return str(out or "")


class Graph:
    """NetworkX DiGraph with label registry, critic splice, internal successors."""

    def __init__(self, registry: SkillRegistry | None = None) -> None:
        self.dg = nx.DiGraph()
        self.registry = registry or SkillRegistry()
        self.label_to_node: dict[str, str] = {}
        self._counter = 0
        self.user_query = ""
        self.critic_fail_counts: dict[str, int] = {}

    def _next_id(self) -> str:
        self._counter += 1
        return f"n:{self._counter}"

    def add_node_from_spec(self, spec: NodeSpec, node_id: str | None = None) -> str:
        nid = node_id or self._next_id()
        if node_id and str(node_id).startswith("n:"):
            try:
                self._counter = max(self._counter, int(str(node_id).split(":", 1)[1]))
            except ValueError:
                pass
        label = str(spec.metadata.get("label") or nid)
        self.dg.add_node(
            nid,
            skill=spec.skill,
            inputs=list(spec.inputs),
            metadata=dict(spec.metadata),
            label=label,
        )
        self.label_to_node[label] = nid
        return nid

    def resolve_label(self, ref: str) -> str | None:
        if ref.startswith("n:"):
            return self.label_to_node.get(ref[2:])
        return None

    def extend_from(
        self,
        src_nid: str,
        result: PlannerOutput | AgentResult,
        *,
        registry: SkillRegistry | None = None,
    ) -> list[str]:
        """Splice dynamic successors, internal_successors, and critic auto-insertion."""
        reg = registry or self.registry
        src_skill = str(self.dg.nodes[src_nid].get("skill") or "")
        specs = result.nodes if isinstance(result, PlannerOutput) else list(result.successors)

        added: list[str] = []
        label_to_id: dict[str, str] = dict(self.label_to_node)
        pending: list[tuple[str, list[str]]] = []

        for spec in specs:
            label = (spec.metadata or {}).get("label")
            new_id = self.add_node_from_spec(NodeSpec(skill=spec.skill, inputs=[], metadata=spec.metadata))
            added.append(new_id)
            if isinstance(label, str) and label:
                label_to_id[label] = new_id
            pending.append((new_id, list(spec.inputs)))

        for new_id, raw_inputs in pending:
            resolved = self._resolve_input_refs(src_nid, raw_inputs, label_to_id)
            self.dg.nodes[new_id]["inputs"] = resolved
            wired = False
            for inp in resolved:
                if inp.startswith("n:") and inp in self.dg.nodes:
                    self.dg.add_edge(inp, new_id)
                    wired = True
            if not wired:
                self.dg.add_edge(src_nid, new_id)

        for nid in list(added):
            skill = str(self.dg.nodes[nid].get("skill") or "")
            try:
                successors = reg.get(skill).internal_successors
            except KeyError:
                continue
            label = str(self.dg.nodes[nid].get("label") or nid)
            for child_skill in successors:
                existing = [s for s in self.dg.successors(nid) if self.dg.nodes[s].get("skill") == child_skill]
                if existing:
                    continue
                spec = NodeSpec(
                    skill=child_skill,
                    inputs=[f"n:{label}"],
                    metadata={"label": f"{child_skill}_{label}", "_auto": True},
                )
                sid = self.add_node_from_spec(spec)
                self.dg.add_edge(nid, sid)
                added.append(sid)

        inserted_critics = self._splice_critics_for_added(added)
        added.extend(inserted_critics)
        return added

    @staticmethod
    def _resolve_input_refs(src_nid: str, raw_inputs: list[str] | None, label_to_id: dict[str, str]) -> list[str]:
        resolved: list[str] = []
        for inp in raw_inputs or [src_nid]:
            if inp.startswith("n:"):
                suffix = inp[2:]
                if suffix in label_to_id:
                    resolved.append(label_to_id[suffix])
                    continue
                if suffix.isdigit() and inp in label_to_id.values():
                    resolved.append(inp)
                    continue
                if inp in label_to_id.values():
                    resolved.append(inp)
                    continue
            if inp in label_to_id:
                resolved.append(label_to_id[inp])
                continue
            if inp in ("USER_QUERY",) or inp.startswith("art:"):
                resolved.append(inp)
                continue
            resolved.append(src_nid)
        return resolved

    def _insert_critic_on_edge(self, src_id: str, child_id: str) -> str | None:
        if self.dg.nodes[child_id].get("skill") == "critic":
            return None
        if any(self.dg.nodes[s].get("skill") == "critic" for s in self.dg.predecessors(child_id)):
            return None
        src_label = self.dg.nodes[src_id].get("label", src_id)
        src_meta = self.dg.nodes[src_id].get("metadata") or {}
        child_meta = self.dg.nodes[child_id].get("metadata") or {}
        critic_meta: dict[str, Any] = {
            "label": f"critic_{src_id}_{child_id}",
            "target": child_id,
            "child": child_id,
            "question": child_meta.get("question") or src_meta.get("question") or "",
        }
        rk = src_meta.get("required_keys")
        if rk:
            critic_meta["required_keys"] = ",".join(rk) if isinstance(rk, list) else str(rk)
        if src_meta.get("syllable_pattern"):
            critic_meta["syllable_pattern"] = src_meta["syllable_pattern"]
        critic_spec = NodeSpec(skill="critic", inputs=["USER_QUERY", src_id], metadata=critic_meta)
        critic_id = self.add_node_from_spec(critic_spec)
        if self.dg.has_edge(src_id, child_id):
            self.dg.remove_edge(src_id, child_id)
        self.dg.add_edge(src_id, critic_id)
        self.dg.add_edge(critic_id, child_id)
        return critic_id

    def splice_critics_on_outgoing_edges(self, source_ids: list[str]) -> list[str]:
        """Insert critic on each outgoing edge from critic:true producers (pre-planned or dynamic)."""
        inserted: list[str] = []
        for src_id in source_ids:
            skill = str(self.dg.nodes[src_id].get("skill") or "")
            if not self.registry.has_critic_splice(skill):
                continue
            for child_id in list(self.dg.successors(src_id)):
                if self.dg.nodes[child_id].get("skill") == "critic":
                    continue
                cid = self._insert_critic_on_edge(src_id, child_id)
                if cid:
                    inserted.append(cid)
        return inserted

    def _splice_critics_for_added(self, node_ids: list[str]) -> list[str]:
        """Insert critic on edges from critic:true producers to their children."""
        sources = [
            nid
            for nid in node_ids
            if self.registry.has_critic_splice(str(self.dg.nodes[nid].get("skill") or ""))
        ]
        return self.splice_critics_on_outgoing_edges(sources)

    extend_from_planner = extend_from

    def _predecessor_satisfied(self, pred_id: str, node_id: str, states: dict[str, NodeState]) -> bool:
        st = states.get(pred_id)
        if not st:
            return False
        if st.status in (NodeStatus.complete, NodeStatus.skipped):
            return True
        meta = self.dg.nodes.get(node_id, {}).get("metadata") or {}
        if meta.get("recovery") and st.status == NodeStatus.failed and self.dg.has_edge(pred_id, node_id):
            return True
        return False

    def ready_nodes(self, states: dict[str, NodeState]) -> list[str]:
        ready: list[str] = []
        for nid in nx.topological_sort(self.dg):
            st = states.get(nid)
            if st is None or st.status != NodeStatus.pending:
                continue
            preds = list(self.dg.predecessors(nid))
            if all(self._predecessor_satisfied(p, nid, states) for p in preds):
                ready.append(nid)
        return ready

    def terminal_complete(self, states: dict[str, NodeState]) -> bool:
        for nid, data in self.dg.nodes(data=True):
            if data.get("skill") == "formatter" and states.get(nid, NodeState(node_id=nid, skill="")).status == NodeStatus.complete:
                return True
        return False

    def formatter_output(self, states: dict[str, NodeState]) -> str | None:
        for nid, data in self.dg.nodes(data=True):
            if data.get("skill") == "formatter":
                st = states.get(nid)
                if st and st.status == NodeStatus.complete and st.output:
                    try:
                        parsed = json.loads(st.output)
                        if isinstance(parsed, dict):
                            for key in ("text", "final_answer", "summary"):
                                if parsed.get(key):
                                    return str(parsed[key])
                    except json.JSONDecodeError:
                        pass
                    return st.output
        return None


class Executor:
    """Walk the DAG: run every ready node concurrently; persist after each wave."""

    def __init__(
        self,
        *,
        registry: SkillRegistry | None = None,
        memory: MemoryManager | None = None,
        action: ActionActuator | None = None,
        artifacts: ArtifactStore | None = None,
    ) -> None:
        self.registry = registry or SkillRegistry()
        self.memory = memory or MemoryManager()
        self.action = action or ActionActuator()
        self.artifacts = artifacts or ArtifactStore()
        self.graph = Graph(self.registry)
        self.states: dict[str, NodeState] = {}
        self.store: SessionStore | None = None
        self.llm: SkillLLMClient | None = None
        self._memory_hits: list[MemoryItem] = []
        self._memory_hits_text = ""
        self._run_start = 0.0
        self._fatal_error: str | None = None
        self._recovered_branches: dict[str, bool] = {}
        self._critic_fail_cap_hit: list[str] = []
        self._is_resume = False

    async def run(self, user_query: str, session_id: str | None = None) -> str:
        sid = session_id or f"dag-{uuid.uuid4().hex[:8]}"
        self.store = SessionStore(sid)
        if self.store.exists():
            raise SessionLoadError(
                f"Session {sid} already exists — resume with: scripts/dag/run_query.py --resume {sid}"
            )
        self.llm = SkillLLMClient(sid)
        self._run_start = time.monotonic()
        self.graph.user_query = user_query
        self._bootstrap(user_query, sid)
        return await self._execute_loop()

    async def resume(self, session_id: str) -> str:
        """Load persisted session; reset running→pending; continue from ready_nodes."""
        self.store = SessionStore(session_id)
        if not self.store.exists():
            raise SessionLoadError(f"No persisted session at {self.store.root}")
        self.llm = SkillLLMClient(session_id)
        self._run_start = time.monotonic()
        self._is_resume = True
        try:
            self._load_session()
            logger.info(
                f"[UI_SESSION_JSON] {json.dumps({'session_id': session_id, 'resumed': True}, ensure_ascii=False)}"
            )
            logger.info(f"[dag] Resumed session {session_id} (running nodes reset to pending)")
            return await self._execute_loop()
        finally:
            self._is_resume = False

    def _bootstrap(self, user_query: str, sid: str) -> None:
        assert self.store is not None
        logger.info(f"[UI_SESSION_JSON] {json.dumps({'session_id': sid}, ensure_ascii=False)}")
        self.store.save_query(user_query)
        root = NodeSpec(skill="planner", inputs=["USER_QUERY"], metadata={"label": "planner"})
        nid = self.graph.add_node_from_spec(root, node_id="n:1")
        self.graph._counter = 1
        self.states[nid] = NodeState(node_id=nid, skill="planner", inputs=["USER_QUERY"], metadata={"label": "planner"})
        self._persist()

    def _load_session(self) -> None:
        assert self.store is not None
        self.graph.dg = self.store.load_graph()
        self.store.reset_running_to_pending()
        self.states = self.store.load_all_node_states()
        self._hydrate_missing_states()
        self.graph.user_query = self.store.load_query()
        self.graph.label_to_node.clear()
        for nid, data in self.graph.dg.nodes(data=True):
            self.graph.label_to_node[str(data.get("label") or nid)] = nid
        nums = [int(n.split(":")[1]) for n in self.graph.dg.nodes if str(n).startswith("n:")]
        self.graph._counter = max(nums) if nums else 0

    def _hydrate_missing_states(self) -> None:
        """SIGKILL mid-wave: graph.json may list coder/formatter before node files exist — keep them pending."""
        assert self.store is not None
        for nid, data in self.graph.dg.nodes(data=True):
            if nid in self.states:
                continue
            st = NodeState(
                node_id=nid,
                skill=str(data.get("skill") or "?"),
                inputs=list(data.get("inputs") or []),
                metadata=dict(data.get("metadata") or {}),
                status=NodeStatus.pending,
            )
            self.states[nid] = st
            self.store.save_node_state(st)
            logger.info(f"[dag] Hydrated missing node state {nid} ({st.skill}) as pending")

    def _restore_memory_hits_from_store(self) -> bool:
        assert self.store is not None
        rows = self.store.load_memory_hits()
        if not rows:
            return False
        hits: list[MemoryItem] = []
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            try:
                hits.append(MemoryItem.model_validate(raw))
            except Exception:
                continue
        if not hits:
            return False
        self._memory_hits = hits
        self._memory_hits_text = format_memory_hits(hits)
        logger.info(f"[dag] Restored {len(hits)} memory hits from session snapshot (resume)")
        return True

    async def _execute_loop(self) -> str:
        assert self.store is not None
        try:
            await self._prime_memory()

            while not self.graph.terminal_complete(self.states):
                if self._fatal_error:
                    break
                ready = self.graph.ready_nodes(self.states)
                if not ready:
                    pending = [n for n, s in self.states.items() if s.status == NodeStatus.pending]
                    if pending and not self._fatal_error:
                        raise RuntimeError(f"DAG deadlock — pending nodes with no ready predecessors: {pending}")
                    break

                _log_node("wave", "executor", f"running {len(ready)} nodes: {ready}")
                await self._run_wave(ready)
                self._persist()
        except asyncio.CancelledError:
            self._revert_running_to_pending()
            raise

        if self._fatal_error:
            raise RuntimeError(self._fatal_error)

        if self._critic_fail_cap_hit:
            logger.warning(
                f"[dag] critic-fail cap hit on {len(self._critic_fail_cap_hit)} branch(es): "
                f"{', '.join(self._critic_fail_cap_hit)}"
            )

        out = self.graph.formatter_output(self.states)
        if not out:
            raise RuntimeError("Formatter did not produce output")
        elapsed = time.monotonic() - self._run_start
        logger.info(f"[dag] RUN_COMPLETE nodes={len(self.states)} wall={elapsed:.2f}s")
        return out

    async def _prime_memory(self) -> None:
        """Single memory.read at session start — same hits threaded to every skill."""
        if self._is_resume and self._restore_memory_hits_from_store():
            return
        self._memory_hits = await asyncio.to_thread(
            self.memory.read,
            self.graph.user_query,
            [],
            top_k=12,
        )
        self._memory_hits_text = format_memory_hits(self._memory_hits)
        if not self._is_resume:
            await asyncio.to_thread(
                self.memory.remember,
                f"DAG run: {self.graph.user_query[:500]}",
                source="dag_session",
                run_id=self.store.session_id if self.store else "",
            )
        if self.store and not self._is_resume:
            await asyncio.to_thread(self.store.save_memory_hits, self._memory_hits)
        elif self.store and self._is_resume and self._memory_hits:
            await asyncio.to_thread(self.store.save_memory_hits, self._memory_hits)

    async def _run_wave(self, ready: list[str]) -> None:
        """Run a parallel wave; on cancel, revert in-flight nodes to pending (resume-safe)."""
        tasks = [asyncio.create_task(self._run_one(nid)) for nid in ready]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._revert_running_to_pending()
            raise

    def _revert_running_to_pending(self) -> None:
        """After wave cancel / resume — never leave nodes stuck as running on disk."""
        for st in self.states.values():
            if st.status != NodeStatus.running:
                continue
            st.status = NodeStatus.pending
            st.started_at = None
            st.finished_at = None
            self._flush_node_state(st)

    async def _run_one(self, node_id: str) -> None:
        """Dispatch one node — identical path for Planner, Researcher, Coder, Formatter."""
        await self._run_node(node_id)

    def _graph_nodes_view(self) -> dict[str, Any]:
        view: dict[str, Any] = {}
        for nid, data in self.graph.dg.nodes(data=True):
            view[nid] = dict(data)
            if "result" not in view[nid] and nid in self.states:
                st = self.states[nid]
                if st.status == NodeStatus.complete:
                    view[nid]["result"] = AgentResult(
                        success=True,
                        agent_name=st.skill,
                        status=st.status.value,
                        output=self._parse_stored_output(st.output),
                        artifact_id=st.artifact_id,
                        error=st.error,
                        elapsed_s=st.elapsed_s,
                    )
        return view

    @staticmethod
    def _parse_stored_output(raw: str | None) -> str | dict[str, Any] | None:
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    def _failure_report_str(self, state: NodeState) -> str | None:
        report = state.metadata.get("failure_report")
        if isinstance(report, dict):
            err = report.get("error")
            if err:
                return str(err)
        if state.metadata.get("recovery"):
            ctx = state.metadata.get("question")
            if ctx:
                return str(ctx)
        return None

    def _skill_run_context(self, state: NodeState) -> SkillRunContext:
        assert self.llm is not None
        return SkillRunContext(
            user_query=self.graph.user_query,
            session_id=self.store.session_id if self.store else "",
            memory_hits=self._memory_hits,
            llm=self.llm,
            action=self.action,
            artifacts=self.artifacts,
            label_map=dict(self.graph.label_to_node),
            failure_report=self._failure_report_str(state),
        )

    async def _run_node(self, node_id: str) -> None:
        data = self.graph.dg.nodes[node_id]
        skill_name = str(data.get("skill") or "")
        skill = self.registry.get(skill_name)
        state = self.states.setdefault(
            node_id,
            NodeState(
                node_id=node_id,
                skill=skill_name,
                inputs=list(data.get("inputs") or []),
                metadata=dict(data.get("metadata") or {}),
            ),
        )
        state.status = NodeStatus.running
        state.started_at = time.monotonic()
        t0 = state.started_at
        self._flush_node_state(state)
        _log_node(node_id, skill_name, "start")

        graph_nodes = self._graph_nodes_view()
        ctx = self._skill_run_context(state)

        try:
            result, prompt = await run_skill(
                skill,
                node_id,
                graph_nodes,
                ctx.session_id,
                self.graph.user_query,
                ctx.failure_report,
                ctx,
            )
            self.graph.dg.nodes[node_id]["result"] = result
            state.output = _state_output_from_result(result)
            state.artifact_id = result.artifact_id
            state.elapsed_s = result.elapsed_s

            if not result.success:
                await self._handle_node_failure(node_id, skill_name, result.error or "skill failed", state)
                return

            state.status = NodeStatus.complete

            if self.registry.has_critic_splice(skill_name):
                critic_ids = self.graph.splice_critics_on_outgoing_edges([node_id])
                for cid in critic_ids:
                    nd = self.graph.dg.nodes[cid]
                    self.states[cid] = NodeState(
                        node_id=cid,
                        skill=str(nd.get("skill")),
                        inputs=list(nd.get("inputs") or []),
                        metadata=dict(nd.get("metadata") or {}),
                    )
                if critic_ids:
                    _log_node(node_id, skill_name, f"auto-spliced {len(critic_ids)} critic(s) on outgoing edges")
                    self._persist()

            if skill_name == "critic":
                raw = state.output or json.dumps(result.output)
                if handle_critic_verdict(
                    node_id,
                    raw,
                    self.graph,
                    self.states,
                    self._recovered_branches,
                    self._critic_fail_cap_hit,
                    fail_cap=CRITIC_FAIL_CAP,
                ):
                    return

            if result.successors and skill.extends_graph:
                successors = list(result.successors)
                if skill_name == "planner":
                    successors = coerce_planner_successors(self.graph.user_query, successors)
                result.successors = successors
                created = self.graph.extend_from(node_id, result)
                for nid in created:
                    nd = self.graph.dg.nodes[nid]
                    self.states[nid] = NodeState(
                        node_id=nid,
                        skill=str(nd.get("skill")),
                        inputs=list(nd.get("inputs") or []),
                        metadata=dict(nd.get("metadata") or {}),
                    )
                _log_node(node_id, skill_name, f"extended graph with {len(created)} nodes")
                if created:
                    self._persist()
        except asyncio.CancelledError:
            if state.status == NodeStatus.running:
                state.status = NodeStatus.pending
                state.started_at = None
            self._flush_node_state(state)
            raise
        except Exception as e:
            await self._handle_node_failure(node_id, skill_name, str(e), state)
            return
        finally:
            state.finished_at = time.monotonic()
            if state.elapsed_s is None:
                state.elapsed_s = state.finished_at - (t0 or state.finished_at)
            self._flush_node_state(state)
            _log_node(node_id, skill_name, f"done in {state.elapsed_s:.2f}s")

    async def _handle_critic(self, critic_id: str, raw_output: str) -> None:
        """Thin wrapper for tests — delegates to recovery.handle_critic_verdict."""
        handle_critic_verdict(
            critic_id,
            raw_output,
            self.graph,
            self.states,
            self._recovered_branches,
            self._critic_fail_cap_hit,
            fail_cap=CRITIC_FAIL_CAP,
        )

    async def _handle_node_failure(
        self,
        node_id: str,
        skill_name: str,
        error: str,
        state: NodeState,
    ) -> None:
        state.status = NodeStatus.failed
        state.error = error
        self._flush_node_state(state)
        decision = plan_recovery(
            failed_skill=skill_name,
            error_text=error,
            failed_node_id=node_id,
        )
        logger.error(f"[dag] {node_id} ({skill_name}) recovery={decision.action} reason={decision.reason}: {error[:300]}")

        if decision.action == "skip":
            if skill_name == "browser" and decision.reason in _BROWSER_GRACEFUL_SKIP:
                self._skip_pending_descendants(node_id)
                self._queue_browser_failure_formatter(node_id, state, decision)
                return
            self._fatal_error = decision.note or error
            return
        if self.graph.dg.number_of_nodes() >= MAX_NODES:
            self._fatal_error = f"MAX_NODES cap ({MAX_NODES}) reached"
            return
        report = decision.failure_report
        recovery_inputs, reused = recovery_planner_inputs(
            self.graph,
            self.states,
            failed_node_id=node_id,
        )
        recovery = NodeSpec(
            skill="planner",
            inputs=recovery_inputs,
            metadata={
                "label": f"recovery_{node_id}",
                "recovery": True,
                "failure_report": report,
            },
        )
        rid = self.graph.add_node_from_spec(recovery)
        self.graph.dg.add_edge(node_id, rid)
        self.states[rid] = NodeState(
            node_id=rid,
            skill="planner",
            inputs=list(recovery.inputs),
            metadata=dict(recovery.metadata),
        )
        self._skip_pending_descendants(node_id, except_ids={rid})
        reused_txt = ", ".join(reused) if reused else "(none)"
        _log_node(
            node_id,
            skill_name,
            f"↪ recovery ({decision.reason}): planner node {rid} queued for {node_id}; "
            f"reusing {len(reused)} prior result(s): {reused_txt}",
        )
        self._persist()

    def _skip_pending_descendants(self, root_id: str, *, except_ids: set[str] | None = None) -> None:
        """After replan, abandon the old branch so recovery planner can run without deadlock."""
        skip = except_ids or set()
        for nid in nx.descendants(self.graph.dg, root_id):
            if nid in skip:
                continue
            st = self.states.get(nid)
            if st and st.status == NodeStatus.pending:
                st.status = NodeStatus.skipped
                self._flush_node_state(st)

    def _queue_browser_failure_formatter(
        self,
        failed_nid: str,
        state: NodeState,
        decision: Any,
    ) -> None:
        """Queue a formatter instead of aborting the whole run when the browser cascade is exhausted."""
        for nid, data in self.graph.dg.nodes(data=True):
            if data.get("skill") != "formatter":
                continue
            st = self.states.get(nid)
            if st and st.status in (NodeStatus.pending, NodeStatus.running):
                st.status = NodeStatus.skipped
                self._flush_node_state(st)

        report = dict(decision.failure_report or {})
        if state.output:
            report.setdefault("partial_output", state.output)
        if state.error:
            report.setdefault("error", state.error)

        spec = NodeSpec(
            skill="formatter",
            inputs=["USER_QUERY"],
            metadata={
                "label": f"browser_graceful_{failed_nid.replace(':', '_')}",
                "question": (
                    "The live browser agent could not fully complete this task. "
                    "Using the user goal and any partial browser notes, produce the best "
                    "comparison table you can and clearly state what could not be verified on-site."
                ),
                "failure_report": report,
            },
        )
        fid = self.graph.add_node_from_spec(spec)
        self.states[fid] = NodeState(
            node_id=fid,
            skill="formatter",
            inputs=list(spec.inputs),
            metadata=dict(spec.metadata),
        )
        _log_node(
            failed_nid,
            "browser",
            f"↪ graceful formatter {fid} queued ({decision.reason})",
        )
        self._persist()

    def _sync_graph_from_states(self) -> None:
        for nid, st in self.states.items():
            if nid not in self.graph.dg:
                continue
            self.graph.dg.nodes[nid]["result"] = AgentResult(
                status=st.status.value,
                output=st.output,
                artifact_id=st.artifact_id,
                error=st.error,
                elapsed_s=st.elapsed_s,
            )

    def _persist(self) -> None:
        assert self.store is not None
        self._sync_graph_from_states()
        self.store.save_graph(self.graph.dg)
        for st in self.states.values():
            self.store.save_node_state(st)

    def _flush_node_state(self, state: NodeState) -> None:
        """Write one node to disk immediately so /api/dag/graph shows run/wait colors during a wave."""
        assert self.store is not None
        if state.node_id in self.graph.dg:
            self.graph.dg.nodes[state.node_id]["result"] = AgentResult(
                status=state.status.value,
                output=state.output,
                artifact_id=state.artifact_id,
                error=state.error,
                elapsed_s=state.elapsed_s,
            )
            self.store.save_graph(self.graph.dg)
        self.store.save_node_state(state)

    async def aclose(self) -> None:
        await self.action.aclose()


def log_final_answer(answer: str) -> None:
    """Emit markers the web UI SSE handler listens for."""
    text = answer or ""
    logger.success(f">>> FINAL ANSWER <<<\n{text}")
    try:
        logger.info("[UI_RESULT_JSON] " + json.dumps({"text": text}, ensure_ascii=False))
    except (TypeError, ValueError):
        logger.info("[UI_RESULT_JSON] " + json.dumps({"text": "(Answer could not be encoded.)"}))


class DagAgent:
    """Web UI entry when AGENT_MODE=dag (default)."""

    async def run(self, user_query: str, session_id: str | None = None) -> None:
        """One fresh Executor (and MCP session) per UI run — avoids stale connections after aclose."""
        executor = Executor()
        try:
            answer = await executor.run(user_query, session_id=session_id)
            log_final_answer(answer)
            logger.info("[agent] RUN_COMPLETE reason=dag_done")
        finally:
            try:
                await executor.aclose()
            except Exception as e:
                logger.warning(f"[dag] cleanup failed: {e}")

    async def resume(self, session_id: str) -> None:
        """Continue a persisted session (SIGKILL-safe): running → pending, then execute."""
        executor = Executor()
        try:
            answer = await executor.resume(session_id.strip())
            log_final_answer(answer)
            logger.info("[agent] RUN_COMPLETE reason=dag_resumed")
        finally:
            try:
                await executor.aclose()
            except Exception as e:
                logger.warning(f"[dag] cleanup failed: {e}")

    async def aclose(self) -> None:
        return
