"""Failure classification and recovery policy for the DAG orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from .dag_schemas import AgentResult, NodeSpec, NodeState
    from .flow import Graph

FailureKind = Literal["transient", "validation_error", "upstream_failure"]
RecoveryAction = Literal["skip", "replan"]

_TRANSIENT_KEYWORDS = (
    "503",
    "502",
    "504",
    "timeout",
    "connection",
    "bad gateway",
    "gateway timeout",
    "ConnectionError",
    "HTTPStatusError",
    "service unavailable",
)

_VALIDATION_KEYWORDS = (
    "malformed",
    "ValidationError",
    "validation error",
)


def classify_failure(error_text: str) -> FailureKind:
    """Map error text to recovery policy (keyword matcher — pinned by unit tests)."""
    text = error_text or ""
    lower = text.lower()
    for kw in _TRANSIENT_KEYWORDS:
        if kw in text or kw.lower() in lower:
            return "transient"
    for kw in _VALIDATION_KEYWORDS:
        if kw in text or kw.lower() in lower:
            return "validation_error"
    return "upstream_failure"


@dataclass
class RecoveryDecision:
    action: RecoveryAction
    reason: str
    note: str = ""
    failure_report: dict[str, Any] = field(default_factory=dict)


def plan_recovery(
    *,
    failed_skill: str,
    error_text: str,
    failed_node_id: str,
) -> RecoveryDecision:
    """Policy gate for node failure — transient/validation skip; upstream replans."""
    lower = (error_text or "").lower()
    if "no module named" in lower or "modulenotfounderror" in lower:
        return RecoveryDecision(
            action="skip",
            reason="missing_dependency",
            note=error_text,
            failure_report={"node_id": failed_node_id, "skill": failed_skill, "error": error_text},
        )
    if failed_skill == "browser" and (
        "executable doesn't exist" in lower
        or "playwright install" in lower
        or "playwright chromium is not installed" in lower
    ):
        return RecoveryDecision(
            action="skip",
            reason="missing_dependency",
            note=error_text,
            failure_report={"node_id": failed_node_id, "skill": failed_skill, "error": error_text},
        )
    if failed_skill == "browser" and (
        "cascade exhausted" in lower
        or "browser cascade failed" in lower
        or "all browser layers failed" in lower
    ):
        return RecoveryDecision(
            action="skip",
            reason="browser_exhausted",
            note=error_text,
            failure_report={"node_id": failed_node_id, "skill": failed_skill, "error": error_text},
        )
    kind = classify_failure(error_text)
    if kind == "transient":
        return RecoveryDecision(action="skip", reason=kind, note=error_text)
    if kind == "validation_error":
        return RecoveryDecision(action="skip", reason=kind, note=error_text)
    if failed_skill == "planner":
        return RecoveryDecision(
            action="skip",
            reason="planner_failure",
            note=error_text,
            failure_report={"node_id": failed_node_id, "skill": failed_skill, "error": error_text},
        )
    return RecoveryDecision(
        action="replan",
        reason=kind,
        note=error_text,
        failure_report={"node_id": failed_node_id, "skill": failed_skill, "error": error_text},
    )


def recovery_planner_inputs(
    graph: Graph,
    states: dict[str, NodeState],
    *,
    failed_node_id: str,
    extra_refs: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Build recovery Planner inputs: USER_QUERY + completed sibling n:* refs + failure context."""
    from .dag_schemas import NodeStatus

    inputs: list[str] = ["USER_QUERY"]
    reused: list[str] = []
    for nid, st in states.items():
        if nid == failed_node_id:
            continue
        if st.status != NodeStatus.complete:
            continue
        skill = str(graph.dg.nodes[nid].get("skill") or "")
        if skill in ("planner", "critic"):
            continue
        label = str(graph.dg.nodes[nid].get("label") or nid)
        ref = f"n:{label}"
        if ref not in inputs:
            inputs.append(ref)
            reused.append(ref)
    for ref in extra_refs or []:
        if ref not in inputs:
            inputs.append(ref)
    f_label = str(graph.dg.nodes[failed_node_id].get("label") or failed_node_id)
    fail_ref = f"n:{f_label}"
    if fail_ref not in inputs:
        inputs.append(fail_ref)
    return inputs, reused


def handle_critic_verdict(
    critic_id: str,
    raw_output: str,
    graph: Graph,
    states: dict[str, NodeState],
    recovered_branches: dict[str, bool],
    critic_fail_cap_hit: list[str],
    *,
    fail_cap: int = 1,
) -> bool:
    """Process critic fail/pass. Returns True when fail was handled (no graph extend)."""
    from .dag_schemas import CriticVerdict, NodeSpec, NodeState as NS, NodeStatus
    from .llm_retry import loads_json_lenient

    data = loads_json_lenient(raw_output)
    verdict = CriticVerdict.model_validate(data)
    if verdict.verdict == "pass":
        return False

    meta = graph.dg.nodes[critic_id].get("metadata") or {}
    target = str(meta.get("target") or meta.get("child") or "")
    succs = list(graph.dg.successors(critic_id))
    if not target and succs:
        target = succs[0]

    for sid in succs:
        if sid in states:
            states[sid].status = NodeStatus.skipped

    count = graph.critic_fail_counts.get(target, 0) + 1
    graph.critic_fail_counts[target] = count
    if count > fail_cap:
        critic_fail_cap_hit.append(target or critic_id)
        recovered_branches[target] = True
        return True

    c_label = graph.dg.nodes[critic_id].get("label", critic_id)
    recovery_inputs, _reused = recovery_planner_inputs(
        graph,
        states,
        failed_node_id=critic_id,
        extra_refs=[f"n:{c_label}"],
    )
    recovery = NodeSpec(
        skill="planner",
        inputs=recovery_inputs,
        metadata={
            "label": f"recovery_{count}",
            "recovery": True,
            "question": verdict.rationale,
            "failure_report": {"critic_id": critic_id, "target": target, "rationale": verdict.rationale},
        },
    )
    rid = graph.add_node_from_spec(recovery)
    graph.dg.add_edge(critic_id, rid)
    states[rid] = NS(
        node_id=rid,
        skill="planner",
        inputs=list(recovery.inputs),
        metadata=dict(recovery.metadata),
    )
    recovered_branches[target] = True
    return True
