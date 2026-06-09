"""Pydantic contracts for DAG orchestration."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class NodeStatus(str, Enum):
    pending = "pending"
    running = "running"
    complete = "complete"
    failed = "failed"
    skipped = "skipped"


class NodeSpec(BaseModel):
    """One node emitted by the Planner — validated before graph extension."""

    model_config = ConfigDict(extra="ignore")

    skill: str
    inputs: list[str] = Field(default_factory=list)
    metadata_json: str = Field(
        default="{}",
        description="JSON object string (label, question, required_keys, verbatim_json, …).",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_metadata(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "metadata" in data and "metadata_json" not in data:
            meta = data.pop("metadata")
            if isinstance(meta, dict):
                data["metadata_json"] = json.dumps(meta, ensure_ascii=False)
            elif isinstance(meta, str):
                data["metadata_json"] = meta
        return data

    @property
    def metadata(self) -> dict[str, Any]:
        try:
            parsed = json.loads(self.metadata_json or "{}")
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        base = super().model_dump(**kwargs)
        base["metadata"] = self.metadata
        return base


class PlannerOutput(BaseModel):
    """Planner JSON output — the program the Executor runs next."""

    model_config = ConfigDict(extra="ignore")

    rationale: str = ""
    nodes: list[NodeSpec] = Field(default_factory=list)


class CriticVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")

    verdict: Literal["pass", "fail"]
    rationale: str = ""


class CoderOutput(BaseModel):
    """Coder emits executable code plus a natural-language summary for the Formatter."""

    model_config = ConfigDict(extra="ignore")

    code: str
    summary: str = ""


BrowserPath = Literal["extract", "deterministic", "a11y", "vision", "gateway_blocked", "failed"]
BrowserErrorCode = Literal[
    "gateway_blocked",
    "extraction_failed",
    "interaction_failed",
    "timeout",
    "vlm_unavailable",
]


class BrowserOutput(BaseModel):
    """Structured Browser skill result — consumed by distiller and replay viewer."""

    model_config = ConfigDict(extra="ignore")

    url: str
    goal: str
    path: BrowserPath
    turns: int = 0
    content: str | None = None
    actions: list[dict[str, Any]] = Field(default_factory=list)
    final_url: str | None = None
    elapsed_s: float | None = None
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class AgentResult(BaseModel):
    """Per-node execution result — typed payload on graph edges and in persistence."""

    model_config = ConfigDict(extra="ignore")

    success: bool = True
    agent_name: str = ""
    status: str = "pending"
    output: str | dict[str, Any] | None = None
    artifact_id: str | None = None
    error: str | None = None
    error_code: BrowserErrorCode | str | None = None
    elapsed_s: float | None = None
    successors: list[NodeSpec] = Field(default_factory=list)


class NodeState(BaseModel):
    """Persisted per-node execution state under ``sessions/<id>/nodes/``."""

    model_config = ConfigDict(extra="ignore")

    node_id: str
    skill: str
    status: NodeStatus = NodeStatus.pending
    inputs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    output: str | None = None
    artifact_id: str | None = None
    error: str | None = None
    elapsed_s: float | None = None
    started_at: float | None = None
    finished_at: float | None = None


class SkillConfig(BaseModel):
    """One row from ``agent_config.yaml``."""

    model_config = ConfigDict(extra="ignore")

    prompt: str
    description: str = ""
    tools: list[str] = Field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 4096
    extends_graph: bool = False
    terminal: bool = False
    critic: bool = False
    internal_successors: list[str] = Field(default_factory=list)
    provider: str | None = None
    model: str | None = None
