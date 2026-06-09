"""
Pydantic v2 contracts for cross-role boundaries (Memory, Perception, Decision, Action).

Core shapes: MemoryItem, Artifact metadata, Goal, Observation, ToolCall, DecisionOutput.
Legacy commerce catalog types (SQLite) remain for Indian PDP caching.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

# --- Memory -------------------------------------------------------------------

MemoryKind = Literal["fact", "preference", "tool_outcome", "scratchpad"]


class MemoryItem(BaseModel):
    """One durable or episodic row in ``state/memory.json``."""

    model_config = ConfigDict(extra="ignore")

    id: str
    kind: MemoryKind
    keywords: list[str] = Field(default_factory=list)
    descriptor: str = ""
    value: dict[str, Any] = Field(default_factory=dict)
    artifact_id: str | None = None
    embedding: list[float] | None = None
    source: str = ""
    run_id: str = ""
    goal_id: str | None = None
    confidence: float = 1.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ArtifactRecord(BaseModel):
    """Metadata sidecar for content-addressable blobs under ``state/artifacts/``."""

    id: str
    content_type: str = "application/octet-stream"
    size_bytes: int = 0
    source: str = ""
    descriptor: str = ""


class Goal(BaseModel):
    """Planner goal with stable ``id`` assigned by the outer loop (not by the LLM)."""

    id: str
    text: str
    done: bool = False
    attach_artifact_id: str | None = None


class Observation(BaseModel):
    """Perception output: ordered goals. Identity is list position + stable ``Goal.id``."""

    goals: list[Goal] = Field(default_factory=list)

    def all_done(self) -> bool:
        return bool(self.goals) and all(g.done for g in self.goals)

    def next_unfinished(self) -> Goal | None:
        for g in self.goals:
            if not g.done:
                return g
        return None


class ToolCall(BaseModel):
    """Single MCP dispatch contract."""

    name: str
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


class DecisionOutput(BaseModel):
    """Planner branch: prefer ``tool_call`` when present; otherwise ``answer``."""

    answer: str | None = None
    tool_call: ToolCall | None = None

    def resolved(self) -> tuple[str | None, ToolCall | None]:
        if self.tool_call is not None and str(self.tool_call.name).strip():
            return None, self.tool_call
        if self.answer is not None:
            return self.answer, None
        return None, None

    @property
    def is_answer(self) -> bool:
        a, t = self.resolved()
        return a is not None and t is None


class DecisionLLMFlat(BaseModel):
    """Flat JSON for Gemini Developer API with explicit reasoning.

    Open-ended maps are **JSON strings**, not ``dict`` fields — Pydantic's ``dict`` schemas
    emit ``additionalProperties``, which the Developer API rejects (Enterprise-only).
    """

    model_config = ConfigDict(extra="ignore")

    reasoning: str = Field(
        description="Step-by-step reasoning explaining the current state, what we learned from history, and why we either call a tool or provide the final answer."
    )
    branch: Literal["answer", "tool"]
    answer_text: str | None = None
    tool_name: str | None = None
    tool_arguments_json: str = Field(
        default="{}",
        description='Tool arguments as one JSON object serialized to a string. Use "{}" for branch answer.',
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_tool_arguments(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "tool_arguments_json" not in data and "tool_arguments" in data:
            ta = data.get("tool_arguments")
            if isinstance(ta, dict):
                try:
                    data["tool_arguments_json"] = json.dumps(ta, ensure_ascii=False)
                except (TypeError, ValueError):
                    data["tool_arguments_json"] = "{}"
            elif isinstance(ta, str):
                data["tool_arguments_json"] = ta
        return data


# --- LLM-facing perception draft (no goal ids; loop merges stable ids) -------

class PerceptionGoalDraft(BaseModel):
    text: str
    done: bool = False
    artifact_index: int | None = Field(
        None,
        description="Index into the enumerated MEMORY HITS list that carry artifact_id; null if none.",
    )


class PerceptionLLMResponse(BaseModel):
    reasoning: str = Field(
        default="",
        description="Step-by-step planning and reconciliation reasoning. Explain what has been completed, what needs to happen next, and if any files/artifacts need to be attached."
    )
    goals: list[PerceptionGoalDraft] = Field(default_factory=list)


# --- LLM-facing memory classification on remember() ---------------------------


class MemoryClassifyLLM(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: MemoryKind
    keywords: list[str] = Field(default_factory=list)
    descriptor: str = ""
    value_json: str = Field(
        default="{}",
        description='Structured payload as one JSON object serialized to a string (e.g. {"text":"..."}).',
    )
    confidence: float = 0.85

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_value(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "value_json" not in data and "value" in data:
            v = data.get("value")
            if isinstance(v, dict):
                try:
                    data["value_json"] = json.dumps(v, ensure_ascii=False)
                except (TypeError, ValueError):
                    data["value_json"] = "{}"
            elif isinstance(v, str):
                data["value_json"] = v
        return data


# --- Partial summary (max iterations) ----------------------------------------


class PartialSummaryMarkdown(BaseModel):
    markdown_answer: str


# --- Commerce DB (optional catalog) -------------------------------------------

class CommerceProduct(BaseModel):
    platform: str
    product_name: str
    base_price: float
    net_price: float
    bank_offers_text: str
    url: str


class CachedProductRow(BaseModel):
    url: str
    platform: str | None = None
    product_name: str | None = None
    base_price: float | None = None
    net_price: float | None = None
    bank_offers_text: str | None = None
    scraped_at: str | None = None


# ``PerceptionModule`` historically imported ``PerceptionState``; keep as alias to ``Observation``.
PerceptionState = Observation