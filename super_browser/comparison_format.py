"""Generic comparison-table parsing, distiller schema, and markdown rendering."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .catalog import load_assignment_queries, min_browser_actions_for_text

_WORD_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

_ROW_ARRAY_KEYS = (
    "rows",
    "items",
    "showtimes",
    "models",
    "tools",
    "institutes",
    "products",
    "entries",
    "results",
    "comparisons",
)

_SUBJECT_ALIASES = (
    "subject",
    "title",
    "name",
    "movie_name",
    "film_title",
    "film",
    "product_name",
    "model_name",
)

_NAME_COLUMN_HINTS = ("movie name", "film name", "product name", "category", "subject", "title")


def _detect_subject_column(columns: list[str]) -> str | None:
    for col in columns:
        low = col.lower().strip()
        if low in _NAME_COLUMN_HINTS:
            return col
        if re.search(r"(movie|film|product|brand|category)\s+name", low):
            return col
        if low in {"subject", "title"}:
            return col
    return None


@dataclass
class ComparisonSpec:
    """Comparison intent inferred from free-text user query."""

    is_comparison: bool = False
    row_count: int = 3
    columns: list[str] = field(default_factory=list)
    column_keys: list[str] = field(default_factory=list)
    subject_column: str | None = None


def _to_snake(label: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", (label or "").lower()).strip("_")
    return text or "value"


def _split_column_list(blob: str) -> list[str]:
    parts = re.split(r"\s*,\s*|\s*\|\s*", blob)
    return [p.strip() for p in parts if p.strip()]


def _extract_columns(query: str) -> list[str]:
    patterns = (
        r"(?:comparison\s+)?table\s+with\s+columns\s*[:\-]\s*([^\n.]+)",
        r"columns?\s*[:\-]\s*([^\n.]+)",
        r"(?:comparison\s+)?table\s*\(([^)]+)\)",
        r"structured\s+comparison\s+table\s*\(([^)]+)\)",
        r"return\s+a\s+(?:comparison\s+)?table\s*\(([^)]+)\)",
        r"return\s+a\s+(?:comparison\s+)?table\s+with\s+columns\s*[:\-]\s*([^\n.]+)",
    )
    for pat in patterns:
        m = re.search(pat, query, re.I)
        if m:
            cols = _split_column_list(m.group(1))
            if cols:
                return cols
    return []


def _extract_row_count(query: str) -> int:
    for pat in (
        r"\bcompare\s+(\d+)\b",
        r"\btop\s+(\d+)\b",
        r"\b(\d+)\s+(?:distinct|different|items|products|theatres|theaters|models|tools|sites|pages|rows|entries)\b",
        r"\b(\d+)\s+[A-Za-z]",
    ):
        m = re.search(pat, query, re.I)
        if m:
            try:
                n = int(m.group(1))
                if 1 <= n <= 20:
                    return n
            except ValueError:
                pass
    for word, n in _WORD_NUMBERS.items():
        if re.search(rf"\b{word}\b", query, re.I):
            return n
    return 3


def parse_comparison_spec(query: str) -> ComparisonSpec:
    """Infer comparison table shape from any user query."""
    blob = (query or "").strip()
    if not blob:
        return ComparisonSpec()

    is_comparison = bool(
        re.search(
            r"\bcompare\b|\bcomparison\s+table\b|\bstructured\s+comparison\b",
            blob,
            re.I,
        )
    ) or min_browser_actions_for_text(blob) > 0

    if not is_comparison:
        return ComparisonSpec(is_comparison=False)

    columns = _extract_columns(blob)
    column_keys = [_to_snake(c) for c in columns]
    return ComparisonSpec(
        is_comparison=True,
        row_count=_extract_row_count(blob),
        columns=columns,
        column_keys=column_keys,
        subject_column=_detect_subject_column(columns),
    )


def match_assignment_query(text: str) -> dict[str, Any] | None:
    """Return corpus row when query text matches a known assignment id."""
    blob = (text or "").strip()
    if not blob:
        return None
    for row in load_assignment_queries():
        corpus_q = str(row.get("query") or "").strip()
        if not corpus_q:
            continue
        qid = str(row.get("id") or "")
        if corpus_q == blob or corpus_q[:140] in blob or blob[:140] in corpus_q:
            return row
        if qid and re.search(rf"\b{re.escape(qid)}\b", blob, re.I):
            return row
    return None


def is_comparison_query(text: str) -> bool:
    return parse_comparison_spec(text).is_comparison


def distiller_metadata_for_query(query: str, row: dict[str, Any] | None = None) -> dict[str, str]:
    """Build distiller metadata from query text (optionally merged with corpus row)."""
    spec = parse_comparison_spec(query)
    if not spec.is_comparison and row:
        spec = parse_comparison_spec(str(row.get("query") or query))
    if not spec.is_comparison:
        return {
            "required_keys": "items",
            "fields": "items: array of comparison rows requested in the user query",
            "question": "Extract structured comparison rows from the browser output.",
            "formatter_hint": "Render the comparison as a markdown table matching the user query.",
        }

    keys = spec.column_keys or ["value"]
    col_text = ", ".join(spec.columns) if spec.columns else ", ".join(keys)
    row_shape = ", ".join(f"{k}: <value>" for k in keys)
    required = "subject,context,rows"
    return {
        "required_keys": required,
        "fields": (
            f"subject (shared title/name if the query names one), "
            f"context (object with city/location/site when mentioned), "
            f"rows: array of {spec.row_count} objects each with keys: {row_shape}"
        ),
        "question": (
            f"Extract a structured comparison from upstream browser text. "
            f"rows must have up to {spec.row_count} entries with fields matching: {col_text}. "
            "Use only values visible upstream — do not invent data. "
            "Prefer LIVE_URL, PRICING_SNIPPETS, and VISIBLE_TEXT sections over general knowledge."
        ),
        "formatter_hint": (
            f"Markdown table with columns: {' | '.join(spec.columns or keys)}. "
            f"Include {spec.row_count} data rows when upstream provides them."
        ),
    }


def comparison_browser_goal_suffix(query: str) -> str:
    spec = parse_comparison_spec(query)
    if not spec.is_comparison:
        return ""
    cols = ", ".join(spec.columns) if spec.columns else "all comparison fields from the query"
    return (
        f"Capture every column needed for the comparison table ({cols}). "
        f"Navigate until {spec.row_count} distinct items or pages are visible."
    )


def browser_goal_suffix(row: dict[str, Any]) -> str:
    return comparison_browser_goal_suffix(str(row.get("query") or ""))


def enrich_planner_nodes(user_query: str, successors: list[Any]) -> list[Any]:
    """Patch planner-emitted nodes with query-derived comparison schemas."""
    from .dag_schemas import NodeSpec

    spec = parse_comparison_spec(user_query)
    if not spec.is_comparison:
        return successors

    row = match_assignment_query(user_query)
    try:
        min_actions = int((row or {}).get("min_browser_actions") or 0)
    except (TypeError, ValueError):
        min_actions = 0
    if min_actions <= 0:
        min_actions = min_browser_actions_for_text(user_query) or 3

    schema = distiller_metadata_for_query(user_query, row)
    suffix = comparison_browser_goal_suffix(user_query)
    out: list[Any] = []
    for node in successors:
        if not isinstance(node, NodeSpec):
            out.append(node)
            continue
        meta = dict(node.metadata or {})
        if node.skill == "browser":
            if row:
                meta.setdefault("query_id", row.get("id"))
            meta.setdefault("min_browser_actions", min_actions)
            goal = str(meta.get("goal") or meta.get("question") or "").strip()
            if suffix and suffix not in goal:
                meta["goal"] = f"{goal}. {suffix}".strip(". ") if goal else suffix
            node = NodeSpec(skill=node.skill, inputs=list(node.inputs), metadata=meta)
        elif node.skill == "distiller":
            for key in ("required_keys", "fields", "question"):
                meta.setdefault(key, schema.get(key, ""))
            meta.setdefault("comparison_spec", {
                "row_count": spec.row_count,
                "columns": spec.columns,
                "column_keys": spec.column_keys,
            })
            node = NodeSpec(skill=node.skill, inputs=list(node.inputs), metadata=meta)
        elif node.skill == "formatter":
            hint = schema.get("formatter_hint")
            if hint:
                existing = str(meta.get("question") or "")
                meta["question"] = f"{existing} {hint}".strip() if existing else hint
            node = NodeSpec(skill=node.skill, inputs=list(node.inputs), metadata=meta)
        out.append(node)
    return out


def _parse_upstream_output(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else {"text": text}
        except json.JSONDecodeError:
            return {"text": text}
    return {}


def _cell(value: Any) -> str:
    text = str(value or "—").strip()
    return text.replace("|", "\\|") or "—"


def _extract_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    for key in _ROW_ARRAY_KEYS:
        raw = data.get(key)
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = None
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
    for value in data.values():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return value
    return []


def _subject_value(data: dict[str, Any], browser_content: str = "") -> str | None:
    for key in _SUBJECT_ALIASES:
        val = data.get(key)
        if val:
            return str(val).strip()
    ctx = data.get("context")
    if isinstance(ctx, dict):
        for key in _SUBJECT_ALIASES:
            val = ctx.get(key)
            if val:
                return str(val).strip()
    for pat in (
        r"(?:subject|title|movie|film|product)\s*[:\-]\s*([^\n|]{2,120})",
        r"^#\s+([^\n]{2,120})$",
    ):
        m = re.search(pat, browser_content, re.I | re.M)
        if m:
            return m.group(1).strip()
    return None


def _context_line(data: dict[str, Any]) -> str:
    ctx = data.get("context")
    bits: list[str] = []
    if isinstance(ctx, dict):
        for key in ("city", "location", "site", "region", "country"):
            val = ctx.get(key)
            if val:
                bits.append(str(val))
    elif isinstance(ctx, str) and ctx.strip():
        bits.append(ctx.strip())
    for key in ("city", "location", "site"):
        val = data.get(key)
        if val and str(val) not in bits:
            bits.append(str(val))
    return ", ".join(bits)


def _row_value(row: dict[str, Any], key: str, aliases: tuple[str, ...] = ()) -> Any:
    if key in row and row[key] not in (None, ""):
        return row[key]
    snake = _to_snake(key)
    if snake in row and row[snake] not in (None, ""):
        return row[snake]
    for alias in aliases:
        if alias in row and row[alias] not in (None, ""):
            return row[alias]
    low_key = snake.lower()
    for rk, rv in row.items():
        if str(rk).lower() == low_key or low_key in str(rk).lower():
            if rv not in (None, ""):
                return rv
    return None


def _infer_columns_from_rows(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    if not rows:
        return [], []
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            sk = str(key)
            if sk not in seen:
                seen.add(sk)
                keys.append(sk)
    labels = [sk.replace("_", " ").title() for sk in keys]
    return labels, keys


def format_comparison_table(
    spec: ComparisonSpec,
    data: dict[str, Any],
    *,
    browser_content: str = "",
) -> str | None:
    """Render a markdown comparison table from structured distiller output."""
    if not spec.is_comparison:
        return None

    rows = _extract_rows(data)
    subject = _subject_value(data, browser_content)
    columns = list(spec.columns)
    keys = list(spec.column_keys)

    if not columns and rows:
        columns, keys = _infer_columns_from_rows(rows)
    if not columns:
        return None

    context = _context_line(data)
    intro_parts: list[str] = []
    if subject:
        intro_parts.append(f"**{_cell(subject)}**")
    if context:
        intro_parts.append(f"({_cell(context)})")
    intro = " ".join(intro_parts) + ":" if intro_parts else "Comparison:"

    header = "| " + " | ".join(_cell(c) for c in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [intro, "", header, sep]

    subject_key = _to_snake(spec.subject_column) if spec.subject_column else None
    rows_to_render = rows[: spec.row_count] if rows else []
    while len(rows_to_render) < spec.row_count:
        rows_to_render.append({})

    for row in rows_to_render:
        cells: list[str] = []
        for col, key in zip(columns, keys):
            if subject and (key == subject_key or col == spec.subject_column):
                cells.append(_cell(subject))
                continue
            val = _row_value(row, key)
            cells.append(_cell(val))
        lines.append("| " + " | ".join(cells) + " |")

    if not rows and not subject:
        return None
    return "\n".join(lines)


def format_comparison_answer(
    user_query: str,
    resolved_inputs: list[dict[str, Any]],
) -> str | None:
    """Build a markdown comparison table from upstream distiller/browser output."""
    spec = parse_comparison_spec(user_query)
    if not spec.is_comparison:
        return None

    distiller: dict[str, Any] = {}
    browser_content = ""
    for item in resolved_inputs:
        if item.get("kind") != "upstream":
            continue
        out = _parse_upstream_output(item.get("output"))
        skill = str(item.get("skill") or "")
        if skill == "distiller" and out:
            distiller = out
        elif skill == "browser":
            browser_content = str(out.get("content") or out.get("text") or "")

    if not distiller and browser_content:
        distiller = {"text": browser_content, "rows": _extract_rows({"text": browser_content})}

    return format_comparison_table(spec, distiller, browser_content=browser_content)


def distiller_prompt_block(query: str) -> str:
    schema = distiller_metadata_for_query(query)
    spec = parse_comparison_spec(query)
    col_line = ", ".join(spec.columns) if spec.columns else "(infer from USER QUERY)"
    return (
        "COMPARISON EXTRACTION (required JSON shape):\n"
        f"- required_keys: {schema.get('required_keys')}\n"
        f"- fields: {schema.get('fields')}\n"
        f"- columns requested: {col_line}\n"
        f"- row_count: {spec.row_count}\n"
        f"- task: {schema.get('question')}"
    )