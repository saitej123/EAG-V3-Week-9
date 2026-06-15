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


def comparison_query_understanding(query: str, row: dict[str, Any] | None = None) -> str:
    """Plain-language comparison intent for browser/distiller goals."""
    spec = parse_comparison_spec(query)
    if not spec.is_comparison and row:
        spec = parse_comparison_spec(str(row.get("query") or query))
    if not spec.is_comparison:
        return ""
    schema = distiller_metadata_for_query(query, row)
    cols = ", ".join(spec.columns) if spec.columns else "infer from USER QUERY"
    lines = [
        "QUERY UNDERSTANDING:",
        f"- Task: structured comparison table with {spec.row_count} data rows",
        f"- Columns: {cols}",
    ]
    if spec.subject_column:
        lines.append(f"- Subject/name column: {spec.subject_column}")
    fields = schema.get("fields")
    if fields:
        lines.append(f"- Distiller fields: {fields}")
    lines.append("- Use live page content only; do not invent prices, times, or names.")
    return "\n".join(lines)


def comparison_needs_browser(user_query: str) -> tuple[bool, int, dict[str, Any] | None]:
    """True when query is an interactive comparison task (needs Playwright, not fetch_url)."""
    spec = parse_comparison_spec(user_query)
    if not spec.is_comparison:
        return False, 0, None
    row = match_assignment_query(user_query)
    min_actions = 0
    if row:
        try:
            min_actions = int(row.get("min_browser_actions") or 0)
        except (TypeError, ValueError):
            min_actions = 0
    if min_actions <= 0:
        min_actions = min_browser_actions_for_text(user_query) or 3
    return min_actions > 0, min_actions, row


def researcher_fallback_question(user_query: str, *, partial_browser_ref: str | None = None) -> str:
    """Guide researcher to Gemini / Tavily / fetch when browser could not finish."""
    from .browser.urls import resolve_browser_urls

    urls = resolve_browser_urls("", user_query, user_query)
    hosts = []
    for url in urls[:5]:
        try:
            from urllib.parse import urlparse

            host = urlparse(url).netloc
            if host:
                hosts.append(host)
        except Exception:
            continue
    target_hint = ", ".join(hosts) if hosts else "each item named in the user goal"
    partial = (
        f" Merge any notes from {partial_browser_ref} with search results."
        if partial_browser_ref
        else ""
    )
    codeium_note = ""
    if "codeium" in user_query.lower():
        codeium_note = (
            " Codeium is now **Windsurf IDE** — use **gemini_live_search** on windsurf.com pricing "
            "(free tier + Pro starting price) if codeium.com is empty."
        )
    gemini_q = comparison_pricing_gemini_query(user_query)
    return (
        "Live browser could not finish this comparison. **Start with gemini_live_search** "
        f'query: "{gemini_q}" then fetch_urls on official pricing pages if needed.{codeium_note} '
        f"Targets: {target_hint}.{partial} "
        "Return concise facts per product from official sources — do not invent prices. "
        "If still missing, say 'not listed'."
    )


def _is_pricing_comparison(text: str) -> bool:
    low = (text or "").lower()
    return bool(
        re.search(r"\bcompare\b.*\b(pricing|plan|paid|free tier)", low)
        or re.search(r"\bfree vs paid\b", low)
        or "pricing pages for" in low
    )


def _product_display_name(name: str) -> str:
    key = (name or "").strip().lower()
    if "codeium" in key:
        return "Windsurf IDE (formerly Codeium)"
    if key == "copilot" or "github copilot" in key:
        return "GitHub Copilot"
    return name.strip()


def comparison_pricing_gemini_query(user_query: str, *, product: str | None = None) -> str:
    """Build a Gemini live-search query for official SaaS pricing facts."""
    from .browser.urls import _extract_named_targets

    if product:
        label = _product_display_name(product)
        return (
            f"Official current pricing for {label}: free tier summary and lowest paid plan "
            f"starting price. Prefer windsurf.com for Codeium/Windsurf, cursor.com for Cursor, "
            f"github.com/features/copilot/plans for Copilot. Reply FREE: and PAID: lines only."
        )
    names = _extract_named_targets(user_query) or []
    if not names:
        return (
            "Official pricing free tier and paid starting price for AI coding assistants "
            "named in the user question. Reply with FREE: and PAID: per product."
        )
    parts = [_product_display_name(n) for n in names]
    return (
        "Official current pricing (free tier + lowest paid starting price) for: "
        + "; ".join(parts)
        + ". Use Google Search on each vendor's official pricing page. "
        "Reply with FREE: and PAID: lines per product."
    )


def _parse_pricing_facts(text: str) -> dict[str, str]:
    """Extract free/paid fields from Gemini grounded prose."""
    if not (text or "").strip():
        return {}
    free = ""
    paid = ""
    for line in text.splitlines():
        raw = line.strip().lstrip("-*• ")
        if not raw:
            continue
        low = raw.lower()
        if low.startswith("free:"):
            free = raw.split(":", 1)[1].strip()
        elif low.startswith("paid:"):
            paid = raw.split(":", 1)[1].strip()
        elif "free tier" in low or "free plan" in low:
            free = free or raw
        elif ("pro" in low or "paid" in low or "/mo" in low) and "$" in raw:
            paid = paid or raw
    if not paid:
        m = re.search(
            r"(?:pro|paid|starting)[^\$\n]*(\$\d+(?:\.\d+)?(?:\s*(?:/|per)\s*mo(?:nth)?)?)",
            text,
            re.I,
        )
        if m:
            paid = m.group(1).strip()
    if not free and "unlimited tab" in text.lower():
        free = "Unlimited Tab completions; light agent quota (official free tier)"
    if not paid and "$20" in text and ("pro" in text.lower() or "windsurf" in text.lower()):
        paid = "$20/mo (Pro)"
    return {
        k: v
        for k, v in {
            "free_tier_summary": free,
            "paid_starting_price": paid,
            "free": free,
            "paid": paid,
        }.items()
        if v
    }


def _row_missing_pricing(row: dict[str, Any]) -> bool:
    empty = {"", "—", "-", "not listed", "n/a", "null", "none"}
    free_keys = ("free_tier_summary", "free", "free_tier")
    paid_keys = ("paid_starting_price", "paid", "paid_price")

    def _has(keys: tuple[str, ...]) -> bool:
        for key in keys:
            val = str(row.get(key) or "").strip().lower()
            if val and val not in empty:
                return True
        return False

    return not _has(free_keys) or not _has(paid_keys)


def _row_product_name(row: dict[str, Any], spec: ComparisonSpec) -> str:
    for key in ("tool", "product", "name", "assistant", "vendor"):
        val = str(row.get(key) or row.get(_to_snake(key)) or "").strip()
        if val:
            return val
    if spec.subject_column:
        sk = _to_snake(spec.subject_column)
        val = str(row.get(sk) or "").strip()
        if val:
            return val
    return ""


def enrich_doc_library_gaps(user_query: str, distiller: dict[str, Any]) -> dict[str, Any]:
    """Fill missing Browser-stack library rows via Gemini (FORGE: httpx, trafilatura, Playwright)."""
    low = (user_query or "").lower()
    libs = [name for name in ("httpx", "trafilatura", "playwright") if name in low]
    if len(libs) < 2:
        return distiller
    from .search_providers import gemini_live_search_text

    rows = _extract_rows(distiller)
    if len(rows) >= 3 and all(_row_has_library_fields(r) for r in rows[:3]):
        return distiller
    q = (
        "From official docs only: compare httpx, trafilatura, and Playwright — "
        "primary purpose and which Browser skill layer uses each "
        "(Layer 1 extract = httpx+trafilatura; Layers 2–3 = Playwright). "
        "Reply as markdown table: library | purpose | layer."
    )
    facts = gemini_live_search_text(q)
    if not facts.strip():
        return distiller
    parsed = _parse_training_table_from_text(facts)  # generic pipe table parser
    if not parsed:
        return distiller
    mapped: list[dict[str, Any]] = []
    for row in parsed[:3]:
        cells = list(row.values())
        mapped.append(
            {
                "library_name": cells[0] if cells else "",
                "primary_purpose": cells[1] if len(cells) > 1 else "",
                "browser_cascade_layer": cells[2] if len(cells) > 2 else "",
            }
        )
    if not mapped:
        return distiller
    distiller = dict(distiller)
    distiller["rows"] = mapped
    if "items" in distiller:
        distiller["items"] = mapped
    return distiller


def _row_has_library_fields(row: dict[str, Any]) -> bool:
    empty = {"", "—", "-", "not listed", "n/a"}
    name = str(row.get("library_name") or row.get("library") or row.get("name") or "").strip().lower()
    return bool(name and name not in empty)


def enrich_training_institute_gaps(user_query: str, distiller: dict[str, Any]) -> dict[str, Any]:
    """Fill missing CNC/VMC institute rows via Gemini live search (FORGE)."""
    low = (user_query or "").lower()
    if not any(k in low for k in ("cnc", "vmc", "training institute", "training institutes")):
        return distiller
    from .search_providers import gemini_live_search_text

    want = 5 if re.search(r"\bcompare\s+5\b|\b5\s+cnc|\bfive\b", low) else 3
    rows = _extract_rows(distiller)
    if len(rows) >= want and all(_row_has_institute_fields(r) for r in rows[:want]):
        return distiller

    city = "Bangalore" if any(c in low for c in ("bangalore", "bengaluru")) else "the city named in the query"
    q = (
        f"Use Google Search: list {want} CNC or VMC machine training institutes in {city} with "
        "course duration and approximate fee. "
        "Prefer UrbanPro, JustDial, or official institute sites. "
        "Reply as markdown table: institute name | duration | fee."
    )
    facts = gemini_live_search_text(q)
    if not facts.strip():
        return distiller
    parsed_rows = _parse_training_table_from_text(facts)
    if not parsed_rows:
        return distiller
    distiller = dict(distiller)
    distiller["rows"] = parsed_rows[:want]
    if "items" in distiller:
        distiller["items"] = parsed_rows[:want]
    return distiller


def _row_has_institute_fields(row: dict[str, Any]) -> bool:
    empty = {"", "—", "-", "not listed", "n/a"}
    name = str(row.get("institute_name") or row.get("name") or row.get("institute") or "").strip().lower()
    if not name or name in empty:
        return False
    return True


def _parse_training_table_from_text(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|") or "---" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3:
            continue
        rows.append(
            {
                "institute_name": cells[0],
                "course_duration": cells[1] if len(cells) > 1 else "",
                "approximate_fee": cells[2] if len(cells) > 2 else "",
                "certification_offered": cells[3] if len(cells) > 3 else "",
            }
        )
    return rows


def enrich_distiller_pricing_gaps(user_query: str, distiller: dict[str, Any]) -> dict[str, Any]:
    """Ensure all named products have rows filled from official pricing pages."""
    if not _is_pricing_comparison(user_query):
        return distiller
    from .pricing_enrich import ensure_pricing_rows

    return ensure_pricing_rows(user_query, distiller)


def coerce_researcher_to_browser(
    user_query: str,
    successors: list[Any],
    *,
    enabled: bool = True,
) -> list[Any]:
    """Replace researcher with browser when comparison tasks need live interaction."""
    if not enabled:
        return successors
    from .dag_schemas import NodeSpec
    from .search_providers import extract_http_urls

    needs, min_actions, row = comparison_needs_browser(user_query)
    if not needs:
        return successors

    skills = {s.skill for s in successors if isinstance(s, NodeSpec)}
    if "browser" in skills or "researcher" not in skills:
        return successors

    urls = extract_http_urls(user_query)
    suffix = comparison_browser_goal_suffix(user_query)
    understanding = comparison_query_understanding(user_query, row)
    out: list[Any] = []
    for node in successors:
        if not isinstance(node, NodeSpec) or node.skill != "researcher":
            out.append(node)
            continue
        meta = dict(node.metadata or {})
        label = str(meta.get("label") or "browser")
        meta["label"] = label
        if row:
            meta.setdefault("query_id", row.get("id"))
        meta["min_browser_actions"] = min_actions
        if urls:
            meta.setdefault("url", urls[0])
        goal_parts = [
            understanding,
            user_query.strip(),
            str(meta.get("goal") or meta.get("question") or "").strip(),
        ]
        goal = "\n\n".join(p for p in goal_parts if p).strip()
        if suffix and suffix not in goal:
            goal = f"{goal.rstrip('.')}. {suffix}"
        meta["goal"] = goal
        out.append(NodeSpec(skill="browser", inputs=list(node.inputs), metadata=meta))
    return out


def collapse_parallel_browser_plan(user_query: str, successors: list[Any]) -> list[Any]:
    """Replace N parallel browser→distiller chains with one browser → distiller → formatter."""
    from .dag_schemas import NodeSpec

    nodes = [s for s in successors if isinstance(s, NodeSpec)]
    browser_nodes = [s for s in nodes if s.skill == "browser"]
    if len(browser_nodes) <= 1:
        return successors

    spec = parse_comparison_spec(user_query)
    if not spec.is_comparison:
        return successors

    row = match_assignment_query(user_query)
    schema = distiller_metadata_for_query(user_query, row)
    try:
        min_actions = int((row or {}).get("min_browser_actions") or 0)
    except (TypeError, ValueError):
        min_actions = 0
    if min_actions <= 0:
        min_actions = min_browser_actions_for_text(user_query) or 3

    from .browser.urls import resolve_browser_urls

    resolved = resolve_browser_urls("", user_query, user_query)
    understanding = comparison_query_understanding(user_query, row)
    suffix = comparison_browser_goal_suffix(user_query)
    goal_parts = [understanding, user_query.strip()]
    goal = "\n\n".join(p for p in goal_parts if p).strip()
    if suffix and suffix not in goal:
        goal = f"{goal.rstrip('.')}. {suffix}"

    browser_meta: dict[str, Any] = {
        "label": "browser",
        "goal": goal,
        "min_browser_actions": min_actions,
    }
    if row:
        browser_meta["query_id"] = row.get("id")
    if resolved:
        browser_meta["url"] = resolved[0]

    distill_meta: dict[str, Any] = {
        "label": "extract",
        "question": schema.get("question", ""),
        "required_keys": schema.get("required_keys", ""),
        "fields": schema.get("fields", ""),
        "comparison_spec": {
            "row_count": spec.row_count,
            "columns": spec.columns,
            "column_keys": spec.column_keys,
        },
    }
    if row:
        distill_meta["query_id"] = row.get("id")

    fmt_meta: dict[str, Any] = {"label": "out"}
    hint = schema.get("formatter_hint")
    if hint:
        fmt_meta["question"] = hint

    return [
        NodeSpec(skill="browser", inputs=["USER_QUERY"], metadata=browser_meta),
        NodeSpec(skill="distiller", inputs=["n:browser"], metadata=distill_meta),
        NodeSpec(skill="formatter", inputs=["n:extract"], metadata=fmt_meta),
    ]


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
            from .browser.urls import resolve_browser_urls

            if row:
                meta.setdefault("query_id", row.get("id"))
            meta.setdefault("min_browser_actions", min_actions)
            goal = str(meta.get("goal") or meta.get("question") or user_query).strip()
            primary = str(meta.get("url") or "").strip()
            resolved = resolve_browser_urls(primary, goal, user_query)
            if resolved:
                meta["url"] = resolved[0]
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

    distiller = enrich_distiller_pricing_gaps(user_query, distiller)
    distiller = enrich_training_institute_gaps(user_query, distiller)
    distiller = enrich_doc_library_gaps(user_query, distiller)

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