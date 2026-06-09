#!/usr/bin/env python3
"""Architecture gate checks for RAG eval submission."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MCP_TOOL_NAMES = [
    "index_document",
    "index_directory",
    "search_knowledge",
    "read_file",
    "list_dir",
    "web_search",
    "fetch_url",
    "fetch_urls",
    "query_database",
    "create_file",
    "update_file",
    "edit_file",
    "analyze_image_url",
    "get_time",
    "currency_convert",
    "gemini_live_search",
]


def perception_prompt_text() -> str:
    import importlib.util

    spec = importlib.util.spec_from_file_location("perception", ROOT / "super_browser" / "perception.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Extract prompt template by reading source — observe() builds prompt inline.
    src = (ROOT / "super_browser" / "perception.py").read_text(encoding="utf-8")
    start = src.find('prompt = f"""')
    if start < 0:
        raise RuntimeError("Could not locate Perception prompt template")
    end = src.find('"""', start + 14)
    return src[start:end + 3]


def check_perception_tool_blindness() -> list[str]:
    """Return list of violations: MCP tool names found inside Perception SYSTEM."""
    prompt = perception_prompt_text()
    violations: list[str] = []
    for name in MCP_TOOL_NAMES:
        if re.search(rf"\b{re.escape(name)}\b", prompt):
            violations.append(name)
    return violations


def check_corpus_manifest() -> tuple[bool, str]:
    manifest = ROOT / "corpus" / "MANIFEST.json"
    if not manifest.is_file():
        return False, "missing corpus/MANIFEST.json"
    import json

    data = json.loads(manifest.read_text(encoding="utf-8"))
    count = int(data.get("item_count", 0))
    if count < 50:
        return False, f"manifest item_count={count} (need >= 50)"
    corpus_dir = ROOT / "sandbox" / "research_papers"
    pdf_count = len(list(corpus_dir.glob("*.pdf")))
    md_count = len(list(corpus_dir.glob("*.md")))
    if pdf_count < 50 or md_count < 50:
        return False, (
            f"sandbox/research_papers has {pdf_count} PDFs and {md_count} sidecars (need >= 50 each)"
        )
    return True, f"corpus ok: {count} manifest items, {pdf_count} PDFs + {md_count} sidecars"


def main() -> int:
    errors: list[str] = []

    violations = check_perception_tool_blindness()
    if violations:
        errors.append(f"Perception SYSTEM contains MCP tool names: {violations}")
    else:
        print("PASS  Perception tool-blindness (zero MCP tool names in SYSTEM)")

    ok, msg = check_corpus_manifest()
    if ok:
        print(f"PASS  {msg}")
    else:
        errors.append(msg)

    if (ROOT / "super_browser" / "memory.py").is_file() and "_format_hits" in (
        (ROOT / "super_browser" / "memory.py").read_text()
    ):
        print("PASS  memory._format_hits present")
    else:
        errors.append("memory._format_hits not found")

    if errors:
        print("\nFAIL architecture checks:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nAll architecture checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
