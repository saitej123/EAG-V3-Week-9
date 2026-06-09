# Browser skill — cascade, cost discipline, replay viewer

Term definitions: [`GLOSSARY.md`](GLOSSARY.md). Corpus: [`corpus/dag/ASSIGNMENT.json`](../corpus/dag/ASSIGNMENT.json) part 6.

## Browser comparison agent + replay viewer

Build a browser-capable agent that performs a **real comparison task** with at least **three visible browser actions** (filter, sort, open cards — not passive snippets). Deliver:

1. Original user goal  
2. Planner DAG  
3. Browser path chosen (`extract` | `deterministic` | `a11y` | `vision`)  
4. Browser actions taken (page-state logs)  
5. Extracted data  
6. Final comparison table (formatter)  
7. Turn count and cost summary  

**Demo query:** **COMP** — Hugging Face top 3 text-generation models (transformers, sorted by likes).

```bash
uv run python scripts/dag/run_query.py COMP
uv run python scripts/browser/seed_browser_sessions.py
uv run python scripts/browser/export_browser_replay.py dag_COMP_ref -o replay.md
# Graph tab → Replay report (grey panel, 8 sections)
```

Layer reference demos: **B1**–**B4** (extract → deterministic → a11y → vision).

## Four-layer cascade

| Layer | Mechanism | LLM cost |
|-------|-----------|----------|
| 1 extract | httpx + trafilatura | $0 |
| 2a deterministic | Playwright + CSS selectors | $0 |
| 2b a11y | Playwright + accessibility tree + text LLM | Low |
| 3 vision | Playwright + set-of-marks + VLM | Per screenshot |

Natural cascade is the default. Opt-in `force_path` metadata skips escalation (debugging only).

## Cost discipline (reference runs)

| Layer | Target | Path | Turns | Cost | Wall |
|-------|--------|------|-------|------|------|
| 1 | news.ycombinator.com | extract | 0 | $0.00 | ~2.1s |
| 2a | amazon.com | deterministic | 0 | $0.00 | ~4.3s |
| 2b | huggingface.co/models | a11y | 5 | $0.00 | ~5.6s |
| 3 | canvas-only.html | vision | 7 | $0.00 | ~29.7s |

Seed reference sessions:

```bash
uv run python scripts/browser/seed_browser_sessions.py
```

## Design choices

- **Stack:** httpx, trafilatura, Playwright, Pillow, Gemini SDK — no LangChain / AutoGen / browser-use.
- **Driver:** shared turn rules in `driver.py`; a11y and vision differ only in `_decide()` loops.
- **Replay:** `build_browser_replay_report()` + Graph tab panel; `BrowserOutput` carries `path`, `actions`, tokens, `cost_usd`.
- **Finding:** six canvas-heavy §9 targets resolved on **a11y**; vision only on adversarial **B4** fixture.

## Code map

```
super_browser/browser/
  skill.py          cascade entry
  extract.py        Layer 1
  deterministic.py  Layer 2a
  a11y.py           Layer 2b
  vision.py         Layer 3
  driver.py         turn rules
  dom.py            clickables + gateway detect
  highlight.py      set-of-marks dedupe + draw
  ledger.py         token/cost fields
  replay.py         replay report + payload
scripts/browser/
  analyze_browser_session.py
  export_browser_replay.py
  seed_browser_sessions.py
sandbox/browser/canvas-only.html
```
