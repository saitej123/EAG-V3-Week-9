# Browser skill — cascade, cost discipline, replay viewer

Term definitions: [`GLOSSARY.md`](GLOSSARY.md). Corpus: [`corpus/dag/ASSIGNMENT.json`](../corpus/dag/ASSIGNMENT.json).

## Why browser (not web_search / fetch_url alone)

`web_search` and `fetch_url` work for **static** pages. They fail on:

- JavaScript-rendered content
- Click-revealed widgets (dropdowns, tabs, accordions)
- Filters, sort controls, search forms
- Multi-step flows where data appears only after interaction

Use the **browser skill** when the task needs ≥3 visible actions (search, filter, sort, open detail pages, switch tabs, expand hidden content, submit forms). Passive scraping from search snippets is **not** accepted.

## Browser comparison agent + replay viewer

Students pick **one** comparison task (examples in corpus):

| Id | Task |
|----|------|
| **COMP** | Top 3 Hugging Face text-generation models by likes |
| **DEAL** | 3 laptops under ₹80,000 (Flipkart) |
| **STACK** | 5 AI coding tools — free vs paid plans |
| **FORGE** | 5 CNC/VMC training institutes in Bangalore |
| **TICKET** | 3 IMAX showtimes in Bengaluru (BookMyShow) |

Deliver an **8-section replay report**:

1. Original user goal  
2. Planner DAG  
3. Browser path chosen (`extract` / `deterministic` / `a11y` / `vision` / `blocked`)  
4. Browser actions taken  
5. Screenshots or page-state logs  
6. Extracted data  
7. Final comparison table  
8. Turn count and cost summary  

**Demo query:** **COMP** — Hugging Face top 3 text-generation models (transformers, sorted by likes).

```bash
uv run python scripts/dag/run_query.py COMP
uv run python scripts/browser/seed_browser_sessions.py
uv run python scripts/browser/export_browser_replay.py dag_COMP_ref -o replay.md
# Graph tab → Replay report (grey panel, 8 sections)
```

Layer reference demos: **B1**–**B4** (extract → deterministic → a11y → vision → blocked).

## Architecture (orchestrator unchanged)

```
extract (httpx) → render (Playwright live text) → agent (indexed LLM loop, browser-use pattern)
  → deterministic → a11y → vision → blocked/failed
```

Multi-URL comparison tasks crawl every resolved pricing URL before the agent loop.

Optional upstream integration: `BROWSER_USE_ENABLED=1` + `pip install "browser-use[core]"` runs the [browser-use](https://github.com/browser-use/browser-use) Agent first.

Optional [BrowserOS](https://github.com/browseros-ai/BrowserOS) (Chromium fork with built-in MCP):

```bash
# Install BrowserOS, then in .env:
BROWSER_BACKEND=browseros
BROWSEROS_MCP_URL=http://127.0.0.1:9239/mcp   # from chrome://browseros/mcp
# Optional CDP — attach Playwright cascade to your running BrowserOS:
# BROWSEROS_CDP_URL=http://127.0.0.1:9222
```

When BrowserOS is running, the cascade tries MCP automation before bundled Chromium.

## Layer reference

```
User goal → Planner → Browser cascade
              extract → render → agent → deterministic → a11y → vision → blocked
              → Distiller → Critic (auto) → Formatter → Replay viewer
```

New behaviour plugs in via `agent_config.yaml` + `super_browser/browser/` only.

## Four-layer cascade

| Layer | Mechanism | LLM cost |
|-------|-----------|----------|
| 1 extract | httpx + trafilatura (+ Playwright render fallback) | $0 |
| 2a deterministic | Playwright + CSS selectors | $0 |
| 2b a11y | Playwright + accessibility tree + text LLM | Low |
| 3 vision | Playwright + set-of-marks + VLM | Per screenshot |
| blocked | Live captcha / bot wall — recover or report | $0 |

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

## Code map

```
super_browser/browser/
  skill.py          cascade entry
  extract.py        Layer 1
  playwright_render.py  JS render fallback
  deterministic.py  Layer 2a
  a11y.py           Layer 2b
  vision.py         Layer 3
  navigation.py     robust goto + fallbacks
  driver.py         turn rules
  dom.py            clickables + gateway detect
  highlight.py      set-of-marks dedupe + draw
  ledger.py         token/cost fields
  browser_use_bridge.py  optional browser-use Agent
  browseros_bridge.py   optional BrowserOS MCP
  browseros_mcp.py        BrowserOS HTTP MCP client
  browser_backend.py      chromium / chrome / browseros selection
scripts/browser/
  analyze_browser_session.py
  export_browser_replay.py
  seed_browser_sessions.py
sandbox/browser/canvas-only.html
```
