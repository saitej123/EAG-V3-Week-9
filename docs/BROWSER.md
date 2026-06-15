# Browser skill — cascade, cost discipline, replay viewer

Term definitions: [`GLOSSARY.md`](GLOSSARY.md). Corpus: [`corpus/dag/ASSIGNMENT.json`](../corpus/dag/ASSIGNMENT.json).

## Why browser (not web_search / fetch_url alone)

`web_search` and `fetch_url` are useful for **static** pages. They fail on:

- JavaScript-rendered content
- Click-revealed widgets (dropdowns, tabs, accordions)
- Multi-page flows where data appears only after navigation
- Filters, sort controls, and forms — useful data appears only after filtering or sorting

Use the **browser skill** when the task needs ≥3 visible actions (search, filter, sort, open detail pages, switch tabs, expand hidden content, submit forms). Passive scraping from search snippets is **not** accepted.

## Browser comparison agent + replay viewer

Students pick **one** comparison task. These four match the course spec (see `corpus/dag/ASSIGNMENT.json`):

| Id | Course example |
|----|----------------|
| **COMP** | Top 3 Hugging Face text-generation models sorted by likes |
| **DEAL** | 3 laptops under ₹80,000 |
| **STACK** | 5 AI coding tools — free plan vs paid plan |
| **FORGE** | 5 CNC/VMC training institutes in Bangalore |

Optional bonus: **TICKET** (GitHub trending repos). Cascade lab: **B1**–**B4**.

Deliver an **8-section replay report**:

1. Original user goal  
2. Planner DAG  
3. Browser path chosen (`extract` / `deterministic` / `a11y` / `vision` / `blocked`)  
4. Browser actions taken  
5. Screenshots or page-state logs (PNG files saved per action; shown in replay UI)  
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

End-to-end pipeline (matches course flowchart):

```
User goal
  → Planner
  → Researcher (optional — find candidate URLs; on browser failure)
  → Browser skill — cheapest correct path:
       extract (httpx + trafilatura)
       → deterministic (CSS selectors)
       → a11y (accessibility tree)
       → vision (set-of-marks + VLM)
       → gateway_blocked (recover or report)
  → Distiller
  → Critic (auto-spliced after distiller)
  → Formatter (comparison table)
  → Replay viewer (8 sections)
```

**Researcher vs Browser:** Researcher runs when the planner needs URL discovery, static fetches, or a browser-failure fallback. Browser runs when clicks, JS rendering, or ≥3 visible actions are required. The orchestrator may upgrade `researcher → browser` for comparison tasks (`flow.py`).

**Where crawl4ai is used (Researcher only — not in the Browser cascade):**

| Tool | crawl4ai? | Mechanism |
|------|-----------|-----------|
| `fetch_url` / `fetch_urls` | Yes | `mcp_server._crawl4ai_fetch` — headless crawl → markdown |
| `web_search` step 2 | Yes | `search_providers.async_crawl4ai_search` (after Tavily) |
| Browser Layer 1 extract | **No** | `httpx` + `trafilatura` (`extract.py`) |
| Browser render / a11y / vision | **No** | Playwright + Pillow + Gemini VLM |

**Implementation extras (same diagram, finer escalation inside Browser):** before deterministic/a11y/vision the cascade may try Playwright live-text **render**, fast **playwright_vlm**, and an indexed **agent** loop — all map to `BrowserOutput.path` values consumed by distiller/replay. Multi-URL tasks (STACK, FORGE) open up to `BROWSER_MAX_URLS` targets in one Playwright session.

**Not in the course diagram:** [browser-use](https://github.com/browser-use/browser-use) is an optional bridge (`browser_use_bridge.py`), **off by default** (`BROWSER_USE_ENABLED=0`). The shipped path is **Playwright + Pillow + httpx + trafilatura** only.

New behaviour plugs in via `agent_config.yaml` + `super_browser/browser/` only.

### Shipped dependencies

| Library | Role in Browser |
|---------|-----------------|
| **httpx** | Layer 1 static fetch |
| **trafilatura** | HTML → markdown extract |
| **Playwright** | Render, a11y, vision, multi-URL navigation |
| **Pillow** | Set-of-marks boxes on screenshots (`highlight.py`) |

### `force_path` (opt-in)

Natural cascade is the default. Set `force_path` in browser metadata only to (1) debug a single layer (**B1**–**B4**), or (2) rare production cases where the caller already knows vision is required (e.g. acting on a screenshot artifact from upstream). See [`VALIDATION.md`](VALIDATION.md) §2.

### Driver layout

Shared turn rules and action execution: `driver.py`. **A11y** and **vision** layers differ only in how they decide the next action (tree + text LLM vs marks + VLM). Experimental driver core was ported with that split preserved. See [`VALIDATION.md`](VALIDATION.md) §3.

## Four-layer cascade (BrowserOutput.path)

| Path | Mechanism | LLM cost |
|------|-----------|----------|
| `extract` | httpx + trafilatura; Playwright render when static fetch is thin | $0 |
| `deterministic` | Playwright + CSS selectors | $0 |
| `a11y` | Playwright + accessibility tree + text LLM | Low |
| `vision` | Playwright + set-of-marks + VLM (`playwright_vlm` fast path, then full marks) | Per screenshot |
| `agent` | Indexed clickables + text LLM loop (implementation layer between render and deterministic) | Low |
| `gateway_blocked` | Live captcha / bot wall — orchestrator queues Researcher fallback or reports | $0 |
| `failed` | All layers exhausted | — |

Natural cascade is the default. Opt-in `force_path` metadata skips escalation — for layer demos **B1**–**B4** or rare production vision-from-screenshot cases. See [`VALIDATION.md`](VALIDATION.md).

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
  browser_use_bridge.py  optional browser-use Agent (runs before local cascade)
  replay.py         replay report + payload
scripts/browser/
  analyze_browser_session.py
  export_browser_replay.py
  seed_browser_sessions.py
sandbox/browser/canvas-only.html
```
