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
  → Browser skill (`super_browser/browser/drivers/`)
       extract (httpx + trafilatura)
       → render (Playwright DOM text)
       → multi-page crawl (when multiple URLs resolved)
       → a11y (A11yDriver — element legend + text LLM)
       → vision (SetOfMarksDriver — marks + Gemini VLM)
       → gateway_blocked (recover or report)
  → Distiller
  → Critic (auto-spliced after distiller)
  → Formatter (comparison table)
  → Replay viewer (8 sections)
```

**LLM:** direct Gemini SDK via `drivers/gemini_client.py` — no gateway URL required.

**Researcher vs Browser:** Researcher runs when the planner needs URL discovery, static fetches, or a browser-failure fallback. Browser runs when clicks, JS rendering, or ≥3 visible actions are required. The orchestrator may upgrade `researcher → browser` for comparison tasks (`flow.py`).

Multi-URL tasks (STACK, FORGE) open up to `BROWSER_MAX_URLS` targets in one Playwright session before the driver loop.

**Where crawl4ai is used (Researcher only — not in the Browser cascade):**

| Tool | crawl4ai? | Mechanism |
|------|-----------|-----------|
| `fetch_url` / `fetch_urls` | Yes | `mcp_server._crawl4ai_fetch` — headless crawl → markdown |
| `web_search` step 2 | Yes | `search_providers.async_crawl4ai_search` (after Tavily) |
| Browser Layer 1 extract | **No** | `httpx` + `trafilatura` (`extract.py`) |
| Browser render / a11y / vision | **No** | Playwright + Pillow + Gemini VLM |

**Implementation:** Playwright live-text **render** and multi-page crawl run before **a11y → vision** drivers. Multi-URL tasks (STACK, FORGE) open up to `BROWSER_MAX_URLS` targets in one Playwright session.

New behaviour plugs in via `agent_config.yaml` + `super_browser/browser/` only.

### Shipped dependencies

| Library | Role in Browser |
|---------|-----------------|
| **httpx** | Layer 1 static fetch |
| **trafilatura** + **lxml** + **lxml-html-clean** | HTML → markdown extract |
| **Playwright** | Render, a11y, vision, multi-URL navigation |
| **Pillow** | Set-of-marks boxes on screenshots (`drivers/marks.py`) |
| **google-genai** | A11y text LLM + vision VLM (direct SDK) |

### `force_path` (opt-in)

Natural cascade is the default. Set `force_path` in browser metadata only to (1) debug a single layer (**B1**–**B4**), or (2) rare production cases where the caller already knows vision is required (e.g. acting on a screenshot artifact from upstream). See [`VALIDATION.md`](VALIDATION.md) §2.

### Driver layout

`super_browser/browser/drivers/interaction.py` — **A11yDriver** and **SetOfMarksDriver**. Element enumeration: `drivers/elements.py`. Mark drawing: `drivers/marks.py`. Gemini: `drivers/gemini_client.py`.

## Cascade paths (BrowserOutput.path)

| Path | Mechanism | LLM cost |
|------|-----------|----------|
| `extract` | httpx + trafilatura; Playwright render when static fetch is thin | $0 |
| `deterministic` | Legacy CSS selectors (force_path demo only) | $0 |
| `a11y` | A11yDriver — enumerated elements + text LLM | Low |
| `vision` | SetOfMarksDriver — screenshot + marks + Gemini VLM | Per screenshot |
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
  skill.py              public API (run_browser_cascade)
  output.py             BrowserOutput builder
  validation.py         layer success + action counting
  gateway.py            captcha / bot-wall detection
  turn_rules.py         action fencing (max 2/turn)
  extract.py            Layer 1 static fetch
  playwright_render.py  JS render fallback
  multi_page.py         multi-URL crawl
  navigation.py         robust goto + fallbacks
  page_capture.py       screenshots + replay logs
  playwright_ctx.py     Chromium session helpers
  ledger.py             token/cost fields
  replay.py             replay report + payload
  urls.py               URL resolution for assignments
  drivers/
    cascade.py          extract → render → a11y → vision
    interaction.py      A11yDriver + SetOfMarksDriver
    elements.py         interactive element enumeration
    marks.py            set-of-marks (Pillow)
    gemini_client.py    direct Gemini SDK
scripts/browser/
  analyze_browser_session.py
  export_browser_replay.py
  seed_browser_sessions.py
sandbox/browser/canvas-only.html
```
