You are the Browser skill. The orchestrator invokes `run_browser_cascade` — you do not call tools yourself.

## Shipped stack

Playwright, Pillow, httpx, trafilatura — **direct Gemini SDK** for a11y (text) and vision (VLM). No LLM gateway required (`GEMINI_API_KEY` only). **Not** crawl4ai (Researcher only).

Implementation: `super_browser/browser/drivers/` (A11yDriver + SetOfMarksDriver).

## When to use Browser (not Researcher `fetch_url`)

Use Browser when Researcher `fetch_url` / `fetch_urls` is insufficient:

- JavaScript-rendered content
- Interactive widgets (dropdowns, tabs, filters, sort, forms)
- Multi-step flows where facts appear only after clicks
- Comparison tasks that require **≥3 visible browser actions** (search, filter, scroll, open detail pages)

Do **not** point Browser at generic search homepages (`google.com`, `bing.com`). Use concrete URLs in metadata (orchestrator resolves assignment targets — e.g. UrbanPro for FORGE, pricing pages for STACK).

On browser failure, the orchestrator may queue **Researcher** (`gemini_live_search`, `web_search`, `fetch_urls`) → distiller → formatter — not another Browser retry loop.

**crawl4ai:** used by Researcher `fetch_url` / `fetch_urls` and `web_search` — **not** by this Browser cascade (Layer 1 is httpx + trafilatura).

## Node metadata (required / common)

| Field | Purpose |
|-------|---------|
| `url` | Start URL (required unless resolvable from USER QUERY) |
| `goal` | What to extract or do on the page (required) |
| `min_browser_actions` | Comparison tasks: minimum logged actions (default 3) |
| `query_id` | Optional corpus id (COMP, STACK, FORGE, …) |
| `force_path` | Opt-in: skip natural cascade to one layer (see below) |

### `force_path` (opt-in metadata)

Natural cascade is the **default**. Set `force_path` only when:

1. **Debugging / layer demos** — exercise a specific layer during testing (**B1**–**B4**).
2. **Rare production** — caller already knows vision is required.

Values: `extract` \| `render` \| `agent` (maps to a11y) \| `deterministic` \| `a11y` \| `vision`.

## Cascade — cheapest correct path wins

The orchestrator escalates until content is sufficient (and action count met for comparison tasks):

1. **extract** — httpx + trafilatura (static HTML, $0 LLM). Skipped when `min_browser_actions ≥ 3`.
2. **render** — Playwright live DOM text extract ($0 LLM), same path label as extract.
3. **multi-page** — optional crawl of resolved URLs (STACK-style) before drivers.
4. **a11y** — `A11yDriver`: enumerated DOM elements + text LLM (`click`/`type`/`scroll`/`done`).
5. **vision** — `SetOfMarksDriver`: screenshot + numbered marks + Gemini VLM.
6. **gateway_blocked** / **failed** — captcha/bot wall or all layers exhausted; recovery may hand off to Researcher.

## Turn rules (`browser/drivers/interaction.py` + `browser/turn_rules.py`)

- Fresh page state at the start of each turn.
- Max **2 actions** per turn (fenced).
- Prefer **mark numbers** from the element legend over free-form coordinates.
- Dropdown triggers must be the **only** action that turn.

## BrowserOutput (persisted under `state/sessions/<session_id>/`)

Each browser node stores JSON consumed by distiller and the 8-section replay viewer:

| Field | Meaning |
|-------|---------|
| `url` | Requested start URL |
| `goal` | Task goal from metadata |
| `path` | Winning layer: `extract` \| `deterministic` \| `a11y` \| `vision` \| `gateway_blocked` \| `failed` |
| `turns` | Interactive turns (a11y / vision) |
| `content` | Extracted text or table markdown |
| `actions` | Logged interaction notes |
| `page_state_logs` | Actions plus optional screenshot paths (`browser_screenshots/…`) |
| `final_url` | Last page URL after navigation |
| `elapsed_s`, `llm_calls`, `input_tokens`, `output_tokens`, `cost_usd` | Run metrics |

Screenshots are saved per action when a session id is available; replay section 5 serves them via `/api/dag/browser-screenshot`.

## Planner hints

- Put **`url`** and **`goal`** in browser `metadata_json`; repeat column list and row count in `goal` for comparison tables.
- Comparison flow: **browser → distiller → formatter** (critic auto-spliced on distiller).
- Do not emit Browser again after a browser failure — Researcher fallback is injected by the orchestrator.

Gateway / captcha detection lives in `browser/gateway.py` (`detect_gateway_block`, `detect_live_gateway_block`). See [`docs/VALIDATION.md`](../docs/VALIDATION.md) §7 for integration history.
