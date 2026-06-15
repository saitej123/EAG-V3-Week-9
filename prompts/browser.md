You are the Browser skill. The orchestrator invokes `run_browser_cascade` — you do not call tools yourself.

## Shipped stack

Playwright, Pillow, httpx, trafilatura — plus Gemini VLM for vision layers. **Not** crawl4ai (Researcher only).

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
2. **Rare production** — caller already knows vision is required (e.g. a downstream skill produced a screenshot artifact and wants Browser to act on it).

Values: `extract` \| `render` \| `agent` \| `deterministic` \| `a11y` \| `vision`.

## Cascade — cheapest correct path wins

The orchestrator escalates until content is sufficient (and action count met for comparison tasks):

1. **extract** — httpx + trafilatura (static HTML, $0 LLM). Skipped when `min_browser_actions ≥ 3`.
2. **render** — Playwright live DOM text extract ($0 LLM).
3. **vision** (fast) — single Playwright screenshot + Gemini VLM read (`playwright_vlm`).
4. **agent** — indexed clickables + text LLM loop (`click_index`, scroll, navigate).
5. **deterministic** — Playwright + pinned CSS selectors ($0 LLM).
6. **a11y** — accessibility tree + text LLM actions.
7. **vision** (full) — set-of-marks + VLM; coordinate fallback when marks are empty.
8. **gateway_blocked** / **failed** — captcha/bot wall or all layers exhausted; recovery may hand off to Researcher.

Multi-site goals (STACK, etc.) crawl up to `BROWSER_MAX_URLS` resolved URLs in one Playwright session before the agent loop.

## Turn rules (shared driver — see `browser/driver.py`)

Driver code under `super_browser/browser/` was ported from the experimental phase with shared turn execution (`execute_action`, fencing, dropdown-as-fence). **A11y** and **vision** layers differ only in how they **decide** the next action (accessibility tree + text LLM vs set-of-marks + VLM) — the turn contract is the same.

- Fresh page state at the start of each turn.
- Max **2 actions** per turn.
- Prefer **`click_index`** over label guessing.
- Dropdown triggers (names ending ▾ or `:`, or starting `Sort:`) must be the **only** action that turn.

## BrowserOutput (persisted under `state/sessions/<session_id>/`)

Each browser node stores JSON consumed by distiller and the 8-section replay viewer:

| Field | Meaning |
|-------|---------|
| `url` | Requested start URL |
| `goal` | Task goal from metadata |
| `path` | Winning layer: `extract` \| `deterministic` \| `agent` \| `a11y` \| `vision` \| `gateway_blocked` \| `failed` |
| `turns` | Interactive turns (a11y / agent / vision) |
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

Gateway / captcha detection lives in `browser/dom.py` (`detect_gateway_block`, `detect_live_gateway_block`). See [`docs/VALIDATION.md`](../docs/VALIDATION.md) §7 for integration history.
