# Validation notes — Browser integration

Cross-checks between the course spec, shipped code, and deliberate deviations.

---

## 1. Shipped Browser stack

The Browser skill uses **Playwright**, **Pillow**, **httpx**, and **trafilatura**. Vision layers add the Gemini SDK. **crawl4ai** is Researcher-only (`fetch_url`, `fetch_urls`, `web_search`) — not part of the Browser cascade.

See [`BROWSER.md`](BROWSER.md) and [`prompts/browser.md`](../prompts/browser.md).

---

## 2. Natural cascade vs `force_path`

The orchestrator escalates until content is sufficient (and action count is met for comparison tasks). That natural cascade is the **default**.

`force_path` in browser node metadata is **opt-in** for two cases:

1. **Debugging** — force a single layer during tests or cascade lab demos (**B1**–**B4**).
2. **Rare production** — caller already knows vision is required (e.g. a downstream skill attached a screenshot artifact and wants Browser to act on it).

---

## 3. Driver core (experimental port)

Driver behaviour under `super_browser/browser/` was ported from the experimental phase with minimal structural change. Shared turn execution lives in `driver.py` (`execute_action`, action fencing, dropdown-as-fence rules). Interactive layers differ only in **decision** logic:

| Layer | Decides via |
|-------|-------------|
| **a11y** | Accessibility tree snapshot + text LLM (`a11y.py`) |
| **agent** | Indexed clickables + text LLM (`agent` loop in cascade) |
| **vision** | Set-of-marks screenshot + VLM; coordinate fallback (`vision.py`, `highlight.py`) |

The experimental **BaseDriver** pattern (SetOfMarksDriver vs A11yDriver differing only in `_decide()`) maps to this split: shared `driver.py` execution, layer-specific prompts and parsers.

---

## 4. BrowserOutput.path values

Replay section 3 and distiller upstream text use `BrowserOutput.path`: `extract`, `render`, `agent`, `deterministic`, `a11y`, `vision`, `gateway_blocked`, `failed`.

---

## 5. Comparison task action floor

Assignment comparisons require **≥3 logged browser actions** when `min_browser_actions` is set (see `corpus/dag/ASSIGNMENT.json`). Layer 1 extract is skipped when that floor applies so Playwright layers can record interactions.

---

## 6. Researcher fallback (not Browser retry loop)

On `gateway_blocked` / `browser_exhausted`, the orchestrator may queue **Researcher** → distiller → formatter — not another Browser node. See `super_browser/recovery.py`.

---

## 7. `detect_gateway_block` placement (spec deviation — resolved)

**Original integration pass:** `detect_gateway_block` was kept in `browser/skill.py` instead of `browser/dom.py` because the pass was instructed to leave the driver core untouched. That was documented here as a small structural deviation.

**Current tree:** the helper lives in **`browser/gateway.py`**, imported by `extract.py`, `navigation.py`, and the cascade. Live-widget detection is `detect_live_gateway_block(page)` in the same module.

No further move is required unless new gateway heuristics are added — extend `dom.py` only.

---

## 8. Optional browser-use bridge

The shipped browser path is httpx + trafilatura + Playwright + direct Gemini (`drivers/`).
