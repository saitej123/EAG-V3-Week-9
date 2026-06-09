<p align="center">
  <img src="Images/app-icon.svg" alt="Super Browser Agent" width="96" height="96"/>
</p>

<h1 align="center">Super Browser Agent</h1>

<p align="center">
  <strong>Four-layer browser cascade · comparison tasks · replay viewer</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#browser-cascade">Browser cascade</a> ·
  <a href="#demo-queries">Demo queries</a> ·
  <a href="#replay">Replay</a>
</p>

A browser-first agent stack: compare and extract from live web pages through a cost-disciplined cascade (extract → deterministic → a11y → vision), orchestrated as a parallel skill DAG with critic recovery and a full replay viewer.

## Quick start

```bash
uv sync
playwright install chromium
cp .env.example .env   # GEMINI_API_KEY
uv run python scripts/browser/seed_browser_sessions.py
./scripts/serve.sh
```

Open **http://127.0.0.1:8080/**

| Action | Where |
|--------|--------|
| Browser comparison | Welcome chip **COMP** or sidebar **Tasks** |
| Layer demos | **B1**–**B4** in sidebar **Tasks** |
| Replay report | Sidebar **Graph & Replay** → **Replay report** |
| Reference runs | Tasks → **Open COMP replay demo** |

```bash
uv run python scripts/dag/run_query.py COMP
uv run python scripts/browser/export_browser_replay.py dag_COMP_ref -o replay.md
```

## Browser cascade

| Layer | Mechanism | When it wins |
|-------|-----------|--------------|
| 1 extract | httpx + trafilatura | Static HTML, news pages |
| 2a deterministic | Playwright + CSS | Product pages, forms |
| 2b a11y | Playwright + a11y tree + text LLM | Filters, dropdowns (HF models) |
| 3 vision | Set-of-marks + VLM | Canvas-only / adversarial UI |

Primary task **COMP**: top-3 Hugging Face model comparison (≥3 visible browser actions). Reference session `dag_COMP_ref` uses path **a11y** with a comparison table in formatter output.

## Demo queries

Browser tasks only in the UI (**COMP**, **B1**–**B4**). Full assignment corpus: `corpus/dag/ASSIGNMENT.json` · [`docs/BROWSER.md`](docs/BROWSER.md)

| Id | Focus |
|----|-------|
| **COMP** | Hugging Face top-3 comparison (≥3 browser actions) |
| **B1**–**B4** | Cascade layer demos (extract, deterministic, a11y, vision) |

## Replay

Eight-section report (Graph UI grey panel + export):

```bash
uv run python scripts/browser/export_browser_replay.py dag_COMP_ref
```

Goal · planner DAG · browser path · actions · page logs · extracted data · comparison table · cost summary.

## Architecture

```
USER_QUERY → Planner → Browser cascade → Distiller → Formatter
                              │
                              └── Replay viewer (/api/dag/browser-replay)
```

| Piece | Location |
|-------|----------|
| Package | `super_browser/` |
| Browser cascade | `super_browser/browser/skill.py` |
| Replay report | `super_browser/browser/replay.py` |
| Orchestrator | `super_browser/flow.py` |
| Skill catalogue | `agent_config.yaml` + `prompts/browser.md` |
| Scripts | `scripts/browser/` |
| Canvas fixture | `sandbox/browser/canvas-only.html` |

Docs: [`docs/BROWSER.md`](docs/BROWSER.md) · [`docs/GLOSSARY.md`](docs/GLOSSARY.md)

## Tests

```bash
uv run pytest tests/test_browser.py tests/test_assignment_spec.py tests/test_dag_queries_api.py -q
```
