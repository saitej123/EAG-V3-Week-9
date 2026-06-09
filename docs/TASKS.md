# DAG demo tasks

This document maps each task part to **query ids** and **verification commands**.

## Architecture (intact)

| Rule | Implementation |
|------|----------------|
| Skills = yaml + prompt | `agent_config.yaml` + `prompts/*.md` |
| Planner emits graph | `extends_graph: true` on planner skill |
| Executor runs ready nodes in parallel | `asyncio.gather` in `super_browser/flow.py` |
| Critic between flagged producer and successor | Auto-splice on `distiller` (`critic: true`) |
| Recovery on critic fail | `_handle_critic` → recovery planner via `classify_failure` |
| Coder → sandbox_executor | `internal_successors: [sandbox_executor]` |
| New skill = yaml edit only | **prosody_analyst** (Part 5 demo) — no Executor changes |
| Catalogue skill (no UI card) | **calculator** — yaml + `prompts/calculator.md`; planner routes arithmetic |

Run architecture tests:

```bash
uv run pytest tests/test_dag_flow.py tests/test_recovery.py tests/test_worked_queries.py tests/test_dag_mcp_tools.py tests/test_assignment_spec.py -q
```

## Run all corpus queries

```bash
uv run python scripts/dag/run_eval.py --fresh
```

Parallel timing proof (after query **P** or **I**):

```bash
uv run python scripts/dag/analyze_session_timing.py dag_P_<timestamp> --json
```

---

## Part 1 — Base queries (hello, A, I, J, K)

Sanity check plus S7 carryover (no behavioural regression) plus resume. **Traces:** `state/sessions/<session_id>/` (`query.txt`, `graph.json`, `nodes/*.json`).

| Id | Role | Expected DAG | Wall bound |
|----|------|--------------|------------|
| **hello** | Minimum DAG | 2 nodes: planner → formatter only | 15s (lecture: &lt;3s) |
| **A** | S7 Shannon Wikipedia | 4+ nodes: researcher → distiller → critic (auto) → formatter | 180s |
| **I** | Parallel fan-out (canonical) | 7 nodes: 3× researcher ∥ → coder → formatter ∥ sandbox_executor | 120s (lecture ~62s) |
| **J** | Graceful failure | 2 nodes: planner → formatter (fail-fast; no tools) | 30s |
| **K** | Resumable execution | Same shape as **I**; kill mid parallel researchers, then resume | 180s |

**Query I** (populations) is the Session 8 headline: three researchers finish on the same `asyncio.gather` barrier; token use is scoped per node (contrast with S7 iteration history). After a run:

```bash
uv run python scripts/dag/analyze_session_timing.py dag_I_<timestamp> --json
```

**Query K** (resume): run until parallel researchers are in flight, then `kill -9` the process. Resume (query read from `query.txt`):

```bash
uv run python scripts/dag/run_query.py --resume dag_K_<timestamp>
# or: uv run python scripts/dag/run_eval.py --ids K --resume  # same session id, no --fresh
```

Expected DAG shapes (structural, no live LLM): `tests/test_worked_queries.py` · `corpus/dag/ASSIGNMENT.json` (`worked_query_ids`: hello–K).

### Optional: Gateway V8 (separate package)

`llm_gatewayV8` is **not** in this repo. DAG mode uses the **Gemini SDK** by default. To route through V8 (agent/session labels, `/v1/cost/by_agent`, pinned providers):

```bash
# In .env — must be GATEWAY_V8_URL (8108), not GATEWAY_URL (8107 / S7 loop only)
GATEWAY_V8_URL=http://127.0.0.1:8108
```

Pins: `agent_routing.yaml`. If the gateway returns 5xx, `SkillLLMClient` retries then falls back to Gemini. Use **Session8StartingCodePatched** (not the older starting zip with the known gateway bug many students hit).

---

## Part 2 — Custom parallel fan-out (≥3 concurrent researchers)

| Id | Query | Verification |
|----|-------|--------------|
| P | Find the current population of Tokyo, Mumbai, and São Paulo and tell me which city has the largest population. | Planner emits 3 researcher nodes in one wave; `analyze_session_timing.py` shows `parallel_confirmed=true` (wall ≈ max branch, not sum) |

---

## Part 3 — Critic pass + fail with recovery

| Id | Query | Expected |
|----|-------|----------|
| C_pass | Validate JSON `{"author":"Ada Lovelace","title":"Notes","year":1843}` — critic verifies keys `author,title,year` via `validate_json_keys` | Critic **pass** → formatter |
| C_fail | Same but JSON missing `year` — critic must fail, recovery planner adds field | Critic **fail** → recovery planner node in session graph |

Critic tools: `validate_json_keys`, `count_syllables` in `super_browser/mcp_server.py`.

---

## Part 4 — Coder + SandboxExecutor

| Id | Query | Expected |
|----|-------|----------|
| M | What is the exact integer value of `(17 * 23 - 4) ** 2 + 1000`? Use coder to compute; sandbox must verify. | DAG: planner → coder → sandbox_executor → formatter; answer **150769** |

Coder prompt: `prompts/coder.md` (JSON `{code, summary}` for SandboxExecutor).

---

## Part 5 — New skill: prosody analyst

**Task goal:** add one skill to `agent_config.yaml` that the existing catalogue did not cover; write its prompt; write **one** query that exercises it. The orchestrator must not need modification (if it did, that would be reportable).

| Item | Location |
|------|----------|
| Skill yaml | `agent_config.yaml` → `prosody_analyst` |
| Prompt | `prompts/prosody_analyst.md` |
| Planner routing | `prompts/planner.md` — syllable comparison → `prosody_analyst` |
| Tool | `count_syllables` in `mcp_server.py` (pre-existing; critic also uses it) |
| **UI demo query** | **PROS** only — three DAG-themed lines; count per line → **B=17** wins (A=11, C=13) |

**Why this skill is new:** the catalogue already had planner, researcher, retriever, distiller, summariser, critic, coder, sandbox_executor, formatter, browser, and **calculator**. None of them perform **multi-line syllable comparison** as a dedicated step. `count_syllables` existed as a critic tool only; `prosody_analyst` gives it a first-class skill + prompt.

**Orchestrator:** no changes to `super_browser/flow.py` — generic `SkillRegistry` dispatch runs `prosody_analyst` like any other skill.

### Calculator — catalogue only (no DAG Queries card)

`calculator` remains in the skill catalogue (`agent_config.yaml`, `prompts/calculator.md`, `safe_calculate` tool). The planner already routes arithmetic through `calculator → formatter`. It does **not** need a separate demo card in the DAG Queries UI — the skill is known to the orchestrator and can be invoked from Chat with any numeric query.

| Item | Location |
|------|----------|
| Skill yaml | `agent_config.yaml` → `calculator` |
| Prompt | `prompts/calculator.md` |
| Tool | `safe_calculate` in `mcp_server.py` |
| Example (Chat, not a corpus id) | `(987654321 ** 0) + ((17 * 23 + 41) / 7)` → **≈ 62.714** |

**Live demo — show (Part 5):**

1. `agent_config.yaml` — both `prosody_analyst` (new) and `calculator` (existing catalogue entry)
2. Run **PROS** from DAG Queries (Part 5 design block) — this is the primary prosody demo
3. Graph tab: `planner → prosody_analyst → formatter`
4. Working panel: three `[dag]` tool calls to `count_syllables`
5. Final answer: Line **B** has the most syllables (17)

```bash
uv run python scripts/dag/run_query.py PROS
# or: uv run python scripts/dag/run_eval.py --fresh --ids PROS
```

Optional ad-hoc calculator check (Chat composer, not in `corpus/dag/ASSIGNMENT.json`):

```text
What is the exact numeric value of (987654321 ** 0) + ((17 * 23 + 41) / 7)? Use the calculator skill.
```

---

## Part 6 — Browser comparison agent + replay viewer

| Item | Location |
|------|----------|
| Browser skill | `super_browser/browser/` — four-layer cascade |
| Catalogue | `agent_config.yaml` → `browser` |
| Primary query | **COMP** — Hugging Face top 3 comparison table |
| Layer demos | **B1**–**B4** (extract, deterministic, a11y, vision) |
| Replay report | `super_browser/browser/replay.py` → `GET /api/dag/browser-replay` |
| Graph UI | Graph tab → click browser node → **Browser replay** panel |
| Docs | [`docs/BROWSER.md`](BROWSER.md), [`docs/GLOSSARY.md`](GLOSSARY.md) |

```bash
uv run python scripts/dag/run_query.py COMP
uv run python scripts/browser/seed_browser_sessions.py
uv run python scripts/browser/analyze_browser_session.py dag_B3_ref --json
```

Orchestrator unchanged — browser plugs in via `skills.py` dispatch only.

---

## YouTube demo checklist

Record one walkthrough showing:

1. Base query **hello** (minimal DAG)
2. Parallel query **P** + timing script output
3. Critic **C_pass** then **C_fail** with recovery node in graph
4. Coder query **M** with sandbox stdout
5. Prosody analyst query **PROS** (new skill demo). Mention **calculator** in yaml only — no UI card.

Video: [YouTube demo](https://www.youtube.com/watch?v=6bLVCm2XcJc) (also linked in README § Demo).
