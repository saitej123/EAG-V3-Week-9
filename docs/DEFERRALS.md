# Deliberate design simplifications

Production RAG systems combine many moving parts. This project implements a **working dense-retrieval stack** end-to-end and **defers** several upgrades to future releases. Each deferral below is intentional and has an explicit forward pointer.

---

## 1. Dense retrieval only (no hybrid sparse partner)

Vector retrieval in this codebase has **no hybrid sparse partner**. Production retrieval systems run a sparse retriever (BM25 or learned-sparse) and a dense retriever in parallel and combine the ranked lists with **Reciprocal Rank Fusion (RRF)**. The current implementation uses **dense retrieval alone** (FAISS + embeddings via `memory.read()`).

**Where it lives:** `super_browser/memory.py` — vector-first `read()` with keyword overlap as fallback only when embeddings or the index are unavailable.

**Forward pointer:** hybrid retrieval + RRF inside `Memory.read()`; same external MCP interface (`search_knowledge`).

---

## 2. Heuristic chunking (sliding window)

Chunking is **heuristic**. The sliding window splits documents at **arbitrary word boundaries** (default **400 words**, **80 overlap** in `index_document` / `indexing.py`). VLM-indexed pages may sub-chunk long page text, but boundaries are still fixed-size, not semantic.

**Where it lives:** `super_browser/indexing.py` — `_chunk_text()`, `index_document_path()`.

**Forward pointer:** semantic chunking (LLM-aware sentence/paragraph/section breaks) as its own typed module.

---

## 3. FAISS reloaded from disk on every read

The FAISS index is **reloaded from disk on every** `memory.read()` call. The cost is small at demo scale. At higher scale, a **memory-mapped index** with file-modification-time invalidation, or an **inter-process lock**, becomes the right pattern. This project does not attempt to model that scale yet.

**Why:** MCP `index_document` runs in a subprocess; in-process FAISS caching would hide subprocess writes from the agent. The implementation trades a few milliseconds of disk I/O per read for **lock-free cross-process consistency**.

**Where it lives:** `super_browser/memory.py` — `_load_faiss_from_disk()` on every `read()`.

**Forward pointer:** mtime-aware mmap cache or shared index service.

---

## 4. Fixed embedding model (pinned semantic space)

The gateway / SDK path **pins the embedding model** (`GEMINI_EMBED_MODEL`, default **`gemini-embedding-2`**) so that all vectors in the FAISS index live in the **same semantic space**. Changing the model **silently invalidates** every vector previously stored.

**Remedy:** delete `state/index.faiss` and `state/index_ids.json`, then rebuild from `state/memory.json` (original text remains in each item's `value` and `descriptor`), or run `scripts/clean.py` and re-index the corpus.

**Where it lives:** `super_browser/llm_env.py` — `gemini_embed_model()`, `try_embed_text()`.

**Forward pointer:** explicit re-embed / index-version migration tooling.

---

## Forward pointer roadmap

| Upgrade | Addresses |
|---------|-----------|
| **Semantic chunking** | Replaces sliding window; chunker as its own typed module |
| **Hybrid retrieval + RRF** | Dense baseline + sparse BM25/learned-sparse path; fusion inside `Memory.read` |
| **Parallel fan-out (DAG agent)** | Sequential `index_document` calls → concurrent nodes in `asyncio.TaskGroup` |
| **Skills abstraction** | Perception attaches capability labels; Decision receives a filtered tool subset |
| **Cross-encoder reranking** | Second-stage ranker over hybrid top-*k* |
| **FAISS mmap + mtime cache** | Scale beyond demo reload-on-every-read |

These deferrals are **deliberate**. This codebase proves the four-role agent loop, durable FAISS memory, and document RAG; future releases harden retrieval quality and operational scale without changing the external tool contract.

---

## 5. Resume at node boundary (not tool-call boundary)

The DAG orchestrator persists the graph and per-node state atomically under `state/sessions/<sid>/`. **Resume** resets `running` nodes to `pending` and re-runs them from the top.

A Researcher killed mid tool-use loop does **not** resume from tool call four — it re-issues every tool call. The cost is real for long-running Researchers.

**Forward pointer:** ~60 lines in an MCP runner to persist in-flight message lists inside `NodeState` and restore on resume.

See also: [`docs/NOTES_RUNS.md`](NOTES_RUNS.md) (worked query notes), [`README.md`](../README.md) (architecture overview).
