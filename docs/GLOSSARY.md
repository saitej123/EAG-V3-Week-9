# Glossary — Browser skill and course context

Terms and libraries used in the Browser skill docs ([`BROWSER.md`](BROWSER.md)).

---

## Python libraries

**httpx** — A modern Python library for making HTTP requests. Similar to the older `requests` library but supports async and HTTP/2. The Browser skill uses it in Layer 1 to download pages without launching a browser.

Docs: [python-httpx.org](https://www.python-httpx.org/)

**trafilatura** — Takes messy HTML from a real web page and pulls out the main article text. Removes navigation, ads, footers, and other clutter. The Browser skill uses it in Layer 1 right after httpx downloads the page.

Docs: [trafilatura.readthedocs.io](https://trafilatura.readthedocs.io/)

**crawl4ai** — Async web crawler that loads pages in a headless browser and returns clean markdown. Used by the **Researcher** skill only (`fetch_url`, `fetch_urls`, and step 2 of the `web_search` pipeline: Tavily → crawl4ai → Gemini → DuckDuckGo). **Not** used in the Browser cascade — Browser Layer 1 uses httpx + trafilatura instead.

Repo: [unclecode/crawl4ai](https://github.com/unclecode/crawl4ai)

**Playwright** — A tool from Microsoft for controlling a real Chromium, Firefox, or WebKit browser from code. You write Python (or JavaScript or other languages); Playwright clicks buttons and types text in an actual browser. The Browser skill uses Playwright for Layers 2a, 2b, and 3.

Docs: [playwright.dev/python](https://playwright.dev/python/)

**Pillow** — The standard Python library for working with images. Open, edit, draw on, and save images. The Browser skill uses Pillow to draw numbered boxes on screenshots for set-of-marks.

Docs: [pillow.readthedocs.io](https://pillow.readthedocs.io/)

**FAISS** — A vector search library from Meta. Given a query vector, it finds the closest matching vectors in a stored collection. Session 7 added FAISS for semantic memory.

Repo: [facebookresearch/faiss](https://github.com/facebookresearch/faiss)

**NetworkX** — A Python library for graphs (nodes connected by edges). Session 8 uses it as the substrate for the multi-agent DAG.

Docs: [networkx.org](https://networkx.org/)

**Pydantic** — Validates data against typed models. Every typed boundary between agents in the course uses Pydantic. `AgentResult`, `BrowserOutput`, and `NodeSpec` are all Pydantic models.

Docs: [docs.pydantic.dev](https://docs.pydantic.dev/)

**SQLite** — A small, file-based database built into Python. The V9 cost ledger writes to a SQLite file. No server is needed.

Docs: [sqlite.org](https://www.sqlite.org/)

---

## Protocols and standards

**MCP (Model Context Protocol)** — An open protocol for an AI model to call external tools. Session 4 covered MCP in depth. The course uses MCP for `web_search`, `fetch_url`, `search_knowledge`, and other tools.

Docs: [modelcontextprotocol.io](https://modelcontextprotocol.io/)

**CDP (Chrome DevTools Protocol)** — The wire format Chrome DevTools uses to talk to Chrome. Playwright uses the same protocol under the hood to control Chromium. The Browser skill uses lower-level CDP only for debugging the full accessibility tree.

Docs: [chromedevtools.github.io/devtools-protocol](https://chromedevtools.github.io/devtools-protocol/)

**ARIA (Accessible Rich Internet Applications)** — A web standard for accessibility labels. Developers add attributes like `role="button"` or `aria-label="Sort by likes"` so screen readers know what each element is. The accessibility tree is built from ARIA plus underlying HTML semantics.

Docs: [W3C ARIA](https://www.w3.org/WAI/standards-guidelines/aria/)

**DOM (Document Object Model)** — The structured representation of a web page that JavaScript can read and modify. DevTools → Elements tab shows the DOM. Pages can be 6 MB of DOM and still only 30 KB of accessibility tree.

Docs: [MDN — Document Object Model](https://developer.mozilla.org/en-US/docs/Web/API/Document_Object_Model)

---

## Concepts

**Accessibility tree (a11y tree)** — A parallel structured view of the page that browsers maintain for screen readers. Contains only meaningful elements: buttons, links, headings, form fields, landmarks. Strips CSS, scripts, hidden elements, and decoration. “a11y” is shorthand for “accessibility” (a, then 11 letters, then y).

**Set-of-marks** — A technique for letting a vision-language model pick an element on a page. Take a screenshot, draw numbered boxes over each clickable element, ask the model to pick a number. Standard input format for visual web agents since ~2024.

Background: [Set-of-Marks paper (arXiv:2310.11441)](https://arxiv.org/abs/2310.11441)

**VLM (Vision-Language Model)** — A model that reads both text and images. Examples in 2026: Gemini 3.1 Pro, GPT-5.5, Claude Opus 4.7, Qwen2.5-VL. The Browser skill uses Gemini 3.1 Flash-Lite by default through V9.

**LLM (Large Language Model)** — A model that reads and writes text. Planners, distillers, and formatters in this course are LLMs. A VLM is an LLM that can also read images.

**DPR (Device Pixel Ratio)** — On high-resolution displays (e.g. Retina), one CSS pixel maps to more than one screen pixel. A button at CSS (200, 300) is at screen (400, 600) on a 2× display. Set-of-marks must account for DPR so boxes land on the right elements in the screenshot.

**Headless browser** — A real browser (Chromium, Firefox, WebKit) running without a visible window. The page loads, JavaScript runs, the DOM is built. Only the window on your screen is missing. Headless is faster and uses less memory than headed browsing.

**CSS selectors** — Rules that pick elements on a page. Examples: `.product-title` (class), `#main-button` (id), `div.card > h2` (direct child). Layer 2a uses hand-written CSS selectors.

Reference: [MDN — CSS selectors](https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_selectors)

**CAPTCHA** — A challenge that asks “are you a human?” (traffic-light squares, wavy text, hold-the-button, etc.). CAPTCHAs block agents at the precondition layer (`gateway_blocked`).

**Popover, dropdown** — Small UI panels that appear when a trigger is clicked. A sort menu showing “Most likes / Most downloads / Newest” is a popover. Items inside do not exist in the DOM until the trigger is clicked — the canonical Layer 2b lesson on Hugging Face.

**DAG (Directed Acyclic Graph)** — A graph of nodes connected by directed edges with no cycles. Session 8 uses a DAG for the plan: nodes are agents, edges are dependencies, execution flows along the edges.

---

## Sites and applications mentioned

| Name | Role in this session | Link |
|------|----------------------|------|
| **Hugging Face** | Layer 2b canonical test — filter model index by tag, sort by popularity | [huggingface.co](https://huggingface.co/) |
| **Excalidraw** | Set-of-marks dedupe diagnostic | [excalidraw.com](https://excalidraw.com/) |
| **tldraw, Photopea, Piskel, OpenProcessing** | §9 canvas-heavy targets; natural cascade chose a11y, not vision | — |
| **Redfin** | Optional gateway-block demo (not in browser demo corpus) | — |

---

## Frameworks the course does NOT use

**LangChain, LlamaIndex, CrewAI, AutoGen** — Third-party agentic frameworks. The shipped path does not use any of them. This repo implements the gateway, skill catalogue, and orchestrator directly so each piece stays understandable.

The Browser skill stack is httpx, trafilatura, Playwright, Pillow, and the Gemini SDK — see [`BROWSER.md`](BROWSER.md).
