# Super Browser Agent — FastAPI Web UI + MCP-backed DAG orchestrator + document RAG
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright \
    CRAWL4AI_BASE_DIRECTORY=/app/.crawl4ai \
    AGENT_MAX_ITERATIONS=3 \
    AGENT_ITERATION_CEILING=4 \
    AGENT_RUN_MAX_SECONDS=900 \
    AGENT_LLM_STEP_TIMEOUT_SEC=60 \
    LLM_RETRY_MAX=3 \
    LLM_RETRY_SLEEP_SEC=2.0 \
    LLM_RETRY_BACKOFF=1.5 \
    VLM_PAGE_BATCH_SIZE=10 \
    VLM_BATCH_SLEEP_SEC=1.0 \
    INDEX_FILE_SLEEP_SEC=0.35 \
    GEMINI_EMBED_MODEL=gemini-embedding-2

WORKDIR /app

# System deps for Playwright/Chromium (crawl4ai fetch fallback) + lxml source-build fallback
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    libxml2-dev libxslt1-dev \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
    libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
    libgbm1 libpango-1.0-0 libcairo2 libasound2 libatspi2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

# Browser for crawl4ai + browser cascade (needed for fetch_url and render layers).
# --with-deps installs Chromium's OS libraries; PLAYWRIGHT_BROWSERS_PATH keeps the
# binary at a fixed location so it is found at runtime.
RUN uv run playwright install --with-deps chromium

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8000}/health" | grep -q '"status"' || exit 1

# Honor the platform-injected $PORT (Railway/Render/Cloud Run); default to 8000 locally.
CMD ["sh", "-c", "uv run uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
