#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Wrong VIRTUAL_ENV (e.g. another repo) makes `uv sync --active` install into the wrong venv.
unset VIRTUAL_ENV

uv sync --extra dev

VENV_PY="$ROOT/.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "error: $VENV_PY not found after uv sync" >&2
  exit 1
fi

# Fail fast before uvicorn — reload subprocess must use this same interpreter.
"$VENV_PY" -c "
import importlib
for mod in ('trafilatura', 'httpx', 'playwright', 'yaml', 'faiss', 'lxml', 'PIL'):
    importlib.import_module(mod)
print('deps ok:', __import__('sys').executable)
"

# Playwright Python package != browser binaries. Install Chromium once if missing.
if ! "$VENV_PY" -c "
from super_browser.browser.playwright_ctx import playwright_chromium_status
ready, err = playwright_chromium_status(refresh=True)
if not ready:
    raise SystemExit(err or 'playwright chromium missing')
print('playwright chromium ok')
" 2>/dev/null; then
  echo "Installing Playwright Chromium (one-time download)..."
  "$VENV_PY" -m playwright install chromium
  if ! "$VENV_PY" -c "
from super_browser.browser.playwright_ctx import playwright_chromium_status
ready, err = playwright_chromium_status(probe_launch=True, refresh=True)
if not ready:
    raise SystemExit(err or 'playwright chromium still missing after install')
print('playwright chromium ok')
"; then
    echo "Playwright launch failed — installing Linux browser dependencies (WSL/Docker)..."
    "$VENV_PY" -m playwright install-deps chromium || true
    "$VENV_PY" -m playwright install chromium
    "$VENV_PY" -c "
from super_browser.browser.playwright_ctx import playwright_chromium_status
ready, err = playwright_chromium_status(probe_launch=True, refresh=True)
if not ready:
    raise SystemExit(err or 'playwright chromium still missing after install-deps')
print('playwright chromium ok')
"
  fi
fi

# Use project venv directly so StatReload workers inherit .venv (not uv's global python).
exec "$VENV_PY" -m uvicorn app:app --reload --host 127.0.0.1 --port "${PORT:-8080}" "$@"
