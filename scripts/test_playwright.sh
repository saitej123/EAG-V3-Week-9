#!/usr/bin/env bash
# Quick Playwright diagnostic — run from project root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
unset VIRTUAL_ENV
uv sync --extra dev -q
VENV_PY="$ROOT/.venv/bin/python"

echo "=== Python ==="
"$VENV_PY" -c "import sys; print(sys.executable)"

echo "=== Package import ==="
"$VENV_PY" -c "import playwright; print('playwright ok')"

echo "=== Chromium binary ==="
"$VENV_PY" -c "
from super_browser.browser.playwright_ctx import playwright_chromium_status
ready, err = playwright_chromium_status(refresh=True)
print('binary:', ready, err or '')
"

echo "=== Launch probe (sync) ==="
"$VENV_PY" -c "
from super_browser.browser.playwright_ctx import playwright_chromium_status
ready, err = playwright_chromium_status(probe_launch=True, refresh=True)
print('launch:', ready, err or '')
if not ready:
    raise SystemExit(1)
"

echo "=== Live navigation ==="
"$VENV_PY" -c "
import asyncio
from super_browser.browser.playwright_ctx import browser_session

async def main():
    async with browser_session() as page:
        await page.goto('https://example.com', wait_until='domcontentloaded', timeout=45000)
        print('title:', await page.title())

asyncio.run(main())
"

echo "=== pytest live suite ==="
uv run pytest tests/test_playwright_live.py -q

echo "Playwright OK"
