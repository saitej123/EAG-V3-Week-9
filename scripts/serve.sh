#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec uv run uvicorn app:app --reload --host 127.0.0.1 --port "${PORT:-8080}" "$@"
