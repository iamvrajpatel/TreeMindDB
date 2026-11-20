#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$PROJECT_DIR/.env" ]]; then
  set -a
  source "$PROJECT_DIR/.env"
  set +a
fi
cd "$PROJECT_DIR/backend"
exec ../.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
