#!/usr/bin/env bash
set -euo pipefail

run_black_fmt() {
  if command -v poetry >/dev/null 2>&1; then
    poetry run black .
  elif command -v black >/dev/null 2>&1; then
    black .
  else
    python -m black .
  fi
}

echo "▶ FORMAT: Black (auto-formato)"
run_black_fmt
echo "✅ Formato aplicado"
