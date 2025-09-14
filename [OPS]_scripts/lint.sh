#!/usr/bin/env bash
set -euo pipefail

# Choose how to run tools: poetry -> binaries -> python -m ...
run_ruff_check() {
  if command -v poetry >/dev/null 2>&1; then
    poetry run ruff check .
  elif command -v ruff >/dev/null 2>&1; then
    ruff check .
  else
    python -m ruff check .
  fi
}

run_black_check() {
  if command -v poetry >/dev/null 2>&1; then
    poetry run black --check .
  elif command -v black >/dev/null 2>&1; then
    black --check .
  else
    python -m black --check .
  fi
}

echo "▶ LINT: Ruff + Black (check)"
run_ruff_check
run_black_check
echo "✅ Lint OK"
