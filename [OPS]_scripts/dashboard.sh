#!/usr/bin/env bash
set -euo pipefail
if [ -f ".env" ]; then export $(grep -v '^#' .env | xargs); fi

PORT="${PORT:-8501}"
echo "▶ DASHBOARD: http://localhost:${PORT}"
poetry run streamlit run 06_dashboard.py --server.port "${PORT}"
