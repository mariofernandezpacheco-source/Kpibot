#!/usr/bin/env bash
set -euo pipefail
if [ -f ".env" ]; then export $(grep -v '^#' .env | xargs); fi

echo "▶ WORKER: lanzando bot…"
poetry run python 05_bot_worker.py --timeframe "${TIMEFRAME:-10min}" --tickers "${TICKERS:-AAPL}"
