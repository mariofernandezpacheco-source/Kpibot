#!/usr/bin/env bash
set -euo pipefail
if [ -f ".env" ]; then export $(grep -v '^#' .env | xargs); fi

echo "▶ CV: validación temporal…"
poetry run python 03_time_series_cv.py --timeframe "${TIMEFRAME:-10min}" --tickers "${TICKERS:-AAPL}"
echo "✅ CV completada"
