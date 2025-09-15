#!/usr/bin/env bash
set -euo pipefail
if [ -f ".env" ]; then export $(grep -v '^#' .env | xargs); fi

EXTRA=""
if [ "${EOD_FLATTEN:-0}" = "1" ]; then EXTRA="--eod_flatten"; fi

echo "▶ BACKTEST: ejecutando…"
poetry run python engine/run_backtest.py \
  --timeframe "${TIMEFRAME:-10min}" \
  --tickers "${TICKERS:-AAPL}" \
  --tp_pct "${TP_PCT:-0.005}" \
  --sl_pct "${SL_PCT:-0.005}" \
  ${EXTRA}
echo "✅ BACKTEST completado"
