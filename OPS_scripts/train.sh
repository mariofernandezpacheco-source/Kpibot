#!/usr/bin/env bash
set -euo pipefail
if [ -f ".env" ]; then export $(grep -v '^#' .env | xargs); fi

echo "▶ TRAIN: entrenamiento…"
poetry run python 04_model_training.py --timeframe "${TIMEFRAME:-10min}" --tickers "${TICKERS:-AAPL}"
echo "✅ TRAIN completado"
