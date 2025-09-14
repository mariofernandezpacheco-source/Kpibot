#!/usr/bin/env bash
set -euo pipefail

# Carga .env si existe
if [ -f ".env" ]; then export $(grep -v '^#' .env | xargs); fi

echo "▶ PREP: descargando/preparando datos…"
poetry run python 01_data_download.py --timeframe "${TIMEFRAME:-10min}" --tickers "${TICKERS:-AAPL}" --data-dir "${DATA_DIR:-./data}"
echo "✅ PREP completado"
