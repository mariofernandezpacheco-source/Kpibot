# DAT_Data_download.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

import settings as settings
S = settings.S

from utils.data_update import (
    get_tickers_from_file,
    update_many,
    last_available_table,
)

def main():
    ap = argparse.ArgumentParser("[DAT] Data Download (IBKR → Parquet incremental)")
    ap.add_argument("--timeframe", default=getattr(S, "timeframe_default", "5mins"))
    ap.add_argument("--tickers-file", default=str(Path(S.config_path) / "sp500_tickers.txt"))
    ap.add_argument("--only-show-last", action="store_true", help="Solo mostrar última fecha por ticker (no descarga)")
    args = ap.parse_args()

    tickers_file = Path(args.tickers_file)
    tickers = get_tickers_from_file(tickers_file)

    if args.only_show_last:
        df = last_available_table(tickers, args.timeframe)
        print(df.to_string(index=False))
        return

    print(f"Descargando {len(tickers)} tickers en {args.timeframe} (incremental, IBKR → Parquet)…")
    res = update_many(tickers, args.timeframe)

    rows = []
    for r in res:
        rows.append({"ticker": r.ticker, "added_rows": r.added_rows, "last_dt_after": r.last_dt_after, "status": "OK" if r.error is None else f"ERR: {r.error}"})
    df_res = pd.DataFrame(rows).sort_values(["status", "ticker"])
    print(df_res.to_string(index=False))

    # resumen última fecha
    print("\nÚltimo dato tras la actualización:")
    df_last = last_available_table(tickers, args.timeframe)
    print(df_last.to_string(index=False))

if __name__ == "__main__":
    main()
