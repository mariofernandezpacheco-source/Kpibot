# utils/F_download_all_tickers_FOR.py
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

import settings as settings

S = settings.S

from ib_insync import IB, Index, Stock, util

from utils.io_utils import atomic_write_csv, safe_read_csv
from utils.parquet_store import write_ohlcv_parquet


def _normalize_timeframe(tf: str) -> str:
    return str(tf).lower().replace(" ", "")


def _bar_size_for(tf: str) -> str:
    return S.bar_size_by_tf.get(tf, "5 mins")


def _csv_paths(base: Path, tf: str, sym: str) -> tuple[Path | None, Path]:
    """
    Devuelve (existing_path, target_path).
    - existing_path: el primer CSV encontrado en las ubicaciones conocidas
    - target_path: dónde vamos a escribir de ahora en adelante (01_data/<tf>/SYM.csv)
    """
    base = Path(base)
    tf_dir = base / tf
    candidates = [
        tf_dir / f"{sym}.csv",  # canónica por timeframe
        base / f"{sym}.csv",  # plana (tu caso actual)
        base / "csv" / f"{sym}.csv",  # por si tenías otra estructura
    ]
    existing = None
    for p in candidates:
        if p.exists():
            existing = p
            break
    target = tf_dir / f"{sym}.csv"
    return existing, target


def _load_csv_any(path: Path | None) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Carga un CSV si existe; devuelve (df_ordenado, last_dt)."""
    if path is None or not path.exists():
        return pd.DataFrame(), None
    try:
        df = pd.read_csv(path)
        if "date" not in df.columns and "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "date"})
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df, (df["date"].max() if not df.empty else None)
    except Exception:
        return pd.DataFrame(), None


def _merge_and_write_to_target(
    existing_df: pd.DataFrame, new_df: pd.DataFrame, target_path: Path
) -> pd.DataFrame:
    out = pd.concat([existing_df, new_df], ignore_index=True)
    out.drop_duplicates(subset="date", inplace=True)
    out.sort_values("date", inplace=True)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target_path, index=False)
    return out


def _contract_and_feeds(symbol: str):
    """
    Devuelve (contract, lista_de_whatToShow a probar en orden).
    Para VIX (IND@CBOE): TRADES -> BID_ASK -> MIDPOINT
    Resto: TRADES
    """
    sym = symbol.upper().strip()
    if sym in {"VIX", "^VIX"}:
        return Index("VIX", "CBOE"), ["TRADES", "BID_ASK", "MIDPOINT"]
    return Stock(sym, "SMART", "USD"), ["TRADES"]


def _load_existing_csv(path: Path) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    if not path.exists():
        return pd.DataFrame(), None
    try:
        df = safe_read_csv(path)
        if "date" not in df.columns and "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "date"})
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        last_dt = df["date"].max() if not df.empty else None
        return df, last_dt
    except Exception:
        return pd.DataFrame(), None


def persist_ohlcv(df, *, ticker: str, timeframe: str, csv_root: Path):
    from utils.parquet_store import write_ohlcv_parquet

    # CSV (solo si está activado)
    if getattr(S, "storage_write_csv", True):
        csv_root.mkdir(parents=True, exist_ok=True)
        csv_path = csv_root / f"{ticker.upper()}.csv"
        try:
            df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"[{ticker}] ⚠️ Error guardando CSV: {e}")
    # Parquet (si está activado)
    if getattr(S, "storage_write_parquet", True):
        try:
            write_ohlcv_parquet(df, ticker=ticker, timeframe=timeframe, mode="append")
        except Exception as e:
            print(f"[{ticker}] ℹ️ Parquet no escrito ({e})")


def _merge_and_write(csv_path: Path, df_old: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
    """Fusiona histórico + nuevas filas (sin duplicados por 'date'), escribe CSV atómicamente y devuelve el combinado."""
    df_combined = pd.concat([df_old, df_new], ignore_index=True)
    df_combined.drop_duplicates(subset="date", inplace=True)
    df_combined.sort_values("date", inplace=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(df_combined, csv_path, index=False)
    return df_combined


@retry(
    reraise=True,
    stop=stop_after_attempt(S.ib_max_retries),
    wait=wait_exponential_jitter(initial=S.ib_backoff_min_s, max=S.ib_backoff_max_s),
    retry=retry_if_exception_type((Exception,)),
)
async def _fetch_one_async(ib: IB, symbol: str, duration_days: int, bar_size: str) -> pd.DataFrame:
    contract, feeds = _contract_and_feeds(symbol)
    # qualify para resolver conId / primaryExch
    try:
        await ib.qualifyContractsAsync(contract)
    except Exception:
        pass

    duration = f"{int(max(1, duration_days))} D"
    last_err = None

    for what in feeds:
        try:
            bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what,
                useRTH=True,
                formatDate=1,
                keepUpToDate=False,
                timeout=float(S.ib_timeout_s),  # Timeout explícito
            )
            if not bars:
                raise RuntimeError(f"Respuesta vacía ({what})")

            df = util.df(bars)
            if "date" not in df.columns:
                df = df.rename(columns={"time": "date"})
            df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

            # Normaliza OHLCV y, si viene BID_ASK, calculamos midpoint -> close
            if what == "BID_ASK":
                if "bid" in df.columns and "ask" in df.columns:
                    df["close"] = (
                        pd.to_numeric(df["bid"], errors="coerce")
                        + pd.to_numeric(df["ask"], errors="coerce")
                    ) / 2
                for c in ["open", "high", "low"]:
                    if c not in df.columns or df[c].isna().all():
                        df[c] = df["close"]
                if "volume" not in df.columns:
                    df["volume"] = 0

            for c in ["open", "high", "low", "close", "volume"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            df = (
                df[["date", "open", "high", "low", "close", "volume"]]
                .dropna(subset=["date"])
                .sort_values("date")
            )
            if df.empty:
                raise RuntimeError(f"DF vacío tras limpieza ({what})")

            return df.reset_index(drop=True)

        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        f"{symbol}: sin datos históricos válidos tras probar {feeds}. Último error: {last_err}"
    )


async def _bounded_fetch(
    ib: IB, sem: asyncio.Semaphore, symbol: str, duration_days: int, bar_size: str
) -> tuple[str, pd.DataFrame, float, Exception | None]:
    start = time.perf_counter()
    async with sem:
        try:
            df = await _fetch_one_async(ib, symbol, duration_days, bar_size)
            return symbol, df, time.perf_counter() - start, None
        except Exception as e:
            return symbol, pd.DataFrame(), time.perf_counter() - start, e


async def main_async():
    tf = _normalize_timeframe(S.timeframe_default)
    bar_size = _bar_size_for(tf)
    data_root = Path(S.data_path)
    config_folder = Path(S.config_path)
    tickers_path = config_folder / "sp500_tickers.txt"

    if not tickers_path.exists():
        print(f"❌ Archivo de tickers no encontrado: {tickers_path}")
        return

    with tickers_path.open("r", encoding="utf-8") as f:
        base_tickers = [line.strip().upper() for line in f if line.strip()]

    universe = sorted(set(base_tickers) | {"SPY", "VIX"})
    print(
        f"📥 Descargando {len(universe)} símbolos | timeframe={tf} | barSize='{bar_size}' → {data_root}"
    )
    data_root.mkdir(parents=True, exist_ok=True)

    # Calcula ventana de días por símbolo (incremental)
    per_symbol_days = {}
    now_utc = pd.Timestamp.now(tz="UTC")

    existing_paths: dict[str, tuple[Path | None, Path]] = {}

    for sym in universe:
        existing_path, target_path = _csv_paths(data_root, tf, sym)
        existing_paths[sym] = (existing_path, target_path)

        df_old, last_dt = _load_csv_any(existing_path)
        if last_dt is None:
            # primera vez: cap para NO pedir 365 D
            base_need = max(int(getattr(S, "days_of_data", 90)), 90)
            per_symbol_days[sym] = min(int(getattr(S, "ib_initial_days_cap", 120)), base_need)
        else:
            per_symbol_days[sym] = max(1, (now_utc - last_dt).days + 1)

    # Conexión IB asíncrona
    ib = IB()
    print(f"🔌 Conectando a IBKR {S.ib_host}:{S.ib_port} (clientId={S.ib_client_id})…")
    await ib.connectAsync(S.ib_host, S.ib_port, clientId=S.ib_client_id, readonly=True)
    if not ib.isConnected():
        print("❌ No se pudo conectar a IBKR.")
        return

    sem = asyncio.Semaphore(S.ib_max_concurrent)
    tasks = [
        _bounded_fetch(ib, sem, sym, per_symbol_days.get(sym, S.days_of_data), bar_size)
        for sym in universe
    ]

    latencies: list[float] = []
    for fut in asyncio.as_completed(tasks):
        sym, df_new, lat, err = await fut
        existing_path, target_path = existing_paths[sym]

        if err is not None:
            print(f"⚠️ {sym}: error → {err}")
            continue

        df_old, last_dt = _load_csv_any(existing_path)
        if last_dt is not None:
            df_new = df_new[df_new["date"] > last_dt]

        if df_new.empty:
            # Mensaje claro para saber desde dónde ha leído
            loc = str(existing_path) if existing_path else "(sin CSV previo)"
            print(f"— {sym}: sin filas nuevas (leído desde {loc})")
        else:
            if existing_path and existing_path != target_path:
                print(f"↪️ {sym}: detectado CSV en {existing_path}; migrando a {target_path}")

            combined = _merge_and_write_to_target(df_old, df_new, target_path)
            rel = target_path.relative_to(data_root)
            print(f"💾 {sym}: +{len(df_new)} filas → {rel}")

            # (opcional) Parquet:
            try:
                write_ohlcv_parquet(
                    combined.tail(5000), ticker=sym, timeframe=tf, mode="overwrite_partition"
                )
            except Exception as e:
                print(f"[{sym}] ℹ️ Parquet no escrito ({e})")

    if latencies:
        avg = sum(latencies) / len(latencies)
        print(f"⏱️ Latencia media por símbolo: {avg:.2f}s")

    if ib.isConnected():
        print("🔌 Desconectando de IBKR.")
        ib.disconnect()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
